"""
pipeline.py —— 检索主链路（Retrieval Pipeline / RetrievalService）

【这个文件做什么】
  把"查询分类 → 更正缓存 → 双路检索 → RRF 融合 → 精排"串成一条完整流水线，
  并记录每阶段耗时、所用策略版本，返回结构化的 RetrievalResponse。

【新手必读 · 面试点：一次请求经历了什么】
    用户问题
      ↓ ① 更正缓存：历史上有用户给过更好的写法？有就用它重搜（used_correction=true）
      ↓ ② 查询分类：EXACT/SEMANTIC/MIXED/NUMERIC
      ↓ ③ 取策略：该类型当前的 bm25/vector 权重（带版本号）
      ↓ ④ 双路并行召回：BM25 检索 + 向量检索（各召回 recall_top_k 条）
      ↓ ⑤ RRF 融合：按排名融合成一份
      ↓ ⑥ Cross-Encoder 精排（可选，失败自动降级）
      ↓
    返回：候选列表 + 总耗时 + 策略快照

【健壮性设计】
  检索中间件可能没启动（ES/Qdrant 没起来）。对**真正必需的**依赖（Embedding / ES / Qdrant），
  本类不做重试/魔法，而是让异常自然抛出，由 API 层的统一异常处理转成清晰的 503 提示
  "先启动依赖服务"。

  但对**可选增强**必须区分对待——更正缓存（Redis）就属于这一类：
    · 它是"命中历史更正就用更好的查询词重搜"，命中不了只是效果差一点，不是不能检索；
    · 少了它，ES/Qdrant 那条主链路依然完整。
  所以 Redis 不可用时降级为"本次没有更正缓存"，并把 correction_available=false 放进
  响应与 Trace —— 让降级**可见**。早期版本没有这层保护，Redis 一抖就会 503 整个 /search，
  与 trace 早就做了静默降级的设计自相矛盾，也让排查方向被带偏
  （调用方以为"检索服务挂了"，其实坏的只是缓存）。
"""
import time
from uuid import uuid4

from app.config import settings
from app.core.fusion import reciprocal_rank_fusion
from app.core.trace import TraceStore, candidates_snapshot
from app.models.retrieval import RetrievalPolicySnapshot, RetrievalResponse
from app.utils.logging import log_json


class RetrievalService:
    def __init__(
        self,
        bm25,
        vector,
        classifier,
        policy,
        feedback_store,
        reranker=None,
        reranker_provider=None,
        trace_store=None,
    ):
        """依赖注入：把各组件从外面传进来，便于测试时替换成假对象。

        reranker           直接给一个精排器实例（测试常用）
        reranker_provider  给一个"返回精排器的函数"，用到时才调用。
        trace_store        Retrieval Trace 存储（可为 None：不落 Trace，仅打日志）

        【为什么要有 reranker_provider 这个看起来多余的东西？】
          精排模型有 1GB，加载很贵。如果在装配 RetrievalService 时就把 reranker
          造出来，那么"用户在请求里传 rerank=false"根本省不掉这次加载——
          服务早就把模型造好了。用 provider（延迟调用）之后，
          只有真正走进精排分支时才会去创建它。
        """
        self.bm25 = bm25
        self.vector = vector
        self.classifier = classifier
        self.policy = policy
        self.feedback_store = feedback_store
        self.reranker = reranker                    # 可为 None
        self.reranker_provider = reranker_provider  # 可为 None
        self.trace_store = trace_store              # 可为 None（不落 Trace）

    def _resolve_reranker(self):
        """真正需要精排时才去拿精排器（可能是懒加载/懒创建的）。"""
        if self.reranker is not None:
            return self.reranker
        if self.reranker_provider is None:
            return None
        return self.reranker_provider()

    def _safe_correction(self, query: str, request_id: str) -> tuple[str | None, bool]:
        """取更正缓存。返回 (改写后的查询 or None, 缓存是否可用)。

        Redis 属于**可选增强**：连不上时返回 (None, False)，而不是让异常冒到 API 层。
        否则"连不上 Redis"会被统一异常处理映射成 503，
        表现为"整个检索服务不可用"——而 ES / Qdrant 其实一直是好的，
        调用方与排查方向都会被带偏。
        """
        try:
            return self.feedback_store.get_correction(query), True
        except Exception as exc:  # noqa: BLE001 —— 缓存不可用属于可降级情形，不该中断检索
            log_json(
                "correction_cache_unavailable",
                request_id=request_id,
                error=type(exc).__name__,
                detail=str(exc)[:200],
            )
            return None, False

    def run_pipeline(
        self,
        query: str,
        top_k: int = 5,
        use_rerank: bool = True,
        document_id=None,
    ) -> RetrievalResponse:
        """执行一次完整检索。

        参数：
          query        用户问题
          top_k        最终返回条数
          use_rerank   本次是否尝试精排（接口层允许用户开关）
          document_id  限定检索范围：给了就只在这一份文档内检索，不传则跨全部文档
        """
        started = time.perf_counter()
        request_id = str(uuid4())
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")

        # ---------- ① 更正缓存：有用户历史更正则用它重搜 ----------
        # 可用性降级：Redis 挂了只是"这次没有更正缓存"，不中断检索（详见文件头【健壮性设计】）
        correction, correction_available = self._safe_correction(query, request_id)
        used_correction = correction is not None
        search_query = correction or query

        # ---------- ② 查询分类 ----------
        query_type = self.classifier.classify(search_query)

        # ---------- ③ 取当前策略 ----------
        policy = self.policy.get(query_type)

        # ---------- ④ 双路检索（各召回一批候选） ----------
        # document_id 会同时下推到 BM25 与向量两路，保证"限定文档"是真的限定：
        # 如果只在融合后过滤，两路各召回 20 条可能全被过滤光，结果为空。
        t = time.perf_counter()
        bm25_results = self.bm25.search(search_query, settings.recall_top_k, document_id)
        bm25_latency = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        vector_results = self.vector.search(search_query, settings.recall_top_k, document_id)
        vector_latency = (time.perf_counter() - t) * 1000

        # ---------- ⑤ RRF 融合（权重来自当前策略） ----------
        t = time.perf_counter()
        fused = reciprocal_rank_fusion(
            [bm25_results, vector_results],
            k=settings.rrf_k,
            weights=[policy.bm25, policy.vector],
        )
        fusion_latency = (time.perf_counter() - t) * 1000
        # 融合后的完整排序快照（精排前），用于 Trace / Failure Analysis 判断
        # "正确结果在融合层就丢了"还是"被精排挤下去了"。
        rrf_candidates = candidates_snapshot(fused)

        # ---------- ⑥ Cross-Encoder 精排（可降级） ----------
        rerank_latency = 0.0
        rerank_applied = False
        reranker = None
        if use_rerank and settings.rerank_enable:
            # 【关键】只有真的要走精排，才去创建/加载模型。
            # 请求里传 rerank=false，或 .env 里 RERANK_ENABLE=false，都在这里直接跳过。
            reranker = self._resolve_reranker()

        if reranker is not None:
            fused, rerank_latency = reranker.rerank(search_query, fused, top_k)
            # 模型没就绪时 reranker 会原样返回候选，rerank_score 全是 None。
            # 用这个判断"本次是否真的精排了"，方便前端和日志看出降级。
            rerank_applied = any(x.rerank_score is not None for x in fused)
        else:
            fused = fused[:top_k]

        total_latency = (time.perf_counter() - started) * 1000

        # ---------- 结构化日志：一次请求一个 JSON 行 ----------
        log_json(
            "retrieval_trace",
            request_id=request_id,
            query_type=query_type.value,
            retrieval_policy={
                "bm25": policy.bm25,
                "vector": policy.vector,
                "version": policy.version,
            },
            bm25_latency_ms=round(bm25_latency, 2),
            vector_latency_ms=round(vector_latency, 2),
            fusion_latency_ms=round(fusion_latency, 2),
            reranker_latency_ms=round(rerank_latency, 2),
            rerank_applied=rerank_applied,
            total_latency_ms=round(total_latency, 2),
            used_correction=used_correction,
            correction_available=correction_available,
        )

        # ---------- 落 Retrieval Trace（供 /trace/{request_id} 与 Debugger 回查） ----------
        # 即便 trace_store 未装配或 Redis 不可用，也不影响检索主链路返回。
        if self.trace_store is not None:
            self.trace_store.save(
                {
                    "request_id": request_id,
                    "query": query,
                    "search_query": search_query,
                    "query_type": query_type.value,
                    "used_correction": used_correction,
                    "correction_available": correction_available,
                    "policy": {
                        "bm25": policy.bm25,
                        "vector": policy.vector,
                        "version": policy.version,
                    },
                    "bm25_candidates": candidates_snapshot(bm25_results),
                    "vector_candidates": candidates_snapshot(vector_results),
                    "rrf_candidates": rrf_candidates,
                    "final_candidates": candidates_snapshot(fused),
                    "rerank_applied": rerank_applied,
                    "latency_breakdown": {
                        "bm25_ms": round(bm25_latency, 2),
                        "vector_ms": round(vector_latency, 2),
                        "fusion_ms": round(fusion_latency, 2),
                        "reranker_ms": round(rerank_latency, 2),
                    },
                    "total_latency_ms": round(total_latency, 2),
                }
            )

        return RetrievalResponse(
            request_id=request_id,
            query=query,
            candidates=fused,
            latency_ms=total_latency,
            retrieval_policy=RetrievalPolicySnapshot(
                bm25=policy.bm25,
                vector=policy.vector,
                version=policy.version,
            ),
            used_correction=used_correction,
            rerank_applied=rerank_applied,
            correction_available=correction_available,
        )
