"""
reranker.py —— Cross-Encoder 精排（重排）

【这个文件做什么】
  对融合后的候选集做一次更精细的排序。

【新手必读 · 面试必考：Bi-Encoder vs Cross-Encoder】
  - Bi-Encoder（双编码器）：查询和文档各自独立编码成向量再比相似度。
      快、可以预先算好所有文档向量，适合大规模召回（Day5 的向量检索就是这个）。
  - Cross-Encoder（交叉编码器）：把"查询 + 文档"拼成一句话送进同一个模型，直接输出相关分。
      更准（模型能看到两者 token 间的交互），但要为每一对重新跑一次模型，慢，
      所以只能作用于小候选集（Top-20），不能作用于全库（Top-1000+）。

【本项目配置】
  默认使用本地模型 BAAI/bge-reranker-base（sentence-transformers 的 CrossEncoder）。
  首次运行会下载模型，之后在本地加载。国内建议在 .env 里配 HF_ENDPOINT=https://hf-mirror.com。

【健壮性设计 —— 这里踩过一个真实的坑，务必看懂】
  早期版本在 Reranker.__init__ 里直接 CrossEncoder(model_name)，有两个致命问题：
    1. 模型约 1GB，要联网下载。若模型未缓存且网络不通，构造会【长时间阻塞】，
       而构造发生在 /search 请求线程内（Depends 装配阶段）→ 整个请求卡死，
       前端最终只看到"超时"，且真正的检索（BM25/向量）一步都没跑到。
    2. sentence_transformers 的 import 本身就要几十秒，放在模块顶层会让服务启动极慢。
  现在的做法：
    - import 延迟到真正加载模型时（函数内 import）；
    - 模型加载放到【后台守护线程】，请求侧只 wait 最多 rerank_timeout_seconds；
    - 超时/失败一律"降级"为直接用融合结果，服务绝不因为精排故障而挂掉；
    - 超时后后台下载仍在继续，下载完的后续请求会自动用上模型。
"""
import threading
import time

from app.config import settings
from app.models.retrieval import Candidate
from app.utils.logging import log_json


class Reranker:
    def __init__(self):
        # 注意：这里【不】加载模型，只准备状态。模型在第一次真正精排时才加载。
        self.model = None
        self._lock = threading.Lock()
        self._loader_started = False
        self._load_done = threading.Event()
        self._load_error: BaseException | None = None
        self._loaded_from: str | None = None     # "cache" / "download"，用于日志
        self._load_note: str | None = None
        self._load_ms: float | None = None

    # ------------------------------------------------------------------
    # 模型加载（后台线程 + 超时）
    # ------------------------------------------------------------------
    def _load_model_worker(self) -> None:
        """在后台线程里真正加载模型；成功/失败都会 set _load_done。

        用守护线程（daemon=True）的原因：如果网络不通导致下载一直挂着，
        进程退出时不会被这个线程拖住。
        """
        started = time.perf_counter()
        try:
            # 延迟 import：sentence_transformers 首次导入实测要 15~40 秒，
            # 放在模块顶层会让 uvicorn 每次启动/热重载都白等。
            from sentence_transformers import CrossEncoder

            kwargs: dict = {"max_length": settings.reranker_max_length}
            if settings.reranker_device:
                kwargs["device"] = settings.reranker_device

            # ① 先只读本地缓存。
            #    模型已经下好时这一步实测只要 3.5 秒，而且完全不联网。
            #    相比之下联网加载会先做 etag 校验/重定向，实测 70 秒以上还可能失败。
            try:
                self.model = CrossEncoder(
                    settings.reranker_model, local_files_only=True, **kwargs
                )
                self._loaded_from = "cache"
                return
            except Exception as cache_exc:  # noqa: BLE001 —— 缓存没命中就往下走去下载
                self._load_note = (
                    f"本地缓存未命中({type(cache_exc).__name__})，改为联网下载"
                )

            # ② 缓存里没有，才允许联网（走 .env 里配的 HF_ENDPOINT 镜像）
            self.model = CrossEncoder(settings.reranker_model, **kwargs)
            self._loaded_from = "download"
        except BaseException as exc:  # noqa: BLE001 —— 加载失败要降级，不能往上炸
            self._load_error = exc
        finally:
            self._load_ms = (time.perf_counter() - started) * 1000
            self._load_done.set()

    def _ensure_model(self):
        """返回已加载的模型；若在超时时间内没加载好，返回 None（调用方降级）。

        设计要点：
          - 加载线程只启动一次（_loader_started 保证），不会每个请求都重启下载；
          - 每个请求最多等 rerank_load_timeout_seconds，等不到就走融合结果；
          - 后台线程继续跑，加载完成后后续请求 wait 会立刻返回 → 自动恢复精排。
        """
        if self.model is not None:
            return self.model

        with self._lock:
            if not self._loader_started:
                self._loader_started = True
                threading.Thread(
                    target=self._load_model_worker,
                    name="reranker-loader",
                    daemon=True,
                ).start()

        timeout = settings.rerank_load_timeout_seconds
        if not self._load_done.wait(timeout=timeout):
            log_json(
                "reranker_load_timeout",
                model=settings.reranker_model,
                timeout_seconds=timeout,
                hint="模型仍在后台加载，本次降级为融合结果；"
                     "先跑 scripts/download_models.py 预下载可消除该等待",
            )
            return None

        if self._load_error is not None:
            log_json(
                "reranker_load_failed",
                model=settings.reranker_model,
                error=f"{type(self._load_error).__name__}: {self._load_error}",
                elapsed_ms=round(self._load_ms or 0, 1),
                hint="精排已降级；可设 RERANK_ENABLE=false 关闭，或配 HF_ENDPOINT 镜像",
            )
            return None

        log_json(
            "reranker_load_ok",
            model=settings.reranker_model,
            source=self._loaded_from,
            elapsed_ms=round(self._load_ms or 0, 1),
            note=self._load_note,
        )
        return self.model

    # ------------------------------------------------------------------
    # 打分（同样带超时）
    # ------------------------------------------------------------------
    @staticmethod
    def _predict_with_timeout(model, pairs, timeout: float):
        """在子线程里跑 model.predict，最多等 timeout 秒，超时抛 TimeoutError。

        为什么不用 ThreadPoolExecutor？它的线程是非守护的，会在解释器退出时被 join；
        万一打分卡住，Ctrl+C 退不掉。守护线程 + Event 更省心。
        """
        box: dict = {}
        done = threading.Event()

        def work():
            try:
                box["scores"] = model.predict(pairs)
            except BaseException as exc:  # noqa: BLE001
                box["error"] = exc
            finally:
                done.set()

        threading.Thread(target=work, name="reranker-score", daemon=True).start()
        if not done.wait(timeout=timeout):
            raise TimeoutError(f"精排打分超过 {timeout}s 未返回")
        if "error" in box:
            raise box["error"]
        return box["scores"]

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def rerank(self, query: str, candidates: list[Candidate], top_k: int):
        """对候选集精排，返回 (重排后的前 top_k 条, 耗时毫秒)。

        任何异常/超时都降级为 (原候选前 top_k 条, 耗时)——即"降级到融合结果"。
        """
        if not candidates:
            return [], 0.0

        started = time.perf_counter()

        def elapsed_ms() -> float:
            return (time.perf_counter() - started) * 1000

        model = self._ensure_model()
        if model is None:
            # 模型还没就绪（正在后台下载 / 加载失败）→ 本次降级
            return candidates[:top_k], elapsed_ms()

        try:
            # CrossEncoder 一次吃一批 (query, doc) 对，返回每对的相关性分数
            scores = self._predict_with_timeout(
                model,
                [(query, x.content) for x in candidates],
                settings.rerank_timeout_seconds,
            )
            ranked = [
                x.model_copy(update={"rerank_score": float(score)})
                for x, score in zip(candidates, scores)
            ]
            ranked.sort(key=lambda x: x.rerank_score or 0.0, reverse=True)
            return ranked[:top_k], elapsed_ms()
        except Exception as exc:  # noqa: BLE001 —— 降级是设计预期，不是 bug
            log_json(
                "reranker_score_failed",
                error=f"{type(exc).__name__}: {exc}",
                hint="本次降级为融合结果",
            )
            return candidates[:top_k], elapsed_ms()
