"""
failure_analysis.py —— 失败分析（Failure Analysis）

【这个文件做什么】
  评测跑完之后，把「没做对」的样本挑出来，逐条归类失败原因、给出根因与修复建议。
  这是"学生 Demo"和"工程项目"的分水岭：Demo 只报一个 Recall 数字，
  工程要能回答「哪一类问题为什么失败、下一步改哪里」。

【失败分类学（docs/开发文档.md 的 WP14）】
  KEYWORD_MISS                关键词路（BM25）没召回，语义路（向量）召回了
  SEMANTIC_MISS               语义路没召回，关键词路召回了
  RERANK_ERROR                融合后还在 Top-K 里，却被精排挤出去了
  QUERY_CLASSIFICATION_ERROR  查询被分错了类型，导致用了不合适的检索权重
  POLICY_ERROR                双路都召回了正确 chunk，但融合后仍没进 Top-K（权重问题）
  WRONG_CHUNK_BOUNDARY        双路都没召回 —— 大概率是切块边界把正确内容切散/切没了
  UNKNOWN                     无法用现有规则归类

【诚实边界】
  这是**基于 Trace 的启发式归类**，不是对失败原因的绝对真理。
  它依赖每次检索留下的链路 Trace（BM25/向量/RRF/精排各阶段的候选快照）；
  没有 Trace 时只能退化到最粗的一层判断（成功/失败，不细分根因）。
  每个结论都要能在面试里自圆其说，而不是当"权威诊断"。
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FailureType(str, Enum):
    KEYWORD_MISS = "KEYWORD_MISS"
    SEMANTIC_MISS = "SEMANTIC_MISS"
    RERANK_ERROR = "RERANK_ERROR"
    QUERY_CLASSIFICATION_ERROR = "QUERY_CLASSIFICATION_ERROR"
    POLICY_ERROR = "POLICY_ERROR"
    WRONG_CHUNK_BOUNDARY = "WRONG_CHUNK_BOUNDARY"
    UNKNOWN = "UNKNOWN"


# 每种失败类型对应的「根因 + 修复方向」模板，方便统一措辞、也方便脚本批量生成报告。
_FAILURE_ADVICE: dict[FailureType, tuple[str, str]] = {
    FailureType.KEYWORD_MISS: (
        "精确实体/术语只靠语义检索召不回，BM25 没抓到关键词。",
        "检查 BM25 分词（中文 cjk 二元组）与精确词特征；必要时把该实体纳入 Query 分类的 EXACT 特征。",
    ),
    FailureType.SEMANTIC_MISS: (
        "同义表达/长问句语义检索没召回，向量模型或语义表达是短板。",
        "换更强中文 embedding 模型，或检查该 chunk 是否被切得太碎/太粗。",
    ),
    FailureType.RERANK_ERROR: (
        "融合阶段正确 chunk 还在 Top-K，精排后却被挤下去了。",
        "检查精排模型的域适配；对比 Hybrid 与 Hybrid+Rerank 的 MRR/NDCG，量化精排是否划算。",
    ),
    FailureType.QUERY_CLASSIFICATION_ERROR: (
        "查询被分错类型，导致用了错误的 BM25/向量权重配比。",
        "在 QueryClassifier 里补该查询的规则特征，再用 Failure Analysis 反哺规则。",
    ),
    FailureType.POLICY_ERROR: (
        "双路都召回了正确 chunk，融合后仍没进 Top-K，权重配比不合理。",
        "调整该类 Query 的 Retrieval Policy 权重，并用 Feedback Before/After 实验验证。",
    ),
    FailureType.WRONG_CHUNK_BOUNDARY: (
        "双路都没召回，正确内容大概率被切块边界切散或切没了。",
        "检查 Structure-aware Chunking：表格是否整块保留、是否在句子/表格行中间截断。",
    ),
    FailureType.UNKNOWN: (
        "现有规则无法归类，需要人工看 case。",
        "人工核查该样本的 chunk 内容与标注，再决定是否新增失败类型或修规则。",
    ),
}


@dataclass
class FailureCase:
    """一条失败案例的完整记录（docs/开发文档.md 的 WP14 要求的最小字段集）。"""

    query: str
    expected_chunks: list[str]
    actual_top_k: list[str]
    failure_type: FailureType
    root_cause: str
    fix: str
    trace: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "expected_chunks": self.expected_chunks,
            "actual_top_k": self.actual_top_k,
            "failure_type": self.failure_type.value,
            "root_cause": self.root_cause,
            "fix": self.fix,
            "extra": self.extra,
        }


def _advice(failure_type: FailureType) -> tuple[str, str]:
    root, fix = _FAILURE_ADVICE[failure_type]
    return root, fix


def _first_hit(chunk_ids: set[str], candidates: list[dict[str, Any]]) -> str | None:
    """在候选快照里找第一个命中 chunk_ids 的 chunk_id（快照已按 rank 排序）。"""
    for c in candidates:
        if c.get("chunk_id") in chunk_ids:
            return c.get("chunk_id")
    return None


def classify_failure(
    query: str,
    relevant_chunk_ids: list[str],
    predicted: list[str],
    trace: dict[str, Any] | None = None,
    annotated_type: str | None = None,
) -> FailureCase | None:
    """判断一条样本是否失败；失败则归类根因并返回 FailureCase，成功返回 None。

    参数：
      query              问题原文
      relevant_chunk_ids 标注的相关 chunk
      predicted          系统最终返回的 chunk 序列（按得分降序）
      trace              本次检索的链路 Trace（含各阶段候选快照），可选
      annotated_type     评测集里人工标注的 query_type（用于查分类错误），可选
    """
    relevant = set(relevant_chunk_ids)
    predicted_set = set(predicted)

    # 成功：任一相关 chunk 进了最终 Top-K，无需分析
    if relevant & predicted_set:
        return None

    failure_type = FailureType.UNKNOWN

    if trace:
        bm25_ids = {c.get("chunk_id") for c in trace.get("bm25_candidates", [])}
        vector_ids = {c.get("chunk_id") for c in trace.get("vector_candidates", [])}
        rrf_ids = {c.get("chunk_id") for c in trace.get("rrf_candidates", [])}
        final_ids = {c.get("chunk_id") for c in trace.get("final_candidates", [])}

        hit_bm25 = relevant & bm25_ids
        hit_vector = relevant & vector_ids
        hit_rrf = relevant & rrf_ids

        # 1) 融合后还在、精排后丢了 → RERANK_ERROR
        if hit_rrf and not (relevant & final_ids):
            failure_type = FailureType.RERANK_ERROR
        # 2) 查询被分错类型（标注为 EXACT/NUMERIC，实际被归成 SEMANTIC）
        elif (
            annotated_type in ("EXACT", "NUMERIC")
            and trace.get("query_type") == "SEMANTIC"
        ):
            failure_type = FailureType.QUERY_CLASSIFICATION_ERROR
        # 3) 关键词路召回、语义路没召回 → SEMANTIC_MISS
        elif hit_bm25 and not hit_vector:
            failure_type = FailureType.SEMANTIC_MISS
        # 4) 语义路召回、关键词路没召回 → KEYWORD_MISS
        elif hit_vector and not hit_bm25:
            failure_type = FailureType.KEYWORD_MISS
        # 5) 双路都召回了、融合后仍没进 Top-K → POLICY_ERROR
        elif hit_bm25 and hit_vector:
            failure_type = FailureType.POLICY_ERROR
        # 6) 双路都没召回 → 大概率切块边界问题
        else:
            failure_type = FailureType.WRONG_CHUNK_BOUNDARY

    root_cause, fix = _advice(failure_type)
    return FailureCase(
        query=query,
        expected_chunks=sorted(relevant),
        actual_top_k=list(predicted),
        failure_type=failure_type,
        root_cause=root_cause,
        fix=fix,
        trace=trace,
    )


def summarize_failures(cases: list[FailureCase]) -> dict[str, Any]:
    """把一批失败案例汇总成「失败类型分布 + 逐条明细」。"""
    from collections import Counter

    distribution = Counter(c.failure_type.value for c in cases)
    return {
        "total_failures": len(cases),
        "distribution": dict(distribution),
        "cases": [c.as_dict() for c in cases],
    }
