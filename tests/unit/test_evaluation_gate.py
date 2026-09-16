"""评测闭环与降级路径的防线测试。

这几条锁的都是"**看起来在正常工作、其实结论是假的**"这类问题：

1. 占位符评测集 —— 能跑完、能打印指标，只是全是 0。
2. 标注指向不存在的 chunk_id —— 语料改过之后，旧标注静默失效，表现同样是 Recall=0。
3. 缺人工标注的样本被当成 0 分计入分母 —— 指标随标注完整度漂移。
4. 用 query 字符串对齐预测 —— 重复问题互相覆盖，两条按同一份结果算分。
5. Redis 挂掉导致整个 /search 不可用 —— 可选增强被升格成硬依赖。
"""
import pytest

from app.config import settings
from app.core.pipeline import RetrievalService
from app.core.query_classifier import QueryType
from app.core.retrieval_policy import RetrievalPolicy
from app.evaluation.ablation import evaluate_predictions, sample_key
from app.evaluation.dataset import EvaluationSample, validate_dataset


# ----------------------------------------------------------------------
# 1~3：评测集前置校验
# ----------------------------------------------------------------------
def test_validate_dataset_rejects_placeholder_ids():
    samples = [EvaluationSample(query="价格是多少",
                                relevant_chunk_ids=["REPLACE_WITH_REAL_CHUNK_ID"])]
    errors, _warnings = validate_dataset(samples)
    assert errors, "占位符必须被拦住：它意味着评测集从没真正标注过"
    assert "占位符" in errors[0]


def test_validate_dataset_rejects_ids_missing_from_corpus():
    samples = [EvaluationSample(query="价格是多少", relevant_chunk_ids=["not-a-real-id"])]
    errors, _warnings = validate_dataset(samples, chunk_universe={"real-id-1"})
    assert errors
    assert "不存在" in errors[0]


def test_validate_dataset_accepts_ids_present_in_corpus():
    samples = [EvaluationSample(query="价格是多少", relevant_chunk_ids=["real-id-1"])]
    errors, warnings = validate_dataset(samples, chunk_universe={"real-id-1"})
    assert errors == []
    assert warnings == []


def test_validate_dataset_warns_instead_of_failing_on_missing_annotation():
    """缺标注只是"无法判定"，不该让整个评测跑不起来——但必须被指出来。"""
    samples = [
        EvaluationSample(query="有标注", relevant_chunk_ids=["real-id-1"]),
        EvaluationSample(query="没标注", relevant_chunk_ids=[]),
    ]
    errors, warnings = validate_dataset(samples, chunk_universe={"real-id-1"})
    assert errors == []
    assert any("没有标注" in w for w in warnings)


# ----------------------------------------------------------------------
# 4：指标口径 —— 无法判定的样本必须从分母里剔除
# ----------------------------------------------------------------------
def test_evaluate_predictions_excludes_unverifiable_from_denominator():
    dataset = [
        EvaluationSample(query="q1", relevant_chunk_ids=["a"]),
        EvaluationSample(query="q2", relevant_chunk_ids=[]),   # 没标注 → 无法判定
    ]
    # q1 命中，q2 的预测即使完全对也不能算分
    result = evaluate_predictions(dataset, {"#0": ["a"], "#1": ["b"]}, k=5)
    assert result["samples"] == 2
    assert result["evaluated"] == 1
    assert result["unverifiable"] == 1
    assert result["Recall@5"] == pytest.approx(1.0), (
        "分母如果算上无法判定的样本，这里会变成 0.5 —— 指标会随标注完整度漂移"
    )


def test_predictions_align_by_position_when_queries_repeat():
    """评测集里出现两条完全相同的问题时，不能互相覆盖。"""
    dataset = [
        EvaluationSample(query="同样的问题", relevant_chunk_ids=["a"]),
        EvaluationSample(query="同样的问题", relevant_chunk_ids=["b"]),
    ]
    assert sample_key(dataset[0], 0) != sample_key(dataset[1], 1)

    # 第一条命中、第二条没命中：按位置对齐时 Recall 应为 0.5
    result = evaluate_predictions(dataset, {"#0": ["a"], "#1": ["a"]}, k=5)
    assert result["Recall@5"] == pytest.approx(0.5)


def test_sample_key_prefers_explicit_id():
    sample = EvaluationSample(query="q", relevant_chunk_ids=["a"], id="my-id")
    assert sample_key(sample, 7) == "my-id"


# ----------------------------------------------------------------------
# 5：检索策略权重真的读 .env（曾经是"文档里写了、代码里没人读"的死配置）
# ----------------------------------------------------------------------
def test_policy_default_weights_match_documented_values():
    policy = RetrievalPolicy(base_bm25=0.5, base_vector=0.5)
    assert policy.get(QueryType.EXACT).bm25 == pytest.approx(0.7)
    assert policy.get(QueryType.NUMERIC).bm25 == pytest.approx(0.7)
    assert policy.get(QueryType.SEMANTIC).vector == pytest.approx(0.7)
    assert policy.get(QueryType.MIXED).bm25 == pytest.approx(0.5)


def test_policy_weights_follow_settings_base():
    """改基准权重必须真的改变各类型权重——否则 .env 里的开关又是假的。"""
    heavy_bm25 = RetrievalPolicy(base_bm25=0.8, base_vector=0.2)
    default = RetrievalPolicy(base_bm25=0.5, base_vector=0.5)
    assert heavy_bm25.get(QueryType.MIXED).bm25 > default.get(QueryType.MIXED).bm25
    assert heavy_bm25.get(QueryType.EXACT).bm25 > default.get(QueryType.EXACT).bm25


def test_policy_weights_are_normalized():
    policy = RetrievalPolicy(base_bm25=0.9, base_vector=0.9)
    for qtype in QueryType:
        item = policy.get(qtype)
        assert item.bm25 + item.vector == pytest.approx(1.0)


# ----------------------------------------------------------------------
# Redis 不可用时的降级：必须"照常返回 + 标记降级"，而不是 503
# ----------------------------------------------------------------------
class _FakeSearch:
    def search(self, query, top_k, document_id=None):
        return []


class _FakeClassifier:
    def classify(self, query):
        return QueryType.MIXED


class _BrokenFeedbackStore:
    """模拟 Redis 挂了：任何调用都抛连接错误。"""

    def get_correction(self, query):
        raise ConnectionError("Error 10061 connecting to localhost:6379")


def _make_service(feedback_store) -> RetrievalService:
    return RetrievalService(
        bm25=_FakeSearch(),
        vector=_FakeSearch(),
        classifier=_FakeClassifier(),
        policy=RetrievalPolicy(),
        feedback_store=feedback_store,
        reranker=None,
        reranker_provider=None,
        trace_store=None,
    )


def test_correction_cache_failure_degrades_instead_of_raising():
    service = _make_service(_BrokenFeedbackStore())

    response = service.run_pipeline("华北区的销售额", top_k=5, use_rerank=False)

    assert response.candidates == []
    assert response.correction_available is False, (
        "降级必须可见：correction_available=false，否则调用方分不清"
        "『这次没有历史更正』和『缓存坏了』"
    )
    assert response.used_correction is False


def test_correction_cache_success_marks_available():
    class _OkStore:
        def get_correction(self, query):
            return None

    response = _make_service(_OkStore()).run_pipeline("华北区的销售额", use_rerank=False)
    assert response.correction_available is True


# ----------------------------------------------------------------------
# 配置自查：settings 里不该再有"声明了但没人读"的字段
# ----------------------------------------------------------------------
def test_no_dead_settings_fields():
    """对所有 settings 字段做一次全项目 grep，抓"文档里写了、代码里没人读"的开关。

    这不是洁癖：`BM25_WEIGHT` / `VECTOR_WEIGHT` 曾经就属于这一类——
    .env.example 在引导用户调它们，而真实权重硬编码在 RetrievalPolicy 里，改了毫无作用。
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    sources: list[str] = []
    for folder in ("app", "scripts"):
        for path in (root / folder).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            sources.append(path.read_text(encoding="utf-8", errors="ignore"))
    blob = "\n".join(sources)

    dead = [
        name
        for name in type(settings).model_fields
        # 字段自身在 config.py 里的定义不算"读取"
        if not re.search(rf"settings\.{re.escape(name)}\b", blob)
    ]
    assert not dead, (
        f"这些配置项在 app/ 与 scripts/ 里都没有读取点：{dead}。"
        "要么真正接上，要么删掉——留着会让人以为改 .env 有效。"
    )
