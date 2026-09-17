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


# ----------------------------------------------------------------------
# 7：分组指标必须能算对"非首块"的分组
# ----------------------------------------------------------------------
def test_evaluate_by_split_handles_group_not_starting_at_zero():
    """分组指标的核心陷阱：子集位置 ≠ 原评测集下标。

    踩过的坑：`evaluate_by_split` 用原下标建 sub_map，而 `evaluate_predictions`
    用子集位置算 sample_key。结果是**只有下标从 0 开始且连续的那一组能对上**：
    47 条字面式（下标 0~46）显示 Recall 1.0，16 条改写式（下标 47~62）显示 0.0——
    一个既不报错、又完全捏造出来的"结论"，比指标偏低危险得多。
    """
    from app.evaluation.ablation import evaluate_by_split

    dataset = [
        EvaluationSample(query="字面一", relevant_chunk_ids=["a"]),
        EvaluationSample(query="字面二", relevant_chunk_ids=["a"]),
        EvaluationSample(query="改写一", relevant_chunk_ids=["b"], paraphrased=True),
        EvaluationSample(query="改写二", relevant_chunk_ids=["b"], paraphrased=True),
    ]
    # 四种预测全部正确
    prediction_map = {"#0": ["a"], "#1": ["a"], "#2": ["b"], "#3": ["b"]}

    splits = evaluate_by_split(dataset, prediction_map, k=5)

    assert splits["字面式问题"]["evaluated"] == 2
    assert splits["改写式问题"]["evaluated"] == 2
    assert splits["字面式问题"]["Recall@5"] == pytest.approx(1.0)
    assert splits["改写式问题"]["Recall@5"] == pytest.approx(1.0), (
        "非首块分组恒为 0.0 说明 sub_map 用的是原下标而不是子集位置"
    )


def test_evaluate_by_split_matches_unsplit_metrics():
    """分组指标按样本数加权，必须能还原成整体指标——否则两组口径与整体口径不一致。"""
    from app.evaluation.ablation import evaluate_by_split

    dataset = [
        EvaluationSample(query="a1", relevant_chunk_ids=["x"]),
        EvaluationSample(query="a2", relevant_chunk_ids=["y"]),
        EvaluationSample(query="p1", relevant_chunk_ids=["y"], paraphrased=True),
        EvaluationSample(query="p2", relevant_chunk_ids=["z"], paraphrased=True),
    ]
    # a1 对、a2 错、p1 对、p2 错 → 整体 Recall@5 = 0.5
    prediction_map = {"#0": ["x"], "#1": ["x"], "#2": ["y"], "#3": ["y"]}

    overall = evaluate_predictions(dataset, prediction_map, k=5)
    splits = evaluate_by_split(dataset, prediction_map, k=5)
    assert overall["Recall@5"] == pytest.approx(0.5)

    total = sum(s["evaluated"] for s in splits.values())
    weighted = sum(s["Recall@5"] * s["evaluated"] for s in splits.values()) / total
    assert weighted == pytest.approx(overall["Recall@5"]), (
        "分组加权应等于整体：不相等说明两组里有分组的预测没对上（key 口径不一致）"
    )


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
# 6：长跑评测必须可续跑 —— 被中断不能整轮白跑
# ----------------------------------------------------------------------
def test_load_progress_skips_partial_line(tmp_path):
    """进程被 kill 时最后一行可能只写了一半，不能因此丢掉前面已完成的记录。"""
    from scripts.run_evaluation import _load_progress

    path = tmp_path / "progress.jsonl"
    path.write_text(
        '{"mode": "hybrid_rerank", "key": "#0", "predicted": ["a"], "latency_ms": 10}\n'
        '{"mode": "hybrid_rerank", "key": "#1", "predicted": ["b"], "latency_ms": 20}\n'
        '{"mode": "hybrid_rerank", "key": "#2", "pred',   # ← 写到一半被终止
        encoding="utf-8",
    )
    done, latencies = _load_progress(path, "hybrid_rerank")
    assert set(done) == {"#0", "#1"}, "半行应被丢弃，但前面两条必须保留"
    assert latencies == [10.0, 20.0]


def test_load_progress_ignores_other_modes(tmp_path):
    """同一份进度文件会被 4 个档位共用，续跑时不能把别的档位当成自己已完成。"""
    from scripts.run_evaluation import _load_progress

    path = tmp_path / "progress.jsonl"
    path.write_text(
        '{"mode": "bm25_only", "key": "#0", "predicted": ["a"], "latency_ms": 1}\n'
        '{"mode": "hybrid", "key": "#1", "predicted": ["b"], "latency_ms": 2}\n',
        encoding="utf-8",
    )
    done, _ = _load_progress(path, "hybrid_rerank")
    assert done == {}


def test_run_mode_resume_only_runs_remaining(tmp_path, monkeypatch):
    """续跑的核心承诺：已完成的样本不再重跑，且结果与从头跑一致。"""
    import scripts.run_evaluation as ev

    monkeypatch.setattr(ev, "get_retrieval_service", lambda: object())
    executed: list[str] = []

    def fake_run_one(mode, service, query, top_k):
        executed.append(query)
        return ["a"], 5.0

    monkeypatch.setattr(ev, "_run_one_query", fake_run_one)

    dataset = [
        EvaluationSample(query="q1", relevant_chunk_ids=["a"]),
        EvaluationSample(query="q2", relevant_chunk_ids=["a"]),
        EvaluationSample(query="q3", relevant_chunk_ids=["a"]),
    ]
    progress = tmp_path / "progress.jsonl"
    progress.write_text(
        '{"mode": "bm25_only", "key": "#0", "predicted": ["a"], "latency_ms": 7}\n',
        encoding="utf-8",
    )

    result = ev.run_mode("bm25_only", dataset, 5, progress_path=progress, resume=True)

    assert executed == ["q2", "q3"], "已完成的 #0 不该被重跑"
    assert result["evaluated"] == 3
    assert result["Recall@5"] == pytest.approx(1.0)
    lines = [l for l in progress.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 3, "续跑是追加，不能把旧记录覆盖掉"


def test_run_mode_without_resume_starts_over(tmp_path, monkeypatch):
    """不续跑时必须覆盖旧进度，否则新旧记录混在一起，结果不可解释。"""
    import scripts.run_evaluation as ev

    monkeypatch.setattr(ev, "get_retrieval_service", lambda: object())
    monkeypatch.setattr(ev, "_run_one_query", lambda *a, **kw: (["a"], 5.0))

    dataset = [EvaluationSample(query="q1", relevant_chunk_ids=["a"])]
    progress = tmp_path / "progress.jsonl"
    progress.write_text(
        '{"mode": "bm25_only", "key": "#0", "predicted": ["zzz"], "latency_ms": 1}\n',
        encoding="utf-8",
    )

    ev.run_mode("bm25_only", dataset, 5, progress_path=progress, resume=False)
    lines = [l for l in progress.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1
    assert "zzz" not in lines[0], "不续跑应覆盖，而不是把上一轮的预测留在文件里"


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
