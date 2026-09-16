"""Failure Analysis 单元测试：验证失败归类规则。"""
from uuid import uuid4

from app.evaluation.failure_analysis import FailureType, classify_failure


def _ids(n):
    return [str(uuid4()) for _ in range(n)]


def test_success_returns_none():
    rel = _ids(2)
    # 最终结果里命中了相关 chunk → 不算失败
    assert classify_failure("q", rel, [rel[0], "x"]) is None


def test_rerank_error():
    rel = _ids(1)
    trace = {
        "rrf_candidates": [{"chunk_id": rel[0], "rank": 1}],
        "final_candidates": [{"chunk_id": "other", "rank": 1}],
        "bm25_candidates": [], "vector_candidates": [],
    }
    case = classify_failure("q", rel, ["other"], trace=trace)
    assert case is not None
    assert case.failure_type == FailureType.RERANK_ERROR


def test_keyword_miss():
    rel = _ids(1)
    trace = {
        "bm25_candidates": [],
        "vector_candidates": [{"chunk_id": rel[0], "rank": 1}],
        "rrf_candidates": [], "final_candidates": [{"chunk_id": "other"}],
    }
    case = classify_failure("q", rel, ["other"], trace=trace)
    assert case.failure_type == FailureType.KEYWORD_MISS


def test_semantic_miss():
    rel = _ids(1)
    trace = {
        "bm25_candidates": [{"chunk_id": rel[0], "rank": 1}],
        "vector_candidates": [],
        "rrf_candidates": [], "final_candidates": [{"chunk_id": "other"}],
    }
    case = classify_failure("q", rel, ["other"], trace=trace)
    assert case.failure_type == FailureType.SEMANTIC_MISS


def test_query_classification_error():
    rel = _ids(1)
    trace = {
        "query_type": "SEMANTIC",
        "bm25_candidates": [], "vector_candidates": [],
        "rrf_candidates": [], "final_candidates": [{"chunk_id": "other"}],
    }
    case = classify_failure("q", rel, ["other"], trace=trace, annotated_type="EXACT")
    assert case.failure_type == FailureType.QUERY_CLASSIFICATION_ERROR


def test_policy_error_when_both_recalled():
    rel = _ids(1)
    trace = {
        "bm25_candidates": [{"chunk_id": rel[0], "rank": 3}],
        "vector_candidates": [{"chunk_id": rel[0], "rank": 5}],
        "rrf_candidates": [{"chunk_id": "other", "rank": 1}],
        "final_candidates": [{"chunk_id": "other"}],
    }
    case = classify_failure("q", rel, ["other"], trace=trace)
    assert case.failure_type == FailureType.POLICY_ERROR


def test_no_trace_falls_back_to_unknown():
    rel = _ids(1)
    case = classify_failure("q", rel, ["other"], trace=None)
    assert case is not None
    assert case.failure_type == FailureType.UNKNOWN
