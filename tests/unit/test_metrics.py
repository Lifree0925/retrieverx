"""test_metrics.py —— 检索指标单元测试（纯计算，不依赖外部服务）。"""
from app.evaluation.metrics import mrr_at_k, ndcg_at_k, recall_at_k


def test_recall_at_k_basic():
    # 5 条相关，前 10 名命中 3 条 → 0.6
    predicted = list(range(20))
    relevant = {0, 3, 7, 100, 200}  # 前 10 名里命中 0,3,7 共 3 条
    assert recall_at_k(predicted, relevant, 10) == 0.6


def test_recall_at_k_empty_relevant():
    # 没有相关标注时，定义返回 0，避免除零
    assert recall_at_k(["a", "b"], [], 5) == 0.0


def test_mrr_first_rank():
    # 第 1 名就命中 → 1.0
    assert mrr_at_k(["a", "b", "c"], ["a"], 5) == 1.0


def test_mrr_second_rank():
    # 第 2 名命中 → 0.5
    assert mrr_at_k(["a", "b", "c"], ["b"], 5) == 0.5


def test_mrr_miss():
    assert mrr_at_k(["a", "b", "c"], ["z"], 5) == 0.0


def test_ndcg_perfect_ranking_is_one():
    # 理想排序下 NDCG = 1.0（数值精度约等于 1）
    assert abs(ndcg_at_k(["a", "b", "c"], ["a", "b", "c"], 3) - 1.0) < 1e-6


def test_ndcg_better_than_random():
    perfect = ndcg_at_k(["a", "b", "c"], ["a", "b", "c"], 3)
    bad = ndcg_at_k(["x", "y", "a"], ["a"], 3)
    assert perfect > bad
