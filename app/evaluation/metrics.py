"""
metrics.py —— 检索评测指标（Recall@K / MRR@K / NDCG@K）

【新手必读 · 三个指标分别回答什么问题】
  Recall@K    "真正相关的内容，前 K 条里捞到了几条？"——关注会不会漏（召回率）
  MRR@K       "第一条相关结果排第几？"——关注第一个对的有多靠前（用户体感）
  NDCG@K      "排序质量好不好？"——位置越靠前越该给高分，综合衡量排序优劣

【评测输入约定】
  predicted_ids   系统返回的候选 chunk id 列表（按得分从高到低）
  relevant_ids    人工标注的"真正相关"的 chunk id 列表
  k               只看前 k 名
"""
import math


def recall_at_k(predicted_ids: list, relevant_ids: list, k: int) -> float:
    """召回率：前 k 条中相关个数 / 相关总数。

    例：5 条相关，前 10 名命中 3 条 → Recall@10 = 3/5 = 0.6
    """
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    hits = len(set(predicted_ids[:k]) & relevant)
    return hits / len(relevant)


def mrr_at_k(predicted_ids: list, relevant_ids: list, k: int) -> float:
    """平均倒数排名：第一条相关结果的 1/rank。

    例：第 2 名就命中 → MRR = 1/2 = 0.5；前 k 条都没命中 → 0
    """
    relevant = set(relevant_ids)
    for rank, chunk_id in enumerate(predicted_ids[:k], start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(predicted_ids: list, relevant_ids: list, k: int) -> float:
    """归一化折损累计增益。

    直觉：
      - DCG：把每条结果的"相关分(1/0)"除以 log2(位置+1)，越靠前权重越大，再求和；
      - IDCG：最理想排序下的 DCG；
      - NDCG = DCG / IDCG，把值归一化到 0~1。
    """
    relevant = set(relevant_ids)
    gains = [1.0 if x in relevant else 0.0 for x in predicted_ids[:k]]

    dcg = sum(g / math.log2(rank + 1) for rank, g in enumerate(gains, start=1))

    # 理想情况：所有相关文档都排在最前面
    ideal_count = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))

    return dcg / idcg if idcg else 0.0
