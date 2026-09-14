"""
ablation.py —— 消融实验（Ablation Study）

【这个文件做什么】
  把评测集 + "预测函数"喂进来，算出整体 Recall/MRR/NDCG。

【新手必读 · 什么是消融实验（Ablation）？】
  消融 = 摘除。为了证明"每个组件都有用"，把系统一个个组件摘掉再测：
    Vector Only      只有向量检索
    BM25 Only        只有关键词检索
    Hybrid(RRF)      双路 + RRF 融合
    Hybrid+Rerank    融合后再 Cross-Encoder 精排
  如果摘掉某组件后指标明显下降，就说明它有贡献——这就是"用数据证明设计有效"。
  真正的对比要配合 scripts/run_evaluation.py 对同一评测集跑多组配置。
"""
from app.evaluation.metrics import mrr_at_k, ndcg_at_k, recall_at_k


def evaluate_predictions(dataset, prediction_map, k: int = 5) -> dict:
    """对评测集整体求平均指标。

    参数：
      dataset        评测样本列表（EvaluationSample）
      prediction_map 字典：{query: [按得分降序的 chunk_id]}
      k              只看前 k 名

    返回：
      {"Recall@k": float, "MRR@k": float, "NDCG@k": float, "samples": n}
    """
    rows = []
    for sample in dataset:
        predicted = prediction_map.get(sample.query, [])
        relevant = set(sample.relevant_chunk_ids)
        rows.append(
            {
                "recall": recall_at_k(predicted, relevant, k),
                "mrr": mrr_at_k(predicted, relevant, k),
                "ndcg": ndcg_at_k(predicted, relevant, k),
            }
        )

    n = len(rows)
    return {
        "samples": n,
        f"Recall@{k}": (sum(r["recall"] for r in rows) / n) if n else 0.0,
        f"MRR@{k}": (sum(r["mrr"] for r in rows) / n) if n else 0.0,
        f"NDCG@{k}": (sum(r["ndcg"] for r in rows) / n) if n else 0.0,
    }
