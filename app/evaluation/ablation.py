"""
ablation.py —— 消融实验（Ablation Study）

【这个文件做什么】
  把评测集 + "预测结果"喂进来，算出整体 Recall/MRR/NDCG。

【新手必读 · 什么是消融实验（Ablation）？】
  消融 = 摘除。为了证明"每个组件都有用"，把系统一个个组件摘掉再测：
    Vector Only      只有向量检索
    BM25 Only        只有关键词检索
    Hybrid(RRF)      双路 + RRF 融合
    Hybrid+Rerank    融合后再 Cross-Encoder 精排
  如果摘掉某组件后指标明显下降，就说明它有贡献——这就是"用数据证明设计有效"。
  真正的对比要配合 scripts/run_evaluation.py 对同一评测集跑多组配置。

【口径说明（三个数，别只看一个）】
  返回里同时给出：
    samples       评测集总条数
    evaluated     **真正参与平均**的条数（有人工标注的那些）
    unverifiable  缺人工标注、无法判定的条数
  指标的分母只算 evaluated。原因：如果拿总条数当分母，
  评测集标注补齐/删减一点，指标就跟着漂，看起来像"模型变差了"，
  其实是"标注没填"。**宁可如实报"有 8 条无法判定"，也不要一个会漂的数。**
"""
from app.evaluation.metrics import mrr_at_k, ndcg_at_k, recall_at_k


def sample_key(sample, index: int) -> str:
    """给一条样本算出稳定的 key。

    不能只用 `query` 字符串：评测集里出现两条**完全相同的问题**时，
    后一条的预测会覆盖前一条，两条都按同一份结果算分——静默算错，
    而且指标看起来还挺正常。所以优先用显式 `id`，没有就用位置序号。
    """
    return getattr(sample, "id", "") or f"#{index}"


def evaluate_predictions(dataset, prediction_map, k: int = 5) -> dict:
    """对评测集整体求平均指标。

    参数：
      dataset        评测样本列表（EvaluationSample）
      prediction_map 字典：{sample_key(sample, index): [按得分降序的 chunk_id]}
      k              只看前 k 名

    返回：
      {"samples", "evaluated", "unverifiable",
       "Recall@k", "MRR@k", "NDCG@k"}
    """
    rows = []
    unverifiable = 0

    for index, sample in enumerate(dataset):
        predicted = prediction_map.get(sample_key(sample, index), [])
        relevant = set(sample.relevant_chunk_ids)
        if not relevant:
            # 没有人标注过这条问题的正确答案 → 无法判定。
            # 既不算通过也不算失败，从分母里剔除（否则等于把它当 0 分）。
            unverifiable += 1
            continue
        rows.append(
            {
                "recall": recall_at_k(predicted, relevant, k),
                "mrr": mrr_at_k(predicted, relevant, k),
                "ndcg": ndcg_at_k(predicted, relevant, k),
            }
        )

    n = len(rows)
    return {
        "samples": len(dataset),
        "evaluated": n,
        "unverifiable": unverifiable,
        f"Recall@{k}": (sum(r["recall"] for r in rows) / n) if n else 0.0,
        f"MRR@{k}": (sum(r["mrr"] for r in rows) / n) if n else 0.0,
        f"NDCG@{k}": (sum(r["ndcg"] for r in rows) / n) if n else 0.0,
    }


def evaluate_by_split(dataset, prediction_map, k: int = 5, split_of=None) -> dict:
    """按分组分别算指标，返回 {组名: 同 evaluate_predictions 的指标字典}。

    【为什么需要分组】一张总平均表会把两类**性质完全不同**的问题混在一起：
      · 字面式问题 —— 问法与语料原文关键词高度重叠，BM25 单靠关键词就能命中；
      · 改写式问题 —— 问法换了说法（"口令"问成"密码"），只有语义检索才稳。
    混在一起看时，前者会把平均分拉得很高，于是"融合/精排有没有用"这件事
    被天花板效应压平，消融实验就看不出差别。
    分开看之后，两组之间的**差距**才是"语义检索在起什么作用"的直接证据。

    split_of：(sample) -> str。默认按 `paraphrased` 字段分成 改写式/字面式。
    """
    if split_of is None:
        def split_of(sample):
            return "改写式问题" if getattr(sample, "paraphrased", False) else "字面式问题"

    groups: dict[str, list] = {}
    for index, sample in enumerate(dataset):
        groups.setdefault(split_of(sample), []).append((index, sample))

    out: dict[str, dict] = {}
    for name, members in groups.items():
        sub_dataset = [s for _i, s in members]
        # 【必须按子集重新编号】members 里存的是样本在**原评测集**里的下标，
        # 而 evaluate_predictions 会拿 `enumerate(sub_dataset)` 的位置去算 sample_key。
        # 如果这里沿用原下标建 sub_map，只有"下标恰好从 0 开始且连续"的那一组能对上，
        # 其余分组会全部查不到预测 → 指标恒为 0.0，**而且不报错**，
        # 表格里会安静地出现一个 0，看起来像"这组问题全错"。
        # 实测踩到过：47 条字面式（下标 0~46）显示 Recall 1.0，
        # 16 条改写式（下标 47~62）显示 0.0——不是模型差，是指标算错了。
        sub_map = {}
        for position, (original_index, sample) in enumerate(members):
            sub_map[sample_key(sample, position)] = prediction_map.get(
                sample_key(sample, original_index), []
            )
        out[name] = evaluate_predictions(sub_dataset, sub_map, k=k)
    return out
