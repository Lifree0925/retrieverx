"""
fusion.py —— RRF 结果融合（Reciprocal Rank Fusion，倒数排名融合）

【这个文件做什么】
  把多路检索结果（BM25 的一列 + 向量的一列）合并成一个统一排序。

【新手必读 · 面试必考：为什么用 RRF 而不是直接加权分数？】
  直接把两路分数相加没有意义：
    - BM25 的分数与向量的余弦相似度，量纲不同、分布不同、取值范围不同；
    - 你无法回答"BM25 的 5.2 分 vs 向量的 0.87 分谁更相关"。
  RRF 的做法是【只看排名，不看原始分】：
    score(c) = Σ_retriever  w / (k + rank_retriever(c))
  即在每路结果里，candidate 排第几，就给它 1/(k+rank) 的贡献；排第 1 贡献最大。
  常数 k 用来压平"第一名和第二名的差距"，避免第一名一枝独秀，常取 60。

【公式直觉】
  rank=1 → 1/(60+1)≈0.0164
  rank=2 → 1/(60+2)≈0.0161
  rank=20 → 1/(60+20)=0.0125
  可见 RRF 对排名差异不敏感、很平滑，这正是它鲁棒的原因。
"""
from collections import defaultdict

from app.models.retrieval import Candidate


def reciprocal_rank_fusion(
    result_lists: list[list[Candidate]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[Candidate]:
    """把若干路 Candidate 列表融合为一个按融合得分降序的列表。

    参数：
      result_lists  每路检索的结果，例如 [bm25_results, vector_results]
      k             RRF 常数，默认 60
      weights       每路检索的权重（Retrieval Policy 用），默认全 1

    返回：
      融合后的 Candidate 列表（每个 Candidate 保留最先出现的那条，累加融合分）
    """
    if not result_lists:
        return []
    weights = weights or [1.0] * len(result_lists)
    assert len(weights) == len(result_lists), "weights 长度必须与结果路数一致"

    merged: dict[str, Candidate] = {}              # chunk_id -> 首个出现的 Candidate
    scores: dict[str, float] = defaultdict(float)  # chunk_id -> 累计 RRF 分数
    methods: dict[str, set] = defaultdict(set)     # chunk_id -> 命中的方法集合

    for i, results in enumerate(result_lists):
        weight = weights[i]
        for rank, candidate in enumerate(results, 1):  # rank 从 1 开始
            key = str(candidate.chunk_id)
            merged.setdefault(key, candidate)           # 保留第一个看到的完整信息
            scores[key] += weight / (k + rank)          # RRF 核心公式
            methods[key].add(candidate.retrieval_method)

    output = [
        candidate.model_copy(
            update={
                "fusion_score": scores[key],
                # 例如同时被 bm25 与 vector 命中 → "bm25+vector"
                "retrieval_method": "+".join(sorted(methods[key])),
            }
        )
        for key, candidate in merged.items()
    ]
    output.sort(key=lambda x: x.fusion_score, reverse=True)
    return output
