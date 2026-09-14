"""
retrieval_policy.py —— 检索策略（Retrieval Policy）

【这个文件做什么】
  不同查询类型使用不同的"BM25/向量权重组合"，即一条策略。
  策略带版本号，可随反馈逐步调整，也能回滚。

【新手必读 · 面试点：为什么策略更新不能"点差评→权重+10%"?】
  一次误操作反馈会把策略带偏。因此本项目策略更新受三重约束：
    1. 最小样本数（min_samples）：样本太少不允许更新；
    2. 单次最大步长（max_step）：每次只能小幅调整，防止剧烈抖动；
    3. 版本号 + 历史记录：出错可以 rollback 回上一个版本。
"""
from dataclasses import dataclass

from app.core.query_classifier import QueryType


@dataclass
class Policy:
    """一条检索策略：BM25 与向量的权重 + 版本号。"""

    bm25: float      # BM25 权重
    vector: float    # 向量权重
    version: int = 1


class RetrievalPolicy:
    def __init__(self, min_samples: int = 20, max_step: float = 0.1):
        """
        min_samples  最少需要多少条反馈才允许调整权重（防误操作）
        max_step     单次最大调整步长（防抖动）
        """
        self.min_samples = min_samples
        self.max_step = max_step

        # 每种查询类型的初始策略（领域直觉，冷启动默认值）：
        #   EXACT/NUMERIC 以精确词为主 → BM25 权重大；
        #   SEMANTIC 以语义为主 → 向量权重大；MIXED 各半。
        self.policies: dict[QueryType, Policy] = {
            QueryType.EXACT: Policy(bm25=0.7, vector=0.3),
            QueryType.SEMANTIC: Policy(bm25=0.3, vector=0.7),
            QueryType.MIXED: Policy(bm25=0.5, vector=0.5),
            QueryType.NUMERIC: Policy(bm25=0.7, vector=0.3),
        }
        # 更新历史：[(query_type, 旧策略, 新策略), ...]，供回滚
        self.history: list[tuple[QueryType, Policy, Policy]] = []

    def get(self, query_type: QueryType) -> Policy:
        """取某类型当前的策略。"""
        return self.policies[query_type]

    def update(
        self,
        query_type: QueryType,
        bm25_delta: float,
        vector_delta: float,
        sample_count: int,
    ) -> Policy:
        """根据反馈统计尝试更新策略。

        参数：
          bm25_delta / vector_delta  期望的调整量（可正可负）
          sample_count               当前已积累的反馈样本数
        约束生效时返回原策略（不更新），满足条件才真正调整。
        """
        # 约束 1：样本不足，拒绝更新
        if sample_count < self.min_samples:
            return self.get(query_type)

        # 约束 2：把单次调整量限制在 [-max_step, +max_step]
        bm25_delta = max(-self.max_step, min(self.max_step, bm25_delta))
        vector_delta = max(-self.max_step, min(self.max_step, vector_delta))

        old = self.get(query_type)
        bm25 = max(0.0, old.bm25 + bm25_delta)
        vector = max(0.0, old.vector + vector_delta)

        # 归一化：保证 bm25 + vector = 1
        total = bm25 + vector
        new = Policy(bm25=bm25 / total, vector=vector / total, version=old.version + 1)

        self.policies[query_type] = new
        self.history.append((query_type, old, new))
        return new

    def rollback(self, query_type: QueryType) -> Policy:
        """回滚该类型到最近一次更新前的版本（从历史里倒着找）。"""
        for qtype, old, _new in reversed(self.history):
            if qtype == query_type:
                self.policies[query_type] = old
                return old
        return self.get(query_type)
