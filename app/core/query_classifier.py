"""
query_classifier.py —— 查询分类（Query Classification）

【这个文件做什么】
  在检索之前，先判断用户的问题属于哪一类，从而决定用什么样的检索策略。

【分类类型】
  EXACT    精确查询：含型号、SKU、编号（如 "PX-4200 多少钱"）→ 偏重 BM25 精确匹配
  NUMERIC  数值查询：含数字、单位、销售额等（如 "2025 年销售额"）→ 偏重精确+聚合相关文本
  SEMANTIC 语义查询：问原因/趋势/解释（如 "为什么最近销售下降"）→ 偏重向量语义检索
  MIXED    混合查询：既有精确词又有语义意图（如 "解释 PX-4200 为什么延迟高"）

【新手必读 · 面试点】
  分类是"策略入口"而不是"理论证明"。规则版零成本、可解释，适合冷启动；
  真正要证明"分类有用"，靠的是评测数据：对比"分类+策略" vs "不分统一策略"的指标。
"""
import re
from enum import Enum


class QueryType(str, Enum):
    EXACT = "EXACT"
    SEMANTIC = "SEMANTIC"
    MIXED = "MIXED"
    NUMERIC = "NUMERIC"


class QueryClassifier:
    # 精确实体特征：SKU-123 / PX-4200 / A123_XYZ 这类"字母数字混合串"
    EXACT_PATTERNS = [
        r"\bSKU[-_ ]?[A-Za-z0-9]+\b",   # SKU A123 / SKU-42
        r"\b[A-Z]{1,5}\d{2,}\b",        # PX-4200 / ISO9001
        r"\b[A-Z0-9]+[-_][A-Z0-9]+\b",  # A123_X / AB-12
    ]
    # 数值/业务指标特征
    NUMERIC_PATTERNS = [
        r"\d+(\.\d+)?%",      # 30%
        r"\d+(\.\d+)?",       # 任意数字
        r"同比", r"环比", r"增长率", r"销售额", r"数量", r"价格", r"成本", r"预算",
    ]
    # 语义提问意图词
    SEMANTIC_WORDS = ["为什么", "原因", "如何", "怎么", "影响", "趋势", "解释", "区别", "对比"]

    def classify(self, query: str) -> QueryType:
        """把一条查询分到四类之一。规则简单可解释，冷启动够用。"""
        exact = any(re.search(p, query, re.I) for p in self.EXACT_PATTERNS)
        numeric = any(re.search(p, query, re.I) for p in self.NUMERIC_PATTERNS)
        semantic = any(word in query for word in self.SEMANTIC_WORDS)

        # 优先级：既含精确词又有语义意图 → MIXED；否则按命中特征归类；都没有 → SEMANTIC
        if exact and semantic:
            return QueryType.MIXED
        if exact:
            return QueryType.EXACT
        if numeric and semantic:
            return QueryType.MIXED
        if numeric:
            return QueryType.NUMERIC
        return QueryType.SEMANTIC
