"""
retrieval.py —— 检索请求 / 候选结果 / 响应 的数据模型

【新手必读】
  这一层模型决定了 FastAPI 接口"长什么样"：
    POST /search 接收 RetrievalRequest，返回 RetrievalResponse。
  一个 Candidate（候选结果）对应一条命中的 Chunk，并记录它来自哪种检索方法、
  各阶段的得分——这些得分是后面做实验、讲"为什么融合有效"的数据基础。
"""
from uuid import UUID

from pydantic import BaseModel, Field


class RetrievalRequest(BaseModel):
    """POST /search 的请求体。FastAPI 会自动校验这些字段。"""

    query: str = Field(min_length=1, max_length=500)  # 查询问题，不能为空
    top_k: int = Field(default=5, ge=1, le=100)        # 返回条数（1~100）
    rerank: bool = True                                # 本次是否启用 Cross-Encoder 精排
    # 限定检索范围：给了就只在这一份文档内检索；不传则跨全部文档
    document_id: UUID | None = None


class Candidate(BaseModel):
    """一条检索结果。"""

    chunk_id: UUID                                     # 命中的 Chunk 唯一标识
    content: str                                       # Chunk 文本内容（用于展示给用户）
    source: str                                        # 原始来源（预留，通常同 retrieval_method）
    retrieval_method: str                              # 命中的方法：bm25 / vector / bm25+vector
    original_score: float = 0.0                        # 该路检索的原始得分（BM25 或余弦相似度）
    fusion_score: float = 0.0                          # RRF 融合后的得分
    rerank_score: float | None = None                  # Cross-Encoder 重排得分（未精排则为 None）
    page_number: int | None = None                     # 来源页码（方便前端展示）
    heading_path: list[str] = Field(default_factory=list)  # 标题路径（溯源展示用）
    content_type: str = "text"                         # text / table
    source_name: str | None = None                     # 来源文件名（前端展示"来自哪份文档"）
    metadata: dict = Field(default_factory=dict)       # 其余元数据


class RetrievalPolicySnapshot(BaseModel):
    """策略快照：本次检索实际使用的策略参数与版本号。

    为什么需要版本号？因为策略会随反馈迭代更新；带上版本，才能回溯"某次查询用了哪个版本"，
    也是实验对比和回滚的依据。
    """

    bm25: float                                        # BM25 权重
    vector: float                                      # 向量权重
    version: int = 1                                   # 策略版本号


class RetrievalResponse(BaseModel):
    """POST /search 的响应体。"""

    query: str                                         # 原始查询
    candidates: list[Candidate]                        # 排序后的候选结果
    latency_ms: float                                  # 整条管线总耗时（毫秒）
    retrieval_policy: RetrievalPolicySnapshot          # 本次使用的策略快照
    used_correction: bool = False                      # 是否命中了用户的更正缓存
    rerank_applied: bool = False                       # 本次是否真的做了精排（false=被降级/未开启）
