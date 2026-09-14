"""
feedback.py —— 用户反馈的数据模型

【新手必读】
  反馈是"让检索系统自我改进"的信号。本项目支持三种反馈：
    positive   结果好（点赞）
    negative   结果差（点踩）
    correction 用户给出正确的查询写法（更正，是信息量最大的反馈）
  一条反馈必须记录它对应的 query、当时的候选结果、反馈类型，
  否则无法用于后续统计与策略调整。
"""
from datetime import datetime, timezone
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class FeedbackEntry(BaseModel):
    """一条用户反馈。"""

    feedback_id: UUID = Field(default_factory=uuid4)       # 反馈唯一标识
    query: str = Field(min_length=1, max_length=500)       # 用户查询的问题
    candidate_ids: list[UUID] = Field(default_factory=list)  # 本次返回给用户的候选 chunk id 列表
    label: str                                             # positive / negative / correction
    correct_answer: str | None = None                      # 仅 correction 时填写：正确写法
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )                                                      # UTC 时间戳

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        """校验 label 只能是三种之一，写错直接报 422。"""
        allowed = {"positive", "negative", "correction"}
        if value not in allowed:
            raise ValueError(f"label 必须是 {sorted(allowed)} 之一，当前是 {value!r}")
        return value
