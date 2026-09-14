"""
routes_feedback.py —— 反馈接口

【接口约定】
  POST /feedback
  请求体：FeedbackEntry（query / candidate_ids / label / correct_answer）
  label 取值：positive（好）、negative（差）、correction（更正，需填 correct_answer）
"""
from fastapi import APIRouter, Depends, status

from app.api.dependencies import get_feedback_store_dep
from app.models.feedback import FeedbackEntry

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("", status_code=status.HTTP_201_CREATED)
def create_feedback(
    feedback: FeedbackEntry,
    store=Depends(get_feedback_store_dep),
):
    """记录一条用户反馈到 Redis。label 非法会自动 422（见 FeedbackEntry 校验）。"""
    store.add(feedback)
    return {"ok": True, "feedback_id": str(feedback.feedback_id)}
