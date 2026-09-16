"""
routes_trace.py —— Retrieval Trace 查询接口

【接口约定】
  GET /trace/{request_id}
  返回该次检索的完整链路 Trace（docs/开发文档.md 的 WP9 / Retrieval Debugger 的数据基础）。

  request_id 从 POST /search 的响应里拿（RetrievalResponse.request_id）。
"""
from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_trace_store_dep

router = APIRouter(prefix="/trace", tags=["trace"])


@router.get("/{request_id}")
def get_trace(request_id: str, trace_store=Depends(get_trace_store_dep)):
    """按 request_id 取回一次检索的完整链路 Trace。"""
    trace = trace_store.get(request_id)
    if trace is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "TRACE_NOT_FOUND",
                "message": (
                    f"未找到 request_id={request_id} 的检索 Trace。"
                    "可能原因：Trace 已过期（见 TRACE_TTL），或该次检索未落 Trace。"
                ),
            },
        )
    return trace
