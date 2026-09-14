"""
routes_search.py —— 检索接口

【接口约定】
  POST /search
  请求体：{"query": "问题", "top_k": 5, "rerank": true, "document_id": "可选，限定文档"}
  返回：RetrievalResponse（见 app/models/retrieval.py）
"""
from fastapi import APIRouter, Depends

from app.api.dependencies import get_retrieval_service_dep
from app.models.retrieval import RetrievalRequest, RetrievalResponse

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=RetrievalResponse)
def search(
    request: RetrievalRequest,
    service=Depends(get_retrieval_service_dep),
):
    """执行一次混合检索。

    Depends 会自动调用 get_retrieval_service_dep() 拿到全局唯一的检索服务。
    Pydantic 会先校验 request 里的字段（query 非空、top_k 范围等），校验失败返回 422。

    document_id 不传 = 跨全部文档检索；传了 = 只在那一份文档内检索。
    """
    return service.run_pipeline(
        request.query,
        request.top_k,
        request.rerank,
        request.document_id,
    )
