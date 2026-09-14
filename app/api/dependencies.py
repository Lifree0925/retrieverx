"""
dependencies.py —— 依赖注入容器（Depends 提供全局单例）

【这个文件做什么】
  FastAPI 路由通过 Depends(get_xxx) 拿服务对象。
  所有"重量级对象"（连接库、模型）都放在这里用 lru_cache 懒加载：
  - 第一次被用到时才创建；
  - 之后复用同一个实例（单例）。

【新手必读 · 为什么这样设计？】
  1. 避免循环 import：路由不再直接 import app.main 里的全局变量；
  2. 启动快：Reranker 模型（约 1GB）不会在服务一启动就下载/加载，
     只有第一次真正用精排时才加载——这就是"懒加载（Lazy Loading）"；
  3. 好测试：测试时可以直接把假对象塞给 RetrievalService，不必连真库。
"""
from functools import lru_cache

from app.config import settings
from app.core.bm25 import BM25Retriever
from app.core.embeddings import OpenAICompatibleEmbedding
from app.core.feedback import FeedbackStore
from app.core.pipeline import RetrievalService
from app.core.query_classifier import QueryClassifier
from app.core.reranker import Reranker
from app.core.retrieval_policy import RetrievalPolicy
from app.core.vector_search import VectorRetriever


# ----------------------------------------------------------------------
# 单例工厂：首次调用创建，之后一直复用
# ----------------------------------------------------------------------
@lru_cache
def get_embedding() -> OpenAICompatibleEmbedding:
    return OpenAICompatibleEmbedding()


@lru_cache
def get_bm25() -> BM25Retriever:
    return BM25Retriever()


@lru_cache
def get_vector() -> VectorRetriever:
    return VectorRetriever(get_embedding())


@lru_cache
def get_feedback_store() -> FeedbackStore:
    return FeedbackStore()


@lru_cache
def get_classifier() -> QueryClassifier:
    return QueryClassifier()


@lru_cache
def get_policy() -> RetrievalPolicy:
    return RetrievalPolicy(
        min_samples=settings.policy_min_samples,
        max_step=settings.policy_max_step,
    )


@lru_cache
def get_reranker():
    """按配置决定是否创建精排器。

    配置关闭（rerank_enable=False）时返回 None，流水线会自动跳过精排。
    注意：Reranker() 本身很轻（只准备状态、不加载模型）；真正的模型加载
    发生在第一次精排时，且带超时保护，不会把请求卡死。
    """
    if not settings.rerank_enable:
        return None
    return Reranker()


@lru_cache
def get_retrieval_service() -> RetrievalService:
    """把全部组件装配成一条检索流水线。

    ⚠️ 这里传的是 get_reranker 这个【函数本身】，而不是它的返回值。
    原因：如果在装配阶段就调用 get_reranker()，那么无论请求里 rerank 传什么值，
    精排模型都会被创建——请求参数就形同虚设了（这正是之前"取消勾选精排也没用"的原因）。
    传函数进去 = 延迟到真正需要精排时才创建。
    """
    return RetrievalService(
        bm25=get_bm25(),
        vector=get_vector(),
        classifier=get_classifier(),
        policy=get_policy(),
        feedback_store=get_feedback_store(),
        reranker_provider=get_reranker,
    )


# ----------------------------------------------------------------------
# FastAPI Depends 使用下面的短名
# ----------------------------------------------------------------------
def get_retrieval_service_dep() -> RetrievalService:
    return get_retrieval_service()


def get_feedback_store_dep() -> FeedbackStore:
    return get_feedback_store()


def get_bm25_dep() -> BM25Retriever:
    return get_bm25()


def get_vector_dep() -> VectorRetriever:
    return get_vector()
