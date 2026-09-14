"""
vector_search.py —— Qdrant 向量检索器

【这个文件做什么】
  把 Chunk 的 embedding 向量写入 Qdrant，并做最近邻（相似度）检索。

【新手必读 · 面试点】
  1. 向量检索找到的是"语义上最接近"的文本，适合自然语言提问；
  2. Qdrant 存向量时同时存 payload（原始字段），这样检索命中后不用再回原库捞文本；
  3. 距离度量用 COSINE（余弦相似度），语义检索的常用选择。
"""
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.config import settings
from app.models.retrieval import Candidate


class VectorRetriever:
    def __init__(self, embedding_provider):
        """
        embedding_provider：提供 embed(texts)->vectors 的对象（见 embeddings.py）。
        把 embedding 作为依赖注入进来，方便测试时替换成假实现（不必真调 API）。
        """
        self.client = QdrantClient(
            url=settings.qdrant_url,
            # 默认 5 秒对 delete_collection、批量 upsert 这类偏重的操作偏紧，
            # 实测会偶发 ResponseHandlingException: timed out。给宽一点，避免"随机失败"。
            timeout=settings.qdrant_timeout_seconds,
        )
        self.embedding_provider = embedding_provider
        self.collection = settings.qdrant_collection

    # ------------------------------------------------------------------
    # 集合管理
    # ------------------------------------------------------------------
    def ensure_collection(self) -> None:
        """集合不存在则创建。向量维度必须等于 settings.embedding_dim。"""
        names = {x.name for x in self.client.get_collections().collections}
        if self.collection not in names:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )

    def delete_collection(self) -> None:
        """删除整个集合（重置数据用）。"""
        names = {x.name for x in self.client.get_collections().collections}
        if self.collection in names:
            self.client.delete_collection(collection_name=self.collection)

    def count_all(self) -> int:
        """返回集合里的点数（即 Chunk 数），供 GET /stats 使用。"""
        names = {x.name for x in self.client.get_collections().collections}
        if self.collection not in names:
            return 0
        return self.client.count(collection_name=self.collection).count

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def index_chunks(self, chunks) -> None:
        """把 Chunk 列表写入向量库：先批量向量化，再 upsert。

        注意：embed 接口一次别传太多文本，分批处理更稳（很多 API 有单次长度限制）。
        """
        if not chunks:
            return
        vectors = self.embedding_provider.embed([x.content for x in chunks])
        points = [
            PointStruct(
                id=str(chunk.chunk_id),       # 用 chunk_id 做 point id，方便去重/覆盖
                vector=vector,
                payload=chunk.model_dump(mode="json"),  # 原始字段全部带进 payload
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int, document_id=None) -> list[Candidate]:
        """把查询向量化后在向量库找最近邻，返回 Candidate 列表。

        参数 document_id：给了就只在那一份文档内检索（前端"限定文档范围"用）。

        若集合不存在（没数据），返回空列表而不是报错。
        """
        names = {x.name for x in self.client.get_collections().collections}
        if self.collection not in names:
            return []

        query_filter = None
        if document_id is not None:
            # payload 里的 document_id 是 chunk.model_dump(mode="json") 存的，即字符串
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id", match=MatchValue(value=str(document_id))
                    )
                ]
            )

        vector = self.embedding_provider.embed([query])[0]
        points = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
            with_payload=True,  # 需要 payload 里的 content 等字段
            query_filter=query_filter,
        ).points

        return [
            Candidate(
                chunk_id=point.payload["chunk_id"],
                content=point.payload["content"],
                source="vector",
                retrieval_method="vector",
                original_score=float(point.score),  # 余弦相似度得分
                page_number=point.payload.get("page_number"),
                heading_path=point.payload.get("heading_path", []),
                content_type=point.payload.get("content_type", "text"),
                source_name=point.payload.get("source_name"),
                metadata=point.payload.get("metadata", {}),
            )
            for point in points
        ]

    # ------------------------------------------------------------------
    # 按文档删除
    # ------------------------------------------------------------------
    def delete_document(self, document_id) -> None:
        """删除某一份文档的全部向量点（按 payload 里的 document_id 过滤）。

        注意不是删整个集合：多文档场景下要能只下线其中一份。
        """
        names = {x.name for x in self.client.get_collections().collections}
        if self.collection not in names:
            return
        self.client.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id", match=MatchValue(value=str(document_id))
                        )
                    ]
                )
            ),
        )
