"""
bm25.py —— Elasticsearch BM25 检索器

【这个文件做什么】
  把 Chunk 文本写入 Elasticsearch，并用 BM25 算法做关键词检索。

【新手必读 · BM25 直觉】
  BM25 判断"查询里的词在文档里有多匹配"，综合考虑：
    - 词频（Term Frequency）：出现次数越多分越高，但有"饱和"——从 10 次到 100 次收益递减；
    - 稀有度（Inverse Document Frequency）：越稀有的词权重越高（"价格"普通，"PX-4200"稀有）；
    - 文档长度归一化：长文档里出现一次关键词，不如短文档里出现一次"值钱"。

【中文注意】
  ES 默认 standard 分词器对中文按单字切，效果一般。生产级中文检索应使用 IK 分词器
  （analyzer: ik_max_word）。本项目先跑通默认配置，中文分词优化作为进阶项。

【字段约定】
  ES 里每条文档存一个 Chunk 的全部字段（见 ensure_index 的 mappings），
  检索时用 multi_match 同时匹配 content 和 heading_path，并给 content 更高权重。
"""
from elasticsearch import Elasticsearch, helpers

from app.config import settings
from app.models.retrieval import Candidate


class BM25Retriever:
    # 中文分词器。
    # ES 默认的 standard 分词器把中文【按单字】切开："版本控制系统" →
    # ['版','本','控','制','系','统']。于是查询"Git有什么用"被切成
    # git/有/什/么/用/处，而"有""用""么"是高频字 —— 几乎任何文档都会命中，
    # 实测查一句话命中 21/23 条，BM25 的区分度基本归零，排序近乎随机。
    #
    # cjk 是 ES 内置分词器（无需安装 IK 插件），把中文切成二元组：
    #   "版本控制系统" → ['版本','本控','控制','制系','系统']
    # 区分度比单字高一个数量级。英文/数字照常按词切（PX-4200 → ['px','4200']）。
    CONTENT_ANALYZER = "cjk"

    # 索引字段定义。抽成类常量，让 create 与 put_mapping 复用同一份定义，避免两处不一致。
    MAPPING_PROPERTIES = {
        "chunk_id": {"type": "keyword"},        # 精确匹配类型，适合等值过滤
        "document_id": {"type": "keyword"},     # 「按文档过滤 / 按文档删除」都靠它
        "source_name": {"type": "keyword"},     # 来源文件名，前端"文档库"列表用
        "content": {
            "type": "text",
            "analyzer": CONTENT_ANALYZER,
            "search_analyzer": CONTENT_ANALYZER,
        },
        # heading_path 本体保持 keyword（前端展示、精确过滤用）；
        # 另存一个 heading_text —— 把标题路径拼成字符串供全文检索。
        # 为什么要多一个字段：keyword 只支持整串精确匹配，
        # 把它塞进 multi_match 是无效的（实测命中 0 条，给的权重形同虚设）。
        "heading_text": {
            "type": "text",
            "analyzer": CONTENT_ANALYZER,
            "search_analyzer": CONTENT_ANALYZER,
        },
        "content_type": {"type": "keyword"},
        "page_number": {"type": "integer"},
        "heading_path": {"type": "keyword"},
        "metadata": {"type": "object", "enabled": True},
    }

    def __init__(self):
        # Elasticsearch 客户端是"懒连接"的：这里只是记住地址，真正发请求时才建连
        self.client = Elasticsearch(settings.elasticsearch_url)
        self.index = settings.elasticsearch_index

    @staticmethod
    def _document_body(chunk) -> dict:
        """把 Chunk 转成待写入 ES 的文档体。

        额外补一个 heading_text：把标题路径拼成字符串。
        （见 MAPPING_PROPERTIES 里的说明：heading_path 是 keyword，不能做全文检索。）
        """
        body = chunk.model_dump(mode="json")
        body["heading_text"] = " ".join(chunk.heading_path or [])
        return body

    # ------------------------------------------------------------------
    # 索引管理
    # ------------------------------------------------------------------
    def ensure_index(self) -> None:
        """确保索引存在且 mapping 完整（幂等操作，可重复调用）。

        索引已存在时会再 put_mapping 一次，用来「补齐新增字段」。
        这样升级版本后（比如新增 source_name）不用重建索引；
        否则新字段只能靠 ES 动态映射，类型不可控（字符串会被同时映射成 text + keyword）。
        如果字段类型与已有 mapping 冲突，put_mapping 会报错，这里吞掉并继续 —— 不影响检索。
        """
        if self.client.indices.exists(index=self.index):
            try:
                self.client.indices.put_mapping(
                    index=self.index, properties=self.MAPPING_PROPERTIES
                )
            except Exception as exc:  # noqa: BLE001 —— 补 mapping 失败不该挡住索引流程
                print(f"[bm25] 补充 mapping 失败（可忽略）: {exc}")
            return
        self.client.indices.create(
            index=self.index, mappings={"properties": self.MAPPING_PROPERTIES}
        )

    def delete_index(self) -> None:
        """删除整个索引（重置数据用）。"""
        if self.client.indices.exists(index=self.index):
            self.client.indices.delete(index=self.index)

    def count_all(self) -> int:
        """统计索引里有多少个 Chunk（供 GET /stats 使用）。"""
        if not self.client.indices.exists(index=self.index):
            return 0
        return int(self.client.count(index=self.index)["count"])

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def index_chunk(self, chunk) -> None:
        """把一个 Chunk 写入 ES。以 chunk_id 作为文档 id，重复写入会覆盖（幂等）。

        单条写入用这个方法；多条请用 index_chunks()。
        """
        self.client.index(
            index=self.index,
            id=str(chunk.chunk_id),
            document=self._document_body(chunk),
            refresh="wait_for",                      # 等刷新完成再返回，保证立即可搜到（演示友好）
        )

    def index_chunks(self, chunks) -> None:
        """批量写入多个 Chunk。

        ⚠️ 必须走 ES 的 bulk API，不要写成 for 循环逐条 index_chunk()。
        早期版本就是 for 循环调用 index_chunk()，而 index_chunk 带
        refresh="wait_for" —— 等于「写一条、就强制刷新一次索引」。
        一份几百个 Chunk 的 PDF 会触发几百次刷新，慢到不可接受。
        改成 bulk 之后，不论多少条都只刷新一次。

        参数 refresh=True 的效果同 index_chunk 的 wait_for：
        写完就立即可搜，方便上传后立刻提问。
        """
        if not chunks:
            return
        actions = [
            {
                "_index": self.index,
                "_id": str(chunk.chunk_id),
                "_source": self._document_body(chunk),
            }
            for chunk in chunks
        ]
        helpers.bulk(self.client, actions, refresh=True)

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int, document_id=None) -> list[Candidate]:
        """BM25 关键词检索，返回 Candidate 列表（得分降序）。

        参数 document_id：给了就只在那一份文档内检索（前端"限定文档范围"用）。

        若索引还不存在（没数据），直接返回空列表而不是报错——更友好。
        """
        if not self.client.indices.exists(index=self.index):
            return []

        bool_query: dict = {
            "must": [
                {
                    "multi_match": {
                        "query": query,
                        # content 权重是 heading_text 的 3 倍：正文相关性更重要，
                        # 但标题命中也很值钱（一小段标题往往就是整节的答案）。
                        # 注意这里用 heading_text 而不是 heading_path ——
                        # 后者是 keyword 类型，参与全文检索是无效的。
                        "fields": ["content^3", "heading_text^2"],
                    }
                }
            ]
        }
        if document_id is not None:
            # filter 不参与打分，只做"圈定范围"，比塞进 must 更合适
            bool_query["filter"] = [{"term": {"document_id": str(document_id)}}]

        response = self.client.search(
            index=self.index,
            size=top_k,
            query={"bool": bool_query},
        )
        return [
            Candidate(
                chunk_id=hit["_source"]["chunk_id"],
                content=hit["_source"]["content"],
                source="bm25",
                retrieval_method="bm25",
                original_score=float(hit.get("_score") or 0),
                page_number=hit["_source"].get("page_number"),
                heading_path=hit["_source"].get("heading_path", []),
                content_type=hit["_source"].get("content_type", "text"),
                source_name=hit["_source"].get("source_name"),
                metadata=hit["_source"].get("metadata", {}),
            )
            for hit in response["hits"]["hits"]
        ]

    # ------------------------------------------------------------------
    # 按文档管理（前端"文档库"用）
    # ------------------------------------------------------------------
    def delete_document(self, document_id) -> int:
        """删除某一份文档的全部 Chunk，返回删除条数。

        用 delete_by_query 按 document_id 精确删除，而不是删掉整个索引 ——
        这样多文档场景下可以只下线其中一份。
        """
        if not self.client.indices.exists(index=self.index):
            return 0
        response = self.client.delete_by_query(
            index=self.index,
            query={"term": {"document_id": str(document_id)}},
            refresh=True,          # 删完立刻生效，前端刷新就能看到
            conflicts="proceed",   # 有并发写入时跳过冲突，不要把整个请求打回
        )
        return int(response.get("deleted", 0))

    def list_documents(self) -> list[dict]:
        """列出索引里有哪些文档（每个 document_id 一条）。

        用聚合而不是把全部 Chunk 拉回来自己统计 —— 后者在几万条 Chunk 时
        会把内存和网络都打满。size=0 表示不需要返回命中文档，只要聚合结果。
        """
        if not self.client.indices.exists(index=self.index):
            return []

        response = self.client.search(
            index=self.index,
            size=0,
            aggs={
                "docs": {
                    "terms": {"field": "document_id", "size": 200},
                    "aggs": {
                        "source": {"terms": {"field": "source_name", "size": 1}},
                        "pages": {"max": {"field": "page_number"}},
                        "types": {"terms": {"field": "content_type", "size": 5}},
                    },
                }
            },
        )

        documents = []
        for bucket in response.get("aggregations", {}).get("docs", {}).get("buckets", []):
            source_buckets = bucket["source"]["buckets"]
            documents.append(
                {
                    "document_id": bucket["key"],
                    "source_name": source_buckets[0]["key"] if source_buckets else "(未知来源)",
                    "chunks": bucket["doc_count"],
                    "pages": int(bucket["pages"]["value"] or 0),
                    "content_types": {
                        item["key"]: item["doc_count"] for item in bucket["types"]["buckets"]
                    },
                }
            )
        # 按文件名排序，前端展示更稳定
        documents.sort(key=lambda item: item["source_name"])
        return documents
