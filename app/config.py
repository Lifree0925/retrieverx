"""
config.py —— RetrieverX 统一配置模块

【这个文件是做什么的】
  集中管理项目所有"可调参数"，例如：连接哪个数据库、用哪个 embedding 模型、
  召回多少条候选、RRF 的 k 值等。代码里绝不写死这些值，统一通过 settings.xxx 读取。

【新手必读】
  1. pydantic-settings 会自动读取项目根目录下的 .env 文件，把大写变量名映射成这里的小写字段。
     例如 .env 里写 EMBEDDING_MODEL=bge-m3  →  这里 settings.embedding_model 就等于 "bge-m3"。
  2. 修改任何配置，都去 .env 改，不要改这个文件里的默认值——方便不同环境（本地/Docker）切换。
"""
import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """项目配置类。每个字段都有默认值，未在 .env 配置时使用默认值。"""

    # ---------- 基础信息 ----------
    app_name: str = "RetrieverX"          # 应用名，用于 FastAPI 标题
    app_env: str = "development"          # development / production
    log_level: str = "INFO"               # 日志级别 DEBUG/INFO/WARNING/ERROR

    # ---------- Embedding（文本向量化）----------
    embedding_model: str = "text-embedding-3-small"   # OpenAI 兼容接口的 embedding 模型名
    embedding_dim: int = 1536                         # 向量维度，必须与模型输出一致！
    embedding_api_key: str | None = None              # API Key，从 .env 注入
    embedding_base_url: str | None = None             # 兼容接口地址（硅基流动/智谱等）

    # ---------- Qdrant（向量数据库）----------
    qdrant_url: str = "http://localhost:6333"         # Docker 部署时改为 http://qdrant:6333
    qdrant_collection: str = "retrieverx_chunks"      # 向量集合名
    # 客户端超时。默认只有 5 秒，实测在 delete_collection / 批量 upsert 时
    # 偶发 ResponseHandlingException: timed out，本机服务给宽一点更稳。
    qdrant_timeout_seconds: float = 30.0

    # ---------- Elasticsearch（BM25 关键词检索）----------
    elasticsearch_url: str = "http://localhost:9200"  # Docker 部署时改为 http://elasticsearch:9200
    elasticsearch_index: str = "retrieverx_chunks"    # ES 索引名

    # ---------- Redis（反馈存储 / 更正缓存 / Trace 存储）----------
    redis_url: str = "redis://localhost:6379/0"
    feedback_ttl: int = 2592000                       # 反馈过期时间（秒），默认 30 天
    trace_ttl: int = 86400                            # Retrieval Trace 保留时长（秒），默认 1 天

    # ---------- Reranker（精排模型）----------
    reranker_model: str = "BAAI/bge-reranker-base"    # 本地 Cross-Encoder 模型
    rerank_enable: bool = True                        # 总开关；关闭则跳过精排（省下载量）
    # 单次【打分】的超时上限。本机 CPU 实测：20 个候选约 16 秒，
    # 所以这个值给太小会导致每次都超时降级。CPU 环境建议 30。
    rerank_timeout_seconds: float = 30.0
    # 模型【加载/下载】的等待上限（一次性开销，和上面的打分超时分开算）。
    # 已缓存时实测 3.5 秒；首次联网下载约 1GB，要给足时间。
    rerank_load_timeout_seconds: float = 90.0
    # 每对 (query, doc) 送进模型前截断到的最大 token 数。
    # 直接决定打分耗时：CPU 上 512 → 20 条约 16 秒；调到 256 大约能快一倍。
    reranker_max_length: int = 512
    reranker_device: str | None = None                # cpu / cuda / 留空=自动选择

    # ---------- HuggingFace 模型下载 ----------
    # 国内直连 huggingface.co 基本不可用，会导致首次加载模型长时间卡住。
    # 填 https://hf-mirror.com 走国内镜像；留空则用官方源。
    hf_endpoint: str | None = None

    # ---------- 检索链路参数 ----------
    default_top_k: int = Field(default=5, ge=1, le=100)     # 对外默认返回条数
                                                            # （app/models/retrieval.py 的 SearchRequest 读它）
    recall_top_k: int = Field(default=20, ge=1, le=200)     # 内部召回条数（先召回再融合再精排）
    rrf_k: int = Field(default=60, ge=1)                    # RRF 融合常数 k

    # 融合权重的**基准比例**，由 RetrievalPolicy 派生各查询类型的具体权重
    # （见 app/core/retrieval_policy.py）。
    #
    # 【为什么要重写成"基准 + 偏移"而不是每类型写死】
    #   原来这两个字段**没有任何代码读**，真实权重硬编码在 RetrievalPolicy 里——
    #   也就是说 .env.example 在引导你调两个不起作用的开关，属于"文档里写了、代码里没有"。
    #   现在改成：这两个值定义基线（默认各 0.5），
    #   再按查询类型做固定偏移（EXACT/NUMERIC 偏 BM25、SEMANTIC 偏向量），
    #   默认值下算出来的结果与原硬编码值完全一致（0.7/0.3、0.3/0.7、0.5/0.5），
    #   但从此**改 .env 真的会生效**。
    bm25_weight: float = Field(default=0.5, ge=0.0, le=1.0)   # 基准权重（BM25 侧）
    vector_weight: float = Field(default=0.5, ge=0.0, le=1.0) # 基准权重（向量侧）

    # ---------- 检索策略（Policy）自适应约束 ----------
    policy_min_samples: int = Field(default=20, ge=1)   # 少于该样本数不允许调策略，防误操作带偏
    policy_max_step: float = Field(default=0.1, ge=0.0, le=1.0)  # 单次调整的最大步长

    # 告诉 pydantic：从 .env 读配置；.env 里多余字段忽略不报错
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    """返回唯一配置实例；lru_cache 保证全程序只解析一次 .env，性能好且状态一致。"""
    return Settings()


# 模块级单例，其他地方统一 `from app.config import settings`
settings = get_settings()

# 【关键时序】huggingface_hub 在「被 import 的那一刻」就把 HF_ENDPOINT 读进常量了，
# 之后再改环境变量不会生效。所以必须在这里（config 是所有模块的公共依赖，最早被导入）
# 就把镜像地址塞进环境变量，否则首次加载 reranker 仍然会去连 huggingface.co 而卡死。
# 用 setdefault：如果你已经在系统里配了 HF_ENDPOINT，以你的为准。
if settings.hf_endpoint:
    os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)


def _bypass_proxy_for_local_services() -> None:
    """把本机地址加进 NO_PROXY，避免访问 localhost 的服务被送去系统代理。

    【为什么必须做这件事】
      Qdrant 客户端和 OpenAI 客户端底层都用 httpx，而 httpx 会读取
      环境变量 HTTP_PROXY / HTTPS_PROXY。如果机器上开着系统级代理
      （Clash、VPN、公司网关等），到 localhost 的请求会被错误地转发给代理，
      表现为连接超时。

      这个 bug 特别难查，因为它只影响部分依赖：
        · Elasticsearch 用 urllib3 —— urllib3 不读代理环境变量 → 正常
        · Qdrant 用 httpx         —— 读取代理变量 → 报 ResponseHandlingException: timed out
      于是你会看到"ES 明明通了，Qdrant 却超时"这种诡异现象。

      容器里跑时地址是 qdrant / elasticsearch 这类主机名，不属于本机，
      所以不会被加进 NO_PROXY，不影响 docker 部署。
    """
    from urllib.parse import urlparse

    local_hosts = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
    touched_local = False
    for url in (
        settings.qdrant_url,
        settings.elasticsearch_url,
        settings.redis_url,
        settings.embedding_base_url,
    ):
        if not url:
            continue
        parsed = urlparse(url if "://" in url else f"//{url}")
        if (parsed.hostname or "").lower() in local_hosts:
            touched_local = True
            break

    if not touched_local:
        return

    existing = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    entries = [e.strip() for e in existing.split(",") if e.strip()]
    for host in ("localhost", "127.0.0.1", "::1"):
        if host not in entries:
            entries.append(host)
    value = ",".join(entries)
    # 大小写两个都设：httpx 读 no_proxy 的变体在不同版本上不一致
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value


_bypass_proxy_for_local_services()
