"""
trace.py —— 检索链路追踪（Retrieval Trace）

【这个文件做什么】
  把一次检索的完整过程（查询类型、策略版本、双路召回、RRF 融合、精排、延迟分解）
  落成一条可回查的结构化记录，供 Retrieval Debugger 和 Failure Analysis 使用。

【新手必读 · 面试点：Trace 和日志有什么区别？】
  日志是"给人看的流水"，按时间顺序打印，适合排查"某一行发生了什么"；
  Trace 是"给机器/调试台看的结构"，按一次请求组织成 JSON，能回答：
    - 这次查询被判成了什么类型？用了哪个策略版本？
    - BM25 和向量各自召回了什么、排第几？
    - RRF 融合后、精排后，排序发生了什么变化？
    - 每一阶段花了多少毫秒？
  这些正是「不仅能看到结果，还能解释结果是怎么来的」的数据基础（docs/开发文档.md 的 WP9 / WP18）。

【为什么用 Redis 而不是进程内 dict？】
  1. uvicorn 多 worker 时，进程内 dict 各存各的，查不到别个 worker 的 trace；
  2. Redis 天然带 TTL，trace 数据可以自动过期，不会无限膨胀。
  Trace 不是业务核心数据，丢了不影响检索，所以即便 Redis 短暂不可用也应当降级而非报错。
"""
import json
from typing import Any

from redis import Redis

from app.config import settings


class TraceStore:
    """Redis 支撑的 Trace 存取。save/get 都做了容错：Redis 不可用时静默降级。"""

    def __init__(self, redis_url: str | None = None) -> None:
        url = redis_url or settings.redis_url
        self.redis = Redis.from_url(url, decode_responses=True)

    def save(self, trace: dict[str, Any]) -> bool:
        """保存一条 trace。request_id 由调用方保证唯一。返回是否真正写入成功。"""
        request_id = trace.get("request_id")
        if not request_id:
            return False
        try:
            # ensure_ascii=False 让中文原样存储；default=str 兜底 UUID/float 等类型
            self.redis.set(
                f"trace:{request_id}",
                json.dumps(trace, ensure_ascii=False, default=str),
                ex=settings.trace_ttl,
            )
            return True
        except Exception:  # noqa: BLE001 —— Trace 不是核心链路，Redis 挂了不影响检索
            return False

    def get(self, request_id: str) -> dict[str, Any] | None:
        """按 request_id 取回一条 trace；不存在或 Redis 不可用返回 None。"""
        try:
            raw = self.redis.get(f"trace:{request_id}")
        except Exception:  # noqa: BLE001
            return None
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None


def candidates_snapshot(candidates: list, limit: int = 10) -> list[dict[str, Any]]:
    """把一列候选结果压缩成 trace 里的紧凑快照。

    只保留排序位置 + chunk_id + 各阶段得分，不重复存整段 content（太占空间）。
    快照里用 rank 字段显式标注"第几名"，比依赖列表顺序更稳（切片后也不会丢位置）。
    """
    return [
        {
            "rank": rank,
            "chunk_id": str(c.chunk_id),
            "retrieval_method": c.retrieval_method,
            "original_score": c.original_score,
            "fusion_score": c.fusion_score,
            "rerank_score": c.rerank_score,
        }
        for rank, c in enumerate(candidates[:limit], start=1)
    ]
