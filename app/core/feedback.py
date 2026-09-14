"""
feedback.py —— Redis 反馈存储（Feedback Store）

【这个文件做什么】
  把用户的反馈（positive/negative/correction）存进 Redis，
  并按查询取回历史反馈；correction（更正）可用来改写后续同款查询。

【新手必读 · 面试点：为什么反馈不直接改 BM25 权重？】
  反馈首先是"数据"，先落库、再做统计和实验，而不是一有差评就立刻动权重。
  落库的数据可以用于：
    1. 查询改写（correction 缓存）；
    2. 扩充评测集；
    3. 达到最小样本数后再小步调整策略（见 retrieval_policy.py）。
"""
from redis import Redis

from app.config import settings
from app.models.feedback import FeedbackEntry
from app.utils.hashing import hash_query


class FeedbackStore:
    def __init__(self):
        # decode_responses=True 让 Redis 直接返回字符串而不是 bytes，省去手动解码
        self.redis = Redis.from_url(settings.redis_url, decode_responses=True)

    # ------------------------------------------------------------------
    # 写入 / 读取
    # ------------------------------------------------------------------
    def add(self, feedback: FeedbackEntry) -> None:
        """把一条反馈追加到对应查询的列表末尾，并刷新过期时间。"""
        key = f"feedback:{hash_query(feedback.query)}"
        self.redis.rpush(key, feedback.model_dump_json())  # Pydantic → JSON 字符串
        self.redis.expire(key, settings.feedback_ttl)      # TTL 防止无限膨胀

    def get(self, query: str) -> list[FeedbackEntry]:
        """取回某查询的全部历史反馈（按时间顺序）。"""
        values = self.redis.lrange(f"feedback:{hash_query(query)}", 0, -1)
        return [FeedbackEntry.model_validate_json(x) for x in values]

    # ------------------------------------------------------------------
    # 更正缓存（Correction Cache）
    # ------------------------------------------------------------------
    def get_correction(self, query: str) -> str | None:
        """若某查询曾经被用户更正过，返回最新的正确写法；否则返回 None。

        注意：倒序遍历（reversed），取时间上最近的一条 correction。
        """
        for item in reversed(self.get(query)):
            if item.label == "correction" and item.correct_answer:
                return item.correct_answer
        return None

    # ------------------------------------------------------------------
    # 统计（供策略实验使用）
    # ------------------------------------------------------------------
    def count_labels(self, query: str) -> dict[str, int]:
        """统计某查询三类反馈的数量，返回 {"positive": n, "negative": n, "correction": n}。"""
        counts = {"positive": 0, "negative": 0, "correction": 0}
        for item in self.get(query):
            if item.label in counts:
                counts[item.label] += 1
        return counts

    def total_feedback_count(self) -> int:
        """粗略统计所有反馈条数（扫描 feedback:* 键）。演示/体检用。

        ⚠️ 用 scan_iter 而不是 KEYS。
        KEYS 会一次性遍历整个 keyspace 并且在遍历期间阻塞 Redis 的单线程，
        数据量一上来就是线上事故，属于明确的生产禁用命令。
        SCAN 是游标式分批扫描，不会长时间阻塞；count 只是每次的建议批量大小。
        另外统计长度直接用 LLEN，不要 LRANGE 全量取回来再 len()。
        """
        total = 0
        for key in self.redis.scan_iter(match="feedback:*", count=100):
            total += self.redis.llen(key)
        return total
