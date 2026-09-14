"""
hashing.py —— 查询文本的规范化哈希工具

【新手必读】
  用户可能用不同大小写/空格写同一个问题，例如 "SKU A123" 与 "sku  a123"。
  为了把同一问题聚到同一条反馈记录上，先把查询规范化（小写、压缩空格），
  再用 SHA-256 生成固定长度指纹，作为 Redis 里反馈记录的 key。
"""
import hashlib


def hash_query(query: str) -> str:
    """把查询规范化后生成 SHA-256 指纹（64 位十六进制字符串）。

    注意：这是单向哈希，不是为了加密，只是为了"同义写法 → 同一个 key"。
    """
    # 小写 + 按空白切分再拼接 → 压缩连续空格
    normalized = " ".join(query.strip().lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
