"""
ids.py —— 内容寻址（content-addressed）的 Chunk ID 派生

【为什么不满足于"页码 + 页内序号"】
  早期实现是 `chunk_id = uuid5(document_id, f"{page}:{ordinal}")`。
  看起来也是确定性的，但它把 ID 绑死在**输出顺序**上，于是有个很隐蔽的后果：

    解析器一改（调整页眉页脚过滤门槛、放宽表格识别条件、改切块阈值），
    同一页里前面的块增加或减少 → 后面所有块的 ordinal **集体位移**
    → **文件一个字节都没改，整批 chunk_id 却变了**。
    索引里于是留下"新 ID + 旧残留"两份同样的内容，检索结果出现重复。

  改成由 **Chunk 自身的内容**派生之后就稳了：
    · 同一段文字，不管排在第几页第几个，ID 都一样
      → 重复索引是"就地覆盖"，而不是"新增一条残留"；
    · 内容真的改了，ID 才变（这正是我们要的：旧内容必须失效）；
    · 同一份文档里出现两段**完全相同**的文字时，用"第几次出现"（occurrence）区分，
      否则后一段会把前一段覆盖掉、静默丢一个 chunk。

【为什么把 document_id 也算进种子】
  不算进去的话，两份不同文档里相同的段落会得到同一个 chunk_id。后果有两个：
    1. "删除其中一份文档"会把另一份还在用的 chunk 一起删掉；
    2. 那条 chunk 的 `document_id` 到底属于谁变得含糊。
  代价是相同内容在不同文档里会各存一份。**这个代价是值得的**——
  检索系统的删除语义必须清晰，去重不能以牺牲它为代价。

【不变量】
  对任何一条最终写进索引的 Chunk 都成立：

      chunk_id == uuid5(NAMESPACE_URL, f"chunk:{document_id}:{内容指纹}:{第几次出现}")

  也就是说，**ID 只由"它自己所在文档 + 它自己的内容"决定**。
  不依赖页码、不依赖解析顺序、不依赖解析器的版本。
"""
import hashlib
import re
from collections import Counter
from uuid import NAMESPACE_URL, UUID, uuid5

# 去掉全部空白（含 PDF 视觉换行）。与 scripts/corpus_utils.py 的 norm() 同一套规则，
# 保证"评测集里的标注"与"索引里的 id"用的是同一口径。
_WHITESPACE = re.compile(r"\s+")


def normalize_content(content: str) -> str:
    """把内容规整成"只比较字符、不比较空白"的形式，作为哈希输入。

    PDF 里的换行是**视觉换行**（排到页宽就断），同一段文字重新解析后
    行尾位置可能不同。去掉空白后，这类"只是排版差分"不会被误判成内容改动。
    """
    return _WHITESPACE.sub("", content or "")


def content_digest(content: str) -> str:
    """内容指纹：sha256(去掉空白后的内容)，返回 64 位十六进制字符串。"""
    return hashlib.sha256(normalize_content(content).encode("utf-8")).hexdigest()


class ChunkIdFactory:
    """按文档维护"同一内容已经出现过几次"，把 (内容, 出现次数) 映射成稳定 ID。

    为什么要有状态：单纯用内容哈希会让"同一文档里两段一模一样的文字"撞成同一个 ID，
    后一段覆盖前一段 —— 一个 chunk 就这样静默消失了。
    所以对每个内容指纹计数，用 `出现次数` 作为区分符。
    """

    def __init__(self, document_id: UUID):
        self.document_id = document_id
        self._seen: Counter[str] = Counter()

    def next(self, content: str) -> UUID:
        """为这段内容分配 ID（同一内容第 n 次调用得到第 n 个 ID）。"""
        digest = content_digest(content)
        self._seen[digest] += 1
        return uuid5(
            NAMESPACE_URL,
            f"chunk:{self.document_id}:{digest}:{self._seen[digest]}",
        )
