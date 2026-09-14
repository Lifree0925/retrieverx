"""
chunker.py —— 结构感知切块（Structure-aware Chunking）

【这个文件做什么】
  把 PDFParser 产出的、大小不一的文本块，进一步合并成"大小合适、保留结构上下文"的
  Chunk（最终写入向量库/搜索引擎的最小单元）。

【新手必读 · 为什么不能 text[i:i+500] 固定切片】
  - 固定切片会拦腰切断一句话/一个表格，语义破碎；
  - 结构感知切块按"标题 → 段落/表格"的边界来切：
      表格整块保留（拆开会失去行列对应关系）；
      同一标题下的若干段落累积到接近目标大小再成块；
      每个 Chunk 继承标题路径，保证"来源可追溯"。
"""
from app.models.document import DocumentChunk
from app.utils.text import join_pdf_lines


class StructureAwareChunker:
    def __init__(self, min_chars: int = 1200, max_chars: int = 2800):
        """
        min_chars  块内容达到该长度就"可以考虑封口"
        max_chars  内容超过该长度则强制封口（避免 Chunk 过长、噪声变多）
        """
        self.min_chars = min_chars
        self.max_chars = max_chars
        # 换节时的封口门槛：上一节至少要攒到这么多字符才值得单独成块。
        # 定太小 → 幻灯片会被切成大量十几字的碎片（Chunk 太碎，向量检索质量反而变差）；
        # 定太大 → 一节的内容会一路拖到下一节，等于没换节。
        self.section_min_chars = max(1, min_chars // 4)

    def chunk(self, blocks: list[DocumentChunk]) -> list[DocumentChunk]:
        """输入解析块，输出切好的 Chunk。

        算法（按顺序处理每个块）：
          1. 表格块：直接单独成为一个 Chunk（表格语义紧凑，不能和其他文本混）；
          2. 换节检测：heading_path 变了就说明进入新的一节 —— 只要上一节已经攒够
             section_min_chars，就立刻封口（见下方说明）；
          3. 文本块：累积进 buffer；到 min_chars 就封口；
             若加一个新块会超过 max_chars，先封口旧 buffer，再开始累积新块。

        【为什么要有第 2 步】
          早期版本只按字符数封口，完全不看标题。结果一段正文如果标题在第 3 段才变，
          整块会在标题已经变了之后才封口，于是"2.2 节"的正文被贴上了"2.1 节"的标题路径
          —— 溯源信息就错了，而且不同小节的内容混在一个 Chunk 里会互相稀释语义。
        """
        output: list[DocumentChunk] = []
        buffer: list[DocumentChunk] = []
        size = 0
        current_section: tuple[str, ...] | None = None

        for block in blocks:
            if block.content_type == "table":
                # 遇到表格：先把已累积的文本封口，再让表格独占一个 Chunk
                if buffer:
                    output.extend(self._flush(buffer))
                    buffer, size = [], 0
                output.append(block)  # 表格块原样保留（它已经是一个完整语义单元）
                current_section = tuple(block.heading_path)
                continue

            section = tuple(block.heading_path)
            # 换节了，且上一节已经攒够内容 → 立刻封口，避免跨节串味
            if (
                buffer
                and section != current_section
                and size >= self.section_min_chars
            ):
                output.extend(self._flush(buffer))
                buffer, size = [], 0
            current_section = section

            # 文本块：若加入会超出 max_chars，先封口旧内容
            if buffer and size + len(block.content) > self.max_chars:
                output.extend(self._flush(buffer))
                buffer, size = [], 0

            buffer.append(block)
            size += len(block.content)

            # 达到 min_chars 即封口：避免 Chunk 太大
            if size >= self.min_chars:
                output.extend(self._flush(buffer))
                buffer, size = [], 0

        if buffer:  # 收尾：清空剩余
            output.extend(self._flush(buffer))

        return output

    @staticmethod
    def _flush(blocks: list[DocumentChunk]) -> list[DocumentChunk]:
        """把 buffer 里若干相邻块合并成一个 Chunk。

        合并规则：
          - content 用 join_pdf_lines 规整：按句末标点分段、中文直接相接、
            英文之间补空格（详见 app/utils/text.py）；
          - 元数据以第一个块的为准（第一个块通常带标题上下文）；
          - source_pages 记录该 Chunk 横跨了哪些页码（可能有跨页的段落）。
        """
        if not blocks:
            return []

        first = blocks[0]
        # 用 join_pdf_lines 而不是 "\n\n".join：
        # 每个块内部可能还是"视觉行"拼的，直接拼会产生大量短行。
        # 这个函数会按标点和中英混排规则还原成真正的段落。
        content = join_pdf_lines([x.content for x in blocks])
        return [
            first.model_copy(
                update={
                    "content": content,
                    "metadata": {
                        **first.metadata,
                        "source_pages": sorted({x.page_number for x in blocks}),
                    },
                }
            )
        ]
