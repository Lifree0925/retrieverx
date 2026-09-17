"""
parser.py —— PDF 结构化解析（PDF Parsing）

【这个文件做什么】
  把一份 PDF 解析成"带结构的文本块"，尽量保留：
    - 标题层级（根据字号梯度推断 H1/H2/H3...）
    - 正文段落
    - 表格（转成 Markdown 表格，单独成块）
    - 页码、标题路径（heading_path）

【新手必读 · 为什么 PDF 不能当纯文本读】
  PDF 存的是"字符画在页面哪个坐标"，而不是"标题/正文/表格"这样的语义结构。
  直接用 page.get_text() 会丢掉标题层级、拆散表格、混入页眉页脚。
  所以我们要在坐标层面自己重建结构，这正是本项目比"只会调 get_text()"更值钱的地方。

【PyMuPDF 关键概念】
  page.get_text("dict") 返回：
    page["blocks"] → 块列表
    block["bbox"]  → 坐标框 (x0,y0,x1,y1)
    block["lines"] → 行列表
    line["spans"]  → 片段列表（同一行可能因字体不同被拆成多个 span）
    span["text"]   → 文本
    span["size"]   → 字号
    span["font"]   → 字体名

【三个踩过的坑，改这个文件前务必看懂】

  ① 标题判定必须用「相对字号」，不能用绝对阈值。
     早期版本写的是 `avg_size >= 14 就算标题`。这在正文 10~11pt 的技术手册上没问题，
     但在幻灯片类 PDF 上会灾难性失效：一份 58 页的 Git 教程，正文是 16.5pt，
     于是【所有正文都被判成标题】塞进了 heading_path，
     结果 content 里一个字都不剩，检索出来全是空表格线。
     现在改成：先统计全文字号直方图（按字符数加权）推断出正文字号，
     只有明显大于正文（>= 1.15 倍）的块才算标题，并把不同字号梯度映射成层级。

  ② heading_path 必须按层级维护，不能无限追加。
     早期版本是 `current_heading = current_heading + [text]`，从不弹出。
     于是"第二章"会被接在"1.1 节"后面变成它的子节点，
     实测一份文档的 heading_path 能涨到 22 个元素。
     现在用栈：压入前先弹出所有层级 >= 当前的标题。

  ③ 表格检测必须加质量门槛，且要按纵向位置取标题。
     - PyMuPDF 的 find_tables() 靠页面线条检测。幻灯片/海报上文本框的边框线
       会被误判成表格，产出大量"1 行 2 列、仅 1 个非空单元格"的假表格。
       实测那份 Git 幻灯片被检出 9 个假表格 —— 它们成了索引里唯一的"内容"。
     - 早期版本在遍历完页面所有文本块之后才调 _extract_tables()，
       传入的 current_heading 已经是"这一页最后一个标题"，
       于是页面中部的表格会被贴上页面末尾的标题（来源标注就错了）。
     现在：先把良构表格挑出来，再把"文本块 + 表格"按纵向位置统一排序后依次处理。
     另外落在表格区域内的文字会被跳过，避免同一内容被索引两遍。
"""
import hashlib
import re
from collections import Counter
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import fitz

from app.models.document import DocumentChunk
from app.utils.ids import ChunkIdFactory
from app.utils.text import join_pdf_lines


class PDFParser:
    """把 PDF 解析成 DocumentChunk 列表（此时还没做切块，块可能偏大/偏小）。"""

    # 页眉页脚过滤：距离页面顶部/底部小于该比例（如 0.08=8%）的文本视为页眉页脚
    HEADER_FOOTER_RATIO = 0.08

    # ---------- 标题判定（相对字号）----------
    HEADING_SIZE_RATIO = 1.15        # 至少比正文大 15% 才算标题
    HEADING_SIZE_TOLERANCE = 0.5     # 浮点字号比较容差
    HEADING_MAX_CHARS = 60           # 标题一般很短；超过就不像标题了
    MAX_HEADING_LEVELS = 4           # 最多识别 4 级标题

    # ---------- 表格质量门槛（挡掉幻灯片边框线造成的假表格）----------
    MIN_TABLE_ROWS = 2
    MIN_TABLE_COLS = 2
    MIN_TABLE_CELLS = 3              # 非空单元格个数
    MIN_TABLE_CHARS = 20             # 全部非空单元格的总字符数

    # 编号标题：字号与正文相同时的兜底判定（如 "1.2.3 安装步骤"）
    NUMBERED_HEADING = re.compile(
        r"^(第[一二三四五六七八九十百0-9]+[章节]"
        r"|\d+(\.\d+)*[\s、.]"
        r"|[（(][一二三四五六七八九十]+[)）])"
    )
    # 以句号/问号/叹号结尾的多半是正文句子，不是标题
    SENTENCE_TAIL = re.compile(r"[。！？；;]$")

    def parse_pdf(
        self, file_path: str | Path, source_name: str | None = None
    ) -> list[DocumentChunk]:
        """解析一个 PDF 文件，返回结构化块列表。

        参数：
          file_path     PDF 路径
          source_name   来源文件名（可选）。会写进每个 Chunk，供前端"文档库"展示；
                        不传则用文件名兜底。

        流程：
          1. 算文件指纹 → 得到确定性的 document_id；
          2. 扫一遍全文，统计字号直方图 → 推断正文字号与"字号→标题层级"映射；
          3. 逐页处理：把文本块与表格按纵向位置排序，边走边维护标题栈，
             遇到标题就压栈，遇到正文/表格就用当前标题路径打标签。
        """
        # 【为什么 ID 要确定性生成，而不是用 uuid4 随机生成】
        #   随机 ID 会导致「重复上传同一份 PDF」时，同一段内容每次都被当成新 Chunk
        #   写进去（chunk_id 不同，upsert 无法覆盖），检索结果里就会出现一模一样的
        #   两条甚至多条。这在开发期反复调参、反复上传时极其常见。
        #   改成"文件内容 sha256 → UUID5 派生 ID"之后：
        #     · 同一份文件反复上传 = 幂等覆盖，不会产生重复；
        #     · 内容改了就得到新的 ID（旧 Chunk 需要 --reset 清掉，见 README）。
        document_id = uuid5(NAMESPACE_URL, self._file_fingerprint(file_path))
        if not source_name:
            source_name = Path(file_path).name
        blocks: list[DocumentChunk] = []

        # chunk_id 由「自身内容」派生，而不是「页码 + 页内序号」。
        # 详见 app/utils/ids.py 的模块注释：绑在顺序上的 ID 会因为解析器一改
        # 就集体位移，文件没动却产生"新 ID + 旧残留"。
        # 这个工厂要**跨页共用**（在 parse_pdf 里建、传给每一页），
        # 否则"同一内容第几次出现"的计数会在每页重置、出现重复 ID。
        id_factory = ChunkIdFactory(document_id)

        with fitz.open(file_path) as pdf:
            # 先验 1：书签（目录）表，作为标题路径的基础
            outline_map = self._build_outline_map(pdf.get_toc(simple=True))
            # 先验 2：字号梯度，决定哪些块算标题、算第几级
            body_size, size_to_level = self._analyze_fonts(pdf)

            for page_index, page in enumerate(pdf):
                page_number = page_index + 1
                blocks.extend(
                    self._parse_page(
                        page=page,
                        page_number=page_number,
                        document_id=document_id,
                        source_name=source_name,
                        size_to_level=size_to_level,
                        base_path=outline_map.get(page_number, []),
                        id_factory=id_factory,
                    )
                )

        return blocks

    @staticmethod
    def _file_fingerprint(file_path: str | Path) -> str:
        """算文件内容指纹（sha256），用作确定性 ID 的种子。

        分块读取而不是一次性 read()：几百 MB 的 PDF 也不至于把内存撑爆。
        """
        digest = hashlib.sha256()
        with open(file_path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()

    # ------------------------------------------------------------------
    # 第一遍：字号分析
    # ------------------------------------------------------------------
    @classmethod
    def _analyze_fonts(cls, pdf) -> tuple[float, dict[float, int]]:
        """统计全文字号直方图，返回 (正文字号, {字号: 标题层级})。

        为什么按「字符数」加权而不是按 span 个数？
          正文 span 数量多但每个可能很短，标题 span 少却可能很长，
          按个数统计容易被标题带偏。按字符数加权更接近"这页大部分字是多大"。

        为什么必须做成自适应的？
          不同来源的 PDF 正文字号差异极大：技术手册正文约 11pt，
          幻灯片正文约 16.5pt。用固定阈值（如 14pt）去判标题，
          在幻灯片上会把整篇正文误判为标题，导致 content 全空。
        """
        hist: Counter = Counter()
        for page in pdf:
            for block in page.get_text("dict").get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "")
                        if text.strip():
                            hist[round(float(span.get("size", 0.0)), 1)] += len(text)

        if not hist:
            return 0.0, {}

        body_size = hist.most_common(1)[0][0]
        bigger = sorted(
            (size for size in hist if size >= body_size * cls.HEADING_SIZE_RATIO),
            reverse=True,
        )[: cls.MAX_HEADING_LEVELS]
        return body_size, {size: i + 1 for i, size in enumerate(bigger)}

    # ------------------------------------------------------------------
    # 第二遍：逐页解析
    # ------------------------------------------------------------------
    def _parse_page(
        self,
        page,
        page_number: int,
        document_id,
        source_name: str,
        size_to_level: dict[float, int],
        base_path: list[str],
        id_factory: ChunkIdFactory,
    ) -> list[DocumentChunk]:
        """解析一页，按纵向顺序处理文本块与表格。

        【为什么要按纵向顺序】
          见文件头注释的坑③：表格必须在"它所在的纵向位置"取当时的标题路径，
          而不是取整页处理完后剩下的最后一个标题。
        """
        # 1) 先挑出良构表格，并记下它们的区域
        tables = self._find_tables(page)
        table_rects = [t["bbox"] for t in tables]

        # 2) 收集本页条目：文本块 + 表格，按 y 坐标排序
        items: list[tuple[str, float, dict]] = []
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:
                continue
            bbox = block.get("bbox", (0.0, 0.0, 0.0, 0.0))
            if any(self._rect_contains(rect, bbox) for rect in table_rects):
                # 落在表格区域内的文字 → 交给表格转 Markdown，避免同一内容索引两遍
                continue
            items.append(("text", bbox[1], block))
        for table in tables:
            items.append(("table", table["bbox"][1], table))
        items.sort(key=lambda item: item[1])

        # 3) 依次处理，维护标题栈
        stack: list[list] = []          # [[level, title], ...]
        output: list[DocumentChunk] = []

        def current_path() -> list[str]:
            # 标题路径 = 书签先验（base_path） + 正文标题栈。
            # 两者会重复：书签里写着"7 Github简介"，页面上同一个标题又会被识别一次，
            # 直接拼接就得到 ['7 Github简介', '7.2 能干嘛', '7 Github简介'] 这种路径。
            # 注意重复项【不相邻】（中间隔着别的标题），所以单纯去连续重复没用。
            # 处理办法：保留每个标题【最后一次】出现的位置 ——
            # 正文标题栈比书签更贴近当前页的真实位置。
            merged = list(base_path) + [title for _level, title in stack]
            last_index = {title: i for i, title in enumerate(merged)}
            return [title for i, title in enumerate(merged) if last_index[title] == i]

        # chunk_id 由**块自身内容**派生（id_factory 由 parse_pdf 建好、跨页共用）。
        # 为什么不再用「页码 + 页内序号」：序号只对"输出的块"递增，
        # 解析器一改（页眉过滤、表格门槛）后面的 ID 就集体位移，
        # 文件没变也会产生"新 ID + 旧残留"。详见 app/utils/ids.py。
        for kind, _y, payload in items:
            if kind == "text":
                text = self._block_text(payload).strip()
                if not text:
                    continue

                level = self._heading_level(payload, size_to_level)
                if level is not None:
                    # 标题不做页眉页脚过滤：幻灯片的大标题常常就贴在最上面
                    self._push_heading(stack, level, text)
                    continue

                # 过滤页眉页脚（公司名/页码等每页重复出现的文本）
                if self._is_header_footer(page.rect, payload.get("bbox", (0, 0, 0, 0))):
                    continue

                output.append(
                    DocumentChunk(
                        chunk_id=id_factory.next(text),
                        document_id=document_id,
                        source_name=source_name,
                        content=text,
                        content_type="text",
                        page_number=page_number,
                        heading_path=current_path(),
                        metadata={"bbox": payload.get("bbox")},
                    )
                )
            else:
                output.append(
                    DocumentChunk(
                        chunk_id=id_factory.next(payload["markdown"]),
                        document_id=document_id,
                        source_name=source_name,
                        content=payload["markdown"],
                        content_type="table",
                        page_number=page_number,
                        heading_path=current_path(),
                        metadata={"table": True, "bbox": payload["bbox"]},
                    )
                )

        return output

    # ------------------------------------------------------------------
    # 标题相关
    # ------------------------------------------------------------------
    @classmethod
    def _heading_level(cls, block: dict, size_to_level: dict[float, int]) -> int | None:
        """判断一个文本块是不是标题；是则返回层级（1 起），否则返回 None。"""
        text = cls._block_text(block).strip()
        if not text or len(text) > cls.HEADING_MAX_CHARS:
            return None

        spans = [
            span
            for line in block.get("lines", [])
            for span in line.get("spans", [])
            if span.get("text", "").strip()
        ]
        if not spans:
            return None

        # 按字符数加权的平均字号
        total_chars = sum(len(span.get("text", "")) for span in spans)
        avg_size = sum(
            float(span.get("size", 0.0)) * len(span.get("text", "")) for span in spans
        ) / max(total_chars, 1)

        for size in sorted(size_to_level, reverse=True):
            if avg_size >= size - cls.HEADING_SIZE_TOLERANCE:
                return size_to_level[size]

        # 兜底：字号和正文一样大，但明显是编号标题且不像句子
        if (
            cls.NUMBERED_HEADING.match(text)
            and not cls.SENTENCE_TAIL.search(text)
            and len(text) <= 40
        ):
            head = text.split()[0] if text.split() else text
            depth = head.count(".") + 1
            return min(depth, cls.MAX_HEADING_LEVELS)

        return None

    @staticmethod
    def _push_heading(stack: list[list], level: int, title: str) -> None:
        """把标题压入标题栈。

        两步：
          ① 如果这个标题【已经】在栈里，先把它以及它之后的所有标题弹掉。
             因为"同一个标题再次出现"意味着我们回到了同一节。
          ② 再弹出所有层级 >= 当前层级 的标题。

        【为什么 ① 必须有】—— 这是幻灯片类 PDF 的典型陷阱：
          同一个标题会以不同字号重复出现（章节引导页用 48pt，后续页的标题栏用 28.9pt），
          于是它会被判成两种不同层级。只在末尾追加的话，路径会变成
              ['4.2 新建|提交', '4.2.3 查看内容', '4.2 新建|提交', '4.2.2 编辑内容']
          自相矛盾 —— 第一节标题又跑到它自己的子节后面去了。
          加上 ① 之后，同样的输入会得到干净的 ['4.2 新建|提交', '4.2.2 编辑内容']。
        """
        for index, entry in enumerate(stack):
            if entry[1] == title:
                del stack[index:]
                break
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append([level, title])

    @classmethod
    def _build_outline_map(cls, outline: list) -> dict[int, list[str]]:
        """把 PDF 书签表转成 {页码: 该页所属标题路径}。

        书签格式：[(level, title, page), ...]，level=1 是一级标题、2 是二级……
        同一页可能命中多个层级的书签，需要按层级正确维护栈。
        """
        result: dict[int, list[str]] = {}
        stacks: dict[int, list[list]] = {}
        for item in outline:
            if len(item) < 3:
                continue
            level, title, page = int(item[0]), str(item[1]), int(item[2])
            stack = stacks.setdefault(page, [])
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append([level, title])
            result[page] = [t for _lvl, t in stack]
        return result

    # ------------------------------------------------------------------
    # 表格相关
    # ------------------------------------------------------------------
    def _find_tables(self, page) -> list[dict]:
        """检测本页的「良构」表格，返回 [{"bbox":..., "markdown":...}]。

        ⚠️ 质量门槛不能去掉。PyMuPDF 靠页面线条检测表格，幻灯片/海报上
        文本框的边框线会被误判：实测一份 58 页的 Git 幻灯片被检出 9 个"表格"，
        每个只有 1 行、1 个非空单元格、3~45 个字符 —— 全是假的。
        去掉门槛的话，这些垃圾会挤掉真正的正文。
        """
        if not hasattr(page, "find_tables"):
            return []
        try:
            finder = page.find_tables()
        except Exception:
            return []

        result: list[dict] = []
        for table in getattr(finder, "tables", []):
            try:
                rows = table.extract()   # rows[i][j] = 单元格文本
            except Exception:
                continue
            rows = rows or []

            n_rows = len(rows)
            n_cols = max((len(row) for row in rows), default=0)
            cells = [
                str(cell).strip()
                for row in rows
                for cell in row
                if cell is not None and str(cell).strip()
            ]
            if (
                n_rows < self.MIN_TABLE_ROWS
                or n_cols < self.MIN_TABLE_COLS
                or len(cells) < self.MIN_TABLE_CELLS
                or sum(len(c) for c in cells) < self.MIN_TABLE_CHARS
            ):
                continue

            markdown = self._table_to_markdown(rows)
            if not markdown:
                continue

            bbox = getattr(table, "bbox", None)
            if not bbox:
                continue
            result.append({"bbox": tuple(bbox), "markdown": markdown})
        return result

    @staticmethod
    def _table_to_markdown(rows: list[list]) -> str:
        """把二维列表转成 Markdown 表格字符串。

        例如 [[产品, 价格], [A, 100]] →
            | 产品 | 价格 |
            |---|---|
            | A | 100 |
        """
        if not rows:
            return ""
        # 空值转空字符串；单元格内的 | 需要转义避免破坏表格语法
        normalized = [
            ["" if v is None else str(v).replace("|", "\\|") for v in row]
            for row in rows
        ]
        width = max(len(row) for row in normalized)
        normalized = [row + [""] * (width - len(row)) for row in normalized]

        lines = [
            "| " + " | ".join(normalized[0]) + " |",
            "| " + " | ".join(["---"] * width) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in normalized[1:])
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 通用辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _block_text(block: dict) -> str:
        """把一个 block 内的多行文本还原成一段正文。

        ⚠️ 不能用 "\\n".join 直接拼。那是 PDF 的【视觉换行】（排到页宽就断），
        不是语义换行。硬保留会同时毁掉显示和检索，详见 app/utils/text.py。
        """
        return join_pdf_lines(
            [
                "".join(span.get("text", "") for span in line.get("spans", []))
                for line in block.get("lines", [])
            ]
        )

    @staticmethod
    def _rect_contains(rect, bbox) -> bool:
        """判断文本块是否落在表格区域内（用中心点判断，避免边缘误差）。"""
        if not rect or not bbox:
            return False
        x0, y0, x1, y1 = rect
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        return x0 <= cx <= x1 and y0 <= cy <= y1

    @staticmethod
    def _is_header_footer(page_rect, bbox) -> bool:
        """按 bbox 的垂直位置判断是否页眉/页脚。

        页眉通常贴在页面顶部，页脚在底部，且占位比例很小。
        只过滤掉位置极靠上/靠下的文本，正文不会误伤。
        """
        if not bbox or page_rect.height <= 0:
            return False
        y_center = (bbox[1] + bbox[3]) / 2
        ratio = y_center / page_rect.height
        return (
            ratio < PDFParser.HEADER_FOOTER_RATIO
            or ratio > 1 - PDFParser.HEADER_FOOTER_RATIO
        )
