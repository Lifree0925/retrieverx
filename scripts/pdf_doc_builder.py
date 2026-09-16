"""
scripts/pdf_doc_builder.py —— 生成"带标题层级 + 带边框表格"的 PDF（供演示与语料构造共用）

【为什么要单独抽出来】
  `make_demo_pdf.py` 里原本内嵌了一份排版逻辑。再写一个"生成企业语料"的脚本时，
  如果把它抄一遍，就会有两份排版代码需要同步维护——改了一处忘了另一处，
  生成的 PDF 结构就会不一致，进而影响切块结果与评测集里的 chunk_id。
  所以这里抽成唯一一份实现，两个脚本都 import 它。

【三个踩过的坑，改这个文件时别踩回去】
  1. **字体整份嵌入 → PDF 30MB+**。PyMuPDF 不支持按需嵌入，用系统字体就要在保存前
     调 `doc.subset_fonts()` 做子集化（本机 PyMuPDF 1.28 自带，不需要额外装 fontTools）。
  2. **标题和正文被提取成同一个块 → heading_path 被正文污染**。
     解决：① 用 insert_textbox 而不是 insert_text（每次插入有独立矩形）；
           ② 标题前后留足间距（实测 16~26pt 即可稳定分块）。
  3. **表格必须有真实边框线**：`PDFParser` / PyMuPDF 的 `find_tables()` 靠线条检测表格区域，
     只画文字不画线，表格会被当成普通正文。

【用法（给其他脚本调用）】
    from scripts.pdf_doc_builder import render_document

    render_document(out_path, [
        ("title", "PX-4200 产品手册"),
        ("h1", "第一章 产品概述"),
        ("p", "……正文……"),
        ("table", (rows, col_widths)),
    ])
"""
from pathlib import Path

import fitz  # PyMuPDF

FONT_BODY = r"C:\Windows\Fonts\Deng.ttf"      # 等线（正文）
FONT_HEAD = r"C:\Windows\Fonts\Dengb.ttf"     # 等线 Bold（标题，字体名含 Bold）

PAGE_W, PAGE_H = fitz.paper_size("a4")
MARGIN = 60
TOP_START = 90          # 避开 PDFParser 的页眉过滤区（页面顶部 8%）
BOTTOM_SAFE = PAGE_H - 70

GAP_BEFORE_HEADING = 22   # 标题前留白，兼作"块分隔"
GAP_AFTER_HEADING = 6
GAP_BETWEEN_PARAGRAPHS = 10


class Drawer:
    """按顺序往页面上排布内容，自动换页。"""

    def __init__(self, doc: fitz.Document):
        self.doc = doc
        self.page = doc.new_page(width=PAGE_W, height=PAGE_H)
        self._register_fonts(self.page)
        self.y = TOP_START

    @staticmethod
    def _register_fonts(page):
        page.insert_font(fontname="body", fontfile=FONT_BODY)
        page.insert_font(fontname="head", fontfile=FONT_HEAD)

    def _new_page(self):
        self.page = self.doc.new_page(width=PAGE_W, height=PAGE_H)
        self._register_fonts(self.page)
        self.y = TOP_START

    def _write(self, text: str, fontname: str, size: float, lineheight: float = 1.4) -> None:
        """写一段文字，返回后把 y 推进到实际占用高度之后。"""
        max_h = BOTTOM_SAFE - self.y
        if max_h < size * 2:            # 当前页放不下，换页
            self._new_page()
            max_h = BOTTOM_SAFE - self.y
        rect = fitz.Rect(MARGIN, self.y, PAGE_W - MARGIN, BOTTOM_SAFE)
        leftover = self.page.insert_textbox(
            rect, text, fontname=fontname, fontsize=size, lineheight=lineheight
        )
        if leftover >= 0:               # 成功：leftover 是没用掉的高度
            used = rect.height - leftover
        else:                           # 溢出（理论上不会发生，rect 给到了页底）
            used = size * lineheight * (len(text) // 60 + 1)
        self.y += max(used, size * lineheight)

    # ------------------------------------------------------------------
    def heading(self, text: str, size: float = 16.0):
        self.y += GAP_BEFORE_HEADING
        self._write(text, "head", size)
        self.y += GAP_AFTER_HEADING

    def paragraph(self, text: str, size: float = 11.0):
        self.y += GAP_BETWEEN_PARAGRAPHS
        self._write(text, "body", size, lineheight=1.55)

    def table(self, rows, col_widths, row_h: float = 24.0, size: float = 10.0):
        """画一个带边框的表格。

        必须画出真实的横竖线：PyMuPDF 的 find_tables() 基于页面线条检测表格，
        没有边框线就识别不到。
        """
        n_rows, n_cols = len(rows), len(rows[0])
        total_w, total_h = sum(col_widths), row_h * n_rows
        if self.y + total_h + 16 > BOTTOM_SAFE:
            self._new_page()

        x0, y0 = MARGIN, self.y
        shape = self.page.new_shape()
        for i in range(n_rows + 1):
            y = y0 + i * row_h
            shape.draw_line(fitz.Point(x0, y), fitz.Point(x0 + total_w, y))
        for j in range(n_cols + 1):
            x = x0 + sum(col_widths[:j])
            shape.draw_line(fitz.Point(x, y0), fitz.Point(x, y0 + total_h))
        shape.finish(color=(0, 0, 0), width=0.8)
        shape.commit()

        for i, row in enumerate(rows):
            cx = x0
            for j, cell in enumerate(row):
                self.page.insert_text(
                    fitz.Point(cx + 5, y0 + i * row_h + row_h * 0.68),
                    str(cell), fontname="body", fontsize=size,
                )
                cx += col_widths[j]
        self.y = y0 + total_h + 6


def render_document(out_path: str | Path, blocks, extra_metadata: dict | None = None) -> Path:
    """按声明式内容列表渲染一份 PDF。

    blocks 是 (kind, payload) 的序列，kind 取值：
      "title" / "h1" / "h2" / "h3"   → 各级标题（字号递减，都会触发标题层级判定）
      "p"                            → 正文段落
      "table"                        → (rows, col_widths)
    """
    doc = fitz.open()
    drawer = Drawer(doc)
    heading_sizes = {"title": 20.0, "h1": 16.0, "h2": 15.0, "h3": 13.5}

    for kind, payload in blocks:
        if kind in heading_sizes:
            drawer.heading(payload, size=heading_sizes[kind])
        elif kind == "p":
            drawer.paragraph(payload)
        elif kind == "table":
            rows, col_widths = payload
            drawer.table(rows, col_widths)
        else:
            raise ValueError(f"未知的块类型: {kind!r}")

    # 关键：子集化嵌入字体，否则 PDF 会有几十 MB
    doc.subset_fonts()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if extra_metadata:
        doc.set_metadata({k: str(v) for k, v in extra_metadata.items()})
    doc.save(out)
    doc.close()
    return out
