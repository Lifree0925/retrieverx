"""
make_demo_pdf.py —— 生成 RetrieverX 演示用 PDF（带标题层级 + 带边框表格）

【为什么需要它】
  data/ 目录里没有可索引的 PDF，无法端到端验证「解析 → 切块 → 双路检索 → 融合 → 精排」。
  这个脚本用 PyMuPDF 画一份结构完整的《PX-4200 企业级服务器产品手册》：
    - 多级标题（字号 >= 14pt，触发 PDFParser 的标题启发式判定）
    - 带真实边框线的表格（PyMuPDF 的 find_tables() 靠线条检测表格区域）
    - 足够长的正文（让 StructureAwareChunker 能切出多个 Chunk）

【两个踩过的坑，改脚本时别踩回去】
  1. 字体整份嵌入 → PDF 32MB。
     PyMuPDF 不支持按需嵌入，用系统字体就要在保存前调 doc.subset_fonts()
     （本机 PyMuPDF 1.28 自带子集化，不需要额外装 fontTools）。
  2. 标题和正文被提取成同一个块 → heading_path 被正文污染。
     解决：① 用 insert_textbox 而不是 insert_text（每次插入有独立矩形）；
           ② 标题前后留足间距。实测 16~26pt 间距即可稳定分块。

【用法】
  cd retrieverx
  venv\\Scripts\\python.exe scripts/make_demo_pdf.py
  输出：data/产品手册-PX4200.pdf
"""
import sys
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

        ⚠️ 必须画出真实的横竖线：PyMuPDF 的 find_tables() 基于页面线条检测表格，
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


def build() -> Path:
    doc = fitz.open()
    d = Drawer(doc)

    d.heading("PX-4200 企业级服务器产品手册", size=20)

    d.heading("第一章 产品概述")
    d.paragraph(
        "PX-4200 是面向中小企业数据中心推出的双路机架式服务器，整机高度 2U，"
        "支持两颗第三代英特尔至强可扩展处理器，单颗最高 32 核心，整机最大 64 核心 128 线程。"
        "产品定位于虚拟化整合、数据库中间件以及中等规模的容器编排场景，"
        "在性能、扩展性与功耗之间取得了较好的平衡。"
    )
    d.paragraph(
        "整机标配 64GB DDR4 ECC 内存，提供 16 个内存插槽，最高可扩展至 1TB。"
        "存储方面前置支持 8 个 2.5 英寸热插拔硬盘位，支持 SAS、SATA 与 NVMe 三种介质混插，"
        "并可选配硬件 RAID 卡，提供 RAID 0/1/5/6/10 级别支持，"
        "在保障数据安全的同时兼顾读写性能。"
    )
    d.paragraph(
        "网络方面，PX-4200 主板集成四口千兆网卡，并预留两个 PCIe 4.0 x16 扩展槽，"
        "可灵活加装万兆光卡、25G 网卡或智能网卡。电源采用 1+1 冗余设计，"
        "单电源额定功率 800W，支持热插拔与在线更换，单路故障不影响业务连续性。"
    )

    d.heading("1.1 规格参数", size=15)
    d.paragraph(
        "下表列出了 PX-4200 系列在售的三个配置版本的处理器、内存、存储与价格信息，"
        "采购时请以配置编码为准，不同配置的价格与交付周期存在差异。"
    )
    d.table(
        [
            ["配置编码", "处理器", "内存", "存储", "价格（元）"],
            ["PX-4200-A", "至强 4310 ×1", "64GB", "2×480G SSD", "32800"],
            ["PX-4200-B", "至强 4314 ×2", "128GB", "4×960G SSD", "58600"],
            ["PX-4200-C", "至强 5318Y ×2", "256GB", "2×1.92T NVMe", "89900"],
        ],
        col_widths=[80, 110, 70, 110, 105],
    )
    d.paragraph(
        "以上价格均为含税单价，不含三年上门服务费用。批量采购十台以上可申请项目折扣，"
        "具体折扣比例由销售经理根据项目规模核定，一般不高于标价的百分之十五。"
    )

    d.heading("第二章 售后服务政策")
    d.paragraph(
        "本章说明 PX-4200 系列整机及随机附件的售后服务范围、响应时限与保修期限。"
        "所有服务条款自设备签收之日起生效，以购机发票与保修卡载明日期为准，"
        "两者不一致时以较晚的日期为准。"
    )

    d.heading("2.1 保修期限", size=15)
    d.paragraph(
        "整机标准保修期为三年，自签收之日起计算。处理器、内存、主板等核心部件随整机享受三年保修；"
        "硬盘、电源、风扇等易损耗部件保修期为两年；随机附带的线缆、导轨等配件保修期为一年。"
        "超出保修期的设备，可以提供有偿维修服务，费用按照官方备件价格加收人工费计算。"
    )
    d.paragraph(
        "以下情形不属于免费保修范围：因人为损坏、进水、跌落导致的故障；"
        "未经授权擅自拆卸、改装或更换非原厂配件导致的问题；"
        "因不可抗力如火灾、地震、雷击造成的设备损坏；"
        "以及使用非官方推荐的电源环境或超出标称范围的环境温湿度所引发的故障。"
    )

    d.heading("2.2 服务响应时限", size=15)
    d.paragraph(
        "客户报障后，服务台会在两个工作小时内完成初步响应与故障分级。"
        "不同服务等级对应的现场到达时限与备件更换承诺如下表所示，"
        "服务等级由客户在购买时选择，服务期内可申请升级。"
    )
    d.table(
        [
            ["服务等级", "响应时间", "现场到达", "备件更换"],
            ["基础服务", "4 小时", "次工作日", "5 个工作日"],
            ["标准服务", "8 小时", "8 小时", "次工作日"],
            ["高级服务", "30 分钟", "4 小时", "4 小时"],
        ],
        col_widths=[90, 100, 100, 110],
    )

    d.heading("2.3 退换货政策", size=15)
    d.paragraph(
        "设备自签收之日起七个自然日内，在包装完整、配件齐全且无外观损伤的前提下，"
        "可以申请无理由退货，往返运费由客户承担。"
        "签收之日起十五个自然日内，若出现非人为因素导致的性能故障，"
        "可以凭官方检测报告申请换货，更换设备为同型号同配置的全新产品。"
    )
    d.paragraph(
        "定制配置机型（包括加装特殊网卡、定制硬盘容量组合、预装指定操作系统等）"
        "不适用无理由退货条款，但同样享受标准保修与故障换新服务。"
        "退换货申请统一通过官方服务热线或客户门户提交，"
        "审核通过后会在三个工作日内安排上门取件。"
    )

    # 关键：子集化嵌入字体，否则 PDF 会有 30MB+
    doc.subset_fonts()

    out = Path(__file__).resolve().parent.parent / "data" / "产品手册-PX4200.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    doc.close()
    return out


if __name__ == "__main__":
    path = build()
    print(f"已生成: {path.name}  ({path.stat().st_size / 1024:.1f} KB)\n")

    # 自检：用项目自己的解析器看结构识别得对不对
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.core.parser import PDFParser

    blocks = PDFParser().parse_pdf(path)
    print(f"PDFParser 解析出 {len(blocks)} 个块：")
    for b in blocks:
        kind = "表格" if b.content_type == "table" else "正文"
        heading = " > ".join(b.heading_path) or "(无标题)"
        print(f"  p{b.page_number} [{kind}] {heading}")
        print(f"        {b.content[:58].replace(chr(10), ' ⏎ ')}...")
