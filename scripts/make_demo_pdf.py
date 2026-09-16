"""
make_demo_pdf.py —— 生成 RetrieverX 演示用 PDF（《PX-4200 企业级服务器产品手册》）

【为什么需要它】
  data/ 目录里没有可索引的 PDF，无法端到端验证「解析 → 切块 → 双路检索 → 融合 → 精排」。
  这个脚本画一份结构完整的文档：多级标题、带真实边框线的表格、足够长的正文，
  用来验证 PDFParser 的标题层级判定、表格识别与 StructureAwareChunker 的切块。

【排版逻辑去哪了】
  已经抽到 `scripts/pdf_doc_builder.py`（Drawer / render_document）。
  原因：`make_enterprise_corpus.py` 也要生成同类 PDF，
  抄两份排版代码必然漂移，而**排版一变 chunk_id 就变**（id 由文件内容哈希派生），
  评测集里的标注会集体失效。所以只留唯一一份实现。

【用法】
  cd retrieverx
  venv\\Scripts\\python.exe scripts/make_demo_pdf.py
  # 想生成到别处（不覆盖仓库里那份）：
  venv\\Scripts\\python.exe scripts/make_demo_pdf.py --out /tmp/demo.pdf
  输出：data/产品手册-PX4200.pdf
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pdf_doc_builder import render_document  # noqa: E402

BLOCKS = [
    ("title", "PX-4200 企业级服务器产品手册"),

    ("h1", "第一章 产品概述"),
    ("p", "PX-4200 是面向中小企业数据中心推出的双路机架式服务器，整机高度 2U，"
          "支持两颗第三代英特尔至强可扩展处理器，单颗最高 32 核心，整机最大 64 核心 128 线程。"
          "产品定位于虚拟化整合、数据库中间件以及中等规模的容器编排场景，"
          "在性能、扩展性与功耗之间取得了较好的平衡。"),
    ("p", "整机标配 64GB DDR4 ECC 内存，提供 16 个内存插槽，最高可扩展至 1TB。"
          "存储方面前置支持 8 个 2.5 英寸热插拔硬盘位，支持 SAS、SATA 与 NVMe 三种介质混插，"
          "并可选配硬件 RAID 卡，提供 RAID 0/1/5/6/10 级别支持，"
          "在保障数据安全的同时兼顾读写性能。"),
    ("p", "网络方面，PX-4200 主板集成四口千兆网卡，并预留两个 PCIe 4.0 x16 扩展槽，"
          "可灵活加装万兆光卡、25G 网卡或智能网卡。电源采用 1+1 冗余设计，"
          "单电源额定功率 800W，支持热插拔与在线更换，单路故障不影响业务连续性。"),

    ("h2", "1.1 规格参数"),
    ("p", "下表列出了 PX-4200 系列在售的三个配置版本的处理器、内存、存储与价格信息，"
          "采购时请以配置编码为准，不同配置的价格与交付周期存在差异。"),
    ("table", (
        [
            ["配置编码", "处理器", "内存", "存储", "价格（元）"],
            ["PX-4200-A", "至强 4310 ×1", "64GB", "2×480G SSD", "32800"],
            ["PX-4200-B", "至强 4314 ×2", "128GB", "4×960G SSD", "58600"],
            ["PX-4200-C", "至强 5318Y ×2", "256GB", "2×1.92T NVMe", "89900"],
        ],
        [80, 110, 70, 110, 105],
    )),
    ("p", "以上价格均为含税单价，不含三年上门服务费用。批量采购十台以上可申请项目折扣，"
          "具体折扣比例由销售经理根据项目规模核定，一般不高于标价的百分之十五。"),

    ("h1", "第二章 售后服务政策"),
    ("p", "本章说明 PX-4200 系列整机及随机附件的售后服务范围、响应时限与保修期限。"
          "所有服务条款自设备签收之日起生效，以购机发票与保修卡载明日期为准，"
          "两者不一致时以较晚的日期为准。"),

    ("h2", "2.1 保修期限"),
    ("p", "整机标准保修期为三年，自签收之日起计算。处理器、内存、主板等核心部件随整机享受三年保修；"
          "硬盘、电源、风扇等易损耗部件保修期为两年；随机附带的线缆、导轨等配件保修期为一年。"
          "超出保修期的设备，可以提供有偿维修服务，费用按照官方备件价格加收人工费计算。"),
    ("p", "以下情形不属于免费保修范围：因人为损坏、进水、跌落导致的故障；"
          "未经授权擅自拆卸、改装或更换非原厂配件导致的问题；"
          "因不可抗力如火灾、地震、雷击造成的设备损坏；"
          "以及使用非官方推荐的电源环境或超出标称范围的环境温湿度所引发的故障。"),

    ("h2", "2.2 服务响应时限"),
    ("p", "客户报障后，服务台会在两个工作小时内完成初步响应与故障分级。"
          "不同服务等级对应的现场到达时限与备件更换承诺如下表所示，"
          "服务等级由客户在购买时选择，服务期内可申请升级。"),
    ("table", (
        [
            ["服务等级", "响应时间", "现场到达", "备件更换"],
            ["基础服务", "4 小时", "次工作日", "5 个工作日"],
            ["标准服务", "8 小时", "8 小时", "次工作日"],
            ["高级服务", "30 分钟", "4 小时", "4 小时"],
        ],
        [90, 100, 100, 110],
    )),

    ("h2", "2.3 退换货政策"),
    ("p", "设备自签收之日起七个自然日内，在包装完整、配件齐全且无外观损伤的前提下，"
          "可以申请无理由退货，往返运费由客户承担。"
          "签收之日起十五个自然日内，若出现非人为因素导致的性能故障，"
          "可以凭官方检测报告申请换货，更换设备为同型号同配置的全新产品。"),
    ("p", "定制配置机型（包括加装特殊网卡、定制硬盘容量组合、预装指定操作系统等）"
          "不适用无理由退货条款，但同样享受标准保修与故障换新服务。"
          "退换货申请统一通过官方服务热线或客户门户提交，"
          "审核通过后会在三个工作日内安排上门取件。"),
]

DEFAULT_OUT = PROJECT_ROOT / "data" / "产品手册-PX4200.pdf"


def build(out_path: Path | None = None) -> Path:
    return render_document(out_path or DEFAULT_OUT, BLOCKS)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成演示 PDF")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="输出路径（默认覆盖 data/产品手册-PX4200.pdf）")
    parser.add_argument("--no-check", action="store_true", help="跳过解析自检")
    args = parser.parse_args()

    path = build(Path(args.out))
    print(f"已生成: {path}  ({path.stat().st_size / 1024:.1f} KB)\n")
    if args.no_check:
        return

    # 自检：用项目自己的解析器看结构识别得对不对
    from app.core.parser import PDFParser

    blocks = PDFParser().parse_pdf(path)
    print(f"PDFParser 解析出 {len(blocks)} 个块：")
    for b in blocks:
        kind = "表格" if b.content_type == "table" else "正文"
        heading = " > ".join(b.heading_path) or "(无标题)"
        print(f"  p{b.page_number} [{kind}] {heading}")
        print(f"        {b.content[:58].replace(chr(10), ' / ')}...")


if __name__ == "__main__":
    main()
