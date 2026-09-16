"""
scripts/run_chunking_ablation.py —— 切块消融：固定切块 vs 结构感知切块

【用法】
  python scripts/run_chunking_ablation.py ./data/产品手册-PX4200.pdf

【做什么】
  对同一份 PDF，分别用「结构感知切块」和「固定大小切块」切一遍，
  对比两者的**结构保留质量**，用数据回答 docs/开发文档.md 的 WP10.5 所提的问题：
  "结构信息保留后，切块质量是否真的改善？"

  对比维度（都是可机械测量、不含主观判断的）：
    1. Chunk 数量与长度分布（固定切块会切出大量尺寸雷同、但语义破碎的块）；
    2. heading_path 覆盖：结构感知切块几乎每个 Chunk 都能溯源到标题层级，
       固定切块则完全没有标题上下文；
    3. 表格整块保留：结构感知切块让每个表格独占一个 Chunk（行列关系不丢），
       固定切块把表格拍平成普通文本、行列关系丢失。

【诚实边界】
  这里的对比是「结构层」的，回答的是"切块质量是否改善"；
  要回答"Recall 是否改善"（docs/开发文档.md 的 WP10.5 的完整命题），需要用同一份评测集
  对两种切块分别建索引、分别跑 Recall 对比——而评测集的 relevant_chunk_ids
  是绑定某一种切块的 chunk_id 的，换切块方式后需要重新标注（脚本不做自动映射，
  避免用一个偷换概念的映射得出虚假结论）。
"""
import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.chunker import FixedChunker, StructureAwareChunker  # noqa: E402
from app.core.parser import PDFParser  # noqa: E402


def _stats(chunks) -> dict:
    lengths = [len(c.content) for c in chunks]
    table_chunks = sum(1 for c in chunks if c.content_type == "table")
    heading_chunks = sum(1 for c in chunks if c.heading_path)
    return {
        "num_chunks": len(chunks),
        "avg_chars": round(statistics.mean(lengths), 1) if lengths else 0,
        "max_chars": max(lengths) if lengths else 0,
        "min_chars": min(lengths) if lengths else 0,
        "heading_path_coverage": round(heading_chunks / len(chunks), 3) if chunks else 0.0,
        "table_chunks_intact": table_chunks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="固定切块 vs 结构感知切块 的结构对比")
    parser.add_argument("pdf", help="要解析的 PDF 文件路径")
    parser.add_argument("--chunk-size", type=int, default=800, help="固定切块的窗口字符数")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(f"找不到文件: {pdf_path}")

    print(f"[1/3] 解析 PDF: {pdf_path.name}")
    blocks = PDFParser().parse_pdf(pdf_path)
    table_blocks = sum(1 for b in blocks if b.content_type == "table")
    print(f"      解析出 {len(blocks)} 个块，其中表格 {table_blocks} 个")

    print("[2/3] 结构感知切块 ...")
    structure_chunks = StructureAwareChunker().chunk(blocks)

    print("[3/3] 固定大小切块（对照基线） ...")
    fixed_chunks = FixedChunker(chunk_size=args.chunk_size).chunk(blocks)

    a = _stats(structure_chunks)
    b = _stats(fixed_chunks)

    print("\n" + "=" * 62)
    print(f"{'指标':<22}{'结构感知':>16}{'固定切块':>16}")
    print("=" * 62)
    rows = [
        ("Chunk 数量", a["num_chunks"], b["num_chunks"]),
        ("平均长度(字符)", a["avg_chars"], b["avg_chars"]),
        ("最大长度(字符)", a["max_chars"], b["max_chars"]),
        ("最小长度(字符)", a["min_chars"], b["min_chars"]),
        ("标题路径覆盖", f"{a['heading_path_coverage']:.1%}", f"{b['heading_path_coverage']:.1%}"),
        ("表格整块保留", a["table_chunks_intact"], b["table_chunks_intact"]),
    ]
    for label, va, vb in rows:
        print(f"{label:<22}{str(va):>16}{str(vb):>16}")

    print("\n解读：")
    print("  - 标题路径覆盖：结构感知切块把标题层级写进 metadata，固定切块为 0 ——")
    print("    后者检索命中后无法溯源『内容来自哪一章哪一节』。")
    print(f"  - 表格整块保留：源文档有 {table_blocks} 个表格，结构感知切块让它们各自独立成块，")
    print("    固定切块把它们拍平成纯文本、行列关系丢失（表格查询会因此召回不到）。")


if __name__ == "__main__":
    main()
