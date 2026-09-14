"""
scripts/index_documents.py —— 命令行索引脚本

【用法】
  python scripts/index_documents.py ./data/产品手册.pdf        # 普通索引
  python scripts/index_documents.py ./data/产品手册.pdf --reset # 先清空旧索引再写入
  python scripts/index_documents.py ./data/产品手册.pdf --dry-run  # 只解析切块，不写库（调试用）

【为什么需要这个脚本？】
  除了在 Streamlit/API 上传 PDF，命令行脚本更方便：
  - 一次性批量索引多份 PDF（可写 for 循环调用）；
  - 配合 --dry-run 先看切块效果，再决定要不要入库；
  - 是面试时展示"我有一条可复现的索引流程"的证据。
"""
import argparse
import sys
from pathlib import Path

# 允许 `python scripts/index_documents.py` 这种直接运行方式找到 app 包。
# 注意：Python 只会把「脚本所在目录」加进 sys.path，不会加当前工作目录，
# 所以不加这一行就会报 ModuleNotFoundError: No module named 'app'。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api.dependencies import get_bm25, get_vector  # noqa: E402
from app.core.chunker import StructureAwareChunker  # noqa: E402
from app.core.parser import PDFParser  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="把 PDF 解析、切块并写入 ES 与 Qdrant")
    parser.add_argument("pdf", help="要索引的 PDF 文件路径")
    parser.add_argument("--reset", action="store_true", help="索引前先清空旧数据")
    parser.add_argument("--dry-run", action="store_true", help="只解析切块并打印，不写库")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(f"找不到文件: {pdf_path}")

    # 1) 解析 + 切块
    print(f"[1/4] 解析 PDF: {pdf_path.name}")
    blocks = PDFParser().parse_pdf(pdf_path)
    chunks = StructureAwareChunker().chunk(blocks)
    print(f"      解析出 {len(blocks)} 个块 → 切出 {len(chunks)} 个 Chunk")

    # 0 个 Chunk 说明这份 PDF 一个字都没提取出来（典型：扫描件/纯图片没有文字层）。
    # 必须立刻报错，否则后面会打印"完成！共写入 0 个 Chunk"，让人误以为成功了。
    if not chunks:
        raise SystemExit(
            f"❌ 解析出 {len(blocks)} 个块、切出 0 个 Chunk，没有内容可索引。\n"
            "   常见原因：PDF 是扫描件或纯图片，没有可提取的文字层。\n"
            "   请改用带文字层的 PDF（用 Word/WPS 另存为 PDF 通常没问题）。"
        )

    if args.dry_run:
        # 调试：打印前 10 个 Chunk 的结构预览
        for c in chunks[:10]:
            print(f"  · [{c.content_type}] p{c.page_number} {c.heading_text} -> {c.content[:50]}...")
        return

    # 2) 准备索引 / 集合
    bm25 = get_bm25()
    vector = get_vector()
    if args.reset:
        print("[2/4] 重置索引（删除旧数据）")
        bm25.delete_index()
        vector.delete_collection()
    print("[3/4] 确保索引/集合存在")
    bm25.ensure_index()
    vector.ensure_collection()

    # 3) 写入
    print("[4/4] 写入 ES(BM25) 与 Qdrant(向量) ...")
    bm25.index_chunks(chunks)       # ES 走 bulk，一次刷新
    vector.index_chunks(chunks)     # 向量路批量转向量再写 Qdrant
    print(f"完成！共写入 {len(chunks)} 个 Chunk。ES={bm25.count_all()}，Qdrant={vector.count_all()}")


if __name__ == "__main__":
    main()
