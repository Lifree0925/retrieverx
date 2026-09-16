"""
scripts/export_evaluation_template.py —— 从已索引数据生成评测集模板

【⚠️ 做正式评测请改用 scripts/build_eval_set.py】
  本脚本保留是为了"手头只有一份任意 PDF、想快速攒一批候选"这种临时场景：
  它按关键词粗筛出候选 chunk，需要**人工核对**后才能用。

  正式评测请用 `scripts/build_eval_set.py`：它面向整套语料，
  按"答案短语"精确定位相关 chunk，且**逐条校验短语确实命中**——
  短语找不到就整体失败退出，不会产出"看着填了、其实指向不存在 chunk"的标注。
  （评测集正确性是可以机械验证的，不该依赖人工核对。）

【原来为什么需要它】
  data/evaluation.jsonl 里的 relevant_chunk_ids 必须是"真实存在的 chunk_id"才能跑评测。
  手写 UUID 容易错。这个脚本把当前索引里的 Chunk 导出成"候选池"，
  你再把每条问题的正确答案 id 填进去。

【用法】
  1. 先索引你的 PDF：python scripts/index_documents.py ./data/产品手册.pdf
  2. 运行：python scripts/export_evaluation_template.py --pdf 产品手册.pdf
     会生成 data/evaluation_template.jsonl，每条样本的 relevant_chunk_ids 已帮你挑好最可能的候选；
  3. 人工检查 / 修改后，另存为 data/evaluation.jsonl 再跑评测。

【说明】
  本脚本"按标题路径 + 关键词粗筛"帮你自动标注候选，属于辅助标注：
  真正的评测集仍建议人工核对，保证相关标注准确。
"""
import argparse
import json
import re
import sys
from pathlib import Path

# 允许 `python scripts/export_evaluation_template.py` 直接运行（同 index_documents.py）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.chunker import StructureAwareChunker  # noqa: E402
from app.core.parser import PDFParser  # noqa: E402
from app.evaluation.dataset import EvaluationSample


def keyword_hits(query: str, chunks) -> list[str]:
    """最简单的粗筛：查询里的词在哪些 chunk 出现最多，返回按命中数排序的 id。"""
    words = re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]{2,}", query.lower())
    scored = []
    for chunk in chunks:
        text = chunk.content.lower()
        count = sum(1 for w in words if w in text)
        if count > 0:
            scored.append((count, str(chunk.chunk_id)))
    scored.sort(reverse=True)
    return [cid for _, cid in scored]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="已索引的 PDF 路径（用于取 chunk 池）")
    parser.add_argument(
        "--questions",
        nargs="*",
        default=[
            "PX-4200 的价格是多少",
            "产品保修政策是什么",
            "怎么降低系统访问延迟",
            "2025 年产品价格表",
        ],
        help="你要评测的问题列表（默认给 4 个示例）",
    )
    parser.add_argument("--output", default="data/evaluation_template.jsonl")
    args = parser.parse_args()

    # 用和索引时一致的流程切块，得到 chunk 池
    blocks = PDFParser().parse_pdf(args.pdf)
    chunks = StructureAwareChunker().chunk(blocks)

    samples = []
    for q in args.questions:
        hits = keyword_hits(q, chunks)[:5]  # 粗筛前 5 个作为候选
        samples.append(EvaluationSample(query=q, relevant_chunk_ids=hits))

    out = Path(args.output)
    with out.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.model_dump(), ensure_ascii=False) + "\n")
    print(f"已生成 {len(samples)} 条评测样本 → {out}")
    print("提示：请人工检查 relevant_chunk_ids 是否确实对应正确答案，再用于评测。")


if __name__ == "__main__":
    main()
