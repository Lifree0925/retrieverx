"""
scripts/corpus_utils.py —— 语料加载与 chunk 池构造（build_eval_set / run_evaluation 共用）

【为什么必须是"共用的一份"】
  chunk_id 由文件内容 sha256 派生，而"文件内容 → chunk_id"这条链路
  要经过 PDFParser 的标题判定、表格识别，以及 StructureAwareChunker 的合并规则。
  只要两次调用的参数不同（比如一处用 min_chars=1200、另一处改了默认值），
  算出来的 chunk_id 就会不一样，于是"评测集校验"会报出一种**根本不存在的**错误：
  标注明明是对的，却提示"chunk_id 找不到"。
  所以解析与切块必须只有一份实现，两边都 import 它。
"""
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"
CORPUS_INDEX = PROJECT_ROOT / "data" / "corpus_index.json"
DEMO_PDF = PROJECT_ROOT / "data" / "产品手册-PX4200.pdf"
DEMO_KEY = "PX4200"


def norm(text: str) -> str:
    """去掉全部空白后再比较。

    PDF 里的换行是**视觉换行**（排到页宽就断），切块时会被 join_pdf_lines 规整。
    所以短语匹配必须对空白不敏感，否则"短语明明在文档里、却匹配不上"，
    会把标注错误误判成"答案不在语料中"。
    """
    return re.sub(r"\s+", "", text or "")


def load_documents() -> dict[str, Path]:
    """收集语料：curated corpus 目录（按 corpus_index.json 的 key）+ 演示文档。"""
    docs: dict[str, Path] = {}
    if CORPUS_INDEX.exists():
        try:
            index = json.loads(CORPUS_INDEX.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            index = {}
        for filename, info in index.items():
            path = CORPUS_DIR / filename
            if path.exists():
                docs[info.get("key") or filename] = path
    if DEMO_PDF.exists():
        docs[DEMO_KEY] = DEMO_PDF
    return docs


def build_chunk_pool(docs: dict[str, Path] | None = None):
    """按"与建索引完全一致"的流程解析 + 切块，返回 {doc_key: [chunk, ...]}。

    必须用 PDFParser + StructureAwareChunker（而不是自己读文本），
    否则 chunk_id 与真正索引进 ES/Qdrant 的对不上，评测就测了另一批数据。
    """
    from app.core.chunker import StructureAwareChunker
    from app.core.parser import PDFParser

    docs = docs if docs is not None else load_documents()
    pdf_parser = PDFParser()
    chunker = StructureAwareChunker()
    return {key: chunker.chunk(pdf_parser.parse_pdf(path)) for key, path in docs.items()}


def build_chunk_universe() -> set[str]:
    """返回"当前语料能派生出的全部 chunk_id"。

    用途：校验评测集里的标注是否仍然有效。
    语料 PDF 被改动过之后，旧标注会指向不存在的 id，
    此时评测会静默地变成"Recall=0"——这个函数就是用来提前发现它的。
    """
    universe: set[str] = set()
    for chunks in build_chunk_pool().values():
        universe.update(str(chunk.chunk_id) for chunk in chunks)
    return universe
