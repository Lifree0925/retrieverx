"""FixedChunker 与 Trace 快照的单元测试。"""
from uuid import uuid4

from app.core.chunker import FixedChunker
from app.core.trace import candidates_snapshot
from app.models.document import DocumentChunk
from app.models.retrieval import Candidate


def _block(text, content_type="text"):
    return DocumentChunk(
        document_id=uuid4(),
        content=text,
        content_type=content_type,
        page_number=1,
        heading_path=["第一章"],
    )


def test_fixed_chunker_slices_flat_text():
    text = "A" * 2500  # 2500 字符，chunk_size=800 → 4 块
    chunks = FixedChunker(chunk_size=800, overlap=0).chunk([_block(text)])
    assert len(chunks) == 4
    # 固定切块不保留标题上下文
    assert all(c.heading_path == [] for c in chunks)
    # 表格被拍平成普通文本
    assert all(c.content_type == "text" for c in chunks)


def test_fixed_chunker_skips_empty():
    chunks = FixedChunker(chunk_size=800).chunk([])
    assert chunks == []


def test_fixed_chunker_overlap():
    text = "B" * 1500
    chunks = FixedChunker(chunk_size=800, overlap=200).chunk([_block(text)])
    # step = 800 - 200 = 600；1500 字符 → 起点 0, 600, 1200 → 3 块
    assert len(chunks) == 3
    # 相邻块有重叠：第二块起点 600 与第一块 [0,800) 重叠
    assert chunks[1].content.startswith(text[600:610])


def test_candidates_snapshot_includes_rank():
    cands = [
        Candidate(chunk_id=uuid4(), content="x", source="bm25", retrieval_method="bm25",
                  original_score=1.0, fusion_score=0.5),
        Candidate(chunk_id=uuid4(), content="y", source="vector", retrieval_method="vector",
                  original_score=0.9, fusion_score=0.4),
    ]
    snap = candidates_snapshot(cands)
    assert len(snap) == 2
    assert snap[0]["rank"] == 1
    assert snap[1]["rank"] == 2
    assert "chunk_id" in snap[0]
    assert snap[0]["original_score"] == 1.0
