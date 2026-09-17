"""chunk_id 的稳定性和上传接口加固的防线测试。

锁四类问题，全都是"不报错、但结果是错的"：

1. **chunk_id 绑在输出顺序上** —— 解析器一改（页眉过滤门槛、表格识别条件），
   后面的 ID 集体位移，文件一个字节没改却产生"新 ID + 旧残留"。
2. **同一文档里两段完全相同的文字撞成同一个 ID** —— 后一段覆盖前一段，静默丢块。
3. **合并后的 Chunk 沿用第一个块的 ID** —— ID 描述的内容和这条 Chunk 的实际内容对不上。
4. **上传接口没有体积上限 / 全量读进内存** —— 一个请求就能把进程打爆。
"""
import io
import os
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.routes_documents import _read_upload_to_temp_file, _rollback_document
from app.config import settings
from app.core.chunker import FixedChunker, StructureAwareChunker
from app.models.document import DocumentChunk
from app.utils.ids import ChunkIdFactory, content_digest, normalize_content


def _block(text: str, document_id, page: int = 1, content_type: str = "text"):
    return DocumentChunk(
        document_id=document_id,
        content=text,
        content_type=content_type,
        page_number=page,
        heading_path=["第一章"],
    )


# ----------------------------------------------------------------------
# 1. 基本不变量：ID 只是内容的函数
# ----------------------------------------------------------------------
def test_content_digest_ignores_whitespace_only_differences():
    """PDF 的换行是"视觉换行"，重新解析后行尾位置可能变；那不该导致 ID 变化。"""
    assert content_digest("保修期为三年") == content_digest("保修期\n为三年")
    assert normalize_content("a\n b\tc ") == "abc"


def test_same_content_same_id_different_content_different_id():
    doc = uuid4()
    factory_a = ChunkIdFactory(doc)
    factory_b = ChunkIdFactory(doc)
    assert factory_a.next("同一段文字") == factory_b.next("同一段文字")
    assert factory_a.next("另一段文字") != factory_b.next("同一段文字")


def test_duplicate_content_in_one_document_gets_distinct_ids():
    """同一文档里两段一模一样的文字，必须拿到两个 ID，不能互相覆盖。"""
    factory = ChunkIdFactory(uuid4())
    first = factory.next("同上")
    second = factory.next("同上")
    assert first != second, "内容相同不代表可以共用 ID：后一段会把前一段覆盖掉、静默丢一个 chunk"


def test_different_documents_do_not_share_ids():
    """不同文档里的相同段落必须各自有 ID：否则删一份文档会连带删掉另一份还在用的块。"""
    assert (
        ChunkIdFactory(uuid4()).next("同样的段落")
        != ChunkIdFactory(uuid4()).next("同样的段落")
    )


# ----------------------------------------------------------------------
# 2. 关键回归：换页 / 换位置不该改变 ID
# ----------------------------------------------------------------------
def test_chunk_id_does_not_depend_on_page_number():
    """这是审查报告 R5 的核心指控：ID 曾绑在「页码 + 页内序号」上。

    把同样的两段文字放到不同页码，合并出来的 Chunk 内容是一样的，
    ID 就必须一样 —— 否则"解析器一改、页码一挪"就会产生一批新 ID + 一批旧残留。
    """
    doc = uuid4()
    on_page_1 = StructureAwareChunker().chunk(
        [_block("第一段内容。", doc, page=1), _block("第二段内容。", doc, page=1)]
    )
    on_page_9 = StructureAwareChunker().chunk(
        [_block("第一段内容。", doc, page=9), _block("第二段内容。", doc, page=9)]
    )
    assert len(on_page_1) == len(on_page_9) == 1
    assert on_page_1[0].content == on_page_9[0].content
    assert on_page_1[0].chunk_id == on_page_9[0].chunk_id


def test_chunk_id_is_derived_from_merged_content_not_first_block():
    """合并后的 Chunk 不能用第一个块的 ID —— 那个 ID 描述的是"第一个块的内容"。"""
    doc = uuid4()
    chunks = StructureAwareChunker().chunk(
        [_block("甲" * 50, doc), _block("乙" * 50, doc)]
    )
    assert len(chunks) == 1
    merged = chunks[0]
    assert merged.content == "甲" * 50 + "乙" * 50
    expected = ChunkIdFactory(doc).next(merged.content)
    assert merged.chunk_id == expected


def test_chunk_id_is_reproducible_across_two_runs():
    """同样输入跑两遍，ID 必须完全一致（幂等覆盖的前提）。"""
    doc = uuid4()
    blocks = [_block("内容一。", doc), _block("内容二。", doc)]
    first = [c.chunk_id for c in StructureAwareChunker().chunk(blocks)]
    second = [c.chunk_id for c in StructureAwareChunker().chunk(blocks)]
    assert first == second and first


def test_fixed_chunker_ids_are_deterministic():
    """FixedChunker 早期没传 chunk_id → 落到 default_factory=uuid4，每次都不一样。"""
    doc = uuid4()
    blocks = [_block("A" * 2500, doc)]
    first = [c.chunk_id for c in FixedChunker(chunk_size=800).chunk(blocks)]
    second = [c.chunk_id for c in FixedChunker(chunk_size=800).chunk(blocks)]
    assert len(first) == 4
    assert first == second, "消融基线也必须确定性生成 ID，否则重复索引幂等在它身上不成立"


def test_identical_blocks_are_not_swallowed():
    """结构感知切块遇到两段相同文字时，最终 Chunk 数不能变少。"""
    doc = uuid4()
    chunks = StructureAwareChunker().chunk(
        [
            _block("重复段落。" * 30, doc),
            _block("| 表头 |\n|---|\n| 值 |", doc, content_type="table"),
            _block("重复段落。" * 30, doc),
        ]
    )
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "同一份文档内出现了重复 ID，说明有 Chunk 会被覆盖掉"


# ----------------------------------------------------------------------
# 3. 上传接口加固
# ----------------------------------------------------------------------
class _FakeUpload:
    """最小可用的 UploadFile 替身：只需要 .file.read(size)。"""

    def __init__(self, payload: bytes):
        self.file = io.BytesIO(payload)


def test_upload_over_limit_is_rejected_and_leaves_no_temp_file(monkeypatch):
    monkeypatch.setattr(settings, "max_upload_mb", 1)
    tmpdir = tempfile.gettempdir()
    before = set(os.listdir(tmpdir))
    try:
        with pytest.raises(HTTPException) as exc:
            _read_upload_to_temp_file(_FakeUpload(b"x" * (2 * 1024 * 1024)), ".pdf")
        assert exc.value.status_code == 413
    finally:
        leaked = set(os.listdir(tmpdir)) - before
    assert not leaked, f"被拒绝的上传留下了临时文件：{leaked}"


def test_upload_within_limit_is_written_completely(monkeypatch):
    monkeypatch.setattr(settings, "max_upload_mb", 5)
    payload = b"%PDF-1.4 " + b"y" * 10000
    path, written = _read_upload_to_temp_file(_FakeUpload(payload), ".pdf")
    try:
        assert written == len(payload)
        assert Path(path).read_bytes() == payload
    finally:
        Path(path).unlink(missing_ok=True)


class _BrokenStore:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.deleted: list = []

    def delete_document(self, document_id):
        if self.fail:
            raise RuntimeError("connection refused")
        self.deleted.append(document_id)
        return 1


def test_rollback_removes_document_from_both_stores():
    bm25, vector = _BrokenStore(), _BrokenStore()
    doc = uuid4()
    ok, error = _rollback_document(bm25, vector, doc)
    assert ok is True and error == ""
    assert bm25.deleted == [doc] and vector.deleted == [doc]


def test_rollback_reports_failure_instead_of_pretending_success():
    """回滚失败必须如实报告——不能让调用方以为索引已经干净了。"""
    ok, error = _rollback_document(_BrokenStore(fail=True), _BrokenStore(), uuid4())
    assert ok is False
    assert "ES" in error and "connection refused" in error
