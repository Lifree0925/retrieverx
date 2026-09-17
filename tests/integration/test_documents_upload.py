"""上传接口的三道加固：走真实 HTTP 层验证（不连 ES/Qdrant，用假存储替换依赖）。

【为什么辅助函数有单测了还要在接口层再测一遍】
  `_read_upload_to_temp_file` / `_rollback_document` 的单测只能证明"函数本身对"，
  证明不了**接线对**——比如"Qdrant 写失败时到底有没有真去回滚"
  "回滚也失败时响应有没有如实说明"。这类"函数都对、拼起来不对"的问题，
  只有从接口层打进去才看得见。
"""
import contextlib
import pathlib

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_bm25_dep, get_vector_dep
from app.config import settings
from app.main import app

DEMO_PDF = pathlib.Path(__file__).resolve().parents[2] / "data" / "产品手册-PX4200.pdf"
PDF_CONTENT_TYPE = "application/pdf"


class _FakeStore:
    """假存储：只记录被调用过什么，不连真库。"""

    def __init__(self, fail_write: bool = False, fail_delete: bool = False):
        self.fail_write = fail_write
        self.fail_delete = fail_delete
        self.written = 0
        self.deleted: list = []

    def ensure_index(self) -> None:  # ES 侧
        pass

    def ensure_collection(self) -> None:  # Qdrant 侧
        pass

    def index_chunks(self, chunks) -> None:
        if self.fail_write:
            raise RuntimeError("connection refused")
        self.written += len(chunks)

    def delete_document(self, document_id) -> int:
        if self.fail_delete:
            raise RuntimeError("connection refused")
        self.deleted.append(document_id)
        return 1


@contextlib.contextmanager
def _client_with(bm25, vector):
    app.dependency_overrides[get_bm25_dep] = lambda: bm25
    app.dependency_overrides[get_vector_dep] = lambda: vector
    try:
        yield TestClient(app)
    finally:
        # 必须清理：依赖覆盖是全局的，留着会污染同一进程里后面的测试
        app.dependency_overrides.clear()


def _upload(client, payload: bytes, filename: str = "demo.pdf"):
    return client.post(
        "/documents/index",
        files={"file": (filename, payload, PDF_CONTENT_TYPE)},
    )


# ----------------------------------------------------------------------
# 正常路径
# ----------------------------------------------------------------------
def test_upload_writes_to_both_stores_and_reports_full_size():
    bm25, vector = _FakeStore(), _FakeStore()
    with _client_with(bm25, vector) as client:
        response = _upload(client, DEMO_PDF.read_bytes())

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is True
    assert body["chunks"] > 0
    # 流式分块读取必须把整份文件写全（少写一段会静默解析出更少的块）
    assert body["size_bytes"] == DEMO_PDF.stat().st_size
    assert bm25.written == vector.written == body["chunks"]


# ----------------------------------------------------------------------
# 加固一：体积上限
# ----------------------------------------------------------------------
def test_upload_rejects_oversized_file(monkeypatch):
    monkeypatch.setattr(settings, "max_upload_mb", 1)
    bm25, vector = _FakeStore(), _FakeStore()
    oversized = b"%PDF-1.4 " + b"x" * (2 * 1024 * 1024)

    with _client_with(bm25, vector) as client:
        response = _upload(client, oversized, filename="big.pdf")

    assert response.status_code == 413
    assert "上限" in response.json()["detail"]
    # 被拒绝时绝不能往索引里写任何东西
    assert bm25.written == 0 and vector.written == 0


def test_upload_rejects_non_pdf():
    bm25, vector = _FakeStore(), _FakeStore()
    with _client_with(bm25, vector) as client:
        response = client.post(
            "/documents/index", files={"file": ("a.txt", b"hello", "text/plain")}
        )
    assert response.status_code == 400
    assert bm25.written == 0


# ----------------------------------------------------------------------
# 加固二：双写失败要回滚
# ----------------------------------------------------------------------
def test_failed_second_write_rolls_back_both_stores():
    """Qdrant 写失败时，ES 里已经写进去的那半份必须一并清掉。

    否则就是经典的"半索引"：BM25 搜得到、向量搜不到，而接口还返回了错误，
    用户以为彻底没写进去。
    """
    bm25, vector = _FakeStore(), _FakeStore(fail_write=True)
    with _client_with(bm25, vector) as client:
        response = _upload(client, DEMO_PDF.read_bytes())

    assert response.status_code == 503
    assert vector.deleted, "写入失败的存储侧要回滚"
    assert bm25.deleted, "已写入的存储侧同样要回滚——不能留下半索引"
    assert "回滚" in response.json()["detail"]


def test_failed_rollback_is_reported_instead_of_pretending_success():
    """回滚本身失败时，响应必须如实说出来，并给出人工清理办法。"""
    bm25, vector = _FakeStore(fail_delete=True), _FakeStore(fail_write=True)
    with _client_with(bm25, vector) as client:
        response = _upload(client, DEMO_PDF.read_bytes())

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "回滚" in detail
    assert "DELETE" in detail, "回滚失败时要告诉调用方怎么手动清理"
