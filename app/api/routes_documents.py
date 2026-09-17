"""
routes_documents.py —— 文档上传 / 索引 / 文档库管理接口

【接口约定】
  POST   /documents/index              上传一个 PDF（multipart/form-data，字段名 file）
                                       服务端：解析 → 切块 → 同时写入 ES 与 Qdrant
                                       返回：{"ok": true, "filename": ..., "blocks": n, "chunks": m}
  GET    /documents                    列出索引里已有的文档（文档库）
  DELETE /documents/{document_id}       只下线其中一份文档（不删整个索引）

【为什么这里是 def 而不是 async def？——这是个容易写错的地方】
  这些接口全程都是【阻塞式】操作：PyMuPDF 解析 50 页 PDF 要几秒，
  写 ES/Qdrant、调 Ollama 算向量也都走同步客户端。

  FastAPI 的规则：
    · `async def` 路由在【事件循环线程】里执行 —— 一旦里面有阻塞调用，
      整个服务在阻塞期间无法处理任何其他请求（不是"这个接口慢"，
      而是"所有人都卡住"）；
    · `def` 路由会被 FastAPI 自动扔进线程池执行 —— 不会堵住事件循环。

  所以这种纯阻塞的接口必须写成 `def`。相应地，读上传文件也不能用
  `await file.read()`（sync 函数里没有 await），改用底层的
  `file.file.read()` —— 它读的就是同一个 SpooledTemporaryFile。

【上传接口的三道加固（都是被审查点出来的真实问题）】
  1. **大小上限**：早期只查后缀，没有任何体积限制 → 一个超大文件就能把内存/磁盘打满。
     现在边写边数字节，超限立刻中断（413），且**不信任 Content-Length**
     （请求头可伪造、分块传输时缺失）。
  2. **流式落盘**：早期是 `file.file.read()` 一次性全量读进内存。
     现在按块读写，几百 MB 的 PDF 也不会把进程撑爆。
  3. **双写失败要回滚**：ES 与 Qdrant 是两个独立存储，先写 ES 再写 Qdrant，
     中间失败就会留下"半索引"（BM25 搜得到、向量搜不到），而接口却返回错误、
     用户以为彻底没写进去。现在任一失败都会把该文档从两边一并清掉，
     让索引回到"要么全有、要么全无"的一致状态。
"""
import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.dependencies import get_bm25_dep, get_vector_dep
from app.config import settings
from app.core.chunker import StructureAwareChunker
from app.core.parser import PDFParser
from app.utils.logging import log_json

router = APIRouter(prefix="/documents", tags=["documents"])

# 流式落盘的块大小。1 MiB 是"系统调用次数"与"单次内存占用"之间的常见折中。
UPLOAD_CHUNK_BYTES = 1 << 20


def _read_upload_to_temp_file(file: UploadFile, suffix: str) -> tuple[str, int]:
    """把上传内容**分块**写到临时文件，返回 (临时文件路径, 字节数)。

    超限时抛 413，并把已经写下的半截临时文件删掉 ——
    "拒绝了请求却留下一个被截断的 PDF"比拒绝本身更糟：
    磁盘上会慢慢堆积一堆看起来正常、其实残缺的文件。
    """
    limit_bytes = settings.max_upload_mb * 1024 * 1024
    written = 0
    temp = NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        with temp:
            while True:
                block = file.file.read(UPLOAD_CHUNK_BYTES)
                if not block:
                    break
                written += len(block)
                if written > limit_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"文件超过 {settings.max_upload_mb} MB 上限"
                            f"（已读取 {written / 1024 / 1024:.1f} MB）。"
                            "如需处理更大的 PDF，调大 .env 里的 MAX_UPLOAD_MB。"
                        ),
                    )
                temp.write(block)
    except BaseException:
        Path(temp.name).unlink(missing_ok=True)
        raise
    return temp.name, written


@router.post("/index")
def index_document(
    file: UploadFile = File(...),
    bm25=Depends(get_bm25_dep),
    vector=Depends(get_vector_dep),
):
    """上传并索引一份 PDF。

    实现要点：
      1. 只收 PDF；**分块**流式落到临时文件（PyMuPDF 需要真实文件路径），并卡体积上限；
      2. 解析 + 切块；切出 0 个 Chunk 直接报 422，不返回"假成功"；
      3. 先确保索引/集合存在，再用 bulk 批量写入 ES（BM25）与 Qdrant（向量）；
         任一路失败就把该文档从两边一起清掉（避免留下"半索引"）；
      4. finally 里清理临时文件，避免磁盘垃圾。
    """
    # 类型校验：文件后缀必须 .pdf
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持 PDF 文件")

    suffix = Path(file.filename).suffix
    temp_path: str | None = None
    try:
        temp_path, size_bytes = _read_upload_to_temp_file(file, suffix)
        if size_bytes == 0:
            raise HTTPException(status_code=400, detail="上传的文件是空的")

        # 1) 解析 PDF → 文本/表格块（把原始文件名带进去，供前端"文档库"展示）
        blocks = PDFParser().parse_pdf(temp_path, source_name=file.filename)
        # 2) 结构感知切块 → 最终 Chunk
        chunks = StructureAwareChunker().chunk(blocks)

        # 【重要】切出 0 个 Chunk 必须报错，不能返回 ok:true。
        # 早期版本会返回 {"ok": true, "blocks": 0, "chunks": 0}，
        # 前端照常显示"索引完成"，用户完全不知道这份 PDF 一个字都没入库
        # （典型场景：扫描件/纯图片 PDF 没有文字层）。
        if not chunks:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"解析出 {len(blocks)} 个块、切出 0 个 Chunk，本次索引为空。"
                    "常见原因：这份 PDF 是扫描件或纯图片，没有可提取的文字层。"
                    "请改用带文字层的 PDF（用 Word/WPS 另存为 PDF 通常没问题）。"
                ),
            )

        document_id = chunks[0].document_id

        # 3) 写索引：确保存在 → 批量写入（两路，任一路失败即回滚）
        bm25.ensure_index()
        vector.ensure_collection()
        try:
            bm25.index_chunks(chunks)        # BM25 路：ES bulk，一次刷新
            vector.index_chunks(chunks)      # 向量路：批量转向量写 Qdrant
        except Exception as exc:  # noqa: BLE001 —— 要把失败原因如实报给调用方
            rolled_back, rollback_error = _rollback_document(bm25, vector, document_id)
            log_json(
                "upload_rollback",
                document_id=str(document_id),
                chunks=len(chunks),
                error=f"{type(exc).__name__}: {exc}",
                rolled_back=rolled_back,
            )
            detail = (
                f"写入索引失败（{type(exc).__name__}: {exc}），"
                "已回滚本次写入，索引中不会残留这份文档的半成品数据。"
            )
            if not rolled_back:
                detail += (
                    f"⚠️ 但回滚也失败了（{rollback_error}），索引可能残留部分数据，"
                    f"请调 DELETE /documents/{document_id} 手动清理后重试。"
                )
            raise HTTPException(status_code=503, detail=detail) from exc

        return {
            "ok": True,
            "filename": file.filename,
            "document_id": str(document_id),   # 前端拿到后可立即用于限定检索
            "size_bytes": size_bytes,          # 落盘字节数（便于确认流式读取的确实是完整文件）
            "blocks": len(blocks),             # 解析出的块数（调试用）
            "chunks": len(chunks),             # 最终 Chunk 数
        }
    finally:
        # 无论成败都删除临时文件
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


def _rollback_document(bm25, vector, document_id) -> tuple[bool, str]:
    """把某份文档从两个索引里都删掉，返回 (是否全部成功, 失败原因)。

    【为什么按 document_id 整份删，而不是删这次写入的那些 chunk_id】
      这份文件的 document_id 由文件内容派生，所以"同一份文件的再上传"会复用同一个
      document_id。此时若只删这次的 chunk，上一条没被覆盖的旧 chunk 仍会残留；
      而按 document_id 整份删，语义是明确的：**这份文档现在不在索引里**，
      调用方重新上传即可。宁可让用户重传一次，也不要留下一个说不清状态的半索引。
    """
    errors: list[str] = []
    try:
        bm25.delete_document(document_id)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"ES: {type(exc).__name__}: {exc}")
    try:
        vector.delete_document(document_id)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Qdrant: {type(exc).__name__}: {exc}")
    return (not errors), "；".join(errors)


@router.get("")
def list_documents(bm25=Depends(get_bm25_dep)):
    """列出索引里已有的文档（前端"文档库"用）。

    以 ES 为准做聚合统计。这份数据是"索引里有什么"，不是"磁盘上有什么"——
    上传是写临时文件的，不会留存原 PDF。
    """
    documents = bm25.list_documents()
    return {"total": len(documents), "documents": documents}


@router.delete("/{document_id}")
def delete_document(
    document_id: uuid.UUID,
    bm25=Depends(get_bm25_dep),
    vector=Depends(get_vector_dep),
):
    """只删除指定文档的全部 Chunk（ES + Qdrant 两边都删）。

    路径参数声明成 uuid.UUID，格式不对 FastAPI 会直接返回 422，不用自己校验。

    【为什么需要这个接口】
      Chunk 的 ID 由文件内容派生，所以"改过的 PDF 重新上传"会得到一批新 ID，
      旧 Chunk 会一直残留在索引里，既污染检索结果也占空间。
      以前只能 `--reset` 把整个索引清空重来，多文档场景下不可接受。
    """
    es_deleted = bm25.delete_document(document_id)
    vector.delete_document(document_id)

    if es_deleted == 0:
        raise HTTPException(
            status_code=404, detail=f"索引里没有找到文档 {document_id}"
        )

    return {
        "ok": True,
        "document_id": str(document_id),
        "deleted_chunks": es_deleted,   # 以 ES 的删除条数为准
    }
