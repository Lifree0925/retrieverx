"""
document.py —— 文档与 Chunk 的数据模型

【新手必读】
  Chunk（文本块）是检索系统的最小单元。PDF 解析 + 切块之后，会得到很多个 DocumentChunk，
  每个 Chunk 都带着"它来自哪一页、属于哪个标题层级、是正文还是表格"等元数据，
  这些信息在检索命中后用来告诉用户"答案来自哪里"。
"""
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """一条切好的文本块。

    字段说明：
      chunk_id       Chunk 唯一标识，由「文件内容指纹 + 页码 + 页内序号」确定性派生，
                     保证同一份文件重复解析得到相同的 ID（重复上传即幂等覆盖）。
      document_id    所属文档标识（由文件 sha256 派生，一份 PDF 的所有 Chunk 共享）。
      source_name    来源文件名（如 "产品手册-PX4200.pdf"），用于前端展示文档库。
      content        文本内容（正文段落，或 Markdown 格式的表格）。
      content_type   内容类型：text=正文段落 / table=表格（已转 Markdown）。
      page_number    来源页码（从 1 开始）。
      heading_path   标题路径，例如 ["产品手册", "服务器", "规格参数"]。
                     它记录了该 Chunk 在文档结构中的位置，用于溯源展示。
      metadata       额外的键值信息（如 bbox 坐标、是否表格），按需扩展。
    """

    chunk_id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    source_name: str | None = None                   # 来源文件名（前端文档库展示用）
    content: str = Field(min_length=1)
    content_type: str = "text"                       # "text" | "table"
    page_number: int = Field(ge=1)
    heading_path: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    @property
    def heading_text(self) -> str:
        """把标题路径拼成可读字符串，例如 '产品手册 > 服务器 > 规格参数'。"""
        return " > ".join(self.heading_path)
