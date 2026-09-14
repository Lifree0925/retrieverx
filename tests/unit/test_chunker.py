from uuid import uuid4
from app.core.chunker import StructureAwareChunker
from app.models.document import DocumentChunk

def block(text, content_type="text"):
    return DocumentChunk(document_id=uuid4(), content=text, content_type=content_type, page_number=1)

def test_table_independent():
    result = StructureAwareChunker().chunk([
        block("paragraph " * 100),
        block("| SKU | Sales |\n|---|---|\n| A01 | 100 |", "table"),
    ])
    assert any(x.content_type == "table" for x in result)
