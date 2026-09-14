from app.core.parser import PDFParser

def test_table_to_markdown():
    markdown = PDFParser._table_to_markdown([
        ["SKU", "销售额", "数量"],
        ["A01", "10000", "120"],
    ])
    assert "| SKU | 销售额 | 数量 |" in markdown
    assert "| A01 | 10000 | 120 |" in markdown
