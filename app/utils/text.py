"""
text.py —— 文本规整工具

【为什么需要这个模块】
  PDF 里的文本是"视觉行"拼出来的：一行排版到页宽就换行，PyMuPDF 会把它们
  作为独立的 line 返回。如果直接把这些行用 \n 拼起来，会得到两种坏结果：

    1. 显示上：一段话被切成很多条短行。前端 white-space: pre-wrap 会把每个
       \n 当真实换行，于是每行只占十几个字，右侧大片空白、整块被纵向撑得极高。
    2. 检索上：换行符割裂了本来的语义单元，BM25 分词和 embedding 都会受影响。

  所以要在解析/切块阶段把"视觉行"还原成"语义段落"：同一段内的行直接接起来，
  真正换段的地方才保留空行。

【连接规则】
  · 上一段以句末标点结尾（。！？；.!?;）→ 认为换段，用空行分隔；
  · 上一段末尾与下一段开头都是中日韩字符 → 直接相接，不加空格
    （中文没有词间空格，硬加一个空格会污染文本）；
  · 其余情况（英文/数字混排）→ 加一个空格，避免单词粘连。
"""

# 句末标点：出现这些就认为一段话说完了
SENTENCE_END = set("。！？；!?;：:")

# 中日韩文字与全角标点的码位区间
_CJK_RANGES = (
    (0x3000, 0x303F),    # CJK 标点
    (0x3400, 0x4DBF),    # 扩展 A
    (0x4E00, 0x9FFF),    # 统一表意文字
    (0xF900, 0xFAFF),    # 兼容表意文字
    (0xFF00, 0xFFEF),    # 全角字符
)


def is_cjk(char: str) -> bool:
    """判断一个字符是否属于中日韩文字/全角标点。"""
    if not char:
        return False
    code = ord(char)
    return any(low <= code <= high for low, high in _CJK_RANGES)


def join_pdf_lines(parts: list[str]) -> str:
    """把若干"视觉行/块"还原成带段落的正文。

    参数 parts 是按阅读顺序排好的文本片段（可以是一行，也可以是一整段）。
    返回规整后的文本：同段相接，换段用空行。
    """
    out = ""
    for raw in parts:
        part = (raw or "").strip()
        if not part:
            continue
        if not out:
            out = part
            continue

        prev_char = out[-1]
        next_char = part[0]

        if prev_char in SENTENCE_END:
            # 上一句已经说完了 → 真换段
            out += "\n\n" + part
        elif is_cjk(prev_char) and is_cjk(next_char):
            # 中文行末接中文行首：直接相接
            out += part
        else:
            # 英文/数字之间需要空格，否则单词会粘连
            out += " " + part
    return out
