"""
dataset.py —— 评测集（Evaluation Dataset）读写与**前置校验**

【评测集是什么】
  一组"问题 → 应该命中哪些 Chunk"的人工标注。评测时把系统返回的结果和标注对比，
  就能算出 Recall/MRR/NDCG——用数据证明检索效果好，而不是嘴上说"效果很好"。

【文件格式】
  每行一个 JSON（JSONL 格式），例如：
    {"query": "PX-4200 的价格", "relevant_chunk_ids": ["<uuid>", ...],
     "query_type": "EXACT", "id": "px-price"}

  字段：
    query               用户问题
    relevant_chunk_ids  人工核对的"真正相关"的 chunk id（**必须是真实存在的 id**）
    query_type          EXACT / SEMANTIC / MIXED / NUMERIC（分层看指标用）
    id                  可选。同一条问题出现多次时必须给，否则靠位置区分

【怎么生成真实 chunk_id】
  不要手写 UUID。跑 `python scripts/build_eval_set.py`：
  它用与建索引相同的解析+切块流程算出 chunk_id，并按"答案短语"定位相关 chunk。
  id 由文件内容 sha256 派生，所以**语料 PDF 一改，旧标注就全部失效**。

【前置校验为什么必须有】
  踩过的坑：`relevant_chunk_ids` 全是 `REPLACE_WITH_REAL_CHUNK_ID` 占位符时，
  评测照样能跑完、照样打印出指标，只是全是 0。
  一个"能跑但毫无意义"的评测比报错危险得多——所以这里显式拦住。
"""
import json
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, Field

# 生成模板里用的占位符：出现在评测集里说明"从没真正标注过"
PLACEHOLDER_IDS = {
    "REPLACE_WITH_REAL_CHUNK_ID",
    "REPLACE_WITH_REAL_ID",
    "TODO",
    "xxx",
}


class EvaluationSample(BaseModel):
    """一条评测样本。"""

    query: str                              # 用户问题
    relevant_chunk_ids: list[str] = Field(default_factory=list)  # 标注的相关 chunk id
    query_type: str = "SEMANTIC"            # EXACT/SEMANTIC/MIXED/NUMERIC（辅助分析用）
    id: str = ""                            # 可选稳定标识；重复问题时必须给
    source_document: str = ""               # 答案来自哪份文档（人工排查用）
    # 是否为"改写式问题"（问法与语料原文用词不同，但答案仍在语料里）。
    # 这个标记很有用：字面式问题 BM25 就能命中，改写式才真正考验语义检索——
    # 两者必须分开看指标，否则一张平均表会把"关键词命中"与"语义命中"混为一谈，
    # 而消融实验要区分的恰恰是这两件事。
    paraphrased: bool = False


def load_dataset(path: str | Path) -> list[EvaluationSample]:
    """读取 JSONL 评测集，逐行解析成 EvaluationSample。"""
    samples = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            samples.append(EvaluationSample.model_validate_json(line))
    return samples


def append_sample(path: str | Path, sample: EvaluationSample) -> None:
    """往评测集追加一条样本（方便边用边攒数据）。"""
    with Path(path).open("a", encoding="utf-8") as file:
        file.write(json.dumps(sample.model_dump(), ensure_ascii=False) + "\n")


def validate_dataset(
    samples: Iterable[EvaluationSample],
    chunk_universe: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    """校验评测集能不能拿来下结论。返回 (errors, warnings)。

    errors   —— 会把结论变成假的，必须拦住：占位符、标注指向不存在的 chunk、
                评测集为空。其中"标注指向不存在的 chunk"几乎总是因为
                **语料 PDF 被改过**（chunk_id 由文件内容哈希派生）。
    warnings —— 不致命，但会削弱结论可信度：缺人工标注（会被排除出指标分母）、
                同一条问题重复出现且没有给 id。

    调用方应当：errors 非空就拒绝跑评测（或要求显式 `--allow-stale-dataset`），
    warnings 照常打印出来让人看到。
    """
    errors: list[str] = []
    warnings: list[str] = []

    samples = list(samples)
    if not samples:
        errors.append("评测集是空的")
        return errors, warnings

    placeholders: list[str] = []
    missing_annotation: list[str] = []
    unknown_ids: list[tuple[str, str]] = []
    seen_queries: dict[str, int] = {}

    for sample in samples:
        seen_queries[sample.query] = seen_queries.get(sample.query, 0) + 1
        if not sample.relevant_chunk_ids:
            missing_annotation.append(sample.query)
            continue
        if any(cid in PLACEHOLDER_IDS for cid in sample.relevant_chunk_ids):
            placeholders.append(sample.query)
        if chunk_universe is not None:
            for cid in sample.relevant_chunk_ids:
                if cid not in chunk_universe:
                    unknown_ids.append((sample.query, cid))

    if placeholders:
        errors.append(
            f"{len(placeholders)} 条样本的 relevant_chunk_ids 还是占位符"
            f"（例如 {placeholders[0]!r}）。这说明评测集从没真正标注过——"
            "请跑 python scripts/build_eval_set.py 生成带真实 chunk_id 的评测集。"
        )
    if unknown_ids:
        example_query, example_id = unknown_ids[0]
        errors.append(
            f"{len(unknown_ids)} 处标注指向当前语料里不存在的 chunk_id"
            f"（例如 {example_query!r} -> {example_id}）。"
            "chunk_id 由 PDF 内容 sha256 派生：语料被改动过就会全部失效。"
            "请重跑 scripts/build_eval_set.py，或用 "
            "`build_eval_set.py --check` 单独体检。"
        )
    if missing_annotation:
        warnings.append(
            f"{len(missing_annotation)} 条样本没有标注相关 chunk，"
            "将被排除在指标分母之外（记为 unverifiable，不算失败）"
        )
    duplicates = {q: n for q, n in seen_queries.items() if n > 1}
    if duplicates:
        warnings.append(
            f"{len(duplicates)} 条问题在评测集里重复出现"
            "（预测按位置对齐，不会互相覆盖，但重复问题会让同一能力被重复计分）"
        )

    return errors, warnings
