"""
dataset.py —— 评测集（Evaluation Dataset）读写

【评测集是什么】
  一组"问题 → 应该命中哪些 Chunk"的人工标注。评测时把系统返回的结果和标注对比，
  就能算出 Recall/MRR/NDCG——用数据证明检索效果好，而不是嘴上说"效果很好"。

【文件格式】
  每行一个 JSON（JSONL 格式），例如：
    {"query": "PX-4200 的价格", "relevant_chunk_ids": ["<uuid>", ...], "query_type": "EXACT"}

【新手提示】
  relevant_chunk_ids 必须是"真实存在的 chunk_id"：
  先索引 PDF，再通过 /search 或 Qdrant 面板找到对应 chunk 的 id，填进来。
  骨架文档里用 "REPLACE_WITH_REAL_CHUNK_ID" 占位是跑不通评测的，要替换成真实 id。
"""
import json
from pathlib import Path

from pydantic import BaseModel, Field


class EvaluationSample(BaseModel):
    """一条评测样本。"""

    query: str                              # 用户问题
    relevant_chunk_ids: list[str] = Field(default_factory=list)  # 标注的相关 chunk id
    query_type: str = "SEMANTIC"            # EXACT/SEMANTIC/MIXED/NUMERIC（辅助分析用）


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
