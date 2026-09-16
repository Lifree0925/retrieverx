"""
scripts/run_feedback_experiment.py —— 反馈驱动的策略更新前后对比（Feedback Before/After）

【用法】
  python scripts/run_feedback_experiment.py                        # 用内置演示反馈集
  python scripts/run_feedback_experiment.py --feedback data/feedback_seed.jsonl

【做什么】（对应 docs/开发文档.md 的 WP15）
  1. 用初始策略在评测集上跑一遍 → "Before" 指标；
  2. 读入一份反馈集，按「反馈 → 策略更新」的规则做**小步长**调整；
  3. 冻结更新后的策略版本，再在同一评测集上跑一遍 → "After" 指标；
  4. 对比 Before / After，并报告负反馈率、更正命中率。

【策略更新规则（刻意做成保守、可解释的启发式）】
  - 只统计「负反馈（negative）」，且按查询类型分组；
  - 某类型负反馈达到 min_samples 且负反馈率 > 0.5 时，才动它的权重：
      EXACT / NUMERIC → BM25 权重 +1 步（精确/数值查询更该靠关键词）；
      SEMANTIC        → 向量权重 +1 步（语义查询更该靠向量）；
      MIXED           → 不动（方向不明确，宁可不调）；
  - 单步大小、最小样本数都走 RetrievalPolicy 自带的约束（min_samples / max_step），
    满足"最小样本量 + 限制步长 + 可回滚"的 WP15 要求。

【诚实边界】
  - 反馈集是**固定测试集**，不能因为"拿它调过策略"就把它当评测集（数据污染）。
    本脚本里"评测集"（--dataset）与"反馈集"（--feedback）是两个独立文件，默认不同源。
  - 内置演示反馈集只有几条，样本极小，只能演示链路，不能当结论。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api.dependencies import get_retrieval_service  # noqa: E402
from app.config import settings  # noqa: E402
from app.core.query_classifier import QueryType  # noqa: E402
from app.evaluation.ablation import evaluate_predictions, sample_key  # noqa: E402
from app.evaluation.dataset import load_dataset  # noqa: E402
from app.models.feedback import FeedbackEntry  # noqa: E402
from app.utils.logging import configure_logging, log_json  # noqa: E402

configure_logging(settings.log_level)

# 内置演示反馈集（极小的冒烟数据，只能演示链路，不能当结论）
DEMO_FEEDBACK = [
    {"query": "PX-4200 的价格", "label": "negative"},
    {"query": "PX-4200 的保修期", "label": "negative"},
    {"query": "为什么近期销售下滑", "label": "negative"},
]


def _run_eval(service, dataset, top_k: int) -> dict:
    """对评测集跑一遍完整链路，返回 Recall/MRR/NDCG。"""
    prediction_map = {}
    for index, sample in enumerate(dataset):
        resp = service.run_pipeline(sample.query, top_k=top_k, use_rerank=True)
        # 按 sample_key 对齐（而不是 query 字符串）：重复问题用 query 做 key 会互相覆盖
        prediction_map[sample_key(sample, index)] = [str(c.chunk_id) for c in resp.candidates]
    return evaluate_predictions(dataset, prediction_map, k=top_k)


def _load_feedback(path: str | None) -> list[FeedbackEntry]:
    """读反馈集（JSONL），没有文件则用内置演示集。"""
    if path and Path(path).exists():
        entries = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(FeedbackEntry.model_validate_json(line))
        return entries

    print("未提供 --feedback 或文件不存在，使用内置演示反馈集（仅演示链路）。")
    return [FeedbackEntry(**item) for item in DEMO_FEEDBACK]


def _apply_feedback_policy(service, feedback: list[FeedbackEntry]) -> None:
    """把反馈集转成策略更新（只认负反馈、按类型分组、小步长）。"""
    policy = service.policy
    classifier = service.classifier

    # 按类型统计负反馈；同时统计该类型的**反馈总数**，作为 min_samples 门槛的分母。
    #
    # 【为什么分母要用总数而不是负反馈条数】
    #   `POLICY_MIN_SAMPLES` 的语义是"这个查询类型攒够多少条反馈样本才允许动策略"，
    #   防的是"一两条差评就调权重"。如果拿负反馈条数当样本数，
    #   就变成"差评攒够 20 条就调"——门槛的含义被悄悄换掉了：
    #   一个只有 3 条反馈、其中 3 条差评的类型（样本极少、却 100% 差评）会更容易触发调整，
    #   而一个 200 条反馈、20 条差评的类型反而更难。这跟"防止样本不足"的初衷正好相反。
    neg_by_type: dict[QueryType, int] = {}
    total_by_type: dict[QueryType, int] = {}
    for item in feedback:
        qtype = classifier.classify(item.query)
        total_by_type[qtype] = total_by_type.get(qtype, 0) + 1
        if item.label == "negative":
            neg_by_type[qtype] = neg_by_type.get(qtype, 0) + 1

    for qtype, neg_count in neg_by_type.items():
        step = policy.max_step
        if qtype in (QueryType.EXACT, QueryType.NUMERIC):
            bm25_delta, vector_delta = step, -step
        elif qtype == QueryType.SEMANTIC:
            bm25_delta, vector_delta = -step, step
        else:  # MIXED：方向不明确，不动
            print(f"  - {qtype.value}：负反馈 {neg_count} 条，方向不明确，跳过")
            continue

        sample_count = total_by_type.get(qtype, neg_count)
        # 【必须先把旧版本号记下来】update() 会把新策略写回并返回它，
        # 之后 policy.get(qtype) 拿到的就是**新的**那份，
        # 于是 `new_policy.version != policy.get(qtype).version` 恒为 False，
        # 日志永远打印"[样本不足/未满足约束，未更新]"——日志在骗人。
        version_before = policy.get(qtype).version
        new_policy = policy.update(qtype, bm25_delta, vector_delta, sample_count=sample_count)
        changed = new_policy.version != version_before
        print(
            f"  - {qtype.value}：负反馈 {neg_count} 条 / 样本 {sample_count} 条 → "
            f"BM25 {new_policy.bm25:.2f} / 向量 {new_policy.vector:.2f} "
            f"(v{new_policy.version})"
            + ("" if changed else "  [样本不足/未满足约束，未更新]")
        )


def _correction_hit_rate(service, feedback: list[FeedbackEntry]) -> float:
    """更正命中率：把 correction 写入更正缓存后，重搜原问题命中 used_correction 的比例。"""
    corrections = [f for f in feedback if f.label == "correction" and f.correct_answer]
    for item in corrections:
        service.feedback_store.add(item)
    if not corrections:
        return 0.0
    hits = 0
    for item in corrections:
        resp = service.run_pipeline(item.query, top_k=5, use_rerank=False)
        if resp.used_correction:
            hits += 1
    return hits / len(corrections)


def main() -> None:
    parser = argparse.ArgumentParser(description="反馈驱动策略更新 前后对比实验")
    parser.add_argument("--dataset", default="data/evaluation.jsonl")
    parser.add_argument("--feedback", default=None, help="反馈集 JSONL 路径")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    feedback = _load_feedback(args.feedback)
    service = get_retrieval_service()

    print(f"评测集 {len(dataset)} 条，反馈集 {len(feedback)} 条\n")

    print("[1/3] Before：初始策略评测 ...")
    before = _run_eval(service, dataset, args.top_k)

    print("[2/3] 应用反馈 → 策略更新 ...")
    _apply_feedback_policy(service, feedback)

    print("[3/3] After：冻结版本重新评测 ...")
    after = _run_eval(service, dataset, args.top_k)

    neg_count = sum(1 for f in feedback if f.label == "negative")
    neg_rate = neg_count / len(feedback) if feedback else 0.0
    corr_rate = _correction_hit_rate(service, feedback)

    k = args.top_k
    print("\n" + "=" * 46)
    print(f"{'指标':<16}{'Before':>12}{'After':>12}")
    print("=" * 46)
    for metric in (f"Recall@{k}", f"MRR@{k}", f"NDCG@{k}"):
        print(f"{metric:<16}{before[metric]:>12.4f}{after[metric]:>12.4f}")
    print(f"{'负反馈率':<16}{neg_rate:>11.2%}{neg_rate:>12.2%}")
    print(f"{'更正命中率':<16}{corr_rate:>11.2%}{corr_rate:>12.2%}")

    report = {
        "before": {m: before[m] for m in (f"Recall@{k}", f"MRR@{k}", f"NDCG@{k}")},
        "after": {m: after[m] for m in (f"Recall@{k}", f"MRR@{k}", f"NDCG@{k}")},
        "negative_feedback_rate": neg_rate,
        "correction_hit_rate": corr_rate,
    }
    log_json("feedback_experiment_finished", **report)

    print("\n注：本实验的反馈集与评测集是两份独立数据，避免数据污染；")
    print("    内置演示反馈集样本极小，结论仅供链路演示。")


if __name__ == "__main__":
    main()
