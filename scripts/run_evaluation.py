"""
scripts/run_evaluation.py —— 评测脚本（跑指标 + 消融实验）

【用法】
  python scripts/run_evaluation.py                          # 默认：混合检索（RRF + 精排）
  python scripts/run_evaluation.py --mode vector_only       # 消融：只看向量
  python scripts/run_evaluation.py --mode bm25_only         # 消融：只看 BM25
  python scripts/run_evaluation.py --mode hybrid            # 消融：RRF 融合（不精排）
  python scripts/run_evaluation.py --mode hybrid_rerank     # 完整链路（默认）
  python scripts/run_evaluation.py --all                    # 一次跑完 4 种模式，输出对比表

【输出什么】
  对评测集（默认 data/evaluation.jsonl）逐条跑检索，统计整体：
    Recall@K / MRR@K / NDCG@K + 平均延迟 + P95 延迟
  跑 --all 就能得到"消融实验表"，这是你简历上"实验数据"的来源。

【跑之前会先体检评测集（这一步很重要）】
  踩过的坑：`relevant_chunk_ids` 全是 `REPLACE_WITH_REAL_CHUNK_ID` 占位符时，
  评测照样跑得完、照样打印指标，只是全是 0——**一个"能跑但毫无意义"的结果
  比直接报错危险得多**。所以这里前置校验：
    · 占位符（说明从没真正标注）        → 直接拒绝运行
    · 标注指向语料里不存在的 chunk_id    → 直接拒绝运行（几乎总是语料被改过）
    · 缺人工标注的样本                   → 警告，并排除出指标分母
  确实想跑"坏数据"看现象，加 --allow-stale-dataset。
"""
import argparse
import statistics
import sys
from pathlib import Path

# 允许 `python scripts/run_evaluation.py` 直接运行（同 index_documents.py）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api.dependencies import get_retrieval_service  # noqa: E402
from app.config import settings  # noqa: E402
from app.core.fusion import reciprocal_rank_fusion  # noqa: E402
from app.evaluation.ablation import evaluate_predictions, sample_key  # noqa: E402
from app.evaluation.dataset import load_dataset, validate_dataset  # noqa: E402
from app.utils.logging import configure_logging, log_json  # noqa: E402
from scripts.corpus_utils import build_chunk_universe  # noqa: E402

configure_logging(settings.log_level)

MODES = ["bm25_only", "vector_only", "hybrid", "hybrid_rerank"]


def _run_one_query(mode: str, service, query: str, top_k: int) -> tuple[list[str], float]:
    """按指定模式跑一次检索，返回 (预测 id 列表, 耗时 ms)。

    消融的核心：其余环节完全一致，只摘掉/保留某个组件，指标差异就是该组件的贡献。
    """
    import time

    start = time.perf_counter()

    if mode == "bm25_only":
        results = service.bm25.search(query, settings.recall_top_k)
        predicted = [str(x.chunk_id) for x in results[:top_k]]
    elif mode == "vector_only":
        results = service.vector.search(query, settings.recall_top_k)
        predicted = [str(x.chunk_id) for x in results[:top_k]]
    elif mode == "hybrid":
        bm25_res = service.bm25.search(query, settings.recall_top_k)
        vec_res = service.vector.search(query, settings.recall_top_k)
        policy = service.policy.get(service.classifier.classify(query))
        fused = reciprocal_rank_fusion(
            [bm25_res, vec_res], k=settings.rrf_k,
            weights=[policy.bm25, policy.vector],
        )
        predicted = [str(x.chunk_id) for x in fused[:top_k]]
    else:  # hybrid_rerank —— 走完整流水线
        response = service.run_pipeline(query, top_k=top_k, use_rerank=True)
        predicted = [str(x.chunk_id) for x in response.candidates]

    latency = (time.perf_counter() - start) * 1000
    return predicted, latency


def run_mode(mode: str, dataset, top_k: int) -> dict:
    """对评测集整体跑一种模式，返回指标 + 延迟统计。"""
    service = get_retrieval_service()
    prediction_map = {}
    latencies = []
    for index, sample in enumerate(dataset):
        predicted, latency = _run_one_query(mode, service, sample.query, top_k)
        # 按 sample_key 对齐，而不是按 query 字符串：
        # 评测集里出现两条完全相同的问题时，用 query 做 key 会互相覆盖，
        # 两条都按同一份预测算分——静默算错。
        prediction_map[sample_key(sample, index)] = predicted
        latencies.append(latency)

    result = evaluate_predictions(dataset, prediction_map, k=top_k)
    result["mean_latency_ms"] = statistics.mean(latencies) if latencies else 0.0
    latencies_sorted = sorted(latencies)
    result["p95_latency_ms"] = (
        latencies_sorted[int((len(latencies_sorted) - 1) * 0.95)] if latencies_sorted else 0.0
    )
    log_json("evaluation_finished", mode=mode, **{k: round(v, 4) for k, v in result.items()})
    return result


def _preflight(dataset, allow_stale: bool) -> None:
    """跑评测前校验评测集。errors 会拦住运行，warnings 只提示。"""
    # 只有能读到语料时才做"id 是否存在"的强校验；语料不在就退化成弱校验。
    try:
        universe = build_chunk_universe()
    except Exception as exc:  # noqa: BLE001 —— 语料缺失不该阻止评测（只是少一层校验）
        universe = None
        print(f"[提示] 无法读取语料构造 chunk 全集，跳过 id 存在性校验（{type(exc).__name__}）")

    errors, warnings = validate_dataset(dataset, chunk_universe=universe)
    for message in warnings:
        print(f"[警告] {message}")
    if errors:
        print("\n" + "=" * 70)
        print("评测集体检未通过，拒绝运行：")
        for message in errors:
            print("  - " + message)
        if not allow_stale:
            print("\n（确实想看坏数据下的现象，加 --allow-stale-dataset 可强行运行）")
            raise SystemExit(2)
        print("\n[警告] --allow-stale-dataset 已指定，继续运行——结果不可用于下结论。")


def main() -> None:
    parser = argparse.ArgumentParser(description="跑评测集指标 / 消融实验")
    parser.add_argument("--dataset", default="data/evaluation.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", choices=MODES, default="hybrid_rerank")
    parser.add_argument("--all", action="store_true", help="跑完四种模式并打印对比表")
    parser.add_argument("--allow-stale-dataset", action="store_true",
                        help="跳过评测集体检（结果不可用于下结论，仅供排查）")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    print(f"评测集 {args.dataset} 共 {len(dataset)} 条，top_k={args.top_k}")
    _preflight(dataset, args.allow_stale_dataset)

    if args.all:
        print(f"\n{'模式':<14}{'Recall@K':>10}{'MRR@K':>10}{'NDCG@K':>10}"
              f"{'平均延迟ms':>12}{'有效样本':>10}{'无法判定':>10}")
        for mode in MODES:
            res = run_mode(mode, dataset, args.top_k)
            print(
                f"{mode:<14}{res[f'Recall@{args.top_k}']:>10.4f}"
                f"{res[f'MRR@{args.top_k}']:>10.4f}{res[f'NDCG@{args.top_k}']:>10.4f}"
                f"{res['mean_latency_ms']:>12.1f}"
                f"{res['evaluated']:>10}{res['unverifiable']:>10}"
            )
        return

    res = run_mode(args.mode, dataset, args.top_k)
    print(res)


if __name__ == "__main__":
    main()
