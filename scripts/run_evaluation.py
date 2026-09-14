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
  对评测集（data/evaluation.jsonl）逐条跑检索，统计整体：
    Recall@K / MRR@K / NDCG@K + 平均延迟 + P95 延迟
  跑 --all 就能得到"消融实验表"，这是你简历上"实验数据"的来源。
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

# 允许 `python scripts/run_evaluation.py` 直接运行（同 index_documents.py）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api.dependencies import get_retrieval_service  # noqa: E402
from app.config import settings  # noqa: E402
from app.core.fusion import reciprocal_rank_fusion  # noqa: E402
from app.evaluation.ablation import evaluate_predictions  # noqa: E402
from app.evaluation.dataset import load_dataset  # noqa: E402
from app.utils.logging import configure_logging, log_json

configure_logging(settings.log_level)


def _run_one_query(mode: str, service, query: str, top_k: int, relevant: set):
    """按指定模式跑一次检索，返回 (预测 id 列表, 耗时 ms)。

    消融的核心：其余环节完全一致，只摘掉/保留某个组件，指标差异就是该组件的贡献。
    """
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
    for sample in dataset:
        predicted, latency = _run_one_query(
            mode, service, sample.query, top_k, set(sample.relevant_chunk_ids)
        )
        prediction_map[sample.query] = predicted
        latencies.append(latency)

    result = evaluate_predictions(dataset, prediction_map, k=top_k)
    result["mean_latency_ms"] = statistics.mean(latencies) if latencies else 0.0
    latencies_sorted = sorted(latencies)
    result["p95_latency_ms"] = (
        latencies_sorted[int((len(latencies_sorted) - 1) * 0.95)] if latencies_sorted else 0.0
    )
    log_json("evaluation_finished", mode=mode, **{k: round(v, 4) for k, v in result.items()})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="跑评测集指标 / 消融实验")
    parser.add_argument("--dataset", default="data/evaluation.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--mode", choices=["bm25_only", "vector_only", "hybrid", "hybrid_rerank"],
        default="hybrid_rerank",
    )
    parser.add_argument("--all", action="store_true", help="跑完四种模式并打印对比表")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    print(f"评测集共 {len(dataset)} 条，top_k={args.top_k}")

    if args.all:
        modes = ["bm25_only", "vector_only", "hybrid", "hybrid_rerank"]
        print(f"\n{'模式':<14}{'Recall@K':>10}{'MRR@K':>10}{'NDCG@K':>10}{'平均延迟ms':>12}")
        results = {}
        for mode in modes:
            res = run_mode(mode, dataset, args.top_k)
            results[mode] = res
            print(
                f"{mode:<14}{res[f'Recall@{args.top_k}']:>10.4f}"
                f"{res[f'MRR@{args.top_k}']:>10.4f}{res[f'NDCG@{args.top_k}']:>10.4f}"
                f"{res['mean_latency_ms']:>12.1f}"
            )
        return

    res = run_mode(args.mode, dataset, args.top_k)
    print(res)


if __name__ == "__main__":
    main()
