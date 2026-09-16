"""
scripts/run_failure_analysis.py —— 失败分析（Failure Analysis）

【用法】
  python scripts/run_failure_analysis.py                    # 用默认评测集
  python scripts/run_failure_analysis.py --dataset data/evaluation.jsonl
  python scripts/run_failure_analysis.py --output data/failure_report.jsonl

【做什么】
  对评测集逐条跑完整检索链路（带 Trace），挑出没做对的样本，按失败类型归类，
  输出失败类型分布 + 逐条案例（问题 / 期望 chunk / 实际 Top-K / 失败类型 / 根因 / 修复建议）。

【前置条件】
  需要 ES / Qdrant / Redis（Trace 落 Redis）已启动，且评测集里的 relevant_chunk_ids
  是真实存在的 chunk_id（否则"没命中"是必然的，分析没有意义）。

【诚实边界】
  失败归类是**基于 Trace 的启发式**，不是绝对诊断；每个结论都要能人工复核。
  没有 Redis 时 Trace 拿不到，只能退化到"成功/失败"两层，无法细分根因。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api.dependencies import get_retrieval_service, get_trace_store  # noqa: E402
from app.config import settings  # noqa: E402
from app.evaluation.dataset import load_dataset  # noqa: E402
from app.evaluation.failure_analysis import classify_failure, summarize_failures  # noqa: E402
from app.utils.logging import configure_logging, log_json  # noqa: E402

configure_logging(settings.log_level)


def main() -> None:
    parser = argparse.ArgumentParser(description="失败分析：归类检索失败案例的根因")
    parser.add_argument("--dataset", default="data/evaluation.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default=None, help="失败案例明细写到的 JSONL 路径")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    service = get_retrieval_service()
    trace_store = get_trace_store()

    failures = []
    successes = 0
    for sample in dataset:
        resp = service.run_pipeline(sample.query, top_k=args.top_k, use_rerank=True)
        predicted = [str(c.chunk_id) for c in resp.candidates]
        trace = trace_store.get(resp.request_id) if resp.request_id else None

        case = classify_failure(
            sample.query,
            sample.relevant_chunk_ids,
            predicted,
            trace=trace,
            annotated_type=sample.query_type,
        )
        if case is None:
            successes += 1
        else:
            failures.append(case)

    report = summarize_failures(failures)
    report["total_samples"] = len(dataset)
    report["successes"] = successes

    print(f"评测集 {len(dataset)} 条，成功 {successes} 条，失败 {len(failures)} 条\n")
    print("失败类型分布：")
    for ftype, count in sorted(report["distribution"].items(), key=lambda kv: -kv[1]):
        print(f"  {ftype:<28} {count}")
    print("\n逐条案例：")
    for c in report["cases"]:
        print(f"\n· 问题：{c['query']}")
        print(f"  期望 chunk：{c['expected_chunks']}")
        print(f"  实际 Top-K：{c['actual_top_k']}")
        print(f"  失败类型：{c['failure_type']}")
        print(f"  根因：{c['root_cause']}")
        print(f"  修复：{c['fix']}")

    if args.output:
        Path(args.output).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n已写入明细：{args.output}")

    log_json("failure_analysis_finished", **report)


if __name__ == "__main__":
    main()
