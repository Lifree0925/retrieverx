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
import json
import statistics
import sys
from pathlib import Path

# 允许 `python scripts/run_evaluation.py` 直接运行（同 index_documents.py）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api.dependencies import get_retrieval_service  # noqa: E402
from app.config import settings  # noqa: E402
from app.core.fusion import reciprocal_rank_fusion  # noqa: E402
from app.evaluation.ablation import (  # noqa: E402
    evaluate_by_split,
    evaluate_predictions,
    sample_key,
)
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


def _load_progress(progress_path: Path, mode: str) -> tuple[dict[str, list[str]], list[float]]:
    """读回已完成样本，用于续跑。

    【为什么需要】精排档在本地 CPU 上单条约 15~20 秒，63 条就是 20 分钟量级。
    如果进程在跑到一半时被意外终止（内存压力、被外部 kill、机器休眠），
    没有逐条落盘就意味着**整轮重跑**。实测踩过一次：跑到第 2 分钟被回收，
    前 3 个档位的成果虽然在日志里，但这一个档位的进度全丢。
    所以长跑评测必须"做一条存一条、下次接着做"。
    """
    done: dict[str, list[str]] = {}
    latencies: list[float] = []
    if not progress_path.exists():
        return done, latencies
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            # 半行（写入过程中被 kill）→ 丢掉这一行，不影响前面已完成的样本
            continue
        if record.get("mode") != mode:
            continue
        done[record["key"]] = record["predicted"]
        latencies.append(float(record.get("latency_ms", 0.0)))
    return done, latencies


def run_mode(
    mode: str,
    dataset,
    top_k: int,
    progress_path: Path | None = None,
    resume: bool = False,
) -> dict:
    """对评测集整体跑一种模式，返回指标 + 延迟统计。

    progress_path 给定时会逐条追加落盘；resume=True 时跳过其中已完成的样本。
    """
    service = get_retrieval_service()
    prediction_map: dict[str, list[str]] = {}
    latencies: list[float] = []
    remaining: list[tuple[int, object]] = []

    for index, sample in enumerate(dataset):
        # 按 sample_key 对齐，而不是按 query 字符串：
        # 评测集里出现两条完全相同的问题时，用 query 做 key 会互相覆盖，
        # 两条都按同一份预测算分——静默算错。
        remaining.append((index, sample))
        prediction_map[sample_key(sample, index)] = []

    skipped = 0
    if progress_path is not None and resume:
        done, latencies = _load_progress(progress_path, mode)
        for key, predicted in done.items():
            prediction_map[key] = predicted
        remaining = [
            (index, sample) for index, sample in remaining
            if not prediction_map[sample_key(sample, index)]
        ]
        skipped = len(done)

    total = len(dataset)
    if skipped:
        print(f"  [{mode}] 续跑：已完成 {skipped}/{total} 条，剩余 {len(remaining)} 条",
              flush=True)

    progress_file = None
    if progress_path is not None:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        # 续跑时是追加；不续跑则覆盖，避免新旧记录混在一起
        progress_file = progress_path.open("a" if resume else "w", encoding="utf-8")

    try:
        for done_count, (index, sample) in enumerate(remaining, 1):
            predicted, latency = _run_one_query(mode, service, sample.query, top_k)
            key = sample_key(sample, index)
            prediction_map[key] = predicted
            latencies.append(latency)
            if progress_file is not None:
                # 一行一 flush：被 kill 时最多丢当前这一条，而不是整轮
                progress_file.write(json.dumps(
                    {"mode": mode, "key": key, "predicted": predicted,
                     "latency_ms": latency},
                    ensure_ascii=False,
                ) + "\n")
                progress_file.flush()
            print(f"  [{mode}] {skipped + done_count}/{total} {key[:36]}"
                  f"  {latency:8.1f}ms", flush=True)
    finally:
        if progress_file is not None:
            progress_file.close()

    result = evaluate_predictions(dataset, prediction_map, k=top_k)
    # 分组指标：字面式 vs 改写式。两者的**差距**才是"语义检索起了多大作用"的直接证据，
    # 只看总平均会被字面式问题的高分拉平（实测：BM25 单档 Recall@5 就有 0.9894）。
    result["splits"] = evaluate_by_split(dataset, prediction_map, k=top_k)
    result["mean_latency_ms"] = statistics.mean(latencies) if latencies else 0.0
    latencies_sorted = sorted(latencies)
    result["p95_latency_ms"] = (
        latencies_sorted[int((len(latencies_sorted) - 1) * 0.95)] if latencies_sorted else 0.0
    )
    # 只把数值字段写进结构化日志。
    # 注意 result 里现在还有 `splits`（嵌套字典），直接 round() 会 TypeError——
    # 这类"加了新字段却没检查消费方"的问题在日志/序列化边界上特别常见。
    numeric = {
        k: round(v, 4)
        for k, v in result.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    log_json("evaluation_finished", mode=mode, **numeric)

    # 分组指标也各打一行结构化日志。
    # 为什么：分组数据（字面式 vs 改写式）才是"语义检索有没有用"的直接证据，
    # 只打印成人读的文本表，生成 README 表格时就得去正则解析排版——
    # 排版一改（列宽、加一列）解析就静默失败，表格里悄悄少一行数据。
    for split_name, part in (result.get("splits") or {}).items():
        log_json(
            "evaluation_split_finished",
            mode=mode,
            split=split_name,
            **{k: round(v, 4) for k, v in part.items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)},
        )
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
    parser.add_argument("--progress-out", default="",
                        help="逐条把预测写入该 jsonl（长跑评测强烈建议指定，被中断后可续跑）")
    parser.add_argument("--resume", action="store_true",
                        help="配合 --progress-out：跳过其中已完成的样本")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    print(f"评测集 {args.dataset} 共 {len(dataset)} 条，top_k={args.top_k}")
    _preflight(dataset, args.allow_stale_dataset)
    progress = Path(args.progress_out) if args.progress_out else None

    if args.all:
        print(f"\n{'模式':<14}{'Recall@K':>10}{'MRR@K':>10}{'NDCG@K':>10}"
              f"{'平均延迟ms':>12}{'有效样本':>10}{'无法判定':>10}")
        collected = {}
        for mode in MODES:
            res = run_mode(mode, dataset, args.top_k, progress_path=progress,
                           resume=args.resume)
            collected[mode] = res
            print(
                f"{mode:<14}{res[f'Recall@{args.top_k}']:>10.4f}"
                f"{res[f'MRR@{args.top_k}']:>10.4f}{res[f'NDCG@{args.top_k}']:>10.4f}"
                f"{res['mean_latency_ms']:>12.1f}"
                f"{res['evaluated']:>10}{res['unverifiable']:>10}"
            )

        # 分组表：字面式 vs 改写式。
        # 为什么必须分开看：字面式问题与语料原文关键词高度重叠，BM25 单档就能拿高分，
        # 它会把总平均拉高、把四个档位的差距压平；改写式问题才真正考验语义检索。
        # 两组之间的**差距**才是"融合/精排有没有用"的证据。
        split_names = sorted({name for res in collected.values() for name in res.get("splits", {})})
        if split_names:
            print(f"\n按问题类型拆分（同样只看前 {args.top_k} 名）：")
            header = f"{'模式':<14}" + "".join(f"{n:>26}" for n in split_names)
            print(header)
            print(f"{'':<14}" + "".join(f"{'Recall@K / MRR@K / 样本':>30}" for _ in split_names))
            for mode in MODES:
                res = collected.get(mode)
                if not res:
                    continue
                cells = ""
                for name in split_names:
                    part = res.get("splits", {}).get(name) or {}
                    cells += (f"{part.get(f'Recall@{args.top_k}', 0):>11.4f}"
                              f"{part.get(f'MRR@{args.top_k}', 0):>10.4f}"
                              f"{part.get('evaluated', 0):>9}")
                print(f"{mode:<14}{cells}")
            print("\n提示：字面式通常接近满分，两者差距大才说明「语义检索确实在起作用」；"
                  "若两者都接近满分，说明问题过于字面化、这张表区分不出组件差异。")
        return

    res = run_mode(args.mode, dataset, args.top_k, progress_path=progress,
                   resume=args.resume)
    print(res)


if __name__ == "__main__":
    main()
