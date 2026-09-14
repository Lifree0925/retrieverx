"""
benchmark.py —— 延迟基准（Latency Benchmark）

【这个文件做什么】
  对一个函数（如"检索一次"）跑若干条查询，统计平均/95 分位/99 分位延迟。

【新手必读 · 面试点：为什么看 P95 而不是只看平均？】
  平均延迟会被偶发慢请求拉高/掩盖问题；P95（95% 请求都在该值以内）
  更能反映"绝大多数用户的真实体验"，是性能评估的行业习惯。
"""
import statistics
import time


def percentile(values: list[float], p: float) -> float:
    """计算分位数。values 升序后取第 p 位置的值；p=0.95 即 P95。"""
    if not values:
        return 0.0
    values = sorted(values)
    index = int((len(values) - 1) * p)
    return values[index]


def benchmark(fn, queries: list[str]) -> dict:
    """对 fn(query) 逐条计时，返回统计结果（毫秒）。"""
    latencies = []
    for query in queries:
        started = time.perf_counter()
        fn(query)  # 执行一次调用
        latencies.append((time.perf_counter() - started) * 1000)

    return {
        "count": len(latencies),
        "mean_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "p95_latency_ms": percentile(latencies, 0.95),
        "p99_latency_ms": percentile(latencies, 0.99),
    }
