"""
logging.py —— 结构化 JSON 日志工具

【新手必读】
  普通 print 打日志很难被程序消费（不好搜索、不好统计）。
  这里把一次检索的关键信息（query、各阶段耗时、命中策略）打包成一个 JSON 对象输出，
  之后可以用日志检索工具（如 ELK）直接分析"慢在哪、错在哪"。
"""
import json
import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """初始化日志。level 支持 DEBUG/INFO/WARNING/ERROR，写错则回退到 INFO。"""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stdout,
        format="%(message)s",  # 只打印消息本身（因为消息已经是 JSON 一行）
    )


def log_json(event: str, **kwargs) -> None:
    """打印一行结构化 JSON 日志。

    用法：
        log_json("retrieval_trace", query="xxx", total_latency_ms=123.4)
    输出：
        {"event": "retrieval_trace", "query": "xxx", "total_latency_ms": 123.4}
    """
    logging.getLogger("retrieverx").info(
        json.dumps(
            {"event": event, **kwargs},
            ensure_ascii=False,   # 保留中文，方便直接阅读
            default=str,          # datetime/UUID 等对象转字符串
        )
    )
