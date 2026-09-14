"""
scripts/check_services.py —— 依赖服务体检

【为什么需要它？】
  RetrieverX 依赖 4 个外部服务（ES / Qdrant / Redis / Ollama）。
  任何一个没起来，检索都会失败——但错误信息可能很绕。
  这个脚本直接按 .env 里配的地址逐个探活，一眼看出是谁没起。

【用法】
  cd retrieverx
  venv\\Scripts\\python.exe scripts\\check_services.py

  退出码 0 = 全部就绪；1 = 有服务没起来。
"""
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

# 允许 `python scripts/check_services.py` 直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402


def _targets() -> list[tuple[str, str, int]]:
    """从 settings 里解析出「名字 / 主机 / 端口」三元组。"""
    items = [
        ("Elasticsearch", settings.elasticsearch_url, 9200),
        ("Qdrant", settings.qdrant_url, 6333),
        ("Redis", settings.redis_url, 6379),
        ("Ollama (embedding)", settings.embedding_base_url, 11434),
    ]
    result = []
    for name, url, default_port in items:
        if not url:
            continue
        parsed = urlparse(url if "://" in url else f"//{url}")
        host = parsed.hostname or "localhost"
        port = parsed.port or default_port
        result.append((name, host, port))
    return result


def main() -> int:
    print("依赖服务体检（地址来自 .env）\n")
    failed: list[str] = []
    for name, host, port in _targets():
        s = socket.socket()
        s.settimeout(3)
        try:
            s.connect((host, port))
            print(f"  [OK]   {name:<20s} {host}:{port}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {name:<20s} {host}:{port}  -> {type(exc).__name__}")
            failed.append(name)
        finally:
            s.close()

    print()
    if failed:
        print(f"有 {len(failed)} 个服务未就绪：{', '.join(failed)}")
        print("三个中间件可用 docker compose 一键起：docker compose up -d")
        print("或按各自官方文档本机安装后启动；对应地址见 .env。")
        print("提示：本机直跑 uvicorn 时，.env 里所有地址都应是 127.0.0.1（不要用 localhost，")
        print("      否则 Windows 会先试 IPv6 的 ::1、每次白等 2 秒）；")
        print("      若报『域名解析失败』，说明写成了容器名，而你的服务不在容器里。")
        return 1

    print("全部就绪，可以启动后端了：uvicorn app.main:app --reload --port 8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
