"""
scripts/download_models.py —— 预下载精排模型（强烈建议首次部署时跑一次）

【为什么要单独跑这个？】
  精排模型（bge-reranker-base 约 1GB）如果在第一次 /search 请求里现下，
  请求会长时间阻塞，前端只看到"超时"。虽然 Reranker 现在已经做了超时降级，
  但"第一次精排要等模型下载"这件事本身还是会拖慢体验。
  所以最佳实践是：部署时先把模型下好，请求路径上永远只做本地加载。

【用法】
  cd retrieverx
  venv\\Scripts\\activate
  python scripts/download_models.py

  国内网络建议先在 .env 里配：
      HF_ENDPOINT=https://hf-mirror.com
  （本脚本会通过 app.config 读取它，并自动注入到环境变量）

【下载到哪里】
  默认是 HuggingFace 的标准缓存目录：
      Windows: C:\\Users\\<你>\\.cache\\huggingface\\hub
  下载完成后，Reranker 会直接命中缓存，不再联网。
"""
import sys
import time
from pathlib import Path

# 允许 `python scripts/download_models.py` 这种直接运行方式找到 app 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402  （必须先 import config 以注入 HF_ENDPOINT）


def main() -> None:
    model = settings.reranker_model
    print(f"[1/3] 目标模型: {model}")
    print(f"      HF_ENDPOINT = {settings.hf_endpoint or '(未配置，走 huggingface.co 官方源)'}")

    # 说明：这里 import 必须在 app.config 之后，否则 HF_ENDPOINT 可能来不及生效
    from huggingface_hub import snapshot_download

    print("[2/3] 开始下载（首次约 1GB，请耐心等待；已缓存则秒过）...")
    started = time.time()
    try:
        local_dir = snapshot_download(repo_id=model)
    except Exception as exc:  # noqa: BLE001
        print(f"\n❌ 下载失败: {type(exc).__name__}: {exc}")
        print("   排查建议：")
        print("     1) 在 .env 里设置 HF_ENDPOINT=https://hf-mirror.com 后重试；")
        print("     2) 确认能访问外网；")
        print("     3) 临时方案：把 .env 的 RERANK_ENABLE 设为 false，先跑通检索主链路。")
        raise SystemExit(1)

    print(f"[3/3] ✅ 完成，耗时 {time.time() - started:.1f}s")
    print(f"      缓存位置: {local_dir}")
    print("\n现在启动服务，第一次精排就只会做本地加载，不会再联网下载。")


if __name__ == "__main__":
    main()
