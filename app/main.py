"""
main.py —— FastAPI 应用入口（启动文件）

【怎么启动】
  cd retrieverx
  venv\\Scripts\\activate
  uvicorn app.main:app --reload --port 8000
  然后浏览器打开 http://localhost:8000/docs 即可看到所有接口。

【这个文件做什么】
  1. 初始化日志；
  2. 创建 FastAPI 实例并挂载三个路由模块（search / feedback / documents）；
  3. 提供 /health（健康检查）与 /metrics（指标占位）；
  4. 注册【全局异常处理器】：任何未捕获异常都转成规范 JSON，
     而不是把 Python 堆栈直接甩给前端；
     其中"连不上 ES/Qdrant/Redis"这类依赖故障单独转成 503 + 可操作提示。

【新手必读 · 为什么组件不在 main.py 里 new？】
  早期版本在 main.py 顶部就 new 出 ES/Qdrant/Redis 客户端和 1GB 的 Reranker 模型——
  一旦某个依赖服务没启动，整个应用直接起不来；Reranker 还会在启动时就开始下载模型。
  现在改成：所有组件都在 app/api/dependencies.py 里"懒加载"（用到才创建），
  这里只负责组装路由与异常处理，职责清晰、启动轻快。
"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.dependencies import get_bm25, get_feedback_store, get_vector
from app.config import settings
from app.utils.logging import configure_logging

# 初始化日志（必须在任何打印之前调用）
configure_logging(settings.log_level)

# 创建应用实例
app = FastAPI(title=settings.app_name, version="2.0.0")

# 挂载路由（模块末尾 import 放在 include 前，避免加载 app.main 时的循环依赖）
from app.api.routes_documents import router as documents_router  # noqa: E402
from app.api.routes_feedback import router as feedback_router    # noqa: E402
from app.api.routes_search import router as search_router        # noqa: E402

app.include_router(search_router)
app.include_router(feedback_router)
app.include_router(documents_router)


@app.get("/")
def root():
    """根路径引导页：新手直接访问 http://localhost:8000/docs 时，
    不再看到 404，而是得到一份"该访问哪个地址"的说明。

    【注意】浏览器请访问 /docs 看交互式接口文档；
    想看图形化演示界面，另开终端跑 streamlit run streamlit_app.py。
    """
    return {
        "service": settings.app_name,
        "message": "服务已启动。本应用是纯 API 服务，没有首页；浏览器请打开 /docs 查看接口文档。",
        "recommended_paths": {
            "/docs": "Swagger 交互式接口文档（推荐入口，可在线测试每个接口）",
            "/health": "健康检查（最简单：返回 {'status': 'ok'}）",
            "/stats": "查看已索引的 Chunk 数与反馈数（需先启动 ES/Qdrant/Redis）",
            "POST /search": "混合检索主接口",
            "POST /feedback": "提交反馈（点赞/点踩）",
            "POST /documents/index": "上传 PDF/文本并解析入库",
        },
        "ui_hint": "图形化演示界面请另开终端执行: streamlit run streamlit_app.py，然后访问 http://localhost:8501",
    }


@app.get("/health")
def health():
    """健康检查：部署/体检时看服务是否活着。"""
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    """指标占位接口：真正的实验指标由 scripts/run_evaluation.py 产出。"""
    return {"status": "ok", "message": "Run evaluation scripts for experiment metrics."}


@app.get("/stats")
def stats():
    """统计当前索引了多少 Chunk / 收到多少反馈（体检与演示用）。"""
    try:
        bm25_total = get_bm25().count_all()
        vector_total = get_vector().count_all()
        feedback_total = get_feedback_store().total_feedback_count()
        return {
            "elasticsearch_chunks": bm25_total,
            "qdrant_chunks": vector_total,
            "feedback_count": feedback_total,
        }
    except Exception as exc:  # 依赖服务没启动时给出明确提示
        return JSONResponse(
            status_code=503,
            content={
                "error": "stats_unavailable",
                "message": f"无法读取统计信息，请确认 ES/Qdrant/Redis 已启动: {exc}",
            },
        )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """全局兜底异常处理：把任何未捕获异常包装成规范 JSON，不暴露堆栈。

    面试点：统一异常结构 = {code, message, path}，前端/监控都好处理。
    """
    return JSONResponse(
        status_code=500,
        content={
            "code": "INTERNAL_ERROR",
            "message": str(exc),
            "path": request.url.path,
        },
    )


# ---------------------------------------------------------------------------
# 依赖服务不可用 → 503（而不是笼统的 500）
# ---------------------------------------------------------------------------
def _dependency_error_types() -> list[type[BaseException]]:
    """收集各中间件客户端抛出的"连不上/超时"异常类型。

    它们没有一个公共基类（elasticsearch / redis / openai / httpx 各自为政），
    所以这里逐个 import 收集；某个库没装也不影响其他库。
    """
    types: list[type[BaseException]] = []
    for module, names in (
        ("elasticsearch", ("ConnectionError", "ConnectionTimeout", "TransportError")),
        ("redis.exceptions", ("ConnectionError", "TimeoutError")),
        ("openai", ("APIConnectionError", "APITimeoutError")),
        ("httpx", ("ConnectError", "ConnectTimeout", "ReadTimeout", "ProxyError")),
        ("qdrant_client.http.exceptions", ("ResponseHandlingException", "UnexpectedResponse")),
    ):
        try:
            mod = __import__(module, fromlist=list(names))
        except Exception:  # noqa: BLE001 —— 可选依赖，缺了就算了
            continue
        for name in names:
            exc_type = getattr(mod, name, None)
            if isinstance(exc_type, type) and issubclass(exc_type, BaseException):
                types.append(exc_type)
    return types


async def dependency_unavailable_handler(request: Request, exc: Exception):
    """把"连不上中间件"翻译成人话，直接告诉你去启动什么、检查哪里。

    对应 pipeline.py 里"由 API 层统一异常处理转成 503"的设计承诺。
    """
    return JSONResponse(
        status_code=503,
        content={
            "code": "DEPENDENCY_UNAVAILABLE",
            "message": (
                "检索依赖服务不可用（Elasticsearch / Qdrant / Redis / Embedding 接口）。"
                "请确认它们已启动，并检查 .env 里的地址与你的部署方式是否一致："
                "本机直跑用 localhost，docker compose 全家桶才用容器名。"
            ),
            "detail": f"{type(exc).__name__}: {exc}",
            "path": request.url.path,
        },
    )


for _exc_type in _dependency_error_types():
    app.add_exception_handler(_exc_type, dependency_unavailable_handler)
