"""
streamlit_app.py —— RetrieverX 可视化演示（Streamlit）

【怎么启动】
  cd retrieverx
  venv\\Scripts\\activate
  streamlit run streamlit_app.py
  然后浏览器访问 http://localhost:8501
  （需要先启动后端 API：uvicorn app.main:app --port 8000）

【这个页面展示什么（面试演示脚本）】
  1. 「文档库」上传 PDF → 看到解析出多少块、切出多少 Chunk，并看到已索引文档列表；
  2. 「检索」输入业务问题 → 看到：
     - 命中的 Chunk 卡片（含来源文档、页码、标题路径、类型、得分）；
     - 一次请求的耗时与所用策略版本；
     - 可以把检索范围限定到某一份文档（多文档场景的关键能力）；
  3. 对结果点 👍/👎，或在「反馈闭环」提交"更正写法"，再次搜索同一问题，
     观察 used_correction=true 且结果被改写——演示反馈闭环。

【新手必读】
  Streamlit 是"用 Python 写网页"的框架：从上到下执行脚本，用 st.* 画界面。
  本文件通过 requests 调用后端 FastAPI，因此前后端是解耦的。
  界面样式有两条来源：
    · 主题色/背景色 → .streamlit/config.toml（Streamlit 官方配置，最稳）
    · 卡片、徽标等细节 → 下面的 CSS 常量（注入到页面里）
"""
import html

import requests
import streamlit as st

st.set_page_config(
    page_title="RetrieverX · 混合检索",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# 样式：整体走"纯白 + 细边框 + 单一强调色"的克制风格
# ---------------------------------------------------------------------------
CSS = """
<style>
.block-container { padding-top: 2.6rem; padding-bottom: 3.5rem; max-width: 1180px; }

/* ---------- 顶部标题区 ---------- */
.rx-header { border-bottom: 1px solid #ECEEF1; padding-bottom: 18px; margin-bottom: 24px; }
.rx-title { font-size: 26px; font-weight: 600; letter-spacing: -0.3px; color: #1F2328; margin: 0; }
.rx-sub { font-size: 13.5px; color: #6B7280; margin-top: 7px; line-height: 1.65; }

/* ---------- 指标卡 ---------- */
.rx-metrics { display: flex; gap: 10px; margin: 4px 0 18px; flex-wrap: wrap; }
.rx-metric { flex: 1 1 0; min-width: 128px; border: 1px solid #E7E9EE; border-radius: 10px;
             padding: 12px 14px; background: #FCFCFD; }
.rx-metric-label { font-size: 12px; color: #6B7280; margin-bottom: 5px; }
.rx-metric-value { font-size: 18px; font-weight: 600; color: #1F2328; font-variant-numeric: tabular-nums; }

/* ---------- 结果卡片 ---------- */
.rx-card { background: #FFFFFF; border: 1px solid #E7E9EE; border-radius: 12px;
           padding: 16px 18px; margin-bottom: 6px; }
.rx-card:hover { border-color: #C9D4E8; }
.rx-head { display: flex; align-items: flex-start; gap: 10px; margin-bottom: 9px; }
.rx-rank { flex: 0 0 auto; font-size: 12px; font-weight: 600; color: #FFFFFF; background: #2F6FEB;
           width: 22px; height: 22px; line-height: 22px; text-align: center; border-radius: 6px; }
.rx-path { font-size: 12.5px; color: #4B5563; line-height: 1.5; padding-top: 2px; }
.rx-content { font-size: 14px; line-height: 1.7; color: #24292F; white-space: pre-wrap;
              word-break: break-word; }
.rx-meta { margin-top: 11px; padding-top: 10px; border-top: 1px dashed #EDEFF3; }
.rx-badge { display: inline-block; font-size: 11.5px; padding: 2px 8px; border-radius: 999px;
            border: 1px solid #E7E9EE; background: #F8F9FB; color: #5B6472; margin: 0 6px 4px 0; }
.rx-badge-key { background: #F1F6FF; border-color: #D6E4FF; color: #2F6FEB; font-weight: 500; }
.rx-badge-score { font-variant-numeric: tabular-nums; }

/* ---------- 文档列表 ---------- */
.rx-doc-name { font-size: 14px; font-weight: 500; color: #1F2328; }
.rx-doc-meta { font-size: 12px; color: #6B7280; margin-top: 3px; }

/* ---------- 侧边栏收敛 ---------- */
section[data-testid="stSidebar"] { border-right: 1px solid #ECEEF1; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 后端调用封装
# ---------------------------------------------------------------------------
def _detail(resp) -> str:
    """把 FastAPI 的错误体（{"detail": "..."}）翻成人能读的字符串。"""
    try:
        body = resp.json()
        if isinstance(body, dict) and body.get("detail"):
            return str(body["detail"])
    except Exception:
        pass
    return resp.text[:400]


def api_get(url: str, path: str, timeout: int = 30):
    """GET 请求；失败返回 (None, 错误说明)。"""
    try:
        resp = requests.get(f"{url}{path}", timeout=timeout)
    except requests.exceptions.RequestException as exc:
        return None, f"连不上后端 {url}：{exc}"
    if resp.ok:
        return resp.json(), None
    return None, f"HTTP {resp.status_code}：{_detail(resp)}"


def api_post(url: str, path: str, payload: dict, timeout: int = 120):
    try:
        resp = requests.post(f"{url}{path}", json=payload, timeout=timeout)
    except requests.exceptions.RequestException as exc:
        return None, f"请求失败：{exc}"
    if resp.ok:
        return resp.json(), None
    return None, f"HTTP {resp.status_code}：{_detail(resp)}"


def load_documents(api_url: str) -> list[dict]:
    data, err = api_get(api_url, "/documents")
    return [] if err else data.get("documents", [])


# ---------------------------------------------------------------------------
# 渲染小工具
# ---------------------------------------------------------------------------
def render_metrics(items: list[tuple[str, str]]) -> None:
    cells = "".join(
        f'<div class="rx-metric"><div class="rx-metric-label">{html.escape(label)}</div>'
        f'<div class="rx-metric-value">{html.escape(value)}</div></div>'
        for label, value in items
    )
    st.markdown(f'<div class="rx-metrics">{cells}</div>', unsafe_allow_html=True)


METHOD_LABEL = {
    "bm25": "BM25 关键词",
    "vector": "向量语义",
    "bm25+vector": "双路命中",
}


def render_card(rank: int, cand: dict) -> None:
    """渲染一条检索结果卡片。

    ⚠️ 拼成一整行 HTML：带缩进的 HTML 会被 Markdown 解析器当成代码块，
    于是页面上就会出现一堆原始标签。
    """
    path = " › ".join(cand.get("heading_path") or []) or "（无标题路径）"
    content = cand.get("content", "")
    shown = content if len(content) <= 700 else content[:700] + " …"

    badges = [
        f'<span class="rx-badge rx-badge-key">{html.escape(cand.get("source_name") or "未知来源")}</span>',
        f'<span class="rx-badge">第 {cand.get("page_number")} 页</span>',
        f'<span class="rx-badge">{"表格" if cand.get("content_type") == "table" else "正文"}</span>',
        f'<span class="rx-badge">{html.escape(METHOD_LABEL.get(cand.get("retrieval_method", ""), cand.get("retrieval_method", "")))}</span>',
        f'<span class="rx-badge rx-badge-score">融合分 {cand.get("fusion_score", 0):.5f}</span>',
    ]
    if cand.get("rerank_score") is not None:
        badges.append(
            f'<span class="rx-badge rx-badge-score">精排分 {cand["rerank_score"]:.4f}</span>'
        )

    st.markdown(
        f'<div class="rx-card"><div class="rx-head"><div class="rx-rank">{rank}</div>'
        f'<div class="rx-path">{html.escape(path)}</div></div>'
        f'<div class="rx-content">{html.escape(shown)}</div>'
        f'<div class="rx-meta">{"".join(badges)}</div></div>',
        unsafe_allow_html=True,
    )
    if len(content) > 700:
        with st.expander("查看完整内容"):
            st.text(content)


# ---------------------------------------------------------------------------
# 顶部标题
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="rx-header"><div class="rx-title">RetrieverX</div>'
    '<div class="rx-sub">面向企业 PDF 的混合检索系统　·　 '
    "PDF 解析 → 结构感知切块 → BM25 + 向量 → RRF 融合 → Cross-Encoder 精排 → 反馈闭环"
    "</div></div>",
    unsafe_allow_html=True,
)

# 记住上一次的检索结果，避免点"反馈"按钮时结果被清空
st.session_state.setdefault("last_result", None)

# ---------------------------------------------------------------------------
# 侧边栏：配置 + 检索范围 + 索引统计
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("#### 配置")
    API_URL = st.text_input("后端 API 地址", value="http://127.0.0.1:8000")
    top_k = st.slider("返回条数 Top-K", 1, 20, 5)
    use_rerank = st.checkbox(
        "启用 Cross-Encoder 精排",
        value=False,
        help="本机 CPU 上单次精排约 4~16 秒（取决于候选数），演示时建议先关掉。",
    )

    st.divider()
    st.markdown("#### 检索范围")
    documents = load_documents(API_URL)
    options: dict[str, str | None] = {"全部文档（跨文档检索）": None}
    for doc in documents:
        options[f"{doc['source_name']} · {doc['chunks']} 块"] = doc["document_id"]
    picked = st.selectbox("限定文档", list(options.keys()))
    scope_id = options[picked]
    if st.button("刷新文档列表", use_container_width=True):
        st.rerun()

    st.divider()
    st.markdown("#### 索引统计")
    stats, err = api_get(API_URL, "/stats")
    if err:
        st.caption(f"⚠️ {err}")
    else:
        st.caption(
            f"Chunk 总数　**{stats['elasticsearch_chunks']}**（ES）/ "
            f"**{stats['qdrant_chunks']}**（Qdrant）"
        )
        st.caption(f"文档数　**{len(documents)}**　·　反馈数　**{stats['feedback_count']}**")

# ---------------------------------------------------------------------------
# 主区域三个标签页
# ---------------------------------------------------------------------------
tab_search, tab_docs, tab_feedback = st.tabs(["检索", "文档库", "反馈闭环"])

# ============================ 检索 ============================
with tab_search:
    with st.form("search_form", border=False):
        col_q, col_btn = st.columns([6, 1])
        query = col_q.text_input(
            "问题",
            placeholder="例如：Git 有什么用？　/　PX-4200 的价格是多少？",
            label_visibility="collapsed",
        )
        submitted = col_btn.form_submit_button("搜索", use_container_width=True)

    if submitted:
        if not query.strip():
            st.warning("请输入问题。")
        else:
            payload = {"query": query, "top_k": top_k, "rerank": use_rerank}
            if scope_id:
                payload["document_id"] = scope_id
            with st.spinner("检索中 ..."):
                result, err = api_post(API_URL, "/search", payload, timeout=180)
            if err:
                st.error(f"检索失败：{err}")
                st.session_state["last_result"] = None
            else:
                st.session_state["last_result"] = result

    result = st.session_state["last_result"]
    if result is None:
        st.caption("先在「文档库」上传一份 PDF 建立索引，然后在这里提问。")
    else:
        policy = result["retrieval_policy"]
        render_metrics(
            [
                ("总耗时", f"{result['latency_ms']:.0f} ms"),
                ("策略 BM25/向量", f"{policy['bm25']} / {policy['vector']}"),
                ("策略版本", f"v{policy['version']}"),
                ("更正缓存", "命中" if result["used_correction"] else "未命中"),
                ("精排", "已生效" if result.get("rerank_applied") else "未生效"),
            ]
        )
        if use_rerank and not result.get("rerank_applied"):
            st.info("本次结果显示为融合排序（精排被降级或未开启），后端日志里有具体原因。")

        # ---- 链路 Trace：不仅看结果，还能解释结果是怎么来的（WP18 Retrieval Debugger） ----
        if result.get("request_id"):
            with st.expander("🔬 查看链路 Trace（各阶段召回与排序）", expanded=False):
                trace, terr = api_get(API_URL, f"/trace/{result['request_id']}")
                if terr:
                    st.caption(f"Trace 不可用：{terr}")
                else:
                    st.caption(
                        f"查询类型 {trace.get('query_type')} · 策略 v{trace.get('policy', {}).get('version')} · "
                        f"BM25 {trace.get('latency_breakdown', {}).get('bm25_ms', 0):.0f}ms / "
                        f"向量 {trace.get('latency_breakdown', {}).get('vector_ms', 0):.0f}ms / "
                        f"融合 {trace.get('latency_breakdown', {}).get('fusion_ms', 0):.0f}ms / "
                        f"精排 {trace.get('latency_breakdown', {}).get('reranker_ms', 0):.0f}ms"
                    )

                    def _stage(name: str, cands: list, key: str):
                        st.markdown(f"**{name}**（{len(cands)} 条）")
                        if not cands:
                            st.caption("（空）")
                            return
                        lines = []
                        for c in cands[:5]:
                            m = c.get("retrieval_method", "")
                            score = c.get(f"{key}_score") if key else ""
                            score_txt = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
                            lines.append(f"{c.get('rank')}. `{c.get('chunk_id', '')[:8]}…` [{m}] {score_txt}")
                        st.code("\n".join(lines), language=None)

                    col1, col2 = st.columns(2)
                    with col1:
                        _stage("BM25 召回 Top-5", trace.get("bm25_candidates", []), "original")
                        _stage("RRF 融合 Top-5", trace.get("rrf_candidates", []), "fusion")
                    with col2:
                        _stage("向量召回 Top-5", trace.get("vector_candidates", []), "original")
                        _stage("最终 Top-5", trace.get("final_candidates", []), "fusion")

        candidates = result["candidates"]
        st.caption(f"命中 {len(candidates)} 条　·　问题：{result['query']}")
        if not candidates:
            st.warning(
                "没有命中任何内容。如果刚切了检索范围，确认那份文档已经建立索引。"
            )

        for rank, cand in enumerate(candidates, 1):
            render_card(rank, cand)
            # 反馈按钮只能放在卡片外面：卡片是纯 HTML，塞不进 Streamlit 控件
            col_pos, col_neg, _pad = st.columns([1, 1, 8])
            chunk_id = cand["chunk_id"]
            if col_pos.button("👍 有用", key=f"pos-{chunk_id}", use_container_width=True):
                _, err = api_post(
                    API_URL,
                    "/feedback",
                    {
                        "query": result["query"],
                        "candidate_ids": [chunk_id],
                        "label": "positive",
                    },
                    timeout=30,
                )
                st.toast("已记录：有用" if not err else f"记录失败：{err}")
            if col_neg.button("👎 没用", key=f"neg-{chunk_id}", use_container_width=True):
                _, err = api_post(
                    API_URL,
                    "/feedback",
                    {
                        "query": result["query"],
                        "candidate_ids": [chunk_id],
                        "label": "negative",
                    },
                    timeout=30,
                )
                st.toast("已记录：没用" if not err else f"记录失败：{err}")

# ============================ 文档库 ============================
with tab_docs:
    st.markdown("#### 上传并建立索引")
    uploaded = st.file_uploader("选择一份 PDF（含标题、表格的效果最好）", type=["pdf"])
    if uploaded is not None and st.button("建立索引", key="index_btn", type="primary"):
        with st.status(f"正在解析并索引 {uploaded.name} ...", expanded=True) as status:
            st.write("解析 PDF → 结构感知切块 …")
            try:
                resp = requests.post(
                    f"{API_URL}/documents/index",
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                    timeout=300,
                )
            except requests.exceptions.RequestException as exc:
                status.update(label="索引失败", state="error")
                st.error(f"请求失败：{exc}")
                st.stop()

            if resp.ok:
                data = resp.json()
                st.write("写入 Elasticsearch（BM25）与 Qdrant（向量）…")
                status.update(
                    label=f"索引完成：{data['blocks']} 个解析块 → {data['chunks']} 个 Chunk",
                    state="complete",
                )
            else:
                status.update(label="索引失败", state="error")
                st.error(f"HTTP {resp.status_code}：{_detail(resp)}")
                if resp.status_code == 422:
                    st.info(
                        "这条通常表示 PDF 里没有可提取的文字（扫描件 / 纯图片）。\n\n"
                        "可以用 `python scripts/index_documents.py xxx.pdf --dry-run` "
                        "先看看能解析出多少个块。"
                    )
                else:
                    st.info("跑 `python scripts/check_services.py` 一键体检依赖服务。")

    st.divider()
    st.markdown("#### 已索引的文档")
    documents = load_documents(API_URL)
    if not documents:
        st.caption("索引里还没有文档。")
    for doc in documents:
        col_info, col_btn = st.columns([5, 1])
        types = "、".join(
            f"{'表格' if k == 'table' else '正文'} {v}"
            for k, v in doc["content_types"].items()
        )
        meta = f'{doc["chunks"]} 个 Chunk　·　最多 {doc["pages"]} 页'
        if types:
            meta += f"　·　{types}"
        col_info.markdown(
            f'<div class="rx-doc-name">{html.escape(doc["source_name"])}</div>'
            f'<div class="rx-doc-meta">{html.escape(meta)}</div>',
            unsafe_allow_html=True,
        )
        if col_btn.button("删除", key=f"del-{doc['document_id']}", use_container_width=True):
            try:
                resp = requests.delete(
                    f"{API_URL}/documents/{doc['document_id']}", timeout=60
                )
            except requests.exceptions.RequestException as exc:
                st.error(f"删除失败：{exc}")
            else:
                if resp.ok:
                    st.toast(
                        f"已删除 {doc['source_name']}"
                        f"（{resp.json()['deleted_chunks']} 个 Chunk）"
                    )
                    st.rerun()
                else:
                    st.error(f"删除失败：{_detail(resp)}")

# ============================ 反馈闭环 ============================
with tab_feedback:
    st.markdown("#### 提交更正")
    st.caption(
        "如果某个问题搜不到想要的内容，在这里写下「正确的问法」或「正确内容」。"
        "之后再搜同一个问题，后端会自动用更正后的写法检索，"
        "并在结果里标出 used_correction=true。"
    )
    with st.form("correction_form"):
        orig_query = st.text_input("原始问题（没搜到想要的内容）")
        corrected = st.text_input("你希望搜到的正确内容 / 正确问法")
        submitted_corr = st.form_submit_button("提交更正", type="primary")
    if submitted_corr:
        if not orig_query.strip() or not corrected.strip():
            st.warning("两个输入框都要填。")
        else:
            _, err = api_post(
                API_URL,
                "/feedback",
                {
                    "query": orig_query,
                    "candidate_ids": [],
                    "label": "correction",
                    "correct_answer": corrected,
                },
                timeout=30,
            )
            if err:
                st.error(f"提交失败：{err}")
            else:
                st.success(
                    "更正已记录。回到「检索」再搜一次原问题，就会看到结果被改写。"
                )
