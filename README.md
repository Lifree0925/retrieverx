# RetrieverX — Feedback-Driven Hybrid RAG

面向企业 PDF 的混合检索系统：PDF 解析 → 结构感知切块 → BM25 + 向量检索 → RRF 融合 → Cross-Encoder 精排 → 反馈闭环 → 检索策略。

## 架构总览

```text
PDF
 ↓
PDFParser（PyMuPDF 结构化解析：标题层级/表格/页码）
 ↓
StructureAwareChunker（结构感知切块，带 metadata）
 ↓
 ┌───────────────┬───────────────┐
 │ BM25(ES)      │ Vector(Qdrant) │   ← QueryClassifier 决定两路权重
 └───────┬───────┴───────┬───────┘
        RRF（按排名融合）
         ↓
   Cross-Encoder 精排（可选，超时自动降级）
         ↓
   Top-K 结果 + 策略版本
         ↓
   Redis 反馈（positive/negative/correction）→ 更正缓存 / 策略迭代
```

## 快速开始

### 1. 准备环境

要求：Python 3.10+、Docker Desktop（用于起 Elasticsearch / Qdrant / Redis）。

```bash
git clone https://github.com/Lifree0925/retrieverx.git
cd retrieverx
python -m venv venv
# Windows: venv\Scripts\activate   Linux/macOS: source venv/bin/activate
pip install -r requirements.txt

copy .env.example .env   # Windows
# cp .env.example .env   # Linux/macOS
# 编辑 .env：填写 EMBEDDING_MODEL / EMBEDDING_DIM / EMBEDDING_API_KEY / EMBEDDING_BASE_URL
#   【本地 Ollama 零成本方案】ollama pull bge-m3，然后
#     EMBEDDING_MODEL=bge-m3 / EMBEDDING_DIM=1024 / EMBEDDING_API_KEY=ollama
#     / EMBEDDING_BASE_URL=http://127.0.0.1:11434/v1
#   ⚠️ 中文场景别用 nomic-embed-text（英文向），实测 Top-1 只有 4/8，bge-m3 是 7/8。
#   同时确认三个中间件地址与你的部署方式一致（见文末「常见问题」），
#   国内建议保留 HF_ENDPOINT=https://hf-mirror.com，否则加载精排模型会卡住。
#   ⚠️ 中间件地址一律写 127.0.0.1，不要写 localhost（Windows 会先试 IPv6 的 ::1，
#      每次请求白等约 2 秒，详见文末「常见问题」）。
```

### 2. 启动依赖服务（ES / Qdrant / Redis）

方式 A · 中间件放 Docker（本机跑 API）：

```bash
docker compose up -d elasticsearch qdrant redis   # 务必指定服务名，别把 api/streamlit 也起来
```

方式 B · 全部原生装在本机（无 Docker）：

按 Elasticsearch / Qdrant / Redis 各自的官方文档在本机装好并启动，再把 `.env` 里的
`ELASTICSEARCH_URL` / `QDRANT_URL` / `REDIS_URL` 改成本机端口即可。起完先体检：

```bash
python scripts/check_services.py   # 四个服务全绿再往下走
```

### 3. 索引 PDF 并启动 API

```bash
# 没有测试 PDF 就先生成一份（带标题层级 + 表格的示例产品手册）
python scripts/make_demo_pdf.py
python scripts/index_documents.py ./data/产品手册-PX4200.pdf
uvicorn app.main:app --reload --port 8000
```

浏览器打开 http://127.0.0.1:8000/docs（Swagger UI，可在线调试接口）。

### 4. 启动可视化演示

```bash
streamlit run streamlit_app.py
```

浏览器打开 http://localhost:8501：上传 PDF → 提问 → 查看带来源的结果 → 反馈。

## 常用命令

| 用途 | 命令 |
|---|---|
| 依赖服务体检（起服务后先跑这个） | `python scripts/check_services.py` |
| 生成演示用 PDF（带标题层级 + 表格） | `python scripts/make_demo_pdf.py` |
| 预下载精排模型（建议部署时跑一次） | `python scripts/download_models.py` |
| 重置索引后重新索引 | `python scripts/index_documents.py xxx.pdf --reset` |
| 只解析不写库（调试切块） | `python scripts/index_documents.py xxx.pdf --dry-run` |
| 生成企业文档语料（6 份手册/制度类 PDF） | `python scripts/make_enterprise_corpus.py` |
| 生成评测集（真实 chunk_id，逐条校验） | `python scripts/build_eval_set.py` |
| 校验评测集是否仍然有效（语料改过就会失效） | `python scripts/build_eval_set.py --check` |
| 跑评测（消融） | `python scripts/run_evaluation.py --all` |
| 跑失败分析（归类失败根因） | `python scripts/run_failure_analysis.py` |
| 跑切块消融（固定 vs 结构感知） | `python scripts/run_chunking_ablation.py xxx.pdf` |
| 跑反馈前后实验 | `python scripts/run_feedback_experiment.py` |
| 跑单元测试 | `pytest tests/unit -v` |

> 没装 Docker、或不确定 ES/Qdrant/Redis 该怎么在 Windows 上跑？
> 最省事的是用本仓库自带的 `docker-compose.yml` 只起这三个中间件，
> 再在本机跑 uvicorn 与 streamlit（对应上面的「方式 A」）。

## 评测闭环：怎么拿到可复现的数字

评测集不是手写的，而是**从语料派生出来的**，这保证了标注里每个 `chunk_id` 都真实存在。

```bash
# 1) 定稿语料（改完就不要再动它）
python scripts/make_enterprise_corpus.py     # 生成 6 份企业文档 · 7 份共 39 个 chunk
# 2) 建索引（把语料写进 ES + Qdrant）
python scripts/index_documents.py data/corpus/*.pdf
# 3) 生成评测集（用与建索引相同的解析/切块流程算出 chunk_id）
python scripts/build_eval_set.py
# 4) 跑评测
python scripts/run_evaluation.py --all
```

**为什么 chunk_id 可以离线算出来**：`document_id` 由文件内容 sha256 派生，
`chunk_id` 再由 `document_id + 页号 + 页内序号` 派生（见 `app/core/parser.py`）。
所以"语料 PDF + 同一套解析切块代码"就能唯一确定全部 chunk_id，
不需要先起 ES/Qdrant 再手工抄 UUID——手抄一批 UUID 是没法维护的。

**标注怎么保证不自欺**：每条问题给一组"答案短语"，脚本在全语料的 chunk 池里定位它们，
命中的 chunk 就标为相关。短语一个都找不到就**整体失败退出、不写文件**——
不会出现"标注看着填了、其实指向不存在的 chunk"这种最隐蔽的错误。

> ⚠️ **顺序不能反**：chunk_id 由文件内容哈希派生，
> **先定稿语料 → 再建索引 → 最后造评测集**。造完评测集又回头改 PDF，
> 旧标注会集体失效，表现为"Recall 突然掉到 0"——很难联想到是 PDF 本身变了。
> 发现它的方法：`python scripts/build_eval_set.py --check`。

### 跑评测前会先体检评测集

`run_evaluation.py` 在连任何服务之前先做前置校验：

| 情况 | 处理 |
|---|---|
| `relevant_chunk_ids` 还是 `REPLACE_WITH_REAL_CHUNK_ID` 占位符 | **拒绝运行**（说明评测集从没真正标注过） |
| 标注指向当前语料里不存在的 chunk_id | **拒绝运行**（几乎总是语料被改过） |
| 某些样本没有人工标注 | 警告，并**排除出指标分母**（记为 `unverifiable`，不算失败） |
| 同一条问题重复出现 | 警告（预测按位置对齐，不会互相覆盖，但会重复计分） |

为什么要这么严：**一个"能跑但毫无意义"的评测比直接报错危险得多**。
占位符评测集照样能跑完、照样打印指标，只是全是 0——
没有这道闸门，很容易把"标注没填"误读成"检索效果差"。

确实想看坏数据下的现象，加 `--allow-stale-dataset`（结果不可用于下结论）。

### 指标口径：三个数一起看

除 `Recall@K / MRR@K / NDCG@K` 外，还会给出：

- `evaluated` —— **真正参与平均**的条数（有人工标注的那些），指标分母就是它；
- `unverifiable` —— 缺人工标注、无法判定的条数。

分母只算 `evaluated` 是刻意的：拿总条数当分母，评测集标注补齐/删减一点，
指标就跟着漂，看起来像"模型变差了"，其实是"标注没填"。

## 常见问题

### 提问时报超时（Timeout / ReadTimeout）

**最典型的原因是精排模型在请求里现下现加载。** 精排模型约 1GB，如果在 `/search`
请求线程里才去下载，请求会长时间阻塞，前端只能看到"超时"，而真正的检索
（BM25 / 向量 / RRF）一步都没跑到。

已做的三层防护：

1. **懒加载**：请求传 `rerank=false`，或 `.env` 设 `RERANK_ENABLE=false`，
   精排器根本不会被创建，也就不会加载模型；
2. **真超时**：模型加载与打分都受 `RERANK_TIMEOUT_SECONDS` 约束，超时自动降级为融合结果，
   同时打印 `reranker_load_timeout` 结构化日志，不会把请求挂死；
3. **可观测**：响应里的 `rerank_applied` 字段直接告诉你本次到底有没有真的精排。

推荐做法：部署时先跑 `python scripts/download_models.py` 把模型下好，
请求路径上就只剩本地加载了。国内务必在 `.env` 配 `HF_ENDPOINT=https://hf-mirror.com`。

精排模型的缓存目录可以用环境变量 `HF_HOME` 指定（默认在 `~/.cache/huggingface`，Windows 即
`C:\Users\<你>\.cache\huggingface`）。模型动辄 2 GB 起，C 盘紧张时把它挪到大盘更划算：

```bash
# 1) 先把整个缓存目录搬过去（HF_HOME 只能整体指定根目录，不支持单独迁移某个模型）
#    2) 设环境变量：HF_HOME=D:\huggingface
#    3) 重启后端进程（HF_HOME 在 huggingface_hub 被 import 时读一次，之后改无效）
```

验证是否生效：`HF_HUB_OFFLINE=1 python -c "from huggingface_hub import constants; print(constants.HF_HUB_CACHE)"`。

### 报 503 DEPENDENCY_UNAVAILABLE

说明连不上 Elasticsearch / Qdrant / Redis / Embedding 接口。检查 `.env` 里的地址
**必须和你的部署方式一致**：

| 部署方式 | QDRANT / ELASTICSEARCH / REDIS | EMBEDDING_BASE_URL |
|---|---|---|
| 本机直跑 `uvicorn`（中间件在本机/Docker） | `127.0.0.1` | `http://127.0.0.1:11434/v1` |
| `docker compose` 全家桶（api 也在容器里） | 容器名 `qdrant` / `elasticsearch` / `redis` | `http://host.docker.internal:11434/v1` |

注意容器内的 `localhost` 指向容器自己，到不了宿主机的 Ollama。

### 检索很慢：BM25 只要 10ms，向量路却要好几秒

两种常见原因，都不是检索算法的问题：

1. **地址写了 `localhost`。** Windows 上 `getaddrinfo("localhost")` 把 IPv6 的 `::1`
   排在 `127.0.0.1` 前面，而 Qdrant 默认只绑 IPv4。于是每个请求都要先花约 2 秒
   去连 `::1` 失败、再退回 IPv4。**把 `.env` 里所有地址改成 `127.0.0.1` 即可。**
   （ES 不受影响，因为它在 `[::1]` 和 `127.0.0.1` 上都有监听。）

2. **系统代理劫持了本机请求。** 开着 Clash / VPN 时，`HTTP_PROXY` / `HTTPS_PROXY`
   会被 httpx 读取（Qdrant、OpenAI 客户端底层都用 httpx），连 `127.0.0.1` 也走代理。
   而 ES 用的 urllib3 不读代理变量 → 表现为"ES 通了、Qdrant 超时"。
   `app/config.py` 已自动把本机地址加进 `NO_PROXY`，无需手工处理。

### 开启精排后每次都降级（rerank_applied=false）

本机 CPU 上精排打分很慢，实测 **20 个候选约 16 秒**。如果 `RERANK_TIMEOUT_SECONDS`
给太小（比如默认之外的 10），就会每次都超时降级。三个调节方向：

- 把 `RERANK_TIMEOUT_SECONDS` 调到 30；
- 把 `RERANKER_MAX_LENGTH` 从 512 降到 256，打分速度快约一倍；
- 把 `RECALL_TOP_K` 从 20 降到 10（候选少一半，速度快一半）。

另外精排默认**先用本地缓存加载**（实测 3.5 秒、不联网）；只有缓存里没有模型时
才会去联网下载（走 `HF_ENDPOINT` 镜像）。所以部署时先跑一次
`python scripts/download_models.py`，请求路径上就永远不会出现"下载"。

### 上传成功但提问检索不到这份 PDF 的内容

先看 `/stats` 的 `elasticsearch_chunks` 有没有涨，再用
`python scripts/index_documents.py xxx.pdf --dry-run` 看能解析出多少块。三种典型原因：

| 现象 | 原因 | 处理 |
|---|---|---|
| 报 422，切出 0 个 Chunk | PDF 是扫描件/纯图片，没有文字层 | 换带文字层的 PDF（Word/WPS 另存为 PDF 即可） |
| 块数远小于预期 | 标题判定把正文吃掉了 | 见下方「标题判定为什么用相对字号」 |
| 内容进去了但排不到前面 | 该 PDF 篇幅小、或 Chunk 太粗 | 调小 `StructureAwareChunker` 的 `min_chars` |

### 标题判定为什么用「相对字号」而不是固定阈值

`PDFParser` 会先扫全文统计字号直方图（按字符数加权），推断出**这份文档自己的正文字号**，
再把明显大于正文（≥1.15 倍）的字号梯度映射成标题层级。原因是实测发现：

- 技术手册正文约 11pt，标题 15~20pt；
- **幻灯片类 PDF 正文就有 16.5pt**（一份 58 页 Git 教程里 83.8% 的字符是 16.5pt）。

如果写死 `avg_size >= 14 就算标题`，幻灯片上的**全部正文都会被误判成标题**塞进
`heading_path`，结果 `content` 里一个字都不剩，检索出来全是空表格线。
同理，`find_tables()` 会把幻灯片文本框的边框误判成表格，所以加了
「≥2 行、≥2 列、≥3 个非空单元格、≥20 字符」的质量门槛来挡掉这些假表格。

### 重复上传同一份 PDF 会不会产生重复数据

**不会——索引是幂等的。** `document_id` 由文件内容 sha256 派生，
`chunk_id` 再由 `document_id + 页号 + 页内序号` 派生，所以同一份文件反复上传
会用相同的 ID 覆盖写入，`/stats` 的数量不会增长。

反过来，**改动了 PDF 内容再上传会得到一批新 ID，旧 Chunk 会残留**，两种处理方式：

```bash
# 方式一：整体重置（只有一份文档时最简单）
python scripts/index_documents.py 新版本.pdf --reset

# 方式二：只删这一份（多文档场景用这个）
#   前端「文档库」标签页有删除按钮；也可以直接调接口：
#   DELETE http://127.0.0.1:8000/documents/{document_id}
#   document_id 可以从 GET /documents 里查
```

### 检索结果不准确，按这个顺序排查

**1. embedding 模型是不是中文向的 —— 影响最大**

`nomic-embed-text` 是英文向模型，中文语义检索偏弱。实测同一份中文语料：

| 模型 | Top-1 准确率 | Top-3 |
|---|---|---|
| `nomic-embed-text`（768 维） | 4/8 | 7/8 |
| `bge-m3`（1024 维） | **7/8** | 7/8 |

```bash
ollama pull bge-m3            # 1.2GB
# 改 .env：EMBEDDING_MODEL=bge-m3、EMBEDDING_DIM=1024
python scripts/index_documents.py xxx.pdf --reset   # 维度变了，必须重建
```

**2. ES 的中文分词器**

ES 默认 `standard` 分词器把中文**按单字**切：`版本控制系统` →
`['版','本','控','制','系','统']`。于是查询"Git有什么用"被切成 `git/有/什/么/用/处`，
而"有""用""么"是高频字 —— 实测一句中文查询能命中 **21/23** 条，BM25 区分度归零。

现在 `content` 与 `heading_text` 都用 ES **内置的 `cjk` 分词器**（切二元组，
`版本控制系统` → `['版本','本控','控制','制系','系统']`），不需要安装 IK 插件。

**3. 标题匹配是否真的生效**

`heading_path` 是 `keyword` 类型，只支持整串精确匹配，把它塞进 `multi_match`
是**无效的**（实测单独查询命中 0 条，给的 `^2` 权重形同虚设）。
所以另外存了一个 `heading_text`（text 类型，内容是把标题路径拼成的字符串）参与检索。

**4. 融合参数 —— 实测不是瓶颈**

把 `RECALL_TOP_K`(20/12/8/5) × `RRF_K`(60/20/10/5) 共 16 种组合都跑了一遍，
Top-1 稳定在 4/8 不变 —— 说明排序问题出在底层检索质量，不在融合参数上。
当前保留 `recall_top_k=20, rrf_k=60`。

> 上面这些数字来自 8 条人工构造的问题，是**粗略代理指标**，不是严格的评测集。
> 要做正式评测请用 `scripts/run_evaluation.py`（配合 `export_evaluation_template.py`
> 生成人工核对过的评测集），产出 Recall@K / MRR / NDCG。

### 正文显示成很多条短行、右侧大片空白

PDF 里的换行是**视觉换行**（排到页宽就断），不是语义换行。早期直接把行用 `\n`
拼起来，前端 `white-space: pre-wrap` 会把它当真实换行 —— 每行只占十几个字、
右侧留白，整块被纵向撑到 1000px 以上。

现在统一走 `app/utils/text.py::join_pdf_lines()`：按句末标点分段、中文直接相接、
英文之间补空格。实测同一份文档结果卡片高度从 **1291/1620/1357px 降到 448/663/281px**。

## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/search` | 混合检索（BM25 + 向量 + RRF + 可选精排）。请求体可带 `document_id` 把范围限定到某一份文档 |
| POST | `/feedback` | 记录反馈（positive / negative / correction） |
| POST | `/documents/index` | 上传 PDF 并索引，返回 `document_id` / `blocks` / `chunks` |
| GET | `/documents` | 列出索引里已有的文档（来源名、Chunk 数、页数、内容类型分布） |
| DELETE | `/documents/{document_id}` | **只删除指定文档**的全部 Chunk（ES + Qdrant 两边都删） |
| GET | `/health` | 健康检查 |
| GET | `/stats` | 索引 / 反馈统计 |
| GET | `/metrics` | 指标占位（真正的实验指标看 run_evaluation.py） |
| GET | `/trace/{request_id}` | 回查某次检索的完整链路 Trace（request_id 见 `/search` 响应） |

`DELETE` 的存在意义：改过的 PDF 重新上传会得到一批新 Chunk ID，旧 Chunk 会残留，
以前只能 `--reset` 清空整个索引，多文档场景下不可接受。

## 前端（Streamlit）三个标签页

```bash
streamlit run streamlit_app.py   # http://localhost:8501
```

- **检索**：提问 → 指标卡（耗时 / 策略 / 更正缓存 / 精排是否生效）+ 结果卡片
  （来源文档、页码、标题路径、类型、双路命中标记、融合分与精排分），每条都能 👍/👎；
  另有一个「链路 Trace」折叠区，把 BM25 / 向量 / RRF / 精排各阶段的 Top-N 与延迟分解摊开
  —— 对应 [开发文档](docs/开发文档.md) 的 **WP18（Retrieval Debugger）**：不仅看结果，还能解释结果是怎么来的。
- **文档库**：上传建索引、查看已索引文档列表、按文档删除；
- **反馈闭环**：提交更正写法，再搜原问题即可看到 `used_correction=true` 与结果改写。

侧边栏可以**限定检索范围到某一份文档**——多文档索引下这是刚需，
否则问 A 文档的问题可能被 B 文档的内容挤掉。

## 诚实原则

评测集的 `relevant_chunk_ids` 必须是**真实存在的 chunk_id**（用 `scripts/build_eval_set.py` 生成，
它按"答案短语"定位并逐条校验；改过语料就用 `--check` 复查）。
Recall/MRR/NDCG、延迟、成本等数字必须来自 `run_evaluation.py` 的真实输出，禁止虚构。
