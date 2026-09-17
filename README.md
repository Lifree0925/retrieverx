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
| 跑评测并支持**断点续跑**（长跑强烈建议） | `python scripts/run_evaluation.py --all --progress-out progress.jsonl --resume` |
| 跑失败分析（归类失败根因） | `python scripts/run_failure_analysis.py` |
| 跑切块消融（固定 vs 结构感知） | `python scripts/run_chunking_ablation.py xxx.pdf` |
| 跑反馈前后实验 | `python scripts/run_feedback_experiment.py` |
| 跑单元测试 | `pytest tests/unit -v` |

> **精排档很慢，务必用 `--progress-out` 续跑。** 本地 CPU 上单条约 20 秒，63 条就是 20 分钟量级；
> 实测被外部中断过两次（进程被回收，日志里没有任何异常，只是停住）。
> `--progress-out` 会**做一条存一条**，被中断后重跑同一命令会自动跳过已完成的样本，
> 不会让 20 分钟白跑。文件里出现半行（写到一半被 kill）也只会丢掉那一条。
>
> 顺带说明：进度文件在**所有档位之间共用**，靠 `mode` 字段区分；
> 不加 `--resume` 时会覆盖重来，避免新旧两轮的预测混在一起。

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
`chunk_id` 再由 **Chunk 自身的内容**派生（见 `app/utils/ids.py`）：

```
chunk_id = uuid5(NAMESPACE_URL, f"chunk:{document_id}:{内容指纹}:{第几次出现}")
```

所以"语料 PDF + 同一套解析切块代码"就能唯一确定全部 chunk_id，
不需要先起 ES/Qdrant 再手工抄 UUID——手抄一批 UUID 是没法维护的。

**标注怎么保证不自欺**：每条问题给一组"答案短语"，脚本在全语料的 chunk 池里定位它们，
命中的 chunk 就标为相关。短语一个都找不到就**整体失败退出、不写文件**——
不会出现"标注看着填了、其实指向不存在的 chunk"这种最隐蔽的错误。

> ⚠️ **顺序不能反**：chunk_id 由文件内容哈希派生，
> **先定稿语料 → 再建索引 → 最后造评测集**。造完评测集又回头改 PDF，
> 旧标注会集体失效，表现为"Recall 突然掉到 0"——很难联想到是 PDF 本身变了。
> 发现它的方法：`python scripts/build_eval_set.py --check`。

> 📌 **chunk_id 为什么是"内容派生"而不是"页码 + 页内序号"**
>
> 早期实现用 `uuid5(document_id, f"{page}:{ordinal}")`。它看起来也是确定性的，
> 但把 ID 绑死在**输出顺序**上，于是有个很隐蔽的后果：解析器一改
> （调整页眉页脚过滤门槛、放宽表格识别条件、改切块阈值），同一页里前面的块
> 增加或减少，**后面所有块的 ordinal 集体位移**——文件一个字节都没改，
> 整批 chunk_id 却变了，索引里于是留下"新 ID + 旧残留"两份同样的内容。
>
> 改成内容派生之后，这个不变量对所有写进索引的 Chunk 都成立：
>
> **chunk_id 只由"它所在文档 + 它自己的内容"决定**——不依赖页码、不依赖解析顺序、
> 不依赖解析器版本。同一段文字无论排在哪里、索引多少次都是同一个 ID（就地覆盖），
> 内容真的改了才换 ID。
>
> 两个容易被忽略的细节：① 同一文档里两段**完全相同**的文字必须用
> "第几次出现"区分，否则后一段会覆盖前一段、静默丢一个 chunk；
> ② `document_id` 必须留在种子里，否则两份文档里相同的段落会共用一个 ID，
> "删除其中一份文档"会连带删掉另一份还在用的块。
> 这两条都有测试盯着（`tests/unit/test_chunk_id_stability.py`）。


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

### 为什么要把问题分成「字面式」和「改写式」

评测集 63 条里有 16 条是**改写式问题**（`paraphrased: true`）：
问法与语料原文用词故意不同（把"口令"问成"密码"、把"回滚方案"问成"退路"），
但答案短语仍逐字出现在语料里。

**这不是为了把题目变难，而是为了让消融实验有区分度。** 实测踩到过：
第一版 47 条问题全是照着原文措辞写的，结果 ——

| 模式 | Recall@5 |
|---|---|
| BM25 Only（仅关键词） | **0.9894** |
| Vector Only（仅向量） | 1.0000 |

单靠关键词匹配就接近满分，四个档位的差距**全被天花板效应压平了**，
这张消融表其实什么也证明不了。分成两组之后，两组之间的**差距**才是
"语义检索到底起了什么作用"的直接证据：字面式两组都高是正常的，
而改写式上 BM25 应当明显掉分、向量/融合应当更稳。

`run_evaluation.py --all` 会同时打印总表与分组表。**看结论请以分组表为准**，
并记住一句话：如果两组都接近满分，说明问题仍然过于字面化，
**这张表不能用来论证组件价值**——那是指标设计的问题，不是组件没用。

### 本地实测数字（63 条，纯 CPU，零成本方案）

跑法：`python scripts/run_evaluation.py --all`，评测集 63 条（47 字面式 + 16 改写式），
0 条无法判定（`unverifiable=0`，即所有指标都基于完整标注，没有靠剔除样本美化的空间）。

**总表**

| 模式 | Recall@5 | MRR@5 | NDCG@5 | 平均延迟 |
|---|---|---|---|---|
| BM25 Only（仅关键词） | 0.9841 | 0.9053 | 0.9176 | 12.1 ms |
| Vector Only（仅向量） | 0.9841 | 0.9101 | 0.9288 | 99.5 ms |
| Hybrid（RRF 融合） | 0.9841 | 0.9259 | 0.9411 | 96.0 ms |
| Hybrid + Rerank（完整链路） | **1.0000** | **0.9577** | **0.9648** | 21432 ms |

> **口径说明（两个数都要说清楚）**
> ① **准确率类指标与机器负载无关，延迟有关。** 同一套配置独立跑过两次
> （中间换掉了 `chunk_id` 的派生方式），Recall / MRR / NDCG 与分组表**逐位一致**，
> 而平均延迟在 **19.9 s ~ 21.4 s** 之间波动。所以延迟只作量级参考，
> 不要拿它做小数点级的比较。
> ② 两次准确率完全一致，本身也是一次有价值的**回归验证**：
> 把 `chunk_id` 从"页码 + 页内序号"改成"由内容派生"之后，检索行为没有任何变化
> —— 这证明那次改动确实只是"换了一套更稳定的 ID"，没有顺手改坏排序。

**分组表**（这才是能看出组件价值的表）

| 模式 | 字面式 Recall@5 / MRR@5 | 改写式 Recall@5 / MRR@5 |
|---|---|---|
| BM25 Only | 0.9894 / 0.9450 | 0.9688 / **0.7885** |
| Vector Only | 1.0000 / 0.9220 | 0.9375 / **0.8750** |
| Hybrid | 1.0000 / 0.9539 | 0.9375 / 0.8438 |
| Hybrid + Rerank | 1.0000 / 0.9787 | **1.0000** / **0.8958** |

环境：Ollama `bge-m3`（向量）+ `BAAI/bge-reranker-v2-m3`（精排），qwen2.5-coder 未参与检索链路。
数字由 `run_evaluation.py` 的结构化日志直接生成，不经过手工转抄。

### 从这组数字里能读出的四件事

**1. 字面式问题确实没有区分度，改写式才有。** 字面式上 BM25 单档 MRR 就有 0.9450，
四档挤在一起；一旦问法换掉措辞，BM25 的 MRR 掉到 0.7885（相对字面式 **−16.6%**）。
这直接印证了"必须把两类问题分开看"——混在一起时，天花板效应会把差距全抹平。

**2. 语义检索的价值被量化了，不是"感觉更准"。** 改写式上向量把 BM25 的 MRR
从 0.7885 拉到 0.8750（**+11.0%**）。这就是"同一个问题换个说法，关键词检索会输"的具体数字。

**3. 精排的性价比是这个项目最该被追问的数字。** MRR 从 0.9259 提到 0.9577（**+3.4%**），
但平均延迟从 **96.0 ms 涨到 21432 ms（约 223 倍）**。结论很明确：
精排**不适合**放在在线默认路径上；它适合"结果数量少、单条价值高"的场景（如问答前取 5 条），
或者用 GPU / 更小的精排模型把延迟压下来。**这是一个成本决策，不是一个"越多越好"的选择。**

**4. 一个反直觉的观察：融合不一定比单路好。**
改写式 MRR 上 `Hybrid` 是 0.8438，**低于** `Vector Only` 的 0.8750。
原因是 RRF 融合把 BM25 在改写式上的噪声一起折进来了（BM25 在这组上 MRR 只有 0.7885）。
只有加上精排才重新超过单路向量（0.8958）。
所以"混合检索一定优于单一检索"是**错的**——它取决于查询类型与两路的质量对比，
这也是本项目的 `QueryClassifier` + 自适应策略要解决的问题。

> 复现方式见上一节；`data/evaluation.jsonl` 与 `data/evaluation_meta.json`（含语料 sha256）
> 都在仓库里，改语料会让评测集失效，用 `scripts/build_eval_set.py --check` 可以发现。

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
`chunk_id` 再由 **Chunk 自身内容**派生，所以同一份文件反复上传会用相同的 ID
覆盖写入，`/stats` 的数量不会增长。

而且值得强调的是：**这个幂等性不依赖"解析器不改变"**。
早期 ID 是"页码 + 页内序号"，于是调一下页眉过滤门槛、改一下表格识别条件，
同一页后面的块 ID 就集体位移 —— 文件没动却产生一批新 ID。
改成内容派生后，只要那段文字没变，ID 就不变（详见上文「chunk_id 为什么是内容派生」）。

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

### 上传接口的三道加固（都是"没有它就会出事"的那类）

上传是**唯一一个把外部输入变成内部数据**的入口，也是唯一没有鉴权的写接口，
所以它的防护要单独说清楚：

| 加固 | 之前的问题 | 现在的做法 |
|---|---|---|
| **体积上限** | 只查后缀，没有任何大小限制 → 一个超大文件就能把内存/磁盘打满 | 边写边数字节，超过 `MAX_UPLOAD_MB`（默认 50）立刻返回 **413**，并且**不信任 `Content-Length`**（请求头可伪造、分块传输时可以没有），同时删掉已经写下的半截临时文件 |
| **流式落盘** | `file.file.read()` 一次性全量读进内存 | 按 1 MiB 分块读写，几百 MB 的 PDF 也不会把进程撑爆 |
| **双写回滚** | ES 与 Qdrant 是两个独立存储，先写 ES 再写 Qdrant；中间失败就留下"半索引"（BM25 搜得到、向量搜不到），而接口还返回错误、用户以为彻底没写进去 | 任一路失败就把该文档从**两边一并清掉**，让索引回到"要么全有、要么全无"；回滚本身失败也会在响应里如实说明并给出清理办法 |

关于回滚的粒度，这里有个刻意的取舍：**按 `document_id` 整份删，而不是只删这次写入的 chunk**。
因为同一份文件的再上传会复用同一个 `document_id`，只删本次写入的话，上次没被覆盖的旧 chunk
仍会残留。整份删的语义是明确的——"这份文档现在不在索引里"，调用方重传一次即可。
**宁可让用户重传一次，也不要留下一个说不清状态的半索引。**

## API 一览

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
