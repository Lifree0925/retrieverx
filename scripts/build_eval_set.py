"""
scripts/build_eval_set.py —— 生成 RetrieverX 评测集（真实 chunk_id + 执行级校验）

【这个脚本解决什么问题】
  `data/evaluation.jsonl` 里原本是 `REPLACE_WITH_REAL_CHUNK_ID` 占位符——
  也就是说"评测闭环从未真正跑通过"，README 里那套"数字必须来自真实输出"的承诺是落空的。
  手工填 UUID 不现实，而这个脚本把它变成可复现的一步：

    语料 PDF --(PDFParser + StructureAwareChunker)--> chunk 池 --(按答案短语定位)--> 真实 chunk_id

【为什么可以离线做，不需要 ES / Qdrant】
  `document_id` 由文件内容 sha256 派生，`chunk_id` 再由 document_id 派生
  （见 app/core/parser.py）。所以**只要输入同一份 PDF、走同一套解析与切块代码**，
  得到的 chunk_id 与"真正建索引时写进 ES/Qdrant 的那些"完全一致。
  这意味着评测集的正确性可以在建索引之前就确定下来。

【标注怎么做才不会自欺】
  每条问题给一组"答案短语"（必须原样出现在正确 chunk 里）。
  脚本把短语在 chunk 池里定位，命中哪些 chunk 就把哪些标为 relevant。
  —— 这是**可机械验证**的标注：短语找不到就直接失败退出，不会出现
     "标注看似填了、其实指向了不存在的 chunk"这种最隐蔽的错误。

【⚠️ 顺序约束】
  必须先定稿语料（make_enterprise_corpus.py）再跑本脚本。
  之后再改 PDF，chunk_id 会集体变化，评测集会失效——
  `--check` 就是用来发现这件事的。

【用法】
  python scripts/build_eval_set.py                       # 生成 data/evaluation.jsonl
  python scripts/build_eval_set.py --check               # 只校验现有评测集是否还有效
  python scripts/build_eval_set.py --out data/evaluation_v2.jsonl
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUT = PROJECT_ROOT / "data" / "evaluation.jsonl"
META_PATH = PROJECT_ROOT / "data" / "evaluation_meta.json"
CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"
DEMO_PDF = PROJECT_ROOT / "data" / "产品手册-PX4200.pdf"

# 演示文档在语料里的"键"，与 corpus_index.json 的 key 对齐
DEMO_KEY = "PX4200"


# ======================================================================
# 题目定义
# ----------------------------------------------------------------------
# 每条：question（自然语言提问）、doc（答案在哪份文档）、
#       phrases（答案短语，必须原样出现在正确 chunk 里，用来定位 relevant chunk）
#       query_type（EXACT / SEMANTIC / MIXED / NUMERIC，供分层看指标）
#
# 选短语的原则：**短、唯一、不含空格**。
# 因为 PDF 里的换行是视觉换行，切块时会被规整掉，
# 带空格的短语在规整后可能匹配不上；纯中文长串最稳。
# ======================================================================
QUESTIONS: list[dict] = [
    # ---------------- QX-8800 产品手册 ----------------
    dict(doc="QX8800", query_type="NUMERIC",
         question="QX-8800 双控制器切换需要多长时间",
         phrases=["切换时间实测小于三秒"]),
    dict(doc="QX8800", query_type="NUMERIC",
         question="QX-8800 的随机读写 IOPS 分别是多少",
         phrases=["随机读 IOPS"]),
    dict(doc="QX8800", query_type="SEMANTIC",
         question="开启压缩和重删对性能有什么影响",
         phrases=["写延迟增加"]),
    dict(doc="QX8800", query_type="NUMERIC",
         question="QX-8800-P 型号的裸容量和报价是多少",
         phrases=["184TB"]),
    dict(doc="QX8800", query_type="EXACT",
         question="关键业务服务等级的现场到达时限是多久",
         phrases=["关键业务"]),
    dict(doc="QX8800", query_type="NUMERIC",
         question="同步复制对两端链路延迟有什么硬性要求",
         phrases=["往返延迟必须低于"]),
    dict(doc="QX8800", query_type="EXACT",
         question="远程复制高级版支持哪些额外能力",
         phrases=["同步复制与三站点级联"]),

    # ---------------- 员工手册 ----------------
    dict(doc="员工手册", query_type="NUMERIC",
         question="未休完的年休假最多能结转多少天",
         phrases=["最多结转五天至次年"]),
    dict(doc="员工手册", query_type="NUMERIC",
         question="产假和陪产假分别有多少天",
         phrases=["产假"]),
    dict(doc="员工手册", query_type="SEMANTIC",
         question="年中入职的员工当年年休假怎么折算",
         phrases=["实际工作月份数"]),
    dict(doc="员工手册", query_type="NUMERIC",
         question="正式员工离职要提前多久通知公司",
         phrases=["提前三十日以书面形式通知"]),
    dict(doc="员工手册", query_type="NUMERIC",
         question="竞业限制期间的补偿金标准是多少",
         phrases=["百分之三十"]),
    dict(doc="员工手册", query_type="NUMERIC",
         question="加班换来的调休有效期是多久",
         phrases=["逾期作废"]),
    dict(doc="员工手册", query_type="SEMANTIC",
         question="医疗期时长是根据什么确定的",
         phrases=["医疗期为三个月"]),

    # ---------------- IT 服务 SOP ----------------
    dict(doc="IT服务", query_type="EXACT",
         question="P1 级别故障的判定标准是什么",
         phrases=["核心业务全停"]),
    dict(doc="IT服务", query_type="SEMANTIC",
         question="哪些情况必须立即升级故障等级",
         phrases=["未定位到根因"]),
    dict(doc="IT服务", query_type="NUMERIC",
         question="生产环境变更申请要提前多久提交",
         phrases=["提前一个工作日提交变更申请"]),
    dict(doc="IT服务", query_type="EXACT",
         question="没有回滚方案的变更能不能审批",
         phrases=["一律不予审批"]),
    dict(doc="IT服务", query_type="NUMERIC",
         question="数据库备份文件保留多长时间",
         phrases=["保留三十天"]),
    dict(doc="IT服务", query_type="SEMANTIC",
         question="恢复演练多久做一次，要记录什么指标",
         phrases=["季度须组织一次恢复演练"]),
    dict(doc="IT服务", query_type="SEMANTIC",
         question="故障复盘文档对改进措施一栏有什么要求",
         phrases=["改进措施"]),

    # ---------------- 信息安全管理规范 ----------------
    dict(doc="信息安全", query_type="NUMERIC",
         question="系统口令的有效期是多久",
         phrases=["有效期九十天"]),
    dict(doc="信息安全", query_type="EXACT",
         question="哪些系统强制要求启用多因素认证",
         phrases=["强制启用多因素认证"]),
    dict(doc="信息安全", query_type="EXACT",
         question="L3 级别的保密数据都有哪些典型内容",
         phrases=["客户名单"]),
    dict(doc="信息安全", query_type="NUMERIC",
         question="受控外发链接的有效期最长是多少天",
         phrases=["最长七个自然日"]),
    dict(doc="信息安全", query_type="NUMERIC",
         question="发现疑似安全事件要在多久内上报",
         phrases=["两小时内"]),
    dict(doc="信息安全", query_type="NUMERIC",
         question="安全审计日志要保存多久，谁能查阅",
         phrases=["保存三年"]),
    dict(doc="信息安全", query_type="SEMANTIC",
         question="为什么制度要规定主动上报安全事件不予追责",
         phrases=["上报免责"]),

    # ---------------- 采购与供应商管理制度 ----------------
    dict(doc="采购制度", query_type="EXACT",
         question="十万到一百万的采购由谁审批",
         phrases=["分管副总"]),
    dict(doc="采购制度", query_type="NUMERIC",
         question="一百万以上的采购需要几家供应商报价",
         phrases=["总经理办公会"]),
    dict(doc="采购制度", query_type="NUMERIC",
         question="单一来源采购占比超过多少需要书面解释",
         phrases=["连续两个季度超过百分之三十"]),
    dict(doc="采购制度", query_type="NUMERIC",
         question="供应商年度评估里交付准时率占多少权重",
         phrases=["交付准时率百分之三十五"]),
    dict(doc="采购制度", query_type="NUMERIC",
         question="设备类采购要留多少质保金",
         phrases=["百分之五作为质保金"]),
    dict(doc="采购制度", query_type="NUMERIC",
         question="紧急采购每季度最多可以走几次",
         phrases=["不得超过三次"]),
    dict(doc="采购制度", query_type="SEMANTIC",
         question="发票内容与实际采购标的不一致会有什么后果",
         phrases=["暂停该供应商全部在途付款"]),

    # ---------------- 财务报销管理规定 ----------------
    dict(doc="财务报销", query_type="EXACT",
         question="住宿费的限额是按什么划分的",
         phrases=["城市类别"]),
    dict(doc="财务报销", query_type="NUMERIC",
         question="费用发生后要在多久内提交报销",
         phrases=["六十个自然日内"]),
    dict(doc="财务报销", query_type="EXACT",
         question="五万元以上的报销需要谁审批",
         phrases=["须总经理审批"]),
    dict(doc="财务报销", query_type="NUMERIC",
         question="借款必须在多久内完成核销",
         phrases=["十五个工作日内完成核销"]),
    dict(doc="财务报销", query_type="EXACT",
         question="哪些个人消费性质的费用不予报销",
         phrases=["不予报销"]),
    dict(doc="财务报销", query_type="EXACT",
         question="单笔一万元以下的报销由谁终审",
         phrases=["财务经理终审"]),
    dict(doc="财务报销", query_type="NUMERIC",
         question="对外赠送礼品单件金额上限是多少",
         phrases=["不超过五百元"]),

    # ---------------- 演示文档（原来的 PX-4200 手册）----------------
    dict(doc=DEMO_KEY, query_type="NUMERIC",
         question="PX-4200 整机的标准保修期是多久",
         phrases=["整机标准保修期为三年"]),
    dict(doc=DEMO_KEY, query_type="NUMERIC",
         question="PX-4200-B 配置的含税价格是多少",
         phrases=["58600"]),
    dict(doc=DEMO_KEY, query_type="NUMERIC",
         question="设备签收后多少天内可以申请无理由退货",
         phrases=["七个自然日内"]),
    dict(doc=DEMO_KEY, query_type="EXACT",
         question="高级服务等级的响应时间是多少",
         phrases=["高级服务"]),
    dict(doc=DEMO_KEY, query_type="SEMANTIC",
         question="哪些情形不属于免费保修范围",
         phrases=["不属于免费保修范围"]),
]


def norm(text: str) -> str:
    """去掉全部空白后再比较。

    PDF 里的换行是**视觉换行**（排到页宽就断），切块时会被 join_pdf_lines 规整。
    所以短语匹配必须对空白不敏感，否则"短语明明在文档里、却匹配不上"，
    会把标注错误误判成"答案不在语料中"。
    """
    return re.sub(r"\s+", "", text or "")


def load_documents() -> dict[str, Path]:
    """收集语料：curated corpus 目录 + 演示文档。"""
    docs: dict[str, Path] = {}
    index_path = PROJECT_ROOT / "data" / "corpus_index.json"
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            index = {}
        for filename, info in index.items():
            path = CORPUS_DIR / filename
            if path.exists():
                docs[info.get("key") or filename] = path
    if DEMO_PDF.exists():
        docs[DEMO_KEY] = DEMO_PDF
    return docs


def build_chunk_pool(docs: dict[str, Path]):
    """按"与建索引完全一致"的流程解析 + 切块，得到 {doc_key: [chunk, ...]}。

    关键点：必须用 PDFParser + StructureAwareChunker（而不是自己读文本），
    否则 chunk_id 与真正索引进 ES/Qdrant 的对不上，评测就测了另一批数据。
    """
    from app.core.chunker import StructureAwareChunker
    from app.core.parser import PDFParser

    pdf_parser = PDFParser()
    chunker = StructureAwareChunker()
    return {key: chunker.chunk(pdf_parser.parse_pdf(path)) for key, path in docs.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="生成/校验 RetrieverX 评测集")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--check", action="store_true",
                        help="只校验指定评测集是否仍然有效（chunk_id 是否还能在语料里找到）")
    args = parser.parse_args()

    docs = load_documents()
    if not docs:
        raise SystemExit(
            "没找到任何语料 PDF。\n"
            "先跑：python scripts/make_enterprise_corpus.py"
        )
    pool = build_chunk_pool(docs)
    total_chunks = sum(len(v) for v in pool.values())
    print(f"语料: {len(docs)} 份文档 / {total_chunks} 个 chunk")
    for key, chunks in pool.items():
        print(f"  {key:<10} {docs[key].name}  ({len(chunks)} chunks)")

    # chunk_id 全集：用来校验"标注指向的 id 是否真实存在"
    all_ids: dict[str, str] = {}
    for key, chunks in pool.items():
        for chunk in chunks:
            all_ids[str(chunk.chunk_id)] = key

    if args.check:
        target = Path(args.out)
        if not target.exists():
            raise SystemExit(f"找不到评测集 {target}")
        samples = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
        stale = []
        for sample in samples:
            for cid in sample.get("relevant_chunk_ids") or []:
                if cid not in all_ids:
                    stale.append((sample.get("query"), cid))
        print(f"\n评测集 {target.name}: {len(samples)} 条")
        if stale:
            print(f"[FAIL] {len(stale)} 处标注指向了不存在的 chunk_id（语料被改过？）：")
            for query, cid in stale[:10]:
                print(f"    {query}  ->  {cid}")
            raise SystemExit(1)
        print("[OK] 全部标注的 chunk_id 都能在当前语料里找到")
        return

    print(f"\n定位 {len(QUESTIONS)} 条问题的相关 chunk ...")
    failures: list[str] = []
    samples: list[dict] = []
    for spec in QUESTIONS:
        key = spec["doc"]
        if key not in pool:
            failures.append(f"[文档缺失] {spec['question']}  (doc={key!r} 不在语料里)")
            continue
        hits: list[str] = []
        for chunk in pool[key]:
            haystack = norm(chunk.content)
            if all(norm(p) in haystack for p in spec["phrases"]):
                hits.append(str(chunk.chunk_id))
        if not hits:
            failures.append(
                f"[短语未命中] {spec['question']}\n"
                f"    文档={key}  短语={spec['phrases']}\n"
                f"    → 要么短语不在语料里，要么这一节被切块切到了别处"
            )
            print(f"  [FAIL] {spec['question']}")
            continue
        samples.append({
            "query": spec["question"],
            "relevant_chunk_ids": hits,
            "query_type": spec["query_type"],
            "source_document": docs[key].name,
        })
        print(f"  [OK]   {spec['question']}  ->  {len(hits)} 个相关 chunk")

    if failures:
        print("\n" + "=" * 70)
        print(f"有 {len(failures)} 条标注未通过，**没有写任何文件**：")
        for f in failures:
            print("  " + f)
        raise SystemExit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")

    by_type: dict[str, int] = {}
    for sample in samples:
        by_type[sample["query_type"]] = by_type.get(sample["query_type"], 0) + 1
    meta = {
        "dataset": out_path.name,
        "samples": len(samples),
        "chunk_universe": total_chunks,
        "query_types": by_type,
        "documents": {
            key: {
                "file": docs[key].name,
                "sha256": hashlib.sha256(docs[key].read_bytes()).hexdigest(),
                "chunks": len(chunks),
            }
            for key, chunks in pool.items()
        },
        "note": "chunk_id 由 PDF 内容 sha256 派生。语料一旦改动，本评测集立即失效，"
                "用 `python scripts/build_eval_set.py --check` 可以发现。",
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"全部 {len(samples)} 条标注通过 → {out_path}")
    print(f"题型分布: {json.dumps(by_type, ensure_ascii=False)}")
    print(f"元信息(含语料 sha256): {META_PATH}")
    print("\n下一步：")
    print("  python scripts/index_documents.py data/corpus/*.pdf   # 建索引")
    print("  python scripts/run_evaluation.py --all                # 跑 Recall/MRR/NDCG")


if __name__ == "__main__":
    main()
