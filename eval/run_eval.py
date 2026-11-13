import argparse, json, pathlib, sys, time
from transformers import AutoTokenizer
from localchart.rag import PatientIndex, build_messages
from localchart.llm import chat, MODEL
from eval.scoring import score, summarize

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["rag", "full"], default="rag")
p.add_argument("--retrieval", choices=["v1", "v2"], default="v1")
p.add_argument("--k", type=int, default=15)
p.add_argument("--offset", type=int, default=0)
p.add_argument("--limit", type=int, default=20)
p.add_argument("--max-ctx", type=int, default=30000)
p.add_argument("--tag")
a = p.parse_args()
tag = a.tag or ("full" if a.mode == "full" else f"rag_{a.retrieval}")
tok = AutoTokenizer.from_pretrained(MODEL)

qs = json.loads(pathlib.Path("eval/questions.json").read_text())
pids = sorted({q["pid"] for q in qs})[a.offset : a.offset + a.limit]
qs = [q for q in qs if q["pid"] in pids]
res = pathlib.Path("eval/results"); res.mkdir(parents=True, exist_ok=True)
out = open(res / f"{tag}.jsonl", "w")

idx, cur, consec_err = None, None, 0
for n, q in enumerate(qs, 1):
    if q["pid"] != cur:
        cur, idx = q["pid"], PatientIndex.from_file(f"data/chunks/{q['pid']}.json")
    if a.mode == "full": ctx = idx.chunks
    elif a.retrieval == "v2": ctx = idx.search_v2(q["question"], k=a.k)
    else: ctx = idx.search(q["question"], k=a.k)
    msgs = build_messages(q["question"], ctx)
    n_tok = len(tok("\n".join(m["content"] for m in msgs))["input_ids"])
    skipped, error = n_tok > a.max_ctx, False
    t0 = time.time()
    if skipped:
        ans, ptok = "SKIPPED: exceeds context window", n_tok
    else:
        try:
            ans, usage = chat(msgs); ptok = usage.prompt_tokens; consec_err = 0
        except Exception as e:
            ans, ptok, error = f"ERROR: {e}"[:300], n_tok, True
            consec_err += 1
    dt = time.time() - t0
    rec = {**q, "answer": ans, "latency": dt, "prompt_tokens": ptok, "ctx_chunks": len(ctx),
           "skipped": skipped, "error": error, **score(q, ans, {c["id"]: c for c in idx.chunks})}
    out.write(json.dumps(rec) + "\n"); out.flush()
    flag = " SKIPPED" if skipped else (" ERROR" if error else "")
    m = f"R={rec['recall']:.2f} P={rec['precision']:.2f}" if "recall" in rec else f"abstain_ok={rec['correct_abstain']}"
    print(f"[{n}/{len(qs)}] {q['qtype']:11} pid={q['pid']:6} {m}  cite={rec['supported']}/{rec['cited']}  {dt:.1f}s  tok={ptok}{flag}")
    if consec_err >= 3:
        sys.exit("ABORT: 3 errors in a row - is the MLX server down? Restart it and rerun.")
out.close()

rows = [json.loads(l) for l in open(res / f"{tag}.jsonl")]
s = summarize(rows, {"tag": tag, "mode": a.mode, "retrieval": a.retrieval, "k": a.k, "patients": len(pids)})
(res / f"{tag}_summary.json").write_text(json.dumps(s, indent=1))
print("\n" + json.dumps(s, indent=1))
