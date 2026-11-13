import argparse, json, pathlib, time
from localchart.rag import PatientIndex, build_messages
from localchart.llm import chat
from eval.scoring import score, summarize

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["rag", "full"], default="rag")
p.add_argument("--retrieval", choices=["v1", "v2"], default="v1")
p.add_argument("--k", type=int, default=15)
p.add_argument("--limit", type=int, default=20)
p.add_argument("--tag")
a = p.parse_args()
tag = a.tag or ("full" if a.mode == "full" else f"rag_{a.retrieval}")

qs = json.loads(pathlib.Path("eval/questions.json").read_text())
pids = sorted({q["pid"] for q in qs})[: a.limit]
qs = [q for q in qs if q["pid"] in pids]
res = pathlib.Path("eval/results"); res.mkdir(parents=True, exist_ok=True)
out = open(res / f"{tag}.jsonl", "w")

idx, cur = None, None
for n, q in enumerate(qs, 1):
    if q["pid"] != cur:
        cur, idx = q["pid"], PatientIndex.from_file(f"data/chunks/{q['pid']}.json")
    if a.mode == "full": ctx = idx.chunks
    elif a.retrieval == "v2": ctx = idx.search_v2(q["question"], k=a.k)
    else: ctx = idx.search(q["question"], k=a.k)
    t0 = time.time()
    ans, usage = chat(build_messages(q["question"], ctx))
    dt = time.time() - t0
    rec = {**q, "answer": ans, "latency": dt, "prompt_tokens": usage.prompt_tokens,
           "ctx_chunks": len(ctx), **score(q, ans, {c["id"]: c for c in idx.chunks})}
    out.write(json.dumps(rec) + "\n"); out.flush()
    m = f"R={rec['recall']:.2f} P={rec['precision']:.2f}" if "recall" in rec else f"abstain_ok={rec['correct_abstain']}"
    print(f"[{n}/{len(qs)}] {q['qtype']:11} pid={q['pid']:6} {m}  cite={rec['supported']}/{rec['cited']}  {dt:.1f}s")
out.close()

rows = [json.loads(l) for l in open(res / f"{tag}.jsonl")]
s = summarize(rows, {"tag": tag, "mode": a.mode, "retrieval": a.retrieval, "k": a.k, "patients": len(pids)})
(res / f"{tag}_summary.json").write_text(json.dumps(s, indent=1))
print("\n" + json.dumps(s, indent=1))
