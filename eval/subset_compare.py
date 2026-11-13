import json, pathlib, sys
from eval.scoring import summarize
res = pathlib.Path("eval/results")
tags = sys.argv[1:]
data = {t: [json.loads(l) for l in open(res / f"{t}.jsonl")] for t in tags}
bad = {r["pid"] for rows in data.values() for r in rows
       if r.get("skipped") or r.get("error") or r["answer"].startswith(("ERROR", "SKIPPED"))}
pids = set.intersection(*[{r["pid"] for r in rows} for rows in data.values()]) - bad
pct = lambda v: "-" if v is None else f"{100*v:.1f}%"
print(f"Comparing on {len(pids)} shared patients (excluded {len(bad)}: {sorted(bad)})\n")
print("| Config | Precision | Recall | F1 | Citation acc. | Abstention acc. | Cold latency (s) | Prompt tokens |")
print("|---|---|---|---|---|---|---|---|")
for t, rows in data.items():
    s = summarize([r for r in rows if r["pid"] in pids], {"tag": t})
    print(f"| {t} | {pct(s['precision'])} | {pct(s['recall'])} | {pct(s['f1'])} | "
          f"{pct(s['citation_accuracy'])} | {pct(s['abstention_acc'])} | {s['cold_latency_s']:.1f} | {s['mean_prompt_tokens']:.0f} |")
