import json, pathlib
res = pathlib.Path("eval/results")
cols = ["Config", "Precision", "Recall", "F1", "Citation acc.", "Abstention acc.",
        "Cold latency (s)", "Mean latency (s)", "Prompt tokens"]
pct = lambda v: "-" if v is None else f"{100 * v:.1f}%"
lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
for sp in sorted(res.glob("*_summary.json")):
    s = json.loads(sp.read_text())
    lines.append("| " + " | ".join([s["tag"], pct(s["precision"]), pct(s["recall"]), pct(s["f1"]),
        pct(s["citation_accuracy"]), pct(s["abstention_acc"]), f"{s['cold_latency_s']:.1f}",
        f"{s['mean_latency_s']:.1f}", f"{s['mean_prompt_tokens']:.0f}"]) + " |")
table = "\n".join(lines)
(res / "table.md").write_text(table + "\n")
print(table)
