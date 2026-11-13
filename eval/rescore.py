import json, pathlib
from eval.scoring import score, summarize
res = pathlib.Path("eval/results")
for jl in sorted(res.glob("*.jsonl")):
    rows = [json.loads(l) for l in open(jl)]
    for r in rows:
        by_id = {c["id"]: c for c in json.loads(pathlib.Path(f"data/chunks/{r['pid']}.json").read_text())}
        for k in ("precision", "recall", "correct_abstain"): r.pop(k, None)
        r.update(score(r, r["answer"], by_id))
    jl.write_text("".join(json.dumps(r) + "\n" for r in rows))
    old = json.loads((res / f"{jl.stem}_summary.json").read_text())
    meta = {"tag": old["tag"], "mode": old["mode"], "retrieval": old.get("retrieval", "v1"),
            "k": old["k"], "patients": old["patients"]}
    s = summarize(rows, meta)
    (res / f"{jl.stem}_summary.json").write_text(json.dumps(s, indent=1))
    print(f"{jl.stem}: P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f}")
