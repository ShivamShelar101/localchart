import re, statistics as st
from eval.common import item_name, sim

CITE_RE = re.compile(r"\[([A-Za-z]+/[A-Za-z0-9\-\.]+)\]")

def claim_of(line):
    return CITE_RE.sub("", line).strip(" -*•").split(",")[0].strip()

def items_of(ans):
    return [l.strip() for l in ans.splitlines()
            if l.strip().startswith(("-", "*", "•")) or CITE_RE.search(l)]

def score(q, ans, by_id):
    items = items_of(ans)
    abstained = "not found in the record" in ans.lower()
    names = [t["name"] for t in q["truth"]]
    matched, tp, cited, supported = set(), 0, 0, 0
    for b in items:
        c = claim_of(b)
        if names:
            i = max(range(len(names)), key=lambda j: sim(c, names[j]))
            if i not in matched and sim(c, names[i]) >= 0.6:
                matched.add(i); tp += 1
        ids = CITE_RE.findall(b)
        if ids:
            cited += 1
            if any(cid in by_id and "status: active" in by_id[cid]["text"]
                   and sim(c, item_name(by_id[cid]["text"])) >= 0.6 for cid in ids):
                supported += 1
    r = {"n_bullets": len(items), "n_truth": len(names), "tp": tp,
         "cited": cited, "supported": supported, "abstained": abstained}
    if names:
        r["precision"] = tp / len(items) if items else 0.0
        r["recall"] = tp / len(names)
    else:
        r["correct_abstain"] = abstained or not items
    return r

def summarize(rows, meta):
    bad = lambda r: r.get('skipped') or r.get('error') or r['answer'].startswith(('ERROR', 'SKIPPED'))
    meta = {**meta, 'excluded_questions': sum(1 for r in rows if bad(r))}
    rows = [r for r in rows if not bad(r)]
    pos = [r for r in rows if r["n_truth"] > 0]
    neg = [r for r in rows if r["n_truth"] == 0]
    P = st.mean(r["precision"] for r in pos) if pos else 0
    R = st.mean(r["recall"] for r in pos) if pos else 0
    cited = sum(r["cited"] for r in rows); bul = sum(r["n_bullets"] for r in rows)
    seen, cold = set(), []
    for r in rows:
        if r["pid"] not in seen:
            seen.add(r["pid"]); cold.append(r["latency"])
    return {**meta, "questions": len(rows),
        "precision": P, "recall": R, "f1": 2 * P * R / (P + R) if P + R else 0,
        "citation_accuracy": sum(r["supported"] for r in rows) / cited if cited else None,
        "uncited_bullet_rate": (bul - cited) / bul if bul else None,
        "abstention_acc": st.mean(r["correct_abstain"] for r in neg) if neg else None,
        "false_abstain_rate": st.mean(r["abstained"] for r in pos) if pos else None,
        "cold_latency_s": st.mean(cold), "mean_latency_s": st.mean(r["latency"] for r in rows),
        "mean_prompt_tokens": st.mean(r["prompt_tokens"] for r in rows),
        "recall_by_type": {t: st.mean(r["recall"] for r in pos if r["qtype"] == t)
                           for t in sorted({r["qtype"] for r in pos})}}
