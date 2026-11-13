import json, pathlib
from collections import Counter
from eval.common import item_name

ABSENT = ["metformin", "warfarin", "insulin", "levothyroxine", "sertraline", "atorvastatin", "lisinopril"]
qs = []
for f in sorted(pathlib.Path("data/chunks").glob("*.json")):
    pid, ch = f.stem, json.loads(f.read_text())
    def truth(t, pred=lambda c: True):
        return [{"id": c["id"], "name": item_name(c["text"])} for c in ch
                if c["type"] == t and "status: active" in c["text"] and pred(c)]
    qs.append({"pid": pid, "qtype": "conditions",
               "question": "What active medical conditions (disorders) does the patient have?",
               "truth": truth("Condition", lambda c: "(disorder)" in c["text"])})
    qs.append({"pid": pid, "qtype": "medications",
               "question": "What medications is the patient currently taking?",
               "truth": truth("MedicationRequest")})
    qs.append({"pid": pid, "qtype": "allergies",
               "question": "What allergies does the patient have?",
               "truth": truth("AllergyIntolerance")})
    meds = " ".join(c["text"].lower() for c in ch if c["type"] == "MedicationRequest")
    drug = next((d for d in ABSENT if d not in meds), None)
    if drug:
        qs.append({"pid": pid, "qtype": "absent_med",
                   "question": f"Is the patient taking {drug}?", "truth": []})

pathlib.Path("eval/questions.json").write_text(json.dumps(qs, indent=1))
print(f"{len(qs)} questions over {len({q['pid'] for q in qs})} patients")
print("by type:", dict(Counter(q["qtype"] for q in qs)))
print("empty-truth (should abstain):", sum(1 for q in qs if not q["truth"]))
