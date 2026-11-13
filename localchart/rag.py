import json, pathlib, re
import numpy as np
from sentence_transformers import SentenceTransformer

_emb = None
def emb():
    global _emb
    if _emb is None:
        _emb = SentenceTransformer("BAAI/bge-small-en-v1.5", device="mps")
    return _emb

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

ROUTES = {
    "Condition": ["condition", "diagnos", "disease", "problem", "disorder", "history"],
    "MedicationRequest": ["medication", "medicine", "drug", "prescri", "taking", " on "],
    "AllergyIntolerance": ["allerg"],
    "Immunization": ["vaccin", "immuniz", "shot"],
    "Procedure": ["procedure", "surgery", "operation"],
}

class PatientIndex:
    def __init__(self, chunks):
        self.chunks = chunks
        self.ids = {c["id"] for c in chunks}
        self.vecs = emb().encode([c["text"] for c in chunks],
                                 normalize_embeddings=True, batch_size=64)

    @classmethod
    def from_file(cls, path):
        return cls(json.loads(pathlib.Path(path).read_text()))

    def search(self, q, k=15):
        ql = f" {q.lower()} "
        routed = {t for t, kws in ROUTES.items() if any(kw in ql for kw in kws)}
        forced = [c for c in self.chunks if c["type"] in routed or c["type"] == "Patient"]
        qv = emb().encode([QUERY_PREFIX + q], normalize_embeddings=True)[0]
        order = np.argsort(-self.vecs @ qv)
        forced_ids = {c["id"] for c in forced}
        extra = [self.chunks[i] for i in order if self.chunks[i]["id"] not in forced_ids][:k]
        return forced + extra

SYSTEM = """You answer questions about ONE patient's medical record.
Rules:
- Use ONLY the provided records.
- End every bullet with the record ID in square brackets, copied exactly, e.g. [Condition/abc-123].
- A condition is ACTIVE only if its record says "status: active". A medication is CURRENT only if it says "status: active". Do not list resolved or stopped items as active or current.
- If the answer is not in the records, reply exactly: Not found in the record.

Example:
Records:
[Condition/c1] Condition: Asthma (status: active, onset 2010-01-01)
[Condition/c2] Condition: Sprain (status: resolved, onset 2019-05-01, resolved 2019-06-01)
[MedicationRequest/m1] Medication: Albuterol inhaler (status: active, prescribed 2024-01-01)
Question: What active conditions and current medications does the patient have?
Answer:
Active conditions:
- Asthma, since 2010-01-01 [Condition/c1]
Current medications:
- Albuterol inhaler, prescribed 2024-01-01 [MedicationRequest/m1]"""

def build_messages(question, ctx):
    context = "\n".join(f"[{c['id']}] {c['text']}" for c in ctx)
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Records:\n{context}\n\nQuestion: {question}\nAnswer:"}]

CITE_RE = re.compile(r"\[([A-Za-z]+/[A-Za-z0-9\-\.]+)\]")

def check_citations(answer, ctx_ids, all_ids):
    cited = CITE_RE.findall(answer)
    bullets = [l for l in answer.splitlines() if l.strip().startswith(("-", "*", "•"))]
    return {
        "n_cited": len(cited),
        "invalid_not_in_record": [c for c in cited if c not in all_ids],
        "not_in_context": [c for c in cited if c in all_ids and c not in ctx_ids],
        "uncited_bullets": [b for b in bullets if not CITE_RE.search(b)],
        "abstained": "not found in the record" in answer.lower(),
    }

from difflib import SequenceMatcher

def _claim(bullet):
    t = CITE_RE.sub("", bullet).strip(" -*•")
    return t.split(",")[0].strip().lower()

def check_support(answer, chunks_by_id, threshold=0.6):
    """For each cited bullet, does the cited record actually mention the claimed item?"""
    bad = []
    for line in answer.splitlines():
        ids = CITE_RE.findall(line)
        if not ids: continue
        claim = _claim(line)
        for cid in ids:
            txt = chunks_by_id.get(cid, {}).get("text", "").lower()
            name = txt.split(":", 1)[-1].split("(")[0].strip()
            score = SequenceMatcher(None, claim, name).ratio()
            if claim not in txt and score < threshold:
                bad.append({"bullet": line.strip(), "cited": cid, "cited_text": txt[:90]})
    return bad

ACTIVE_WORDS = ("active", "current", "currently", "taking", " now")

def _inactive(c):
    return "status:" in c["text"] and "status: active" not in c["text"]

def _search_v2(self, q, k=15):
    ctx = self.search(q, k=k)
    if any(w in f" {q.lower()} " for w in ACTIVE_WORDS):
        ctx = [c for c in ctx if not _inactive(c)]
    return ctx

PatientIndex.search_v2 = _search_v2
