from collections import Counter, defaultdict

KEEP = {"Patient", "Condition", "MedicationRequest", "AllergyIntolerance",
        "Observation", "Encounter", "Immunization", "Procedure"}

def _cc(c):
    if not c: return "unknown"
    if c.get("text"): return c["text"]
    for cd in c.get("coding", []):
        if cd.get("display"): return cd["display"]
    return "unknown"

def _status(c):
    return ((c or {}).get("coding") or [{}])[0].get("code", "unknown")

def _d(s):
    return (s or "?")[:10]

def _qty(q):
    v = q.get("value")
    v = round(v, 2) if isinstance(v, (int, float)) else v
    return f"{v} {q.get('unit', '')}".strip()

def _obs_value(r):
    if "valueQuantity" in r: return _qty(r["valueQuantity"])
    if "valueCodeableConcept" in r: return _cc(r["valueCodeableConcept"])
    if "valueString" in r: return r["valueString"]
    parts = [f"{_cc(c.get('code'))} {_qty(c['valueQuantity'])}"
             for c in r.get("component", []) if "valueQuantity" in c]
    return "; ".join(parts) or None


import re as _re
def _clean_med(n):
    m = _re.search(r"\[([^\]]+)\]\s*$", n)
    if n.startswith("{") and m:
        ings = sorted({i for i in _re.findall(r"([a-z][a-z ]+?) [\d.]+ MG", n) if "inert" not in i})
        return f"{m.group(1)} pack ({' / '.join(ings)})" if ings else f"{m.group(1)} pack"
    return n

def to_chunk(r):
    t = r["resourceType"]
    if t not in KEEP: return None
    rid = f"{t}/{r['id']}"
    date, key = None, None
    if t == "Patient":
        name = (r.get("name") or [{}])[0]
        nm = " ".join((name.get("given") or []) + [name.get("family", "")])
        dec = f", deceased {_d(r.get('deceasedDateTime'))}" if r.get("deceasedDateTime") else ""
        txt = f"Patient: {nm}, {r.get('gender')}, born {r.get('birthDate')}{dec}"
    elif t == "Condition":
        date = _d(r.get("onsetDateTime"))
        end = f", resolved {_d(r.get('abatementDateTime'))}" if r.get("abatementDateTime") else ""
        txt = f"Condition: {_cc(r.get('code'))} (status: {_status(r.get('clinicalStatus'))}, onset {date}{end})"
    elif t == "MedicationRequest":
        date = _d(r.get("authoredOn"))
        txt = f"Medication: {_clean_med(_cc(r.get('medicationCodeableConcept')))} (status: {r.get('status')}, prescribed {date})"
    elif t == "AllergyIntolerance":
        txt = f"Allergy: {_cc(r.get('code'))} (status: {_status(r.get('clinicalStatus'))})"
    elif t == "Observation":
        val = _obs_value(r)
        if val is None: return None
        date, key = _d(r.get("effectiveDateTime")), _cc(r.get("code"))
        txt = f"Observation: {key} = {val} on {date}"
    elif t == "Encounter":
        date = _d(r.get("period", {}).get("start"))
        txt = f"Encounter: {_cc((r.get('type') or [None])[0])} on {date}"
    elif t == "Immunization":
        date = _d(r.get("occurrenceDateTime"))
        txt = f"Immunization: {_cc(r.get('vaccineCode'))} on {date}"
    elif t == "Procedure":
        date = _d(r.get("performedPeriod", {}).get("start") or r.get("performedDateTime"))
        txt = f"Procedure: {_cc(r.get('code'))} on {date}"
    return {"id": rid, "type": t, "date": date, "key": key, "text": txt}

def flatten_record(resources, obs_per_code=3, max_encounters=25):
    raw = Counter(r["resourceType"] for r in resources)
    chunks = [c for c in (to_chunk(r) for r in resources) if c]
    obs = defaultdict(list)
    other, enc = [], []
    for c in chunks:
        if c["type"] == "Observation": obs[c["key"]].append(c)
        elif c["type"] == "Encounter": enc.append(c)
        else: other.append(c)
    kept_obs = [c for group in obs.values()
                for c in sorted(group, key=lambda x: x["date"], reverse=True)[:obs_per_code]]
    kept_enc = sorted(enc, key=lambda x: x["date"], reverse=True)[:max_encounters]
    final = other + kept_obs + kept_enc
    stats = {"raw_resources": len(resources), "raw_by_type": dict(raw),
             "chunks_before_dedup": len(chunks), "chunks_final": len(final)}
    return final, stats
