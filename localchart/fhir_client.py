import json, pathlib, requests

BASE = "http://localhost:8080/fhir"
CACHE = pathlib.Path("data/cache")

def list_patients(n=200):
    r = requests.get(f"{BASE}/Patient", timeout=60,
                     params={"_count": n, "_elements": "id,name,birthDate,gender,deceased"})
    r.raise_for_status()
    return [e["resource"] for e in r.json().get("entry", [])]

def display_name(p):
    n = (p.get("name") or [{}])[0]
    return " ".join((n.get("given") or []) + [n.get("family", "")]).strip()

def patient_everything(pid, use_cache=True):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{pid}.json"
    if use_cache and f.exists():
        return json.loads(f.read_text())
    url, params, out = f"{BASE}/Patient/{pid}/$everything", {"_count": 500}, []
    while url:
        r = requests.get(url, params=params, timeout=300)
        r.raise_for_status()
        b = r.json()
        out += [e["resource"] for e in b.get("entry", [])]
        url = next((l["url"] for l in b.get("link", []) if l["relation"] == "next"), None)
        params = None
    f.write_text(json.dumps(out))
    return out
