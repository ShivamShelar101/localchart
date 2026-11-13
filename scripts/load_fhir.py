import glob, json, sys, time, requests

BASE = "http://localhost:8080/fhir"
files = sorted(glob.glob("data/output/fhir/*.json"))
first = [f for f in files if "hospitalInformation" in f or "practitionerInformation" in f]
rest = [f for f in files if f not in first]
print(f"{len(first)} org/practitioner bundles, {len(rest)} patient bundles")

ok = fail = 0
t0 = time.time()
for f in first + rest:
    with open(f) as fh:
        bundle = json.load(fh)
    r = requests.post(BASE, json=bundle,
                      headers={"Content-Type": "application/fhir+json"}, timeout=600)
    if r.ok:
        ok += 1
        print(f"ok   {f.split('/')[-1]}")
    else:
        fail += 1
        print(f"FAIL {f.split('/')[-1]} {r.status_code} {r.text[:300]}", file=sys.stderr)
print(f"done: {ok} ok, {fail} failed, {time.time()-t0:.0f}s")
