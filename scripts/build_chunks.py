import json, pathlib, time
from localchart.fhir_client import list_patients, patient_everything, display_name
from localchart.flatten import flatten_record

OUT = pathlib.Path("data/chunks"); OUT.mkdir(parents=True, exist_ok=True)
pts = list_patients()
print(f"{len(pts)} patients")
tot_raw = tot_final = 0
t0 = time.time()
for p in pts:
    res = patient_everything(p["id"])
    chunks, s = flatten_record(res)
    (OUT / f"{p['id']}.json").write_text(json.dumps(chunks, indent=1))
    tot_raw += s["raw_resources"]; tot_final += s["chunks_final"]
    print(f"{display_name(p)[:28]:28} raw={s['raw_resources']:6}  chunks={s['chunks_final']:4}")
print(f"\nTOTAL raw resources={tot_raw}  final chunks={tot_final}  "
      f"reduction={100*(1-tot_final/tot_raw):.1f}%  time={time.time()-t0:.0f}s")

first = json.loads((OUT / f"{pts[0]['id']}.json").read_text())
print("\nSample chunks:")
for c in first[:15]: print(f"  [{c['id']}] {c['text']}")
