import argparse, pathlib, time
from localchart.rag import PatientIndex, build_messages, check_citations
from localchart.llm import chat

p = argparse.ArgumentParser()
p.add_argument("question", nargs="?",
               default="What active conditions and current medications does this patient have?")
p.add_argument("--patient", type=int, default=0)
p.add_argument("--mode", choices=["rag", "full"], default="rag")
p.add_argument("--k", type=int, default=15)
a = p.parse_args()

f = sorted(pathlib.Path("data/chunks").glob("*.json"))[a.patient]
idx = PatientIndex.from_file(f)
ctx = idx.search(a.question, k=a.k) if a.mode == "rag" else idx.chunks

t0 = time.time()
ans, usage = chat(build_messages(a.question, ctx))
dt = time.time() - t0
rep = check_citations(ans, {c["id"] for c in ctx}, idx.ids)

print(f"[{a.mode}] patient={f.stem}  context chunks={len(ctx)}/{len(idx.chunks)}")
print(f"\nQ: {a.question}\n\n{ans}\n")
print(f"prompt tokens={usage.prompt_tokens}  output={usage.completion_tokens}  time={dt:.1f}s")
print(f"citations={rep['n_cited']}  invalid={len(rep['invalid_not_in_record'])}  "
      f"uncited bullets={len(rep['uncited_bullets'])}  abstained={rep['abstained']}")
for b in rep["uncited_bullets"]: print("  UNCITED:", b.strip())
for c in rep["invalid_not_in_record"]: print("  INVALID:", c)
from localchart.rag import check_support
bad = check_support(ans, {c["id"]: c for c in idx.chunks})
print(f"unsupported citations={len(bad)}")
for b in bad: print(f"  WRONG CITE: {b['bullet']}\n     -> {b['cited']} says: {b['cited_text']}")
