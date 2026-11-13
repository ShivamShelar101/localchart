import json, pathlib, sys, time
from localchart.llm import chat

files = sorted(pathlib.Path("data/chunks").glob("*.json"))
f = files[int(sys.argv[1]) if len(sys.argv) > 1 else 0]
chunks = json.loads(f.read_text())
question = sys.argv[2] if len(sys.argv) > 2 else \
    "What active conditions and current medications does this patient have?"

SYSTEM = """You answer questions about ONE patient's medical record.
Use ONLY the provided records. After every claim, cite the record ID in brackets,
e.g. [Condition/abc123]. If the answer is not in the records, say exactly:
"Not found in the record." Never guess."""

context = "\n".join(f"[{c['id']}] {c['text']}" for c in chunks)
t0 = time.time()
ans, usage = chat([{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": f"Records:\n{context}\n\nQuestion: {question}"}])
dt = time.time() - t0
print(f"Patient file: {f.name}  |  chunks in context: {len(chunks)}")
print(f"\nQ: {question}\n\nA: {ans}\n")
if usage:
    print(f"prompt tokens={usage.prompt_tokens}  output tokens={usage.completion_tokens}  "
          f"total time={dt:.1f}s  ~{usage.completion_tokens/dt:.1f} output tok/s (incl. prompt processing)")
