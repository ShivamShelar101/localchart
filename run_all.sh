set -e
uv run python -m scripts.build_chunks
grep -h "pack (" data/chunks/*.json | head -3
uv run python -m eval.build_questions
uv run python -m eval.run_eval --mode full --tag full --limit 20
uv run python -m eval.run_eval --mode rag --retrieval v1 --tag rag_v1 --limit 20
uv run python -m eval.run_eval --mode rag --retrieval v2 --tag rag_v2 --limit 20
uv run python -m eval.compare
