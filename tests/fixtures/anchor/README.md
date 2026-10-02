# Vendored-RAGAS prompt renderings (PC1, D-025)

Each `ragas_*.txt` file is a prompt exactly as RAGAS itself rendered it. The renderings came from running the RAGAS code vendored in GraphRAG-Bench/GraphRAG-Benchmark at commit e6305f5 (2025-06-09) under this repo's pydantic. That covers `ragas/prompt/pydantic_prompt.py` (`PydanticPrompt.to_string` and `FixOutputFormat`) and the prompt classes from `ragas/metrics/_faithfulness.py` and `_answer_correctness.py`, executed from their source unchanged. The imports that only matter for calling a model (LangChain, RAGAS callbacks) were stubbed.

`tests/test_anchor_ragas.py` checks that `ape.anchor.ragas.to_string` reproduces each file byte for byte, on these inputs:

| File | Prompt | Input |
|---|---|---|
| `ragas_statement_plain.txt` | statement generator | a plain question and answer |
| `ragas_statement_escapes.txt` | statement generator | quotes, newlines, a backslash, non-ASCII text |
| `ragas_classifier.txt` | correctness classifier | two answer statements, one ground-truth statement |
| `ragas_classifier_empty.txt` | correctness classifier | no answer statements |
| `ragas_fix_output_format.txt` | the fix-format re-ask | a broken output string and its prompt |

Do not edit these files by hand. If pydantic changes how it renders JSON Schema or JSON, regenerate them the same way and record the change.
