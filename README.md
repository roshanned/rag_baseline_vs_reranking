# RAG Assignment: Baseline vs. Re-Ranking

This repo compares two retrieval-augmented generation pipelines on the IIIT-H *Syllabus for Courses of Monsoon 2026* (295 pages, 123 courses).

- **Pipeline 1 (baseline):** bi-encoder finds the top 10 chunks, the top 3 go to the LLM, giving Answer 1.
- **Pipeline 2 (re-ranking):** the same top 10 are re-scored by a cross-encoder, the new top 3 go to the LLM with the same prompt, giving Answer 2.
## Files

- `RAG_Assignment_Baseline_vs_Reranking__gemma_final.ipynb`: the full notebook with outputs.

## Models

| Role | Model |
|---|---|
| Bi-encoder | `sentence-transformers/all-MiniLM-L6-v2` |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | `google/gemma-3-4b-it` (falls back to `Qwen/Qwen2.5-3B-Instruct`) |

## How to run

1. Open the notebook in Google Colab and set the runtime to **T4 GPU**.
2. Gemma is a gated model. Accept its license on Hugging Face, then add your token as a Colab secret named `HF_TOKEN`.
3. Upload the syllabus PDF to Colab and name it `syllabus.pdf`.
4. Run all cells from top to bottom.

## What the notebook does

1. Loads the PDF and splits it by course into 100-word chunks with 20-word overlap.
2. Embeds the chunks into a 384-dimensional vector index.
3. Runs both pipelines with an identical grounded prompt.
4. Tests 31 queries and records where the right chunk ranks before and after re-ranking.
5. Compares Answer 1 and Answer 2 on the failure queries.
6. Profiles the latency of both pipelines and estimates the cost of cross-encoding everything.

## Results

- 31 queries: 22 already fine, 2 fixed by re-ranking, 7 recall failures, 0 made worse.
- Retrieval time on a T4 GPU: 9.6 ms for Pipeline 1 and 43.2 ms for Pipeline 2.
- LLM answer generation takes about 2.7 s, the same for both pipelines.
