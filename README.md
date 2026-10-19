# RAG Assignment: Baseline vs. Re-Ranking

This repo compares two retrieval-augmented generation pipelines on the IIIT-H *Syllabus for Courses of Monsoon 2026* (295 pages, 123 courses).

- **Pipeline 1 (baseline):** a bi-encoder finds the top 10 chunks, the top 3 go to the LLM, giving Answer 1.
- **Pipeline 2 (re-ranking):** the same top 10 are re-scored by a cross-encoder, the new top 3 go to the LLM with the same prompt, giving Answer 2.

## Files

- `RAG_Assignment_Baseline_vs_Reranking__gemma_final.ipynb`: the full notebook with outputs.
- `report.pdf`: the written analysis (rank shifts, failure types, latency, scaling).

The syllabus PDF is not included in this repo. Download it from the course page and name it `syllabus.pdf` before running.

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

1. Loads the PDF and splits it by course into 100-word chunks with 20-word overlap (1128 chunks).
2. Embeds the chunks into a 384-dimensional vector index.
3. Runs both pipelines with an identical grounded prompt.
4. Tests 31 queries and records where the right chunk ranks before and after re-ranking.
5. Compares Answer 1 and Answer 2 on the failure queries.
6. Profiles the latency of both pipelines and estimates the cost of cross-encoding everything.

## Results

**Overall (31 queries):** 22 already fine (gold chunk in the baseline top 3), 2 fixed by re-ranking, 7 recall failures (gold chunk outside the top 10), 0 made worse.

| Failure type | Queries | Already fine | Fixed | Recall failure |
|---|---|---|---|---|
| Keyword trap | 8 | 6 | 0 | 2 |
| Exception | 7 | 6 | 1 | 0 |
| Negation | 11 | 8 | 0 | 3 |
| Buried fact | 5 | 2 | 1 | 2 |
| **Total** | **31** | **22** | **2** | **7** |

**Examples**
- *Keyword trap + buried chunk:* the UAV Design chunk moved from rank 4 to rank 1. Answer 1 refused, Answer 2 was correct.
- *Exception:* the Autonomous Software Engineering intake-limit chunk moved from rank 5 to rank 1, but both answers still refused. Rewording the question to match the document ("In Take Limit") gave the correct answer, 30.
- *Negation / recall failures:* queries such as "no mid-semester exam" (rank 475) are never retrieved, so re-ranking cannot help.

**Latency (T4 GPU, mean over 31 queries x 10 repeats)**

| Stage | Time (ms) |
|---|---|
| Pipeline 1: embed query + score all chunks + top 10 | 9.6 |
| Re-ranking step only: cross-encoder on 10 pairs | 31.9 |
| Pipeline 2: baseline retrieval + re-ranking | 43.2 |
| LLM answer generation (same for both) | 2714.4 |

Re-ranking makes retrieval about 4.5x slower, but it adds only about 34 ms next to roughly 2.7 s of LLM time.

**Scaling:** the cross-encoder takes about 2.82 ms per (query, chunk) pair. On all 1128 chunks that is about 3.18 s per query, roughly 331x slower than the bi-encoder. For 1,000,000 chunks it would be about 47 minutes for one query, so the cross-encoder is only used to re-rank a short list.
