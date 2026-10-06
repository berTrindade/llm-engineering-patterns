# rag-poc

Retrieval-augmented answering over GOV.UK private renting guidance, with citations and a measured
refusal rate.

## Problem

A renter (or landlord) asking a question about private renting in England needs an answer that is
grounded in real guidance, cites where it came from, and says so plainly when the guidance does
not cover the question, rather than guessing. This repo measures how well a small local RAG
pipeline does that over six GOV.UK renting guidance pages.

## Run it

```
make setup   # uv sync
make data    # fetch GOV.UK pages, chunk them, write data/snapshot.json (already committed)
make index   # embed the chunks with nomic-embed-text, write index/embeddings.json
make run     # answer one example question end to end
make eval    # score the full eval set, write results/<date>.json, print a table
make test    # pytest, no network or model
make lint    # ruff check
```

Needs a local Ollama with `qwen3:8b` and `nomic-embed-text` pulled, or any OpenAI-compatible
endpoint configured through `.env` (copy `.env.example`).

## Result

Full run on 2026-09-16: 40 questions (30 answerable, 10 unanswerable), `qwen3:8b` for
answers and `nomic-embed-text` for retrieval, through local Ollama, with no failed calls.
Raw output is in [results/2026-09-16.json](results/2026-09-16.json).

| Metric | Value | Over |
|---|---|---|
| Hit rate at 5 | 29/30 (96.7%) | answerable questions |
| Citation correctness | 77.8% | mean share of cited chunks from an expected page, answerable questions |
| Correct refusal rate | 9/10 | unanswerable questions |
| False refusal rate | 0/30 | answerable questions |
| Mean latency | 3.8 s | all 40 |
| Mean tokens | 973 (872 prompt, 101 completion) | all 40 |

- **Retrieval was rarely the problem.** An expected page was in the top 5 for 29 of 30
  answerable questions. The miss was q24 (tenant damage to a property).
- **Citations were the weakest step.** 21 of 30 answers cited only chunks from an expected
  page. Nine cited at least one chunk from elsewhere, and three of those (q04 on eviction
  notice, q13 on rental income tax, q18 on student rights) cited none from an expected page.
  The results file does not keep which chunks were cited, so whether those were wrong or
  just from a different page that also covers the topic is not recorded.
- **The model refused when it should, with one exception.** It declined 9 of the 10
  unanswerable questions and none of the answerable ones. The miss was q34, which asks about
  options against a specific named letting agent over a holding deposit. The guidance covers
  holding deposits in general but not that agent, and the model answered instead of
  refusing.
- **Every refusal came from the model, not the similarity threshold.** No question scored
  below 0.3, so the pre-model refusal path never fired on this set.

What this does not measure: an answerable question counts as correct when it is not
refused. Nothing checks that the answer text is right; that would need a reference answer
per question or a judge (see `evals-and-judge-poc`).

## How it works

- **Data**: `make data` fetches six GOV.UK pages through the Content API
  (`https://www.gov.uk/api/content/<path>`), each a multi-part guide. See
  [data/README.md](data/README.md) for the source list, fetch date and licence attribution.
- **Chunking**: each part's HTML is split on `h2`/`h3` headings (stdlib `html.parser`), with a
  1000-character cap and 150-character overlap for oversized sections. Every chunk keeps its
  `source_url`, `page_title` and `heading_path` as its citation id.
- **Embeddings**: `nomic-embed-text` via `/v1/embeddings`. It expects task prefixes, so chunks are
  embedded as `"search_document: " + text` and queries as `"search_query: " + question`. Vectors
  are stored as a flat JSON file (`index/embeddings.json`, gitignored, rebuilt by `make index`);
  cosine similarity is plain Python, fine at ~200 chunks.
- **Answering**: the top 5 chunks by cosine similarity go to `CHEAP_MODEL` with a `json_schema`
  response format (`{answer, citations, refused}`), instructed to answer only from the given
  chunks and cite their ids. If the best similarity is below 0.3, the pipeline refuses without
  calling the model at all.
- **Eval**: [evals/questions.jsonl](evals/questions.jsonl) has 40 hand-written questions (30
  answerable, 10 plausible-but-unanswerable, for example Scotland-specific rules or a named
  letting agent dispute), each with `expected_sources`. `evals/run.py` scores hit rate at 5,
  citation correctness, correct/false refusal rate, mean latency and mean tokens, and keeps
  failing case ids with reasons in the results JSON.

## Limits

- Six GOV.UK pages only, England-focused. Anything about Scotland, Wales, Northern Ireland,
  commercial lets, or a named third party is out of scope by design (part of the unanswerable
  eval set).
- The snapshot is a point-in-time copy; GOV.UK guidance may have changed since the fetch date.
- Pure-Python cosine similarity over a JSON file, not a vector database. Fine at ~200 chunks, not
  meant to scale further.
- The 0.3 similarity refusal threshold and the 1000/150 character chunk size/overlap are not
  tuned against the eval set; they are reasonable starting values.
