# LLM engineering patterns

Seven small, runnable proofs of concept for the problems that decide whether an LLM feature works in production: evaluation, retrieval, tool calling, human approval, cost, structured output and context limits.

Each one asks a single question, answers it against a fixed dataset, and publishes the measured result, failures included. They run against any OpenAI-compatible endpoint and default to a local model through [Ollama](https://ollama.com), so no paid API key is needed to reproduce them.

| Pattern | Question it answers | Headline result |
|---|---|---|
| [evals-and-judge](evals-and-judge/) | Can an LLM judge be trusted to grade a support bot, and how would you know? | Judge agreed with designed labels on 54 of 60 replies (90%) |
| [rag](rag/) | Can answers over GOV.UK renting guidance cite their source and refuse when the guidance is silent? | Retrieval hit rate at 5 of 29/30, citation correctness 77.8%, correct refusals 9/10 |
| [tool-calling](tool-calling/) | Can a hand-written agent loop, with no framework, schedule meetings through tools and recover from its own mistakes? | Success rates per category, e.g. 9 of 10 simple requests |
| [human-in-the-loop](human-in-the-loop/) | How do you make an agent's proposal always wait for a person, even across a crash? | Durable pause in SQLite, append-only audit log, accuracy matrix |
| [cost-and-routing](cost-and-routing/) | When is a cheap model good enough, and what does routing to a strong one cost? | Accuracy, cost per correct answer and latency for three strategies |
| [structured-outputs](structured-outputs/) | Prompting, JSON mode or a schema: which turns messy invoices into valid records? | 49 of 50 valid first try, 50 of 50 after one repair pass |
| [context-window](context-window/) | When a support thread outgrows the context window, which trimming strategy keeps the facts? | Fact recall against token cost per strategy |

## Running one

Each folder is a self-contained Python project managed with [uv](https://docs.astral.sh/uv/):

```bash
cd evals-and-judge
make setup   # uv sync
make eval
```

Every folder's README covers the problem, the design decisions, how to run it, and the full results.

## History

These started as separate repositories and were merged with `git subtree`, so each folder keeps its own commit history.

## Licence

MIT. See [LICENSE](LICENSE).
