# Data provenance

Everything under `data/` is invented. There are no real customers, no real support
tickets, and no data taken from any other project.

- `threads/*.json`: 20 fictional Ledgerline support ticket threads, produced entirely by
  [`src/context_window_poc/generate_threads.py`](../src/context_window_poc/generate_threads.py)
  with a fixed seed. No LLM was used to generate them. Each thread plants four facts
  (account id, plan name, error code, time zone) in its first two exchanges, fills the
  middle with unrelated invented support conversation, and ends with a customer question
  that requires recalling those four facts. Regenerate with
  `uv run python -m context_window_poc.generate_threads`.
- `calibration.json`: the fitted chars-per-token ratio from `make calibrate`, produced by
  sending a handful of invented prompts to the local model and recording real usage. Not
  a dataset, just an estimator parameter.

"Ledgerline" is an invented product name used only for this proof of concept.
