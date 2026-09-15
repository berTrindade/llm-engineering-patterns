# Data provenance

All data in this directory is invented for this proof of concept.

- `policy.md` is a Fernway Travel customer support policy. Fernway Travel is not a real
  company. Every fare name, fee, allowance and timeline in this document was written for
  this repository and does not describe any real airline's policy.
- `questions.jsonl` is generated from `generate_questions.py`, a seeded (`random.Random(42)`)
  pure-Python script that builds 60 questions from templates over the facts in `policy.md`.
  Each question carries `required_facts` (the short strings a correct answer must contain;
  an entry written as `"a|b|c"` is satisfied by any one of those alternative wordings) and a
  `difficulty` tag (`simple_lookup`, `multi_step_calculation`, `policy_exception`) assigned by
  which template produced it.

Regenerate the dataset with:

```bash
uv run python data/generate_questions.py
```
