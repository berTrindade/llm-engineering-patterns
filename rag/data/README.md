# Data

`snapshot.json` is a cleaned, chunked copy of six GOV.UK private renting guidance pages, fetched
through the GOV.UK Content API (`https://www.gov.uk/api/content/<path>`). It is committed so the
eval is reproducible offline. Rebuild it with `make data`.

Contains public sector information licensed under the Open Government Licence v3.0:
https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/

## Sources (fetched 2026-09-15)

| Page | Source URL |
|---|---|
| Private renting | https://www.gov.uk/private-renting |
| Tenancy deposit protection | https://www.gov.uk/tenancy-deposit-protection |
| Deposit protection schemes and landlords | https://www.gov.uk/deposit-protection-schemes-and-landlords |
| Renting out your property | https://www.gov.uk/renting-out-a-property |
| Evicting tenants in England | https://www.gov.uk/evicting-tenants |
| Checking your tenant's right to rent | https://www.gov.uk/check-tenant-right-to-rent-documents |

Each of these pages is a multi-part GOV.UK guide. Every chunk in `snapshot.json` records its own
`source_url` pointing at the specific part (for example
`https://www.gov.uk/private-renting/repairs`), plus `page_title`, `heading_path` (page > part >
heading) and `fetch_date`.

Guidance may have changed on GOV.UK since the fetch date above. This snapshot is a point-in-time
copy for reproducibility, not a live mirror.

`government/publications/how-to-rent` was checked and excluded: it is a withdrawn guide whose body
is a short stub pointing at a PDF, with no chunkable guidance text of its own.

## Eval questions

`evals/questions.jsonl` was authored for this repository from this snapshot: 30 answerable
questions drawn from real headings in the six pages above, and 10 unanswerable-but-plausible
questions (for example Scotland/Wales/Northern Ireland-specific rules, or a dispute with a named
letting agent) that this England-focused snapshot does not cover.
