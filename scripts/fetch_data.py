"""`make data`: fetch GOV.UK renting guidance pages and write data/snapshot.json."""

import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_poc.chunk import chunk_page
from rag_poc.fetch import PAGES, fetch_page

SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "data" / "snapshot.json"


def main() -> None:
    fetch_date = datetime.date.today().isoformat()
    all_chunks = []

    for path in PAGES:
        page_json = fetch_page(path)
        chunks = chunk_page(page_json)
        for chunk in chunks:
            record = chunk.model_dump()
            record["fetch_date"] = fetch_date
            all_chunks.append(record)
        print(f"{path}: {len(chunks)} chunks")

    SNAPSHOT_PATH.write_text(json.dumps(all_chunks, indent=2))
    print(f"wrote {len(all_chunks)} chunks to {SNAPSHOT_PATH}")


if __name__ == "__main__":
    main()
