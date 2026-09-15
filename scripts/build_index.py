"""`make index`: embed data/snapshot.json chunks and write index/embeddings.json."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_poc.chunk import Chunk
from rag_poc.index import build_index, save_index

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_PATH = ROOT / "data" / "snapshot.json"
INDEX_PATH = ROOT / "index" / "embeddings.json"


def main() -> None:
    embed_model = os.environ.get("EMBED_MODEL", "nomic-embed-text")
    records = json.loads(SNAPSHOT_PATH.read_text())
    chunks = [Chunk(**{k: v for k, v in r.items() if k != "fetch_date"}) for r in records]

    entries = build_index(chunks, embed_model=embed_model)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_index(entries, INDEX_PATH)
    print(f"wrote {len(entries)} vectors ({len(entries[0].vector)} dims) to {INDEX_PATH}")


if __name__ == "__main__":
    main()
