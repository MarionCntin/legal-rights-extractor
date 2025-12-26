import json
from pathlib import Path
from typing import Iterator
from .normalize import Document, Page  # your existing types

def read_ingest_jsonl(path: Path) -> Iterator[Document]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            pages = [Page(page_index=p["page_index"], text=p["text"]) for p in obj["pages"]]
            yield Document(
                doc_id=obj["doc_id"],
                source_path=obj["source_path"],
                file_type=obj["file_type"],
                pages=pages,
            )
