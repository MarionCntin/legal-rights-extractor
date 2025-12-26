from pydantic import BaseModel, Field
from typing import List, Optional
from pathlib import Path
import hashlib

class Page(BaseModel):
    page_index: int
    text: str

class Document(BaseModel):
    doc_id: str
    source_path: str
    file_type: str
    language_hint: Optional[str] = None
    pages: List[Page] = Field(default_factory=list)

def compute_doc_id(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]
