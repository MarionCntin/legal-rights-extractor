# src/lre/chunking/chunker.py
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------

PAGE_BREAK_TOKEN = "<<<PAGE_BREAK>>>"

_HEADING_MAX_LEN = 200
_HEADING_MIN_LEN = 4


@dataclass(frozen=True)
class DocumentPage:
    """Step 2 input: one page of text."""
    source_file: str
    page: int  # 1-indexed
    text: str


@dataclass(frozen=True)
class Section:
    doc_id: str
    source_file: str
    title: str
    level: int
    text: str
    page_start: int
    page_end: int
    start_char: int
    end_char: int


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    company_id: str
    doc_id: str
    source_file: str
    page_start: int
    page_end: int
    section_title: str
    text: str
    section_start_char: int
    section_end_char: int
    sha256: str


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------

def write_jsonl(path: Path, rows: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            if hasattr(r, "__dict__"):
                payload = r.__dict__
            else:
                payload = r
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def make_doc_id(source_file: str) -> str:
    return hashlib.sha1(source_file.encode("utf-8", errors="ignore")).hexdigest()[:16]


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def normalize_whitespace(text: str) -> str:
    # Keep newlines (important for heading detection), but normalize spaces/tabs.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    # remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_page_breaks(text: str) -> str:
    # Keep readability: drop explicit token lines
    text = text.replace(PAGE_BREAK_TOKEN, "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _page_range_for_span(
    page_spans: List[Tuple[int, int, int]],
    span_start: int,
    span_end: int,
) -> Tuple[int, int]:
    """
    page_spans: list of (char_start, char_end, page_number) for each page in concat_text.
    """
    p_start: Optional[int] = None
    p_end: Optional[int] = None

    for s, e, p in page_spans:
        if e <= span_start:
            continue
        if s >= span_end:
            break
        if p_start is None:
            p_start = p
        p_end = p

    if p_start is None:
        # fallback
        p_start = page_spans[0][2]
        p_end = page_spans[-1][2]
    assert p_end is not None
    return p_start, p_end

# ---------------------------------------------------------------------
# Heading heuristics (balanced rules)
# ---------------------------------------------------------------------

# Detect annex headings (real ones)
_RX_ANNEX = re.compile(
    r"(?im)^\s*(Annexe|ANNEXE|Annex|APPENDIX|Schedule)\s*"
    r"(\(?[A-Z0-9]+(?:\.[0-9]+)*\)?)?"
    r"(?:\s*[-–:]\s*.*|\s+.*)?\s*$"
)

# Main section headings: "1. DEFINITIONS", "4. ADMINISTRATION DE LA SOCIETE", "4.1 Le Président"
_RX_NUM = re.compile(r"(?m)^\s*(\d+(?:\.\d+)*)\.\s+(.{2,200}?)\s*$")

# Dotted leader / TOC-like entry: "... 4" at end (with dots/spaces before a page number)
_RX_TOC_DOTTED = re.compile(r"(?i)^\s*(.+?)\s*\.{3,}\s*\d+\s*$")

# Generic "ends with page number" TOC entry (covers annex summary lines too)
_RX_TOC_TRAILING_PAGE = re.compile(r"(?i)^\s*(Annexe|ANNEXE|Annex|APPENDIX|Schedule|\d+(\.\d+)*)\b.*\s+\d+\s*$")

# Money / amount-like line (avoid false headings like "3.000.000 €.")
_RX_MONEYISH = re.compile(r"(?i)^\s*[\d\.\s]+(€|eur)\.?\s*$")


def _uppercase_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    upp = sum(1 for c in letters if c.isupper())
    return upp / len(letters)


def _looks_like_toc_line(line: str) -> bool:
    # Dotted leader lines: "1. DEFINITIONS ..... 4"
    if _RX_TOC_DOTTED.match(line):
        return True
    # Broader: line ends with a page number and begins like a title marker
    if _RX_TOC_TRAILING_PAGE.match(line) and len(line) < 220:
        return True
    return False


def _looks_like_party_list_or_body(line: str) -> bool:
    # Avoid capturing body lines like:
    # "1. Avril Industrie, société par actions simplifiée..."
    bad = re.search(
        r"(?i)\b(société|capital|siège|immatricul|registre|représent|rcs|fonds professionnel)\b",
        line,
    )
    if bad and ("," in line or len(line) > 80):
        return True
    return False


def _looks_like_valid_numeric_heading(full_line: str, level: int) -> bool:
    """
    Balanced rule:
    - For top-level (level=1) we strongly prefer ALL-CAPS headings (like legal docs).
    - For sub-level (level>=2) we accept Title Case too ("4.1 Le Président").
    - Reject long sentence-case items ("3. Mise à jour du business plan ...") as headings.
    """
    if _RX_MONEYISH.match(full_line):
        return False

    if _looks_like_toc_line(full_line):
        return False

    if _looks_like_party_list_or_body(full_line):
        return False

    # Remove numbering prefix from heuristics
    cleaned = re.sub(r"^\s*\d+(?:\.\d+)*\.\s+", "", full_line).strip()

    if len(cleaned) < _HEADING_MIN_LEN or len(cleaned) > _HEADING_MAX_LEN:
        return False

    # If line is very long and contains lots of lowercase, it's likely a list item not a heading.
    # This is the key to stop annex decision lists from becoming headings.
    lower = sum(1 for c in cleaned if c.isalpha() and c.islower())
    upper = sum(1 for c in cleaned if c.isalpha() and c.isupper())
    if len(cleaned) > 60 and lower > upper:
        return False

    # Level 1 headings in these docs are usually all caps.
    if level == 1:
        if _uppercase_ratio(cleaned) < 0.65:
            return False

    return True


def _heading_level_from_number(num: str) -> int:
    # "1" -> 1, "4.1" -> 2, "4.1.2" -> 3, etc.
    return num.count(".") + 1


def detect_headings(text: str, ignore_spans: Optional[List[Tuple[int, int]]] = None) -> List[Tuple[int, int, str, int]]:
    """
    Returns list of (start, end, title, level).
    """
    ignore_spans = ignore_spans or []

    def in_ignored(pos: int) -> bool:
        return any(a <= pos < b for (a, b) in ignore_spans)

    matches: List[Tuple[int, int, str, int]] = []

    # 1) Annex headings
    for m in _RX_ANNEX.finditer(text):
        s, e = m.span()
        if in_ignored(s):
            continue
        line = (m.group(0) or "").strip()
        if len(line) < _HEADING_MIN_LEN or len(line) > _HEADING_MAX_LEN:
            continue
        if _looks_like_toc_line(line):
            # e.g., "Annexe 1 Définitions  51" inside a summary
            continue
        matches.append((s, e, line, 1))

    # 2) Numeric headings
    for m in _RX_NUM.finditer(text):
        s, e = m.span()
        if in_ignored(s):
            continue
        full = (m.group(0) or "").strip()
        num = (m.group(1) or "").strip()
        level = _heading_level_from_number(num)

        if not _looks_like_valid_numeric_heading(full, level=level):
            continue

        matches.append((s, e, full, level))

    # Sort & dedupe overlaps (keep earliest)
    matches.sort(key=lambda x: (x[0], x[3], -(x[1] - x[0])))

    dedup: List[Tuple[int, int, str, int]] = []
    last_start = -1
    for s, e, title, lvl in matches:
        if s == last_start:
            continue
        dedup.append((s, e, title, lvl))
        last_start = s

    return dedup


# ---------------------------------------------------------------------
# Section builder
# ---------------------------------------------------------------------

def build_sections(
    pages: List[DocumentPage],
    doc_id: Optional[str] = None,
) -> List[Section]:
    """
    Build document sections.
    GUARANTEE: always returns at least one section if pages is non-empty.
    """
    if not pages:
        return []

    source_file = pages[0].source_file
    doc_id = doc_id or make_doc_id(source_file)

    # Concatenate pages with explicit markers and keep spans
    concat_parts: List[str] = []
    page_spans: List[Tuple[int, int, int]] = []
    cursor = 0

    pb = f"\n\n{PAGE_BREAK_TOKEN}\n\n"

    for i, p in enumerate(pages):
        txt = normalize_whitespace(p.text)
        start = cursor
        cursor += len(txt)
        end = cursor
        page_spans.append((start, end, p.page))
        concat_parts.append(txt)
        if i < len(pages) - 1:
            concat_parts.append(pb)
            cursor += len(pb)

    concat_text = "".join(concat_parts)

    # Ignore spans (TOC page range only, not “global”)
    ignore_spans: List[Tuple[int, int]] = []

    # Ignore "TABLE DES MATIERES" block until next page break (or safe window).
    toc = re.search(r"(?i)\bTABLE\s+DES\s+MATIERES\b", concat_text)
    if toc:
        toc_start = toc.start()
        pb_idx = concat_text.find(pb, toc_start)
        toc_end = pb_idx if pb_idx != -1 else min(len(concat_text), toc_start + 2500)
        ignore_spans.append((toc_start, toc_end))

    headings = detect_headings(concat_text, ignore_spans=ignore_spans)

    # FAIL-SOFT: no headings → one big section
    if not headings:
        pstart, pend = _page_range_for_span(page_spans, 0, len(concat_text))
        return [
            Section(
                doc_id=doc_id,
                source_file=source_file,
                title="(Full document)",
                level=0,
                text=strip_page_breaks(concat_text),
                page_start=pstart,
                page_end=pend,
                start_char=0,
                end_char=len(concat_text),
            )
        ]

    sections: List[Section] = []

    def add_section(start: int, end: int, title: str, level: int) -> None:
        body = strip_page_breaks(concat_text[start:end]).strip()
        if not body:
            return
        pstart, pend = _page_range_for_span(page_spans, start, end)
        sections.append(
            Section(
                doc_id=doc_id,
                source_file=source_file,
                title=title,
                level=level,
                text=body,
                page_start=pstart,
                page_end=pend,
                start_char=start,
                end_char=end,
            )
        )

    # Preface
    first_start = headings[0][0]
    if first_start > 0:
        add_section(0, first_start, "(Preface)", 0)

    for i, (s, _e, title, lvl) in enumerate(headings):
        next_start = headings[i + 1][0] if i + 1 < len(headings) else len(concat_text)
        add_section(s, next_start, title, lvl)

    return sections


# ---------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------

def _split_by_char_budget(text: str, max_chars: int, overlap_chars: int) -> List[Tuple[int, int]]:
    """
    Simple, auditable chunk splitter by character count.
    Returns list of (start,end) spans in 'text'.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be >= 0")
    if not text:
        return []

    spans: List[Tuple[int, int]] = []
    n = len(text)
    start = 0

    while start < n:
        end = min(n, start + max_chars)
        spans.append((start, end))
        if end >= n:
            break
        start = max(0, end - overlap_chars)

    return spans


def build_chunks(
    sections: Sequence[Section],
    company_id: str,
    doc_id: str,
    max_chars: int = 4500,
    overlap_ratio: float = 0.12,
) -> List[Chunk]:
    """
    Produce chunks from sections.
    (Char-based is fine for now; you can swap to tokenizer later.)
    """
    chunks: List[Chunk] = []
    overlap_chars = int(max_chars * overlap_ratio)

    for s_idx, sec in enumerate(sections):
        spans = _split_by_char_budget(sec.text, max_chars=max_chars, overlap_chars=overlap_chars)
        for c_idx, (a, b) in enumerate(spans):
            txt = sec.text[a:b].strip()
            if not txt:
                continue
            chunk_id = f"{doc_id}_s{s_idx:04d}_c{c_idx:04d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    company_id=company_id,
                    doc_id=doc_id,
                    source_file=sec.source_file,
                    page_start=sec.page_start,
                    page_end=sec.page_end,
                    section_title=sec.title,
                    text=txt,
                    section_start_char=sec.start_char,
                    section_end_char=sec.end_char,
                    sha256=sha256_text(txt),
                )
            )

    return chunks


def step2_chunking(
    pages: List[DocumentPage],
    company_id: str,
    doc_id: str,
    max_chars: int = 4500,
    overlap_ratio: float = 0.12,
    max_tokens: Optional[int] = None,  # NEW: accept token budget from caller
) -> Tuple[List[Section], List[Chunk]]:
    # If caller passes max_tokens, convert to an approximate char budget
    # (rule of thumb: 1 token ≈ 4 characters in English/French legal text)
    if max_tokens is not None:
        max_chars = int(max_tokens * 4)

    sections = build_sections(pages=pages, doc_id=doc_id)
    chunks = build_chunks(
        sections=sections,
        company_id=company_id,
        doc_id=doc_id,
        max_chars=max_chars,
        overlap_ratio=overlap_ratio,
    )
    return sections, chunks

