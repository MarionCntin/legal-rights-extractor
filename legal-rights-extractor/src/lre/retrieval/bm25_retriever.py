# src/lre/retrieval/bm25_retriever.py
"""
Step 3 - Retrieval (BM25 lexical retrieval) — Path 3 (no torch, no onnxruntime, no faiss)

Why this choice
---------------
You’re on Python 3.14.2. Many ML wheels (onnxruntime, faiss-cpu, torch) lag behind
new Python versions.

So we implement retrieval with BM25:
- Pure Python dependency set (rank-bm25 + numpy)
- Deterministic, fast, and strong on legal docs (exact term match matters)
- Perfectly auditable: query terms -> hit list

We keep the SAME public functions as the FAISS version:
- run_retrieval(...)
- infer_default_paths(...)

So your Option A CLI integration does not change.

Input contract (chunks.jsonl)
-----------------------------
Each JSONL line at least:
- chunk_id: str
- text: str

Recommended metadata:
- company_id, doc_id, source_file, page_start, page_end, section_title

Output contract (retrieval.jsonl)
---------------------------------
Each JSONL line = record for (company_id, clause_family) with:
- queries
- top_k
- hits: chunk_id, score, text, provenance
"""

from __future__ import annotations
import numpy as np
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from rank_bm25 import BM25Okapi

_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+")

# ---------------------------------------------------------------------
# Query packs (your main recall lever)
# ---------------------------------------------------------------------
DEFAULT_CLAUSE_QUERIES: Dict[str, List[str]] = {
    # -----------------------------------------------------------------
    # VETO / GOVERNANCE / RESERVED MATTERS
    # -----------------------------------------------------------------
    "veto": [
        # Core governance terms
        "décisions collectives associés actionnaires",
        "décisions réservées décisions soumises à approbation",
        "accord préalable consentement préalable autorisation préalable",
        "approbation préalable des investisseurs",
        "majorité qualifiée majorité renforcée unanimité quorum",

        # Bodies / governance architecture
        "assemblée générale AGO AGE consultation écrite",
        "conseil d'administration conseil de surveillance directoire",
        "comité stratégique comité d'investissement comité consultatif",

        # Typical reserved matters buckets
        "approbation budget business plan plan d'affaires",
        "investissements capex dépenses exceptionnelles",
        "endettement emprunt garanties sûretés nantissement",
        "nomination révocation dirigeants président directeur général",
        "émission de titres augmentation réduction de capital",
        "modification des statuts objet social siège social",
        "fusion scission apport partiel d'actif cession d'actifs",
        "distribution dividendes mise en réserve",

        # English fallback
        "reserved matters investor consent prior approval",
        "board approval shareholder approval qualified majority quorum",
        "appointment removal CEO executives budget debt capex",
    ],

    # -----------------------------------------------------------------
    # LIQUIDITY / TRANSFER RESTRICTIONS / ROFR-ROFO / APPROVAL / LOCK-UP
    # -----------------------------------------------------------------
    "liquidity": [
        # Transfer / assignment terms
        "cession de titres cession d'actions cession de parts",
        "transfert de titres mutation transmission",
        "cession à un tiers cession intra-groupe",

        # Approval / agrément
        "clause d'agrément agrément préalable refus d'agrément",
        "procédure d'agrément notification projet de cession",
        "refus d'agrément rachat substitution",

        # Pre-emption / ROFR / ROFO
        "droit de préemption clause de préemption",
        "priorité d'achat offre de cession",
        "droit de premier refus right of first refusal ROFR",
        "right of first offer ROFO droit de première offre",

        # Lock-up / inalienability
        "inaliénabilité clause d'inaliénabilité lock-up",
        "interdiction de céder période d'indisponibilité",

        # Exit mechanics linked to liquidity (not liquidation preference)
        "sortie conjointe clause de sortie conjointe tag along",
        "clause d'entraînement drag along entraînement",
        "promesse de vente promesse d'achat",
        "clause de liquidité liquidité des titres",

        # Price / valuation mechanics often attached to transfers
        "prix de cession expert indépendant détermination du prix",
        "évaluation expert désigné prix plancher prix plafond",

        # English fallback
        "transfer restriction share transfer approval clause",
        "pre-emption right right of first refusal ROFR",
        "lock-up inalienability transfer of shares",
        "tag along drag along",
    ],

    # -----------------------------------------------------------------
    # EXIT / LIQUIDATION / CHANGE OF CONTROL / IPO / SALE PROCESS
    # -----------------------------------------------------------------
    "exit": [
        # Liquidation / dissolution
        "liquidation dissolution boni de liquidation",
        "répartition du boni liquidation partage de l'actif",
        "dissolution anticipée liquidation amiable",

        # Change of control / sale
        "changement de contrôle cession de contrôle",
        "cession de la société vente de la société",
        "cession d'actifs substantiels cession d'activité",
        "fusion acquisition prise de contrôle",

        # IPO / listing
        "introduction en bourse IPO admission aux négociations",
        "marché réglementé Euronext cotation",

        # Structured exit process
        "processus de sortie mandat banque d'affaires",
        "fenêtre de liquidité liquidité organisée",
        "cession forcée entraînement drag along",
        "sortie conjointe tag along",

        # English fallback
        "change of control sale of the company IPO listing",
        "sale process liquidity event",
    ],

    # -----------------------------------------------------------------
    # ANTI-DILUTION / PRICE ADJUSTMENT / RATINGS / WARRANTS (BSA/BSAR)
    # -----------------------------------------------------------------
    "anti_dilution": [
        # Core anti-dilution vocabulary
        "anti-dilution antidilution dilution",
        "clause de relution relution",

        # Price adjustment / down round mechanics
        "ajustement prix d'émission prix de souscription",
        "ajustement du prix en cas d'émission à un prix inférieur",
        "ajustement prix de conversion conversion",
        "émission à un prix inférieur down round",

        # Ratchet mechanics
        "ratchet full ratchet weighted average",
        "moyenne pondérée weighted average anti dilution",

        # Instruments frequently used in French docs
        "bons de souscription BSA BSAR BSPCE",
        "prix d'exercice ajustement du prix d'exercice",
        "parité de conversion nombre d'actions conversion",

        # English fallback
        "anti-dilution full ratchet weighted average conversion price adjustment",
        "warrants adjustment exercise price conversion ratio",
    ],
}

VALID_FAMILIES = tuple(DEFAULT_CLAUSE_QUERIES.keys())
DEFAULT_MODEL_LABEL = "bm25-okapi"


# ---------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    company_id: str = "unknown"
    doc_id: str = "unknown"
    source_file: str = "unknown"
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section_title: Optional[str] = None

    @staticmethod
    def from_json(obj: Dict[str, Any]) -> "Chunk":
        if "chunk_id" not in obj:
            raise ValueError("Chunk missing required field: chunk_id")
        if "text" not in obj:
            raise ValueError("Chunk missing required field: text")

        return Chunk(
            chunk_id=str(obj["chunk_id"]),
            text=str(obj["text"]),
            company_id=str(obj.get("company_id", "unknown")),
            doc_id=str(obj.get("doc_id", "unknown")),
            source_file=str(obj.get("source_file", obj.get("source", "unknown"))),
            page_start=_safe_int(obj.get("page_start")),
            page_end=_safe_int(obj.get("page_end")),
            section_title=_safe_str(obj.get("section_title")),
        )


def _safe_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _safe_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _get(ch, name, default=None):
    return getattr(ch, name, default)

def _key_doc_idx(ch):
    return (_get(ch, "doc_id", ""), int(_get(ch, "chunk_index", 0)))

def _tokenize(s: str) -> List[str]:
    return _WORD_RE.findall((s or "").lower())

def bm25_rank(chunks: List[Chunk], query: str, top_k: int = 12) -> List[tuple[Chunk, float]]:
    """
    Rank chunks for a query using BM25Okapi.
    Returns list of (Chunk, score) sorted desc.
    """
    tokenized_corpus = [_tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    q_tokens = _tokenize(query)
    scores = bm25.get_scores(q_tokens)  # numpy array

    # Take top_k indices
    # (argsort ascending -> take tail)
    idxs = np.argsort(scores)[-top_k:][::-1]

    out = []
    for i in idxs:
        sc = float(scores[i])
        # optional: skip pure zeros
        if sc <= 0:
            continue
        out.append((chunks[int(i)], sc))
    return out


def _doc_order_key(ch) -> tuple:
    """
    Best-effort stable ordering within a document.
    Prefer (page_start, chunk_id). If you have chunk_index, use it instead.
    """
    page = getattr(ch, "page_start", None)
    if page is None:
        page = 10**9
    # chunk_id should exist; fallback to source_file/text hash if needed
    return (page, getattr(ch, "chunk_id", ""))


def _index_by_doc(chunks):
    by_doc = {}
    for ch in chunks:
        doc_id = getattr(ch, "doc_id", "") or "unknown"
        by_doc.setdefault(doc_id, []).append(ch)
    for doc_id in by_doc:
        by_doc[doc_id].sort(key=_doc_order_key)
    return by_doc


def _pos_in_doc(by_doc):
    # doc_id -> {chunk_id -> position}
    out = {}
    for doc_id, lst in by_doc.items():
        out[doc_id] = {getattr(c, "chunk_id", ""): i for i, c in enumerate(lst)}
    return out


def _expand_with_window(core_hits, by_doc, window: int = 2):
    """
    Add +/- window neighbors around each core hit in the same doc.
    """
    pos = _pos_in_doc(by_doc)
    selected = set()

    for h in core_hits:
        doc_id = getattr(h, "doc_id", "") or "unknown"
        cid = getattr(h, "chunk_id", "")
        if doc_id not in by_doc or cid not in pos.get(doc_id, {}):
            continue
        i = pos[doc_id][cid]
        lo = max(0, i - window)
        hi = min(len(by_doc[doc_id]) - 1, i + window)
        for j in range(lo, hi + 1):
            selected.add((doc_id, j))

    expanded = []
    for doc_id, j in sorted(selected):
        expanded.append(by_doc[doc_id][j])
    return expanded


def _merge_contiguous_by_position(expanded, by_doc):
    """
    Merge contiguous neighbors into passages (dicts).
    Contiguous = consecutive in the per-doc sorted list.
    """
    if not expanded:
        return []

    pos = _pos_in_doc(by_doc)
    expanded_sorted = sorted(
        expanded,
        key=lambda c: (
            getattr(c, "doc_id", "") or "unknown",
            pos.get(getattr(c, "doc_id", "") or "unknown", {}).get(getattr(c, "chunk_id", ""), 10**9),
        ),
    )

    merged = []
    cur = None
    cur_doc = None
    cur_pos = None

    for ch in expanded_sorted:
        doc_id = getattr(ch, "doc_id", "") or "unknown"
        cid = getattr(ch, "chunk_id", "")
        p = pos.get(doc_id, {}).get(cid)

        ch_d = {
            "chunk_id": cid,
            "text": getattr(ch, "text", "") or "",
            "company_id": getattr(ch, "company_id", "") or "",
            "doc_id": getattr(ch, "doc_id", "") or "",
            "source_file": getattr(ch, "source_file", "") or "",
            "page_start": getattr(ch, "page_start", None),
            "page_end": getattr(ch, "page_end", None),
            "section_title": getattr(ch, "section_title", None),
            "article": getattr(ch, "article", None) if hasattr(ch, "article") else None,
        }

        if cur is None:
            cur, cur_doc, cur_pos = ch_d, doc_id, p
            continue

        # contiguous if same doc and positions consecutive
        if cur_doc == doc_id and cur_pos is not None and p is not None and p == cur_pos + 1:
            cur["text"] = (cur["text"] + "\n" + ch_d["text"]).strip()
            # extend page_end conservatively
            if cur.get("page_end") is None:
                cur["page_end"] = ch_d.get("page_end")
            elif ch_d.get("page_end") is not None:
                cur["page_end"] = max(int(cur["page_end"]), int(ch_d["page_end"]))
            cur_pos = p
        else:
            merged.append(cur)
            cur, cur_doc, cur_pos = ch_d, doc_id, p

    if cur is not None:
        merged.append(cur)

    return merged

# ---------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------
def infer_default_paths(out_dir: Path) -> Tuple[Path, Path]:
    """
    Infer default flat-layout paths used by the CLI.

    Returns:
      (chunks_path, retrieval_out_path)

    Flat layout expected:
      <out_dir>/chunks.jsonl
      <out_dir>/retrieval.jsonl
    """
    out_dir = Path(out_dir)
    chunks_path = out_dir / "chunks.jsonl"
    retrieval_out_path = out_dir / "retrieval.jsonl"
    return chunks_path, retrieval_out_path


def load_chunks_jsonl(path: str | Path) -> List[Chunk]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"chunks file not found: {path}")

    chunks: List[Chunk] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                chunks.append(Chunk.from_json(obj))
            except Exception as e:
                raise ValueError(f"Invalid JSONL at line {i} in {path}: {e}") from e

    if not chunks:
        raise ValueError(f"No chunks loaded from {path}")
    return chunks

def write_retrieval_jsonl(records: Iterable[Dict[str, Any]], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def _build_chunks_by_doc(all_chunks):
    by_doc = {}
    for ch in all_chunks:
        by_doc.setdefault(ch["doc_id"], []).append(ch)
    for doc_id in by_doc:
        by_doc[doc_id].sort(key=lambda x: x["chunk_index"])
    return by_doc

def _index_by_doc(chunks: List[Chunk]) -> Dict[str, List[Chunk]]:
    by_doc: Dict[str, List[Chunk]] = {}
    for ch in chunks:
        by_doc.setdefault(ch.doc_id or "unknown", []).append(ch)

    # stable order within doc (best effort)
    for doc_id in by_doc:
        by_doc[doc_id].sort(key=lambda c: (c.page_start or 10**9, c.chunk_id))
    return by_doc


def _pos_in_doc(by_doc: Dict[str, List[Chunk]]) -> Dict[str, Dict[str, int]]:
    """
    Returns mapping: doc_id -> {chunk_id -> position in sorted list}
    """
    out: Dict[str, Dict[str, int]] = {}
    for doc_id, lst in by_doc.items():
        out[doc_id] = {c.chunk_id: i for i, c in enumerate(lst)}
    return out


def _expand_with_window(core_hits: List[Chunk], by_doc: Dict[str, List[Chunk]], window: int = 2) -> List[Chunk]:
    pos = _pos_in_doc(by_doc)
    selected: set[tuple[str, int]] = set()

    for h in core_hits:
        doc_id = h.doc_id or "unknown"
        if doc_id not in by_doc or h.chunk_id not in pos.get(doc_id, {}):
            continue
        i = pos[doc_id][h.chunk_id]
        for j in range(max(0, i - window), min(len(by_doc[doc_id]), i + window + 1)):
            selected.add((doc_id, j))

    expanded: List[Chunk] = []
    for doc_id, j in sorted(selected):
        expanded.append(by_doc[doc_id][j])
    return expanded


def _merge_contiguous_by_position(chunks: List[Chunk], by_doc: Dict[str, List[Chunk]]) -> List[Dict[str, Any]]:
    """
    Merge adjacent chunks (neighbors in the sorted per-doc list) into passages.
    Returns list of dicts (hits) compatible with your downstream JSONL.
    """
    if not chunks:
        return []

    pos = _pos_in_doc(by_doc)
    chunks_sorted = sorted(chunks, key=lambda c: (c.doc_id or "unknown", pos.get(c.doc_id or "unknown", {}).get(c.chunk_id, 10**9)))

    merged: List[Dict[str, Any]] = []
    cur = None
    cur_pos = None

    for ch in chunks_sorted:
        doc_id = ch.doc_id or "unknown"
        p = pos.get(doc_id, {}).get(ch.chunk_id)

        ch_d = {
            "chunk_id": ch.chunk_id,
            "text": ch.text,
            "company_id": ch.company_id,
            "doc_id": ch.doc_id,
            "source_file": ch.source_file,
            "page_start": ch.page_start,
            "page_end": ch.page_end,
            "section_title": ch.section_title,
        }

        if cur is None:
            cur, cur_pos = ch_d, p
            continue

        # contiguous if same doc and positions are consecutive
        if doc_id == cur["doc_id"] and cur_pos is not None and p is not None and p == cur_pos + 1:
            cur["text"] = (cur["text"] + "\n" + ch_d["text"]).strip()
            # extend page_end conservatively
            if cur["page_end"] is None:
                cur["page_end"] = ch_d["page_end"]
            elif ch_d["page_end"] is not None:
                cur["page_end"] = max(int(cur["page_end"]), int(ch_d["page_end"]))
            cur_pos = p
        else:
            merged.append(cur)
            cur, cur_pos = ch_d, p

    if cur is not None:
        merged.append(cur)

    return merged

# ---------------------------------------------------------------------
# Tokenization (simple, robust)
# ---------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    # Lowercase + keep alphanum words (works fine for French)
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def _dedup_keep_best(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for h in hits:
        cid = h["chunk_id"]
        if cid not in best or h["score"] > best[cid]["score"]:
            best[cid] = h
    return list(best.values())


def _safe_sigmoid(x: float) -> float:
    # Optional: map BM25 scores to 0..1 for readability while preserving ordering.
    # BM25 scores are unbounded. This makes logs nicer.
    if x > 50:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------
# Retrieval core (BM25)
# ---------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+")

def _tokenize(s: str):
    return _WORD_RE.findall((s or "").lower())

def retrieve_for_company(chunks: List[Chunk], top_k: int = 12) -> List[Dict[str, Any]]:
    """
    Step 3 retrieval per company:
    - BM25 core hits (k_core)
    - window expansion (+/- window neighbors)
    - merge contiguous neighbors into passages
    """
    if not chunks:
        return []

    # knobs (good starting point)
    k_core = min(6, top_k)     # how many "true hits" per family
    window = 2                # neighbors on each side
    k_final = top_k           # final number of merged passages to keep

    by_doc = _index_by_doc(chunks)

    # Build BM25 once per company
    tokenized_corpus = [_tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    records: List[Dict[str, Any]] = []

    for clause_family in ["veto", "liquidity", "exit", "anti_dilution"]:
        queries = DEFAULT_CLAUSE_QUERIES.get(clause_family, [clause_family])

        # ---- 1) multi-query BM25 -> scores aggregated by max ----
        best_score_by_idx = {}

        for q in queries:
            q_tokens = _tokenize(q)
            scores = bm25.get_scores(q_tokens)
            for i, sc in enumerate(scores):
                sc = float(sc)
                if sc <= 0:
                    continue
                if i not in best_score_by_idx or sc > best_score_by_idx[i]:
                    best_score_by_idx[i] = sc

        # core hit indices
        core_idxs = sorted(best_score_by_idx.keys(), key=lambda i: best_score_by_idx[i], reverse=True)[:k_core]
        core_hits = [chunks[i] for i in core_idxs]

        # ---- 2) window expansion (don’t skip between) ----
        expanded = _expand_with_window(core_hits, by_doc, window=window)

        # ---- 3) merge contiguous neighbors into passages ----
        merged_hits = _merge_contiguous_by_position(expanded, by_doc)

        # ---- 4) cap final hits ----
        hits = merged_hits[:k_final]

        records.append(
            {
                "company_id": chunks[0].company_id,
                "clause_family": clause_family,
                "hits": hits,
            }
        )

    return records


def run_retrieval(
    chunks_path: str | Path,
    out_path: str | Path,
    model_name: str = DEFAULT_MODEL_LABEL,  # ignored (BM25)
    top_k: int = 12,
    batch_size: int = 0,  # ignored
    split_by_company: bool = True,
) -> None:
    chunks_path = Path(chunks_path)
    out_path = Path(out_path)

    chunks = load_chunks_jsonl(chunks_path)

    if not split_by_company:
        records = retrieve_for_company(chunks=chunks, top_k=top_k)
        write_retrieval_jsonl(records, out_path)
        return

    by_company = {}
    for c in chunks:
        by_company.setdefault(c.company_id, []).append(c)

    all_records = []
    for _, group in sorted(by_company.items()):
        recs = retrieve_for_company(chunks=group, top_k=top_k)
        all_records.extend(recs)

    write_retrieval_jsonl(all_records, out_path)
