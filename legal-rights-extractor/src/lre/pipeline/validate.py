# pipeline/step5_validation.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Optional
import pandas as pd
from lre.schemas.right_item import RightItem


WORD_RE = re.compile(r"\S+")


def count_words(s: str) -> int:
    return len(WORD_RE.findall(s or ""))


def normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = s.replace("\n", " ")
    s = s.replace("\t", " ")
    s = re.sub(r"\s+", " ", s)
    # normalize common symbols
    s = s.replace("%", " percent")
    s = s.replace("€", " eur ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def token_jaccard(a: str, b: str) -> float:
    ta = set(normalize_text(a).split())
    tb = set(normalize_text(b).split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def deterministic_validate(item: RightItem, min_conf_ambiguous: float = 0.6, max_quote_words: int = 25) -> RightItem:
    # Always keep "Not Found" items, but enforce consistency for present=True.
    # Never allow missing evidence for present=True.
    if item.present:
        if not item.evidence or not (item.evidence.quote or "").strip():
            item.flags.append("missing_evidence")
        if not item.evidence or not (item.evidence.source_file or "").strip():
            item.flags.append("missing_source")

        q = (item.evidence.quote or "").strip()
        if q and count_words(q) > max_quote_words:
            item.flags.append("quote_too_long")

        # If evidence missing/invalid => downgrade
        if any(f in item.flags for f in ["missing_evidence", "missing_source", "quote_too_long"]):
            item.present = False

    # confidence triage flag only (doesn't flip present by itself)
    if item.confidence < min_conf_ambiguous:
        if "ambiguous" not in item.flags:
            item.flags.append("ambiguous")

    # strip newlines in quote (audit-friendly)
    if item.evidence and item.evidence.quote:
        item.evidence.quote = normalize_text(item.evidence.quote)

    # normalize strings (keep originals minimal but consistent)
    item.trigger = item.trigger.strip()
    item.effect = item.effect.strip()
    if item.thresholds is not None:
        item.thresholds = item.thresholds.strip() or None

    # de-dup flags
    item.flags = list(dict.fromkeys(item.flags))
    return item


@dataclass(frozen=True)
class ConflictKey:
    company_id: str
    clause_family: str
    clause_type: str


def conflict_key(item: RightItem) -> ConflictKey:
    return ConflictKey(item.company_id, item.clause_family, normalize_text(item.clause_type))


def conflict_score(a: RightItem, b: RightItem) -> float:
    """
    Higher means "same"; lower means "different".
    We compare trigger/effect/thresholds with token overlap.
    """
    s1 = token_jaccard(a.trigger, b.trigger)
    s2 = token_jaccard(a.effect, b.effect)
    s3 = token_jaccard(a.thresholds or "", b.thresholds or "")
    return (0.45 * s1) + (0.45 * s2) + (0.10 * s3)


def flag_conflicts(items: List[RightItem], similarity_threshold: float = 0.75) -> List[RightItem]:
    """
    For each group (company_id, clause_family, clause_type), if multiple PRESENT items differ -> conflicting.
    We do not delete anything; we flag them.
    """
    groups: Dict[ConflictKey, List[RightItem]] = {}
    for it in items:
        groups.setdefault(conflict_key(it), []).append(it)

    for key, group in groups.items():
        present_items = [x for x in group if x.present]
        if len(present_items) <= 1:
            continue

        # compare all pairs
        n = len(present_items)
        conflict = False
        for i in range(n):
            for j in range(i + 1, n):
                if conflict_score(present_items[i], present_items[j]) < similarity_threshold:
                    conflict = True
                    break
            if conflict:
                break

        if conflict:
            for x in present_items:
                if "conflicting" not in x.flags:
                    x.flags.append("conflicting")

    return items


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def to_dataframe(items: List[RightItem]) -> pd.DataFrame:
    rows = []
    for it in items:
        ev = it.evidence or None
        rows.append(
            {
                "company_id": it.company_id,
                "clause_family": it.clause_family,
                "clause_type": it.clause_type,
                "present": it.present,
                "beneficiary": it.beneficiary,
                "trigger": it.trigger,
                "effect": it.effect,
                "thresholds": it.thresholds,
                "evidence_quote": (ev.quote if ev else None),
                "source_file": (ev.source_file if ev else None),
                "page": (ev.page if ev else None),
                "article": (ev.article if ev else None),
                "confidence": it.confidence,
                "flags": ",".join(it.flags or []),
            }
        )
    df = pd.DataFrame(rows)
    # stable sort for review
    sort_cols = ["company_id", "clause_family", "clause_type", "present"]
    for c in sort_cols:
        if c not in df.columns:
            return df
    df = df.sort_values(sort_cols, ascending=[True, True, True, False]).reset_index(drop=True)
    return df


def export_xlsx(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="portfolio")
        ws = writer.sheets["portfolio"]
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

        # Basic column sizing (rough)
        for col in ws.columns:
            max_len = 0
            col_letter = col[0].column_letter
            for cell in col[:200]:  # cap scan
                v = "" if cell.value is None else str(cell.value)
                max_len = max(max_len, len(v))
            ws.column_dimensions[col_letter].width = min(60, max(10, max_len + 2))


def run_step5(
    extraction_jsonl: Path,
    out_dir: Path,
    min_conf_ambiguous: float = 0.6,
    max_quote_words: int = 25,
    conflict_similarity_threshold: float = 0.75,
) -> Tuple[Path, Path, Path]:
    """
    Returns (validated_jsonl_path, csv_path, xlsx_path)
    """
    raw = list(read_jsonl(extraction_jsonl))

    items: List[RightItem] = []
    for r in raw:
        # Accept two possible shapes:
        # - {"items":[...], "company_id": "..."} (family batch output)
        # - direct RightItem dict per line
        if "items" in r and isinstance(r["items"], list):
            for x in r["items"]:
                # allow top-level fallback company_id
                if "company_id" not in x and "company_id" in r:
                    x["company_id"] = r["company_id"]
                items.append(RightItem.model_validate(x))
        else:
            items.append(RightItem.model_validate(r))

    # 1) deterministic validation
    items = [deterministic_validate(it, min_conf_ambiguous, max_quote_words) for it in items]

    # 2) conflict flagging
    items = flag_conflicts(items, similarity_threshold=conflict_similarity_threshold)

    # write validated jsonl
    validated_path = out_dir / "validated_items.jsonl"
    write_jsonl(validated_path, (it.model_dump() for it in items))

    # export portfolio table
    df = to_dataframe(items)
    csv_path = out_dir / "portfolio_table.csv"
    xlsx_path = out_dir / "portfolio_table.xlsx"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    export_xlsx(df, xlsx_path)

    return validated_path, csv_path, xlsx_path
