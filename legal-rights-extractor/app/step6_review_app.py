#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


# ----------------------------
# Config
# ----------------------------

DEFAULT_OUT_ROOT = Path("data/out-fake")
REVIEW_DIRNAME = "review"  # will write under out_root/review/<company>/...


# ----------------------------
# Helpers
# ----------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clamp(s: str, n: int = 80) -> str:
    s = (s or "").replace("\n", " ").strip()
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def normalize_text(s: str) -> str:
    return "".join(ch.lower() for ch in (s or "") if ch.isalnum() or ch.isspace()).strip()


def item_key(company: str, docset: str, clause_family: str, item_id: str) -> str:
    # stable unique key used for review persistence
    return f"{company}||{docset}||{clause_family}||{item_id}"


# ----------------------------
# Discovery (out-fake structure)
# ----------------------------

def discover_companies(out_root: Path) -> List[str]:
    if not out_root.exists():
        return []
    companies = []
    for p in out_root.iterdir():
        if p.is_dir() and p.name not in {REVIEW_DIRNAME, "raw"}:
            companies.append(p.name)
    return sorted(companies)


def discover_docsets(out_root: Path, company: str) -> List[str]:
    base = out_root / company
    if not base.exists():
        return []
    docsets = [p.name for p in base.iterdir() if p.is_dir()]
    return sorted(docsets)


# ----------------------------
# Load & flatten extraction outputs
# ----------------------------

@dataclass
class FlatItem:
    company_id: str
    docset: str
    doc_type: str
    clause_family: str
    item_id: str
    present: bool
    clause_type: str
    beneficiary: str
    trigger: str
    effect: str
    thresholds: Optional[str]
    evidence_quote: str
    evidence_source_file: str
    evidence_page: Optional[int]
    evidence_article: Optional[str]
    confidence: float
    flags: List[str]

    # review state (step6)
    review_status: str = "pending"   # pending|accepted|edited|rejected
    review_comment: str = ""
    reviewer: str = ""
    review_timestamp: str = ""


def load_flat_items_from_docset(out_root: Path, company: str, docset: str) -> List[FlatItem]:
    extraction_path = out_root / company / docset / "extraction.jsonl"
    rows = read_jsonl(extraction_path)

    items: List[FlatItem] = []
    for row in rows:
        clause_family = row.get("clause_family")
        doc_type = row.get("doc_type", "")
        for it in row.get("items", []):
            ev = it.get("evidence", {}) or {}
            items.append(FlatItem(
                company_id=row.get("company_id", company),
                docset=row.get("docset", docset),
                doc_type=doc_type,
                clause_family=clause_family,
                item_id=it.get("item_id", ""),
                present=bool(it.get("present", False)),
                clause_type=it.get("clause_type", ""),
                beneficiary=it.get("beneficiary", ""),
                trigger=it.get("trigger", ""),
                effect=it.get("effect", ""),
                thresholds=it.get("thresholds"),
                evidence_quote=ev.get("quote", ""),
                evidence_source_file=ev.get("source_file", ""),
                evidence_page=ev.get("page"),
                evidence_article=ev.get("article"),
                confidence=float(it.get("confidence", 0.0) or 0.0),
                flags=list(it.get("flags", []) or []),
            ))
    return items


def load_flat_items(out_root: Path, company: str, docset_filter: str) -> List[FlatItem]:
    docsets = discover_docsets(out_root, company)
    if docset_filter != "ALL":
        docsets = [docset_filter] if docset_filter in docsets else []
    all_items: List[FlatItem] = []
    for ds in docsets:
        all_items.extend(load_flat_items_from_docset(out_root, company, ds))
    return all_items


# ----------------------------
# Review persistence
# ----------------------------

def review_paths(out_root: Path, company: str) -> Tuple[Path, Path]:
    base = out_root / REVIEW_DIRNAME / company
    return base / "reviewed_rights.jsonl", base / "review_log.jsonl"


def load_review_state(out_root: Path, company: str) -> Dict[str, Dict[str, Any]]:
    reviewed_path, _ = review_paths(out_root, company)
    rows = read_jsonl(reviewed_path)
    by_key: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        k = r.get("_key")
        if k:
            by_key[k] = r
    return by_key


def save_review_state(out_root: Path, company: str, state: Dict[str, Dict[str, Any]]) -> None:
    reviewed_path, _ = review_paths(out_root, company)
    write_jsonl(reviewed_path, list(state.values()))


def append_audit(out_root: Path, company: str, event: Dict[str, Any]) -> None:
    _, log_path = review_paths(out_root, company)
    append_jsonl(log_path, event)


def merge_review(items: List[FlatItem], review_state: Dict[str, Dict[str, Any]]) -> None:
    for it in items:
        k = item_key(it.company_id, it.docset, it.clause_family, it.item_id)
        r = review_state.get(k)
        if not r:
            continue
        it.review_status = r.get("review_status", "pending")
        it.review_comment = r.get("review_comment", "")
        it.reviewer = r.get("reviewer", "")
        it.review_timestamp = r.get("review_timestamp", "")

        # If edited, overwrite editable fields from reviewed record
        if it.review_status == "edited":
            it.clause_type = r.get("clause_type", it.clause_type)
            it.beneficiary = r.get("beneficiary", it.beneficiary)
            it.trigger = r.get("trigger", it.trigger)
            it.effect = r.get("effect", it.effect)
            it.thresholds = r.get("thresholds", it.thresholds)


# ----------------------------
# Conflict detection (lightweight)
# ----------------------------

def compute_conflicts(items: List[FlatItem]) -> Dict[str, List[str]]:
    """
    Returns mapping conflict_group_id -> list of item keys.
    Heuristic: present=true items across docsets with same (clause_family, clause_type)
    but different thresholds OR different normalized effect => conflict.
    """
    groups: Dict[Tuple[str, str], List[FlatItem]] = {}
    for it in items:
        if not it.present:
            continue
        key = (it.clause_family, it.clause_type.strip() or "(missing)")
        groups.setdefault(key, []).append(it)

    conflict_groups: Dict[str, List[str]] = {}
    cg_idx = 1
    for (fam, ctype), group_items in groups.items():
        if len(group_items) < 2:
            continue

        thr_set = set((g.thresholds or "").strip() for g in group_items)
        eff_set = set(normalize_text(g.effect) for g in group_items)

        is_conflict = (len(thr_set) > 1) or (len(eff_set) > 1)
        if is_conflict:
            cg_id = f"cg_{cg_idx:03d}_{fam}"
            cg_idx += 1
            conflict_groups[cg_id] = [
                item_key(g.company_id, g.docset, g.clause_family, g.item_id) for g in group_items
            ]
    return conflict_groups


# ----------------------------
# UI
# ----------------------------

def render_kpis(df: pd.DataFrame) -> None:
    pending = int((df["Review"] == "pending").sum())
    flagged = int(df["Flags"].apply(lambda x: len(x) > 0).sum())
    conflicts = int(df["Flags"].apply(lambda x: "conflicting" in x).sum())
    avg_conf = float(df["Confidence"].mean()) if len(df) else 0.0

    c1, c2, c3, c4 = st.columns(4)

    def card(col, label, value, sub="", tone="neutral"):
        dot = {"good":"good","warn":"warn","bad":"bad","neutral":"neutral"}[tone]
        col.markdown(
            f"""
<div class="kpi">
  <div class="label">{label}</div>
  <div class="value">{value}</div>
  <div class="sub"><span class="chip"><span class="dot {dot}"></span>{sub}</span></div>
</div>
""",
            unsafe_allow_html=True,
        )

    card(c1, "Pending", pending, "Needs decision", "warn" if pending else "good")
    card(c2, "Flagged", flagged, "Ambiguous / missing", "warn" if flagged else "good")
    card(c3, "Conflicts", conflicts, "Multi-doc mismatch", "bad" if conflicts else "good")
    card(c4, "Avg confidence", f"{avg_conf:.2f}", "Model signal", "neutral")




def build_table_df(items: List[FlatItem], conflicts: Dict[str, List[str]]) -> pd.DataFrame:
    # invert conflict mapping -> per item key conflict_group_id
    item_to_cg: Dict[str, str] = {}
    for cg_id, keys in conflicts.items():
        for k in keys:
            item_to_cg[k] = cg_id

    rows = []
    for it in items:
        k = item_key(it.company_id, it.docset, it.clause_family, it.item_id)
        flags = list(it.flags)
        if k in item_to_cg and "conflicting" not in flags:
            flags.append("conflicting")

        rows.append({
            "Select": False,
            "_key": k,
            "Company": it.company_id,
            "Docset": it.docset,
            "Type": it.doc_type,
            "Family": it.clause_family,
            "Clause Type": it.clause_type,
            "Beneficiary": it.beneficiary,
            "Trigger": clamp(it.trigger, 60),
            "Effect": clamp(it.effect, 60),
            "Thresholds": it.thresholds or "",
            "Evidence Ref": f'{Path(it.evidence_source_file).name} p.{it.evidence_page or ""} {("Art."+str(it.evidence_article)) if it.evidence_article else ""}'.strip(),
            "Confidence": it.confidence,
            "Flags": flags,
            "Review": it.review_status,
            "Conflict Group": item_to_cg.get(k, ""),
        })
    df = pd.DataFrame(rows)
    return df


def pick_selected_key(df: pd.DataFrame) -> Optional[str]:
    selected = df[df["Select"] == True]
    if len(selected) == 0:
        return None
    # take first selected
    return str(selected.iloc[0]["_key"])


def get_item_by_key(items: List[FlatItem], key: str) -> Optional[FlatItem]:
    for it in items:
        if item_key(it.company_id, it.docset, it.clause_family, it.item_id) == key:
            return it
    return None


def upsert_review_record(
    review_state: Dict[str, Dict[str, Any]],
    it: FlatItem,
    decision: str,
    edited_fields: Dict[str, Any],
    reviewer: str,
    comment: str,
) -> Dict[str, Any]:
    k = item_key(it.company_id, it.docset, it.clause_family, it.item_id)
    prev = review_state.get(k, {})
    prev_status = prev.get("review_status", "pending")

    rec = {
        "_key": k,
        "company_id": it.company_id,
        "docset": it.docset,
        "doc_type": it.doc_type,
        "clause_family": it.clause_family,
        "item_id": it.item_id,

        # store current "business" fields (may be edited)
        "present": it.present,
        "clause_type": edited_fields.get("clause_type", it.clause_type),
        "beneficiary": edited_fields.get("beneficiary", it.beneficiary),
        "trigger": edited_fields.get("trigger", it.trigger),
        "effect": edited_fields.get("effect", it.effect),
        "thresholds": edited_fields.get("thresholds", it.thresholds),

        # evidence is immutable + always stored for audit
        "evidence": {
            "quote": it.evidence_quote,
            "source_file": it.evidence_source_file,
            "page": it.evidence_page,
            "article": it.evidence_article,
        },

        "confidence": it.confidence,
        "flags": it.flags,

        "review_status": decision,
        "review_comment": comment or "",
        "reviewer": reviewer,
        "review_timestamp": now_iso(),

        "previous_review_status": prev_status,
    }

    review_state[k] = rec
    return rec


def validate_decision(it: FlatItem, decision: str, edited_fields: Dict[str, Any], comment: str, conflicts_for_key: bool) -> Optional[str]:
    """
    Return error string if invalid, else None.
    """
    if decision in {"rejected"} and not (comment or "").strip():
        return "Reject requires a comment (why you rejected it)."

    if decision in {"accepted", "edited"}:
        # present=true requires evidence.quote + source_file
        if it.present:
            if not it.evidence_quote.strip() or not it.evidence_source_file.strip():
                return "This item is present=true but has missing evidence. You cannot accept it."
        # if conflicts exist, require edited (or reject) to force explicit resolution
        if conflicts_for_key and decision == "accepted":
            return "This item belongs to a conflict group. Please use 'Accept with edits' (or Reject) to resolve explicitly."

    if decision == "edited":
        # if edited, make sure editable fields aren't empty nonsense
        if not (edited_fields.get("clause_type") or "").strip():
            return "Edited item must have a clause_type."
    return None



# ----------------------------
# Streamlit App
# ----------------------------

def main() -> None:
    st.set_page_config(page_title="Step 6 — Review UI (HITL)", layout="wide")
    st.markdown(
    """
    <style>
    /* ---------- Global ---------- */
    :root{
    --bg: #0B0F14;
    --panel: rgba(255,255,255,0.04);
    --panel2: rgba(255,255,255,0.06);
    --border: rgba(255,255,255,0.08);
    --text: rgba(255,255,255,0.92);
    --muted: rgba(255,255,255,0.62);
    --accent: #7C3AED;         /* violet */
    --good: #10B981;           /* green */
    --warn: #F59E0B;           /* amber */
    --bad:  #EF4444;           /* red */
    --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    }

    html, body, [data-testid="stAppViewContainer"]{
    background: radial-gradient(1200px 900px at 10% 10%, rgba(124,58,237,0.12), transparent 55%),
                radial-gradient(900px 700px at 90% 20%, rgba(16,185,129,0.09), transparent 55%),
                var(--bg) !important;
    color: var(--text) !important;
    }

    [data-testid="stHeader"]{ background: transparent !important; }
    [data-testid="stToolbar"]{ opacity: .35; }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"]{
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02)) !important;
    border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] * { color: var(--text); }

    /* ---------- Typography ---------- */
    h1, h2, h3 { letter-spacing: -0.02em; }
    p, label, span, div { color: var(--text); }
    small, .muted { color: var(--muted) !important; }

    /* ---------- Panels ---------- */
    .panel{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 16px 16px;
    box-shadow: 0 10px 35px rgba(0,0,0,0.35);
    }
    .panel.tight{ padding: 12px 12px; }
    .panel2{
    background: var(--panel2);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 16px 16px;
    }

    /* ---------- KPI cards ---------- */
    .kpi{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 14px 14px;
    }
    .kpi .label{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }
    .kpi .value{ font-size: 24px; font-weight: 650; margin-top: 6px; }
    .kpi .sub{ color: var(--muted); font-size: 12px; margin-top: 4px; }

    /* ---------- Chips / badges ---------- */
    .chip{
    display:inline-flex;
    align-items:center;
    gap:8px;
    padding: 4px 10px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: rgba(255,255,255,0.04);
    font-size: 12px;
    color: var(--muted);
    }
    .dot{ width:8px; height:8px; border-radius:999px; display:inline-block; }
    .dot.good{ background: var(--good); }
    .dot.warn{ background: var(--warn); }
    .dot.bad{ background: var(--bad); }
    .dot.neutral{ background: rgba(255,255,255,0.35); }

    /* ---------- Buttons ---------- */
    .stButton>button{
    border-radius: 14px !important;
    border: 1px solid var(--border) !important;
    background: rgba(255,255,255,0.04) !important;
    color: var(--text) !important;
    padding: 10px 12px !important;
    }
    .stButton>button:hover{
    border-color: rgba(124,58,237,0.55) !important;
    background: rgba(124,58,237,0.12) !important;
    }

    /* ---------- Inputs ---------- */
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stSelectbox"] div[role="combobox"]{
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    }

    /* ---------- Data editor/table ---------- */
    [data-testid="stDataFrame"], [data-testid="stDataEditor"]{
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    overflow: hidden;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


    st.title("Review UI & Human-in-the-Loop Validation")
    st.caption("Data source: extraction.jsonl in data/out-fake/<company>/<docset>/")

    st.markdown(
        """
    <div class="panel">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:16px;">
        <div>
        <div style="font-size:36px;font-weight:750;line-height:1.1;letter-spacing:-0.03em;">
            Review UI & Human-in-the-Loop Validation
        </div>
        <div class="muted" style="margin-top:10px;">
            Data source: <span style="font-family:var(--mono);">extraction.jsonl</span> in <span style="font-family:var(--mono);">data/out-fake/&lt;company&gt;/&lt;docset&gt;/</span>
        </div>
        </div>
        <div style="display:flex;flex-direction:column;gap:8px;align-items:flex-end;">
        <span class="chip"><span class="dot neutral"></span> Demo mode</span>
        <span class="chip"><span class="dot good"></span> Evidence-first</span>
        <span class="chip"><span class="dot warn"></span> Conflicts surfaced</span>
        </div>
    </div>
    </div>
    """,
        unsafe_allow_html=True
    )

    out_root = Path(st.sidebar.text_input("Out root", value=str(DEFAULT_OUT_ROOT)))
    reviewer = st.sidebar.text_input("Reviewer name", value=os.getenv("USER", "reviewer"))

    companies = discover_companies(out_root)
    if not companies:
        st.error(f"No companies found under {out_root}. Did you generate out-fake?")
        st.stop()

    company = st.sidebar.selectbox("Company", companies)
    docsets = discover_docsets(out_root, company)
    docset_filter = st.sidebar.selectbox("Docset", ["ALL"] + docsets)

    # Load items
    items = load_flat_items(out_root, company, docset_filter)

    if not items:
        st.warning("No extraction items found (missing extraction.jsonl?).")
        st.stop()

    # Merge review state
    review_state = load_review_state(out_root, company)
    merge_review(items, review_state)

    # Compute conflicts & inject conflicting flag in display
    conflict_groups = compute_conflicts(items)
    conflict_keys = set(k for keys in conflict_groups.values() for k in keys)

    # Build DF
    df = build_table_df(items, conflict_groups)

    # KPIs
    render_kpis(df)

    # Filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Triage filters")
    only_pending = st.sidebar.checkbox("Only pending", value=True)
    only_flagged = st.sidebar.checkbox("Only flagged", value=False)
    max_conf = st.sidebar.slider("Max confidence", min_value=0.0, max_value=1.0, value=1.0, step=0.01)

    fams = ["ALL"] + sorted(df["Family"].unique().tolist())
    fam_filter = st.sidebar.selectbox("Clause family", fams)

    view = df.copy()
    if only_pending:
        view = view[view["Review"] == "pending"]
    if only_flagged:
        view = view[view["Flags"].apply(lambda x: len(x) > 0)]
    view = view[view["Confidence"] <= max_conf]
    if fam_filter != "ALL":
        view = view[view["Family"] == fam_filter]

    # Sort: conflicts first, then low confidence
    view = view.sort_values(
        by=["Conflict Group", "Confidence"],
        ascending=[False, True],
        kind="mergesort"
    ).reset_index(drop=True)

    st.subheader("Portfolio rights table")
    st.write("Select one row (checkbox) to review it in the detail panel below.")

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Portfolio rights table")
    st.write("Select one row (checkbox) to review it in the detail panel below.")
    st.markdown('<div class="panel2">', unsafe_allow_html=True)
    st.subheader("Detail review panel")
    # ... evidence + fields + decisions
    st.markdown("</div>", unsafe_allow_html=True)

    # ... your st.data_editor(...)
    st.markdown("</div>", unsafe_allow_html=True)

    edited_df = st.data_editor(
        view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Select": st.column_config.CheckboxColumn(required=False),
            "_key": st.column_config.TextColumn(disabled=True),
            "Flags": st.column_config.ListColumn(),
            "Confidence": st.column_config.NumberColumn(format="%.2f"),
        },
        disabled=["_key", "Company", "Docset", "Type", "Family", "Evidence Ref", "Confidence", "Flags", "Review", "Conflict Group"],
        key="table_editor",
    )

    selected_key = pick_selected_key(edited_df)
    st.divider()

    st.subheader("Detail review panel")
    if not selected_key:
        st.info("Select a row above to review.")
        return

    it = get_item_by_key(items, selected_key)
    if not it:
        st.error("Selected item not found (internal key mismatch).")
        return

    in_conflict = selected_key in conflict_keys

    # Evidence box
    st.markdown("### Evidence")
    ev_ref = f"**{Path(it.evidence_source_file).name}**"
    if it.evidence_page:
        ev_ref += f", p.{it.evidence_page}"
    if it.evidence_article:
        ev_ref += f", Art.{it.evidence_article}"
    st.write(ev_ref)
    st.code(it.evidence_quote or "(no evidence quote)", language="text")

    # Conflict preview
    if in_conflict:
        st.warning("This item is in a **conflict group** (multiple docsets disagree). Review below before deciding.")
        cg = edited_df.loc[edited_df["_key"] == selected_key, "Conflict Group"].iloc[0]
        if cg:
            st.caption(f"Conflict group: {cg}")
            # show other items in same conflict group
            other_keys = [k for k in conflict_groups.get(cg, []) if k != selected_key]
            if other_keys:
                st.markdown("**Other conflicting items:**")
                for ok in other_keys:
                    oit = get_item_by_key(items, ok)
                    if not oit:
                        continue
                    st.write(f"- Docset: `{oit.docset}` | Thresholds: `{oit.thresholds or ''}` | Confidence: `{oit.confidence:.2f}`")
                    st.code(oit.evidence_quote or "(no evidence quote)", language="text")

    # Editable fields
    st.markdown("### Structured fields")
    c1, c2 = st.columns(2)
    with c1:
        clause_type = st.text_input("clause_type", value=it.clause_type)
        beneficiary = st.selectbox("beneficiary", ["investor", "founders", "company", "mixed", "unclear"], index=["investor","founders","company","mixed","unclear"].index(it.beneficiary) if it.beneficiary in ["investor","founders","company","mixed","unclear"] else 4)
        thresholds = st.text_input("thresholds", value=it.thresholds or "")
    with c2:
        present = st.checkbox("present (locked)", value=it.present, disabled=True)
        confidence = st.number_input("confidence (locked)", value=float(it.confidence), disabled=True)

    trigger = st.text_area("trigger", value=it.trigger, height=80)
    effect = st.text_area("effect", value=it.effect, height=80)

    st.markdown("### Decision")
    decision_col1, decision_col2, decision_col3 = st.columns(3)

    comment = st.text_input("Comment (required for Reject; recommended for Edit)", value=it.review_comment or "")

    edited_fields = {
        "clause_type": clause_type,
        "beneficiary": beneficiary,
        "thresholds": thresholds.strip() or None,
        "trigger": trigger,
        "effect": effect,
    }

    def handle(decision: str) -> None:
        err = validate_decision(it, decision, edited_fields, comment, conflicts_for_key=in_conflict)
        if err:
            st.error(err)
            return

        rec = upsert_review_record(
            review_state=review_state,
            it=it,
            decision=decision,
            edited_fields=edited_fields,
            reviewer=reviewer,
            comment=comment,
        )
        save_review_state(out_root, company, review_state)

        append_audit(out_root, company, {
            "event": "review_decision",
            "timestamp": now_iso(),
            "reviewer": reviewer,
            "key": rec["_key"],
            "company_id": company,
            "from": rec.get("previous_review_status", "pending"),
            "to": decision,
            "comment": comment or "",
        })

        st.success(f"Saved decision: {decision}")
        st.rerun()

    with decision_col1:
        if st.button("✅ Accept as-is", use_container_width=True):
            handle("accepted")

    with decision_col2:
        if st.button("✏️ Accept with edits", use_container_width=True):
            handle("edited")

    with decision_col3:
        if st.button("❌ Reject", use_container_width=True):
            handle("rejected")

    st.divider()
    st.subheader("Export")

    # export only accepted/edited
    export_rows = []
    for rec in review_state.values():
        if rec.get("review_status") in {"accepted", "edited"}:
            export_rows.append({
                "Company": rec.get("company_id"),
                "Docset": rec.get("docset"),
                "DocType": rec.get("doc_type"),
                "ClauseFamily": rec.get("clause_family"),
                "ClauseType": rec.get("clause_type"),
                "Beneficiary": rec.get("beneficiary"),
                "Trigger": rec.get("trigger"),
                "Effect": rec.get("effect"),
                "Thresholds": rec.get("thresholds"),
                "EvidenceQuote": (rec.get("evidence") or {}).get("quote"),
                "SourceFile": (rec.get("evidence") or {}).get("source_file"),
                "Page": (rec.get("evidence") or {}).get("page"),
                "Article": (rec.get("evidence") or {}).get("article"),
                "Confidence": rec.get("confidence"),
                "Flags": ",".join(rec.get("flags") or []),
                "ReviewedBy": rec.get("reviewer"),
                "ReviewedAt": rec.get("review_timestamp"),
            })

    export_df = pd.DataFrame(export_rows)
    st.write(f"Reviewed rows ready for export: **{len(export_df)}**")

    if len(export_df):
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name=f"{company}_reviewed_export.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("No accepted/edited rows yet. Review items first.")


if __name__ == "__main__":
    main()
