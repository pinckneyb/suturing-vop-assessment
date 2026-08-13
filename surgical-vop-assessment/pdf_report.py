#!/usr/bin/env python3
"""
Per-student PDF report generation for the Vascular Anastomosis VOP Assessment.

Pure functions: a canonical record dict -> PDF bytes (no Streamlit / IO side
effects), so the output is easily testable. Uses reportlab platypus.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)

try:
    # Reuse the canonical item labels / core-item set from the scoring module.
    from gemini_vision_client import ITEM_LABELS, CORE_PROFICIENCY_ITEMS
except Exception:  # pragma: no cover - defensive fallback for standalone use
    ITEM_LABELS = {
        1: "Oblique incision 2x graft lumen with #11 blade",
        2: "Avoids back wall injury to artery",
        3: "Cuts end of graft into spatula shape",
        4: "Places double-ended suture at heel of anastomosis",
        5: "Securely ties knot outside anastomosis at heel",
        6: "Uses right angle to pass suture to opposite side of heel",
        7: "Completes back wall (sutures ~1 mm apart, no gaping)",
        8: "Trims toe of graft to appropriate size at apex",
        9: "Completes front wall (sutures ~1 mm apart, no gaping)",
        10: "Securely ties knot outside anastomosis",
        11: "Economy of Time and Motion",
        12: "Final Rating / Demonstrates Proficiency",
        13: "Other Summative Comments",
    }
    CORE_PROFICIENCY_ITEMS = {2, 7, 9, 10}


# ---- Palette -----------------------------------------------------------------
COLOR_YES = colors.HexColor("#2e7d32")
COLOR_NO = colors.HexColor("#c62828")
COLOR_NULL = colors.HexColor("#616161")
COLOR_HEADER = colors.HexColor("#16213e")
COLOR_ACCENT = colors.HexColor("#0d47a1")
COLOR_COACHING = colors.HexColor("#1565c0")
COLOR_MASTERY = colors.HexColor("#2e7d32")
COLOR_ROW_ALT = colors.HexColor("#f2f5fb")
COLOR_GRID = colors.HexColor("#c5cbd6")


def _score_color(score: Any) -> colors.Color:
    if score == "YES":
        return COLOR_YES
    if score == "NO":
        return COLOR_NO
    if score == "NULL" or score is None:
        return COLOR_NULL
    return COLOR_HEADER


def _score_label(score: Any) -> str:
    if score in ("YES", "NO", "NULL"):
        return str(score)
    if score is None:
        return "--"
    return str(score)


def _styles() -> Dict[str, ParagraphStyle]:
    ss = getSampleStyleSheet()
    styles: Dict[str, ParagraphStyle] = {}
    styles["title"] = ParagraphStyle(
        "vopTitle", parent=ss["Title"], fontSize=17, leading=21,
        textColor=COLOR_HEADER, spaceAfter=2,
    )
    styles["subtitle"] = ParagraphStyle(
        "vopSubtitle", parent=ss["Normal"], fontSize=10, leading=13,
        textColor=colors.HexColor("#5a6472"), spaceAfter=2,
    )
    styles["h2"] = ParagraphStyle(
        "vopH2", parent=ss["Heading2"], fontSize=13, leading=16,
        textColor=COLOR_ACCENT, spaceBefore=12, spaceAfter=6,
    )
    styles["body"] = ParagraphStyle(
        "vopBody", parent=ss["Normal"], fontSize=9, leading=12, alignment=TA_LEFT,
    )
    styles["cell"] = ParagraphStyle(
        "vopCell", parent=ss["Normal"], fontSize=8.5, leading=11, alignment=TA_LEFT,
    )
    styles["cell_head"] = ParagraphStyle(
        "vopCellHead", parent=ss["Normal"], fontSize=9, leading=11,
        textColor=colors.white, alignment=TA_LEFT,
    )
    styles["coaching"] = ParagraphStyle(
        "vopCoaching", parent=ss["Normal"], fontSize=8.5, leading=11,
        textColor=COLOR_COACHING,
    )
    styles["mastery"] = ParagraphStyle(
        "vopMastery", parent=ss["Normal"], fontSize=8.5, leading=11,
        textColor=COLOR_MASTERY,
    )
    styles["stat"] = ParagraphStyle(
        "vopStat", parent=ss["Normal"], fontSize=10, leading=14,
    )
    return styles


def _esc(text: Any) -> str:
    """Escape text for reportlab paragraph mini-markup."""
    s = "" if text is None else str(text)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _get_item(items: List[Dict[str, Any]], item_id: int) -> Dict[str, Any]:
    for it in items:
        if isinstance(it, dict) and it.get("item_id") == item_id:
            return it
    return {}


def _proficiency_display(proficiency: str) -> tuple[str, colors.Color]:
    if proficiency == "PROFICIENT":
        return "PROFICIENT", COLOR_YES
    if proficiency == "NOT_PROFICIENT":
        return "NOT PROFICIENT", COLOR_NO
    return "INSUFFICIENT EVIDENCE", colors.HexColor("#e65100")


def build_pdf_report(record: Dict[str, Any]) -> bytes:
    """Render a single candidate record to a formatted PDF (returns bytes).

    Pure function: no filesystem or network side effects. Legacy records
    (record['is_legacy']) are still rendered with a minimal notice.
    """
    styles = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        title="VOP Assessment Report",
    )
    story: List[Any] = []

    candidate_id = record.get("candidate_id", "") or "Unknown"
    video_id = record.get("video_id", "") or record.get("case_id", "") or "Unknown"
    ts = record.get("timestamp", "")
    date_str = _format_date(ts)

    # ---- Header --------------------------------------------------------------
    story.append(Paragraph("Vascular Anastomosis VOP Assessment &mdash; 2023 Rubric", styles["title"]))
    story.append(Paragraph(
        f"Candidate: <b>{_esc(candidate_id)}</b> &nbsp;&bull;&nbsp; Video: {_esc(video_id)} "
        f"&nbsp;&bull;&nbsp; Date: {_esc(date_str)}",
        styles["subtitle"],
    ))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.2, color=COLOR_ACCENT))
    story.append(Spacer(1, 8))

    if record.get("is_legacy"):
        story.append(Paragraph(
            "This is a legacy (pre-2023) record and cannot be rendered under the "
            "current 13-item rubric.", styles["body"]))
        doc.build(story)
        return buf.getvalue()

    items = record.get("items", [])

    # ---- Summary stats block -------------------------------------------------
    story.extend(_summary_block(record, styles))

    # ---- Checklist items 1-10 ------------------------------------------------
    story.append(Paragraph("Checklist Items (1&ndash;10)", styles["h2"]))
    story.append(_checklist_table(items, styles))

    # ---- Economy (item 11) ---------------------------------------------------
    story.append(Paragraph("Economy of Time and Motion (Item 11)", styles["h2"]))
    story.extend(_economy_block(_get_item(items, 11), record.get("economy_score"), styles))

    # ---- Proficiency (item 12) ----------------------------------------------
    story.append(Paragraph("Final Rating / Proficiency (Item 12)", styles["h2"]))
    story.extend(_proficiency_block(record, styles))

    # ---- Summative comments (item 13) ---------------------------------------
    story.append(Paragraph("Summative Comments (Item 13)", styles["h2"]))
    story.extend(_comments_block(_get_item(items, 13), record.get("coaching_tags", []), styles))

    doc.build(story)
    return buf.getvalue()


def _format_date(ts: Any) -> str:
    if not ts:
        return "N/A"
    try:
        return datetime.fromisoformat(str(ts)).strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return str(ts)


def _summary_block(record: Dict[str, Any], styles) -> List[Any]:
    checklist_yes = record.get("checklist_yes_count", 0)
    null_count = record.get("null_count", 0)
    economy = record.get("economy_score")
    coverage = record.get("coverage", {}) or {}
    cov_pct = coverage.get("observed_percent", 0)
    proficiency = record.get("proficiency", "INSUFFICIENT_EVIDENCE")
    prof_text, prof_color = _proficiency_display(proficiency)
    rationale = record.get("proficiency_rationale", "") or ""

    econ_display = f"{economy} / 5" if isinstance(economy, int) else "-- / 5"
    null_note = f" ({null_count} not observed)" if null_count else ""

    stat_rows = [
        [
            Paragraph("<b>Checklist YES</b>", styles["cell"]),
            Paragraph(f"{checklist_yes} / 10{null_note}", styles["cell"]),
            Paragraph("<b>Economy</b>", styles["cell"]),
            Paragraph(econ_display, styles["cell"]),
        ],
        [
            Paragraph("<b>Coverage</b>", styles["cell"]),
            Paragraph(f"{cov_pct:.0f}%", styles["cell"]),
            Paragraph("<b>Proficiency</b>", styles["cell"]),
            Paragraph(
                f'<font color="#{prof_color.hexval()[2:]}"><b>{prof_text}</b></font>',
                styles["cell"],
            ),
        ],
    ]
    tbl = Table(stat_rows, colWidths=[1.3 * inch, 1.9 * inch, 1.3 * inch, 2.6 * inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), COLOR_ROW_ALT),
        ("BOX", (0, 0), (-1, -1), 0.75, COLOR_GRID),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    out: List[Any] = [tbl]
    if rationale:
        out.append(Spacer(1, 4))
        out.append(Paragraph(f"<b>Rationale:</b> {_esc(rationale)}", styles["body"]))

    core_obs = coverage.get("core_observed", {}) or {}
    if core_obs:
        badges = []
        for iid in ["2", "7", "9", "10"]:
            observed = core_obs.get(iid, False)
            col = "#2e7d32" if observed else "#616161"
            badges.append(
                f'<font color="{col}">&#9632; Item {iid}: '
                f'{"observed" if observed else "not observed"}</font>'
            )
        out.append(Spacer(1, 3))
        out.append(Paragraph("<b>Core domains:</b> " + " &nbsp; ".join(badges), styles["body"]))
    out.append(Spacer(1, 2))
    return out


def _checklist_table(items: List[Dict[str, Any]], styles) -> Table:
    header = [
        Paragraph("<b>#</b>", styles["cell_head"]),
        Paragraph("<b>Item</b>", styles["cell_head"]),
        Paragraph("<b>Score</b>", styles["cell_head"]),
        Paragraph("<b>Evidence &amp; Coaching</b>", styles["cell_head"]),
    ]
    data = [header]
    coaching_row_idxs: List[int] = []

    for i in range(1, 11):
        entry = _get_item(items, i)
        score = entry.get("score")
        label = ITEM_LABELS.get(i, f"Item {i}")
        evidence = entry.get("evidence", "")
        if isinstance(evidence, dict):
            evidence = "; ".join(f"{k}: {v}" for k, v in evidence.items())
        coaching = entry.get("coaching", "")

        core_tag = " <font color='#c62828'>[core]</font>" if i in CORE_PROFICIENCY_ITEMS else ""
        sc_col = _score_color(score).hexval()[2:]

        detail = f"{_esc(evidence)}" if evidence else "<i>No evidence recorded.</i>"
        if i == 6:
            subs = entry.get("subitems", {}) or {}
            sa = subs.get("6a_right_angle_method", {})
            sb = subs.get("6b_safe_transfer_outcome", {})
            if sa or sb:
                detail += (
                    f"<br/><b>6a Method:</b> {_esc(sa.get('score', '--'))} &nbsp; "
                    f"<b>6b Outcome:</b> {_esc(sb.get('score', '--'))}"
                )
        # Coaching only for NO (observed failure); NULL shows a not-observed note.
        if score == "NO" and coaching:
            detail += f"<br/><font color='#1565c0'>&#8594; <b>Coaching:</b> {_esc(coaching)}</font>"
        elif score == "NULL":
            detail += "<br/><font color='#8a8f98'><i>Not observed on video</i></font>"

        row = [
            Paragraph(str(i), styles["cell"]),
            Paragraph(f"{_esc(label)}{core_tag}", styles["cell"]),
            Paragraph(f'<font color="#{sc_col}"><b>{_score_label(score)}</b></font>', styles["cell"]),
            Paragraph(detail, styles["cell"]),
        ]
        data.append(row)
        if score == "NO":
            coaching_row_idxs.append(len(data) - 1)

    tbl = Table(data, colWidths=[0.3 * inch, 2.4 * inch, 0.7 * inch, 3.7 * inch], repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), COLOR_HEADER),
        ("BOX", (0, 0), (-1, -1), 0.75, COLOR_GRID),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, COLOR_GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    for r in range(1, len(data)):
        if r % 2 == 0:
            style.append(("BACKGROUND", (0, r), (-1, r), COLOR_ROW_ALT))
    tbl.setStyle(TableStyle(style))
    return tbl


def _economy_block(entry: Dict[str, Any], economy_score: Any, styles) -> List[Any]:
    out: List[Any] = []
    score = entry.get("score", economy_score)
    evidence = entry.get("evidence", {}) or {}
    coaching = entry.get("coaching", "")

    if score == "NULL" or score is None:
        out.append(Paragraph(
            '<font color="#616161"><b>Score: NULL</b></font> '
            "(not observed; does not gate proficiency)", styles["body"]))
    else:
        col = _score_color(5 if score == 5 else "NO").hexval()[2:] if not isinstance(score, int) else None
        # Colour scale by score value.
        econ_colors = {1: "#c62828", 2: "#e65100", 3: "#f9a825", 4: "#7cb342", 5: "#2e7d32"}
        col = econ_colors.get(score if isinstance(score, int) else 0, "#616161")
        out.append(Paragraph(
            f'<font color="{col}"><b>Score: {score} / 5</b></font> '
            "(out of 5; does not gate proficiency)", styles["body"]))

    if isinstance(evidence, dict):
        flow = evidence.get("flow_organization", "")
        ei = evidence.get("economy_index")
        wasted = evidence.get("wasted_motion_events", []) or []
        meta = []
        if flow and flow != "unknown":
            meta.append(f"<b>Flow:</b> {_esc(flow)}")
        if ei is not None:
            try:
                meta.append(f"<b>Economy Index:</b> {float(ei):.1f}")
            except (ValueError, TypeError):
                meta.append(f"<b>Economy Index:</b> {_esc(ei)}")
        if meta:
            out.append(Spacer(1, 2))
            out.append(Paragraph(" &nbsp;&bull;&nbsp; ".join(meta), styles["body"]))
        if wasted:
            out.append(Spacer(1, 2))
            out.append(Paragraph(f"<b>Wasted-motion events ({len(wasted)}):</b>", styles["body"]))
            for idx, evt in enumerate(wasted, 1):
                if isinstance(evt, dict):
                    etype = evt.get("type", "other")
                    count = evt.get("count_estimate", 1)
                    note = evt.get("note", "")
                    line = f"{idx}. {_esc(etype)} (x{_esc(count)})"
                    if note:
                        line += f": {_esc(note)}"
                else:
                    line = f"{idx}. {_esc(evt)}"
                out.append(Paragraph(line, styles["cell"]))

    if coaching:
        out.append(Spacer(1, 3))
        if score == 5:
            out.append(Paragraph(f"&#9733; <b>{_esc(coaching)}</b>", styles["mastery"]))
        else:
            out.append(Paragraph(f"&#8594; <b>Coaching:</b> {_esc(coaching)}", styles["coaching"]))
    return out


def _proficiency_block(record: Dict[str, Any], styles) -> List[Any]:
    out: List[Any] = []
    proficiency = record.get("proficiency", "INSUFFICIENT_EVIDENCE")
    prof_text, prof_color = _proficiency_display(proficiency)
    out.append(Paragraph(
        f'<font color="#{prof_color.hexval()[2:]}"><b>{prof_text}</b></font>', styles["body"]))

    red_lines = record.get("red_lines_triggered", []) or []
    missing = record.get("missing_core_domains", []) or []
    if red_lines:
        out.append(Spacer(1, 3))
        out.append(Paragraph("<b>Red-line failures:</b>", styles["body"]))
        for rl in red_lines:
            out.append(Paragraph(f'<font color="#c62828">&#9888; {_esc(rl)}</font>', styles["cell"]))
    if missing:
        out.append(Spacer(1, 3))
        out.append(Paragraph("<b>Missing core domains:</b>", styles["body"]))
        for mc in missing:
            out.append(Paragraph(f'<font color="#e65100">&#9888; {_esc(mc)}</font>', styles["cell"]))
    if not red_lines and not missing:
        out.append(Paragraph("No red-line failures or missing core domains.", styles["cell"]))
    return out


def _comments_block(entry: Dict[str, Any], coaching_tags: List[str], styles) -> List[Any]:
    out: List[Any] = []
    # Item 13's summative comment is carried in its "score" (free text) or evidence.
    comment = entry.get("score", "")
    if isinstance(comment, str) and comment.strip() and comment not in ("YES", "NO", "NULL"):
        out.append(Paragraph(_esc(comment), styles["body"]))
    else:
        evidence = entry.get("evidence", {})
        text = ""
        if isinstance(evidence, dict):
            text = evidence.get("summary") or evidence.get("comment") or ""
        elif isinstance(evidence, str):
            text = evidence
        out.append(Paragraph(_esc(text) if text else "<i>No summative comments recorded.</i>", styles["body"]))

    # Coaching tags may live on the record or on item 13's evidence.
    tags = list(coaching_tags or [])
    ev = entry.get("evidence", {})
    if isinstance(ev, dict):
        for t in ev.get("coaching_tags", []) or []:
            if t not in tags:
                tags.append(t)
    if tags:
        out.append(Spacer(1, 4))
        badges = " &nbsp; ".join(
            f'<font color="#0d47a1">&#9642; {_esc(t)}</font>' for t in tags
        )
        out.append(Paragraph(f"<b>Coaching tags:</b> {badges}", styles["body"]))
    return out


def candidate_pdf_filename(record: Dict[str, Any]) -> str:
    """Suggested filename for a candidate's PDF report."""
    cid = str(record.get("candidate_id", "") or "unknown")
    vid = str(record.get("video_id", "") or record.get("case_id", "") or "")
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in f"{cid}_{vid}".strip("_"))
    return f"vop_report_{safe or 'candidate'}.pdf"
