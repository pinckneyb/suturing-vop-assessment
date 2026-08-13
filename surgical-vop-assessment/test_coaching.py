"""Tests for the per-item coaching contract (items 1-10).

Run with: python test_coaching.py  (or pytest test_coaching.py)
"""

from gemini_vision_client import (
    enforce_item_coaching,
    ITEM_COACHING_FALLBACKS,
    _build_v4_record,
)


def _item(item_id, score, coaching=None, observability=None):
    e = {
        "item_id": item_id,
        "score": score,
        "observability": observability or ("NOT_OBSERVED" if score == "NULL" else "OBSERVED"),
        "evidence": f"evidence for item {item_id}",
    }
    if coaching is not None:
        e["coaching"] = coaching
    return e


def test_normal_path_llm_supplied_tips():
    items = [
        _item(1, "NO", coaching="Angle the blade for an oblique cut."),
        _item(2, "YES", coaching="Should be stripped."),
        _item(3, "NULL", coaching="Partially seen; check spatula geometry."),
        _item(4, "NULL"),
    ]
    out = {i["item_id"]: i for i in enforce_item_coaching(items)}
    assert out[1]["coaching"] == "Angle the blade for an oblique cut."
    assert "coaching" not in out[2], "YES items must never carry a tip"
    assert out[3]["coaching"] == "Partially seen; check spatula geometry."
    assert "coaching" not in out[4], "NULL without caveat gets no synthesized tip"


def test_missing_tip_synthesized_for_no():
    items = [_item(i, "NO") for i in range(1, 11)]
    out = enforce_item_coaching(items)
    for e in out:
        assert e["coaching"] == ITEM_COACHING_FALLBACKS[e["item_id"]], (
            f"Item {e['item_id']} scored NO must get a fallback tip"
        )


def test_blank_or_invalid_tip_replaced():
    items = [_item(2, "NO", coaching="   "), _item(7, "NO", coaching=None)]
    items[1]["coaching"] = 123  # non-string
    out = {i["item_id"]: i for i in enforce_item_coaching(items)}
    assert out[2]["coaching"] == ITEM_COACHING_FALLBACKS[2]
    assert out[7]["coaching"] == ITEM_COACHING_FALLBACKS[7]


def test_non_checklist_items_never_carry_coaching():
    # Items 12/13 never carry coaching; item 11 has its own economy contract.
    items = [
        {"item_id": 12, "score": "NOT_PROFICIENT", "observability": "DERIVED", "evidence": {}, "coaching": "nope"},
        {"item_id": 13, "score": "summary", "observability": "OBSERVED", "evidence": {}, "coaching": "nope"},
    ]
    out = enforce_item_coaching(items)
    for e in out:
        assert "coaching" not in e


def test_economy_coaching_contract():
    from gemini_vision_client import ECONOMY_MASTERY_TEXT
    items = [
        {"item_id": 11, "score": 5, "observability": "OBSERVED", "evidence": {}, "coaching": "wrong text"},
    ]
    out = enforce_item_coaching(items)
    assert out[0]["coaching"] == ECONOMY_MASTERY_TEXT

    items = [
        {"item_id": 11, "score": 3, "observability": "OBSERVED", "evidence": {}, "coaching": "Reduce instrument repositioning."},
    ]
    out = enforce_item_coaching(items)
    assert out[0]["coaching"] == "Reduce instrument repositioning."

    # Below 5 without a supplied tip -> synthesized recommendation
    items = [
        {"item_id": 11, "score": 2, "observability": "OBSERVED",
         "evidence": {"flow_organization": "disorganized", "wasted_motion_events": [], "economy_index": 0}},
    ]
    out = enforce_item_coaching(items)
    assert isinstance(out[0].get("coaching"), str) and out[0]["coaching"].strip()

    # NULL economy carries no coaching
    items = [
        {"item_id": 11, "score": "NULL", "observability": "NOT_OBSERVED", "evidence": {}, "coaching": "nope"},
    ]
    out = enforce_item_coaching(items)
    assert "coaching" not in out[0]


def test_action_plan_includes_economy_after_core():
    """End-to-end: build a vop_2023_v1 record with core failures and economy
    below 5, and verify the Action Plan lists core items first and includes
    the Item 11 economy coaching."""
    items = [_item(i, "NO" if i in (2, 3, 7) else "YES") for i in range(1, 11)]
    items += [
        {"item_id": 11, "score": 3, "observability": "OBSERVED",
         "evidence": {"flow_organization": "mixed", "wasted_motion_events": [], "economy_index": 4},
         "coaching": "Plan suture sequence to avoid regrasping."},
        {"item_id": 12, "score": "NOT_PROFICIENT", "observability": "DERIVED",
         "evidence": {"red_lines_triggered": [], "missing_core_domains": []}},
        {"item_id": 13, "score": "Summary.", "observability": "OBSERVED",
         "evidence": {"coaching_tags": []}},
    ]
    scored = {"rubric_version": "vop_2023_v1", "evidence_based": {"items": items}}
    record = _build_v4_record({"case_id": "VOP-TEST-03"}, video_id="test", scored_output=scored)

    from app_simple import build_action_plan
    plan = build_action_plan(record["items"])
    plan_ids = [p[0] for p in plan]
    assert plan_ids == [2, 7, 3, 11], f"unexpected plan order: {plan_ids}"
    econ_tip = dict((p[0], p[2]) for p in plan)[11]
    assert econ_tip == "Plan suture sequence to avoid regrasping."
    # every entry carries a non-empty tip
    for iid, label, tip in plan:
        assert isinstance(tip, str) and tip.strip(), f"empty tip for item {iid}"


def test_stage2_failure_deterministic_path_gets_tips():
    """When Stage-2 scoring fails, _build_v4_record falls back to the
    deterministic engine; every NO checklist item must still carry a tip."""
    for_data = {
        "case_id": "VOP-TEST-01",
        # Explicit back-wall injury -> item 2 scored NO deterministically.
        "back_wall": {"catch_observed": True},
    }
    record = _build_v4_record(for_data, video_id="test", scored_output=None)
    assert record["scoring_path"] == "deterministic"
    for entry in record["items"]:
        iid = entry.get("item_id")
        if iid not in range(1, 11):
            assert "coaching" not in entry
            continue
        score = entry.get("score")
        if score == "NO":
            assert isinstance(entry.get("coaching"), str) and entry["coaching"].strip(), (
                f"NO item {iid} missing coaching tip on deterministic path"
            )
        elif score == "YES":
            assert "coaching" not in entry


def test_llm_path_contract_enforced_end_to_end():
    items = [_item(i, "NO", coaching=("Use the fallback." if i == 1 else None)) for i in range(1, 11)]
    items[1]["score"] = "YES"
    items[1]["coaching"] = "should vanish"
    items += [
        {"item_id": 11, "score": 3, "observability": "OBSERVED",
         "evidence": {"flow_organization": "organized", "wasted_motion_events": [], "economy_index": 0}},
        {"item_id": 12, "score": "NOT_PROFICIENT", "observability": "DERIVED",
         "evidence": {"red_lines_triggered": [], "missing_core_domains": []}},
        {"item_id": 13, "score": "Summary.", "observability": "OBSERVED",
         "evidence": {"coaching_tags": []}},
    ]
    scored = {"rubric_version": "vop_2023_v1", "evidence_based": {"items": items}}
    record = _build_v4_record({"case_id": "VOP-TEST-02"}, video_id="test", scored_output=scored)
    assert record["scoring_path"] == "llm_vop_2023_v1"
    out = {e["item_id"]: e for e in record["items"]}
    assert out[1]["coaching"] == "Use the fallback."
    assert "coaching" not in out[2]
    for iid in range(3, 11):
        assert out[iid]["coaching"] == ITEM_COACHING_FALLBACKS[iid]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
