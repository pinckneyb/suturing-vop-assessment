# VOP Retrospective-Aligned Rubric Spec (v5 — Official 2023 Rubric)

## 1. Scope and Intent

**Source:** This spec implements the official **2023 VOP rubric** (13 items) for
vascular end-to-side anastomosis (see `attached_assets/2023_Rubric_*.docx`).
It supersedes the prior 14-item spec: the old Item 4 ("Selects correct suture
5-0 Prolene/Surgipro") has been **removed entirely** and all subsequent items
renumbered.

**Purpose:** Generate outputs that are apples-to-apples with surgeon
retrospective VOP checklists, while producing stable, auditable AI results.
`rubric_version = "vop_2023_v1"`.

### Key Alignment Principles (from surgeon patterns)

A. **Proficiency is driven by red-line failures** (sequence/anchoring and clear
leak/approximation risks), not perfect stylistic form. Toe anchoring can produce
a "Not Proficient" even when other elements are decent.

B. **"Method" items are not always gating.** Surgeons may mark the method "No"
yet still rate overall proficient. Item 6 (right-angle transfer) is explicitly
non-gating unless it creates downstream technical failure.

C. **Spacing is not literal millimeters.** The 2023 rubric describes wall sutures
as "approx 1 mm apart," but this is captured in item *labels* only. Score spacing
functionally by uniformity + approximation + no gaping. **Never** emit numeric
"mm" spacing claims.

D. **Coaching themes are captured as structured tags** (e.g., ARTERIOTOMY_TOO_SHORT,
SUTURE_GRASPED_WITH_FORCEPS_OR_PICKUPS, TIE_DIRECTLY_ON_TOE). Tags are non-gating
unless they trigger a red-line.

## 2. Two-Stage Architecture

### Stage 1: Factual Observation Record (FOR)
- Produced by a video model (Gemini).
- **Purely descriptive:** actions, sequence, visibility. No "good/bad," no scores, no "proficient."
- Hard bans on evaluative language.

### Stage 2: Scoring Engine
- Applies this rubric with app-side validators (LLM scoring + deterministic fallback).
- Maps FOR observations to YES/NO/NULL scores for Items 1–10.
- Derives economy score (Item 11, out of 5) from flow_organization + wasted_motion_events.
- Derives proficiency (Item 12) from red-line rules.
- Generates coaching tags (Item 13) from FOR patterns.

## 3. Global Scoring Policies

### 3.1 Tri-State Visibility Policy
Uses YES / NO / NULL. Unobserved aspects score **NULL** (NOT_OBSERVED); do not
default to YES.

### 3.2 Spacing Policy
Scored by: consistency + approximation + gaping/leak-risk, **not by millimeters**.
Hard ban: no numeric "mm" spacing claims (the "~1 mm apart" phrasing lives in labels only).

### 3.3 Technique Equivalence Policy
Items phrased as a specific instrument/method (Items 4, 6, 8) are scored by
intent/outcome and may be satisfied by equivalent technique, unless a red-line
failure is triggered.

## 4. Checklist Items 1–13: Operational Definitions

### Items 1–3 (setup steps; often off-screen)

**Item 1: Oblique incision in artery 2x diameter of graft lumen with #11 blade**
- YES if observed with linear/oblique shape, or result visible later showing a clean oblique cut.

**Item 2: Avoids back wall injury to "artery"** *(core)*
- YES unless clear evidence of posterior wall catch/through-and-through injury.

**Item 3: Cuts end of graft into a spatula shape**
- YES if spatulated shape visible (pointed toe, rounded heel), whether or not the cut is seen on camera.

### Items 4–6 (heel strategy; equivalence enabled)

**Item 4: Correctly places double-ended suture at heel of anastomosis**
- YES if heel correctly established and stable, regardless of single- vs double-armed.
- NO only if anchoring at toe (red-line E) or heel alignment wrong/unstable.

**Item 5: Securely ties knot on outside of anastomosis at heel**
- YES if knot is outside/extraluminal and appears seated.

**Item 6: Uses right angle to pass one end of the suture to opposite side of heel** (NON-GATING)
Split into subitems; overall score = 6b (outcome).
- `6a_right_angle_method`: right-angle instrument use.
- `6b_safe_transfer_outcome`: no twist/crossing, no snag/tear, progression achieved.
- **Note: Item 6 is non-gating unless it creates downstream technical failure.**

### Items 7–10 (product/outcome; most predictive)

**Item 7: Completes back wall of anastomosis (sutures approx 1 mm apart)** *(core; functional spacing rule)*
- YES if completed with uniform/mixed spacing, good approximation, no gaping.
- NO if high-confidence gaping/malapposition/leak risk.

**Item 8: Trims toe of graft to appropriate size at the apex of anastomosis**
- YES if fit appropriate (no bunching/dog-ear/shortage).
- NO only if fit clearly inappropriate.

**Item 9: Completes front wall of anastomosis (sutures approx 1 mm apart)** *(core; same spacing rule as Item 7)*
- YES if uniform/mixed + well-approximated + no gaping.
- NO if high-confidence gaps/edge mismatch/gross irregularity.

**Item 10: Securely ties knot on outside of anastomosis** *(core)*
- YES if knot outside and appears seated/secure.

### Item 11: Economy of Time and Motion (out of 5)

Economy ≠ speed. Do not penalize careful slowness unless it produces wasted motion.

Rubric anchors:
- 1 = "Many unnecessary/disorganized movements"
- 3 = "Organized time/motion, some unnecessary movements"
- 5 = "Maximum economy of movement and efficiency"

Derived from FOR economy_markers via an **Economy Index (EI)** = Σ(weight × count):
- instrument_search = 2, pause_reset = 2, failed_pass_sequence = 2,
  excess_regrasp_cluster = 1, other = 1.

EI → score mapping (out of 5):
- EI 0 **and** flow_organization = organized → **5**
- EI 0–1 → **4**
- EI 2–3 → **3**
- EI 4–6 → **2**
- EI ≥7 → **1**
- No markers at all → NULL (NOT_OBSERVED).

Economy is **non-gating** (never affects proficiency). No per-minute normalization.

### Item 12: Final Rating / Demonstrates Proficiency (DERIVED in code)

Core domains: **Items 2, 7, 9, 10**.
- NOT_PROFICIENT if ANY core domain is observed NO, or any red-line is triggered.
- INSUFFICIENT_EVIDENCE if ANY core domain is NULL (not observed).
- PROFICIENT otherwise.

| Red-line | Trigger |
|----------|---------|
| RL-A | Back wall gaping/malapposition (Item 7 = NO) |
| RL-B | Front wall gaping/malapposition (Item 9 = NO) |
| RL-C | Back wall injury (Item 2 = NO) |
| RL-D | Intraluminal/insecure final knot (Item 10 = NO) |
| RL-E | Heel establishment failure (Item 4 = NO, observed consequence) |

Item 6 and Item 11 are non-gating.

### Item 13: Other Summative Comments + Coaching Tags

Output:
A. One-sentence outcome summary.
B. 1–3 coaching tags from taxonomy.

## 5. Coaching Tags Taxonomy

| Tag | Description |
|-----|-------------|
| ARTERIOTOMY_TOO_SHORT | Arteriotomy appears short |
| SPATULATION_CONCAVE_GEOMETRY | Recommend convex spatulation |
| SUTURE_GRASPED_WITH_FORCEPS_OR_PICKUPS | Risk fray/integrity |
| TIE_DIRECTLY_ON_TOE | Recommend just beyond apex |
| NEEDLE_DIRECTION_REVERSED | Out-to-in / in-to-out mismatch |
| ASSISTANT_QUALITY_IMPACT | Benefitted/hindered by assistant |
| SPACING_TOO_CLOSE | Spacing tighter than needed |
| SPACING_CAN_BE_WIDER | Could space bites further |

## 6. Canonical Output Schema

```json
{
  "case_id": "string",
  "rubric_version": "vop_2023_v1",
  "evidence_based": {
    "items": [
      { "item_id": 1, "score": "YES|NO|NULL", "observability": "OBSERVED|NOT_OBSERVED", "evidence": "string" },
      { "item_id": 2, "score": "YES|NO|NULL", "observability": "OBSERVED|NOT_OBSERVED", "evidence": "string" },
      { "item_id": 3, "score": "YES|NO|NULL", "observability": "OBSERVED|NOT_OBSERVED", "evidence": "string" },
      { "item_id": 4, "score": "YES|NO|NULL", "observability": "OBSERVED|NOT_OBSERVED", "evidence": "string" },
      { "item_id": 5, "score": "YES|NO|NULL", "observability": "OBSERVED|NOT_OBSERVED", "evidence": "string" },
      { "item_id": 6, "score": "YES|NO|NULL", "observability": "OBSERVED|NOT_OBSERVED", "evidence": "string",
        "subitems": {
          "6a_right_angle_method": { "score": "YES|NO|NULL", "evidence": "string" },
          "6b_safe_transfer_outcome": { "score": "YES|NO|NULL", "evidence": "string" }
        }
      },
      { "item_id": 7, "score": "YES|NO|NULL", "observability": "OBSERVED|NOT_OBSERVED", "evidence": "string" },
      { "item_id": 8, "score": "YES|NO|NULL", "observability": "OBSERVED|NOT_OBSERVED", "evidence": "string" },
      { "item_id": 9, "score": "YES|NO|NULL", "observability": "OBSERVED|NOT_OBSERVED", "evidence": "string" },
      { "item_id": 10, "score": "YES|NO|NULL", "observability": "OBSERVED|NOT_OBSERVED", "evidence": "string" },
      { "item_id": 11, "score": 3, "observability": "OBSERVED",
        "evidence": {
          "flow_organization": "organized|mixed|disorganized",
          "wasted_motion_events": [ { "type": "string", "count_estimate": 0, "note": "string" } ],
          "economy_index": 0
        }
      },
      { "item_id": 12, "score": "PROFICIENT|NOT_PROFICIENT|INSUFFICIENT_EVIDENCE", "observability": "DERIVED",
        "evidence": { "red_lines_triggered": ["string"], "missing_core_domains": ["string"] } },
      { "item_id": 13, "score": "string", "observability": "OBSERVED",
        "evidence": { "coaching_tags": ["TAG1", "TAG2"] } }
    ],
    "coverage": {
      "observed_count": 0,
      "total_count": 10,
      "observed_percent": 0,
      "core_observed": { "2": true, "7": true, "9": true, "10": true }
    }
  }
}
```
