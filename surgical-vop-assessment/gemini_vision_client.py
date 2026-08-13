#!/usr/bin/env python3
"""
Gemini Vision Client for Vascular Anastomosis VOP Retrospective Assessment
v4.1: Two-stage architecture (FOR + LLM scoring with app-side validators).
Stage 1: Gemini extracts Factual Observation Record (FOR) — purely descriptive.
Stage 2: Gemini scores from FOR JSON, with app-side validators enforcing correctness.
Red-line proficiency model (RL-A through RL-E). Coaching tags taxonomy.
Deterministic scoring engine retained as fallback.
"""

import os
import re
import json
import time
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception, before_sleep_log, before_sleep

from google import genai
from google.genai import types

MAX_VIDEO_SIZE_MB = 1000

# Upload reliability settings: files over the threshold are re-encoded to a
# lower bitrate before upload (real trainee videos run 400-700MB and large
# uploads have been observed to stall silently). Uploads get a hard timeout
# and are retried instead of hanging forever.
UPLOAD_COMPRESS_THRESHOLD_MB = 150
UPLOAD_TIMEOUT_SECONDS = 600          # hard cap per upload attempt
UPLOAD_MAX_ATTEMPTS = 3               # attempts per file (timeouts/transient errors)
FILE_PROCESSING_TIMEOUT_SECONDS = 900  # cap on Gemini-side "PROCESSING" wait
# Split thresholds lowered to ~2/3 (~30 -> ~20 min / ~25 -> ~16 min segments) to
# compensate for the extra frame tokens introduced by 3 fps sampling
# (accuracy prioritized over cost).
MAX_VIDEO_DURATION_SECONDS = 1200
SEGMENT_DURATION_SECONDS = 1000

# Video frames sampled at this rate before being sent to the model.
VIDEO_SAMPLING_FPS = 3

GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

# gemini-3-pro-preview was retired by Google (API returns 404); 3.1 Pro is its successor.
MODEL_VERSION = "gemini-3.1-pro-preview"
RUBRIC_VERSION = "vop_2023_v1"


def is_rate_limit_error(exception: BaseException) -> bool:
    error_msg = str(exception).lower()
    return (
        "429" in error_msg
        or "ratelimit" in error_msg
        or "rate_limit" in error_msg
        or "resource_exhausted" in error_msg
        or "quota" in error_msg
        or "rate limit" in error_msg
        or "exhausted" in error_msg
    )


def is_retryable_error(exception: BaseException) -> bool:
    """Rate limits plus transient Gemini server errors (500/503/504)."""
    if is_rate_limit_error(exception):
        return True
    error_msg = str(exception).lower()
    return (
        "500" in error_msg and "internal" in error_msg
        or "503" in error_msg
        or "504" in error_msg
        or "deadline_exceeded" in error_msg
        or "deadline expired" in error_msg
        or "unavailable" in error_msg
        or "overloaded" in error_msg
    )


def _before_retry_log(retry_state: Any) -> None:
    attempt = retry_state.attempt_number
    wait = retry_state.next_action.sleep if hasattr(retry_state, 'next_action') and retry_state.next_action else 0
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    print(f"  Rate limit hit (attempt {attempt}), waiting {wait:.0f}s before retry... ({exc})")


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

CHECKLIST_ITEMS = list(range(1, 11))
ECONOMY_ITEM = 11
PROFICIENCY_ITEM = 12
COMMENTS_ITEM = 13

VALID_OBSERVABILITIES = {"OBSERVED", "NOT_OBSERVED", "DERIVED"}

CORE_PROFICIENCY_ITEMS = {2, 7, 9, 10}

ECONOMY_EVENT_WEIGHTS = {
    "instrument_search": 2,
    "pause_reset": 2,
    "failed_pass_sequence": 2,
    "excess_regrasp_cluster": 1,
    "regrasp": 1,
    "other": 1,
}

COACHING_TAGS_TAXONOMY = [
    "ARTERIOTOMY_TOO_SHORT",
    "SPATULATION_CONCAVE_GEOMETRY",
    "SUTURE_GRASPED_WITH_FORCEPS_OR_PICKUPS",
    "TIE_DIRECTLY_ON_TOE",
    "NEEDLE_DIRECTION_REVERSED",
    "ASSISTANT_QUALITY_IMPACT",
    "SPACING_TOO_CLOSE",
    "SPACING_CAN_BE_WIDER",
]

RED_LINE_LABELS = {
    "RL-A": "Back wall gaping/malapposition (Item 7 = NO)",
    "RL-B": "Front wall gaping/malapposition (Item 9 = NO)",
    "RL-C": "Back wall injury (Item 2 = NO)",
    "RL-D": "Intraluminal/insecure final knot (Item 10 = NO)",
    "RL-E": "Heel establishment failure (Item 4 = NO, observed consequence)",
}

SCORING_PROMPT = """You are scoring a vascular anastomosis performance using the official 2023 retrospective checklist (Items 1–13). You will be given a Factual Observation Record (FOR) in JSON. You must score ONLY using what is in the FOR; do not invent facts.

CRITICAL: This uses a TRI-STATE scoring model: YES / NO / NULL.
- YES = criterion met and OBSERVED
- NO = criterion failed and OBSERVED
- NULL = not enough evidence to determine (NOT_OBSERVED)
Do NOT force YES or NO when the FOR does not clearly support either. Use NULL.

ACTIONABLE COACHING (applies to checklist items 1–10):
- For every item scored NO (an OBSERVED failure), you MUST include a "coaching" string: one concrete, action-oriented sentence in the imperative voice telling the learner exactly what to DO differently, grounded in the FOR evidence (e.g. "Rotate the wrist to follow the needle's curve so the tip exits perpendicular to the vessel wall."). Say what to do, not just what went wrong.
- For every item scored NULL (NOT_OBSERVED), you MUST omit "coaching" or set it to "" (empty). You cannot advise on something that was not seen; the NULL score itself conveys "not observed."
- For every item scored YES, omit "coaching" or set it to "" (empty). Do not praise; a met criterion needs no coaching.
- Keep coaching specific to that item and that performance — never generic filler.

Critical constraints:
1) Output ONLY valid JSON matching the schema below. No extra text.
2) Do NOT introduce new categories or domains beyond items 1–13.
3) If the FOR does not clearly support YES or NO for an item, set score="NULL" and observability="NOT_OBSERVED".
4) Spacing items (7 and 9) are functional: consistency + approximation + no visible gaping. The rubric describes sutures "approx 1 mm apart" but you must NOT estimate millimeters — score by uniformity/approximation only.
5) Technique wording is not method-policing. Allow equivalence pathways for Items 4, 6, and 8 as defined below.
6) Item 12 (Proficiency) must be DERIVED by code; output your best assessment and the app will recompute.
7) Item 11 (Economy) is NOT speed. Slow pace alone is not a penalty unless it creates wasted-motion events.

INPUT:
You will receive a FOR JSON object.

OUTPUT:
Return a JSON object following this schema:

{
  "case_id": string,
  "rubric_version": "vop_2023_v1",
  "evidence_based": {
    "items": [
      { "item_id": 1, "score": "YES"|"NO"|"NULL", "observability": "OBSERVED"|"NOT_OBSERVED", "evidence": string, "coaching": string },
      ...
      { "item_id": 6, "score": "YES"|"NO"|"NULL", "observability": "OBSERVED"|"NOT_OBSERVED", "evidence": string, "coaching": string,
        "subitems": {
          "6a_right_angle_method": { "score": "YES"|"NO"|"NULL", "evidence": string },
          "6b_safe_transfer_outcome": { "score": "YES"|"NO"|"NULL", "evidence": string }
        }
      },
      { "item_id": 11,
        "score": 1|2|3|4|5|"NULL",
        "observability": "OBSERVED"|"NOT_OBSERVED",
        "coaching": string,
        "evidence": {
          "flow_organization": "organized"|"mixed"|"disorganized"|"unknown",
          "wasted_motion_events": [
            { "type": "instrument_search"|"pause_reset"|"failed_pass_sequence"|"excess_regrasp_cluster"|"other",
              "count_estimate": number,
              "note": string
            }
          ],
          "economy_index": number_or_null
        }
      },
      { "item_id": 12, "score": "PROFICIENT"|"NOT_PROFICIENT"|"INSUFFICIENT_EVIDENCE", "observability": "DERIVED",
        "evidence": { "red_lines_triggered": [string], "missing_core_domains": [string] } },
      { "item_id": 13, "score": string, "observability": "OBSERVED", "evidence": { "coaching_tags": [string] } }
    ],
    "coverage": {
      "observed_count": number,
      "total_count": 10,
      "observed_percent": number,
      "core_observed": { "2": boolean, "7": boolean, "9": boolean, "10": boolean }
    }
  }
}

ITEM DEFINITIONS:

Item 1 (Oblique arteriotomy incision, 2x graft lumen, #11 blade):
- YES (OBSERVED) if the incision action is seen AND produces a linear/oblique shape without jagged edges, preferably with #11 blade extended with scissors.
- YES (OBSERVED) also if the incision action is NOT seen on camera BUT the arteriotomy is visible in subsequent frames showing a clean, linear/oblique shape. The learner should not be penalized for performing the incision off-screen if the result is correct.
- NO (OBSERVED) if the incision is seen and produces a transverse or jagged cut, OR the arteriotomy result is visible later and shows a transverse/jagged shape.
- NULL (NOT_OBSERVED) only if neither the incision action NOR the resulting arteriotomy shape can be seen at any point in the video.

Item 2 (Avoids back wall injury to artery):
- YES if FOR explicitly indicates no catch/through-and-through.
- NO if catch or through-and-through observed.
- NULL if unknown.

Item 3 (Cuts end of graft into spatula shape):
- YES (OBSERVED) if spatulation action is seen AND the graft has a spatulated shape (pointed toe, rounded heel), performed with blade or scissors.
- YES (OBSERVED) also if spatulation is NOT seen on camera BUT the graft is visible subsequently with a correct spatulated shape (pointed toe, rounded heel). The learner should not be penalized for performing spatulation off-screen if the result is correct.
- NO (OBSERVED) if the graft is seen (during spatulation or subsequently) and has a blunt or irregular shape without proper spatulation.
- NULL (NOT_OBSERVED) only if neither the spatulation action NOR the resulting graft shape can be seen at any point in the video.

Item 4 (Places double-ended suture at heel of anastomosis):
- YES if start at heel OR equivalent progression shows stable heel alignment with no twist/mismatch/gaping attributable to start.
- NO only if OBSERVED persistent heel defect OR toe anchoring causes demonstrated heel defect.
- NULL if initial fixation and downstream heel alignment are not visible enough to judge.

Item 5 (Securely ties knot outside anastomosis at heel):
- YES if outside observed, NO if inside observed, NULL if unknown.

Item 6 (Uses right angle to pass suture to opposite side of heel — split into subitems):
6a RIGHT_ANGLE_METHOD:
- YES if right-angle instrument use is clearly observed.
- NO if clearly not used during an otherwise visible transfer step.
- NULL if the transfer step or instrument is not visible.

6b SAFE_TRANSFER_OUTCOME:
- YES if no twist/crossing and no tissue snag/tear and progression to opposite side achieved.
- NO if twist/crossing or snag/tear is observed and consequential.
- NULL if you cannot evaluate twist/trauma/progression.

Item 6 overall score = 6b score (outcome, not method). Do not gate proficiency on item 6.

Items 7 and 9 (wall scoring with gap confidence; rubric says sutures ~1 mm apart — score functionally, no mm estimates):
- YES (OBSERVED) if: completed=true AND visible_gaps=false AND spacing_uniformity in {uniform, mixed}.
- NO (OBSERVED) only for HIGH-CONFIDENCE failure:
  - completed=false, OR
  - visible_gaps=true AND gap_confidence="high" (clear/persistent/multiple gaps or malapposition), OR
  - spacing_uniformity="irregular" AND described as grossly irregular with malapposition/leak-risk.
- NULL (NOT_OBSERVED) if:
  - Wall completion or gap status is uncertain, briefly seen, or obstructed.
  - visible_gaps=true AND gap_confidence="low" (subtle/brief/uncertain gaps → NULL, NOT NO).
  - spacing_uniformity="irregular" but no high-confidence malapposition evidence.
  - Evidence contains "possible," "subtle," "hard to tell," "momentary," or "limited view."

Item 8 (Trims toe of graft to appropriate size at apex):
- YES if fit appropriate and no dog-ear/redundancy.
- NO if fit inappropriate or dog-ear/redundancy.
- NULL if unknown.

Item 10 (Securely ties final knot outside anastomosis):
- YES if outside and seated observed.
- NO if inside or not seated observed.
- NULL if unknown.

Item 11 (Economy of Time and Motion — event-based, out of 5, no clip-length required):
Count wasted-motion events by type. Each type has a weight:
- instrument_search: weight 2
- pause_reset: weight 2
- failed_pass_sequence: weight 2 (≥2 failed attempts before success)
- excess_regrasp_cluster: weight 1 (>3 regrips on same load/positioning episode)
- other: weight 1 (only if clearly wasted)
Needle micro-adjustments: multiple small angle adjustments count as 1 excess_regrasp_cluster at most, unless they cause a failed_pass_sequence or pause_reset.
Compute economy_index (EI) = sum(weight * count_estimate).
Do NOT use clip_minutes or per-minute normalization. Score directly from EI (out of 5):
- EI 0 AND flow_organization=organized AND no hesitation/wasted motion at all -> 5 (maximum economy of movement and efficiency). A 5 is RARE — award it only when there is truly ZERO wasted motion; if in doubt, award 4.
- EI 0–1 -> 4
- EI 2–3 -> 3 (organized time/motion, some unnecessary movements)
- EI 4–6 -> 2
- EI ≥7 -> 1 (many unnecessary/disorganized movements)
If economy markers are entirely absent (no flow_organization and no events) -> score "NULL", observability=NOT_OBSERVED.
ECONOMY COACHING: put a "coaching" string on item 11.
- If score < 5: give ONE concrete, actionable recommendation tied to the observed wasted-motion events (e.g. "Preload both needle drivers before starting the back wall so you stop reaching for instruments mid-suture."). Imperative voice.
- If score == 5: the coaching string must be exactly: "Mastery achieved — maximum economy of movement and efficiency."
- If score is NULL: leave coaching empty.

Item 12 (Proficiency; DERIVED):
Core domains: items 2, 7, 9, 10.
- NOT_PROFICIENT if ANY core domain is observed NO.
- PROFICIENT if ALL core domains are observed YES AND no observed red-line heel defect (Item 4 NO due to observed defect).
- INSUFFICIENT_EVIDENCE if ANY core domain is NULL (not observed).
Economy never affects proficiency. Item 6 never affects proficiency.
Evidence must list red_lines_triggered and missing_core_domains.

Item 13 (Other Summative Comments + coaching tags):
- One sentence summarizing outcome.
- Up to 3 coaching_tags from: ARTERIOTOMY_TOO_SHORT, SPATULATION_CONCAVE_GEOMETRY, SUTURE_GRASPED_WITH_FORCEPS_OR_PICKUPS, TIE_DIRECTLY_ON_TOE, NEEDLE_DIRECTION_REVERSED, ASSISTANT_QUALITY_IMPACT, SPACING_TOO_CLOSE, SPACING_CAN_BE_WIDER.
If FOR does not support a tag, do not include it.

COVERAGE:
Compute coverage over Items 1–10: observed_count = number with score != NULL. observed_percent = observed_count / 10 * 100.
Also compute core_observed flags for items 2, 7, 9, 10."""

CSV_FIELDS = [
    "case_id", "video_id", "candidate_id", "segment_start",
    "item_1", "item_1_observability", "item_1_coaching",
    "item_2", "item_2_observability", "item_2_coaching",
    "item_3", "item_3_observability", "item_3_coaching",
    "item_4", "item_4_observability", "item_4_coaching",
    "item_5", "item_5_observability", "item_5_coaching",
    "item_6", "item_6_observability", "item_6a_method", "item_6b_outcome", "item_6_coaching",
    "item_7", "item_7_observability", "item_7_coaching",
    "item_8", "item_8_observability", "item_8_coaching",
    "item_9", "item_9_observability", "item_9_coaching",
    "item_10", "item_10_observability", "item_10_coaching",
    "item_11_economy", "item_11_flow_organization",
    "item_11_economy_index", "item_11_wasted_events_count", "item_11_coaching",
    "item_12_proficient", "proficiency_rationale",
    "observed_count", "observed_percent", "core_observed",
    "red_lines", "missing_core_domains", "coaching_tags", "comments",
]

FOR_PROMPT = """You are a surgical video fact-extractor. Your job is to describe ONLY observable events and states in the video. Do NOT assess quality. Do NOT give advice. Do NOT assign scores. Do NOT use evaluative words.

Hard bans (do not output these words or their synonyms): proficient, not proficient, good, poor, correct, incorrect, adequate, inadequate, efficient, inefficient, sloppy, excellent, weak, strong, should, needs to, improvement.

Output MUST be valid JSON and MUST match the schema below exactly. Output ONLY the JSON (no markdown, no commentary).

Rules:
1) If something is not clearly visible, set it to "unknown" (do NOT guess).
2) Do not estimate millimeters or numeric spacing. Use: "uniform" | "mixed" | "irregular" | "unknown".
3) For gaps: use "visible_gaps" (true/false/unknown). If visible_gaps=true, you MUST also set gap_confidence:
   - "high" = clear, persistent, or multiple gaps, or clear malapposition
   - "low" = subtle, brief, partially obscured, or uncertain gaps
   - "unknown" = you cannot tell
   Use gap_notes to describe what you see (e.g., "brief view; partially obstructed").
4) If the video starts mid-procedure, reflect that in video_coverage.begins_at and set relevant fields to unknown.
5) You may include short timestamp strings if available (e.g., "00:32"), but do not invent timestamps.
6) Arteriotomy (Item 1): If you do not see the incision being made but the arteriotomy is visible in later frames showing a clean, linear/oblique shape, set incision_observed=false, incision_shape_visible_later=true, and describe the shape. If neither the action nor the result is visible, set shape="unknown".
7) Graft spatulation (Item 3): If you do not see the spatulation being performed but the graft is visible later with a spatulated shape (pointed toe, rounded heel), set spatulation_observed=false, graft_shape_visible_later=true, and describe the shape. If neither the action nor the result is visible, set shape="unknown".
8) Economy events: Use ONLY these event types:
   - instrument_search: learner searches for or fumbles with instruments
   - pause_reset: learner pauses and resets position or approach
   - failed_pass_sequence: ≥2 failed needle pass attempts before success
   - excess_regrasp_cluster: >3 regrips on same needle load/positioning episode (NOT normal needle adjustments between bites)
   - other: only if clearly wasted motion that doesn't fit above categories
   Normal needle adjustments between suture bites are NOT wasted motion events.

Schema (must match):
{
  "case_id": string,
  "video_coverage": {
    "begins_at": "before_first_stitch"|"after_first_stitch"|"mid_anastomosis"|"unknown",
    "shows_arteriotomy_creation": true|false,
    "shows_graft_spatulation": true|false,
    "shows_suture_packaging": true|false,
    "arteriotomy_result_visible": true|false,
    "graft_shape_visible": true|false
  },
  "observations": {
    "arteriotomy": {
      "incision_observed": true|false,
      "incision_shape_visible_later": true|false,
      "shape": "linear_oblique"|"transverse"|"jagged"|"unknown",
      "instrument": "11_blade"|"scissors"|"other"|"unknown",
      "extended_with_scissors": true|false|"unknown"
    },
    "graft_preparation": {
      "spatulation_observed": true|false,
      "graft_shape_visible_later": true|false,
      "shape": "spatulated_pointed_toe_rounded_heel"|"blunt"|"irregular"|"unknown",
      "instrument": "blade"|"scissors"|"other"|"unknown"
    },
    "initial_fixation": {
      "start_location": "heel"|"toe"|"other"|"unknown",
      "suture_configuration": "double_armed"|"single_armed"|"unknown",
      "heel_knot_location": "outside"|"inside"|"unknown",
      "heel_defect_observed": true|false|"unknown"
    },
    "progression_transfer": {
      "separate_transfer_maneuver_observed": true|false|"unknown",
      "instrument_used": "right_angle"|"forceps"|"needle_driver"|"hands"|"unknown",
      "twist_or_crossing_observed": true|false|"unknown",
      "tissue_snag_or_tear_observed": true|false|"unknown"
    },
    "posterior_wall": {
      "completed": true|false|"unknown",
      "spacing_uniformity": "uniform"|"mixed"|"irregular"|"unknown",
      "visible_gaps": true|false|"unknown",
      "gap_confidence": "high"|"low"|"unknown",
      "gap_notes": string
    },
    "anterior_wall": {
      "completed": true|false|"unknown",
      "spacing_uniformity": "uniform"|"mixed"|"irregular"|"unknown",
      "visible_gaps": true|false|"unknown",
      "gap_confidence": "high"|"low"|"unknown",
      "gap_notes": string
    },
    "toe_apex": {
      "trimming_observed": true|false|"unknown",
      "fit_appears_appropriate": true|false|"unknown",
      "dog_ear_or_redundancy": true|false|"unknown"
    },
    "back_wall_injury_signs": {
      "posterior_wall_catch_observed": true|false|"unknown",
      "through_and_through_observed": true|false|"unknown"
    },
    "final_knot": {
      "knot_location": "outside"|"inside"|"unknown",
      "appears_seated": true|false|"unknown"
    },
    "economy_markers": {
      "flow_organization": "organized"|"mixed"|"disorganized"|"unknown",
      "wasted_motion_events": [
        { "type": "instrument_search"|"pause_reset"|"failed_pass_sequence"|"excess_regrasp_cluster"|"other",
          "count_estimate": number,
          "note": string
        }
      ]
    }
  }
}"""


FOCUSED_ITEMS_1_3_PROMPT = """You are a surgical video observer. Examine this video of a simulated vascular anastomosis and report ONLY on the arteriotomy incision and graft spatulation. Do NOT assess any other aspects. Do NOT assign scores.

Output MUST be valid JSON matching the schema below exactly. Output ONLY the JSON.

Rules:
1) If the action is performed OFF-SCREEN but the RESULT is visible in subsequent frames, describe the result.
2) If neither the action nor the result is visible, set fields to "unknown" or false.
3) Do NOT guess. Describe only what you observe.

Schema:
{
  "case_id": string,
  "arteriotomy": {
    "incision_observed": true|false,
    "incision_shape_visible_later": true|false,
    "shape": "linear_oblique"|"transverse"|"jagged"|"unknown",
    "instrument": "11_blade"|"scissors"|"other"|"unknown",
    "extended_with_scissors": true|false|"unknown",
    "notes": string
  },
  "graft_preparation": {
    "spatulation_observed": true|false,
    "graft_shape_visible_later": true|false,
    "shape": "spatulated_pointed_toe_rounded_heel"|"blunt"|"irregular"|"unknown",
    "instrument": "blade"|"scissors"|"other"|"unknown",
    "notes": string
  }
}

Arteriotomy: A linear incision along the path of the "artery", not transverse. Look for a nice straight line without jagged edges. Preferably made with a #11 blade then extended with scissors. If you do not see the incision being made but can see the arteriotomy in later frames, describe its shape.

Graft spatulation: The graft should have a spatulated shape with a pointed toe and rounded heel. Can be performed with blade or scissors. If spatulation happens off-screen, look at the graft shape when it becomes visible."""


FOCUSED_ITEMS_1_3_SCORING_PROMPT = """Score ONLY Items 1 and 3 from the observation data below. Use tri-state scoring: YES / NO / NULL.

Item 1 (Oblique arteriotomy incision):
- YES if: incision observed on camera with linear/oblique shape, OR incision not seen but result visible later showing clean linear/oblique shape.
- NO if: incision shape is transverse or jagged (seen during incision or in later frames).
- NULL if: neither incision action nor resulting shape can be seen anywhere.

Item 3 (Graft spatulation):
- YES if: spatulation observed with correct shape (pointed toe, rounded heel), OR spatulation not seen but graft visible later with correct spatulated shape.
- NO if: graft shape is blunt or irregular (seen during spatulation or in later frames).
- NULL if: neither spatulation action nor resulting graft shape visible anywhere.

Output valid JSON:
{
  "case_id": string,
  "items": [
    {"item_id": 1, "score": "YES"|"NO"|"NULL", "observability": "OBSERVED"|"NOT_OBSERVED", "evidence": string},
    {"item_id": 3, "score": "YES"|"NO"|"NULL", "observability": "OBSERVED"|"NOT_OBSERVED", "evidence": string}
  ]
}

INPUT:
"""


def score_items_1_3_from_observations(obs_data: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic scoring fallback for Items 1 & 3 focused mode."""
    art = obs_data.get("arteriotomy", {})
    graft = obs_data.get("graft_preparation", {})

    art_action = art.get("incision_observed") is True
    art_result = art.get("incision_shape_visible_later") is True
    art_shape = art.get("shape", "unknown")

    if art_action or art_result:
        if art_shape in ("linear_oblique",):
            item1 = _make_item(1, "YES", "OBSERVED",
                               f"Arteriotomy {'performed on camera' if art_action else 'result visible later'}: "
                               f"linear/oblique shape. Instrument: {art.get('instrument', 'unknown')}.")
        elif art_shape in ("transverse", "jagged"):
            item1 = _make_item(1, "NO", "OBSERVED",
                               f"Arteriotomy shape is {art_shape}.")
        elif art_action:
            item1 = _make_item(1, "NULL", "OBSERVED",
                               "Arteriotomy creation observed but shape indeterminate.")
        else:
            item1 = _make_item(1, "NULL", "NOT_OBSERVED",
                               "Arteriotomy result visible but shape indeterminate.")
    else:
        item1 = _make_item(1, "NULL", "NOT_OBSERVED",
                           "Neither arteriotomy action nor result visible.")

    spat_action = graft.get("spatulation_observed") is True
    spat_result = graft.get("graft_shape_visible_later") is True
    graft_shape = graft.get("shape", "unknown")

    if spat_action or spat_result:
        if graft_shape in ("spatulated_pointed_toe_rounded_heel",):
            item3 = _make_item(3, "YES", "OBSERVED",
                               f"Graft {'spatulation on camera' if spat_action else 'shape visible later'}: "
                               f"spatulated with pointed toe/rounded heel. Instrument: {graft.get('instrument', 'unknown')}.")
        elif graft_shape in ("blunt", "irregular"):
            item3 = _make_item(3, "NO", "OBSERVED",
                               f"Graft shape is {graft_shape}, not properly spatulated.")
        elif spat_action:
            item3 = _make_item(3, "NULL", "OBSERVED",
                               "Graft spatulation observed but shape indeterminate.")
        else:
            item3 = _make_item(3, "NULL", "NOT_OBSERVED",
                               "Graft shape visible but could not determine if properly spatulated.")
    else:
        item3 = _make_item(3, "NULL", "NOT_OBSERVED",
                           "Neither spatulation action nor graft shape visible.")

    return {
        "case_id": obs_data.get("case_id", ""),
        "items": [item1, item3],
    }


def extract_video_id(filename: str) -> str:
    return Path(filename).stem


def parse_candidate_ids_from_filename(filename: str) -> Tuple[str, List[str]]:
    """Parse year and candidate IDs from filename.

    Patterns:
      '2018 1518.m4v'           -> year='2018', candidates=['1518']
      '2020 2001 and 2005.m4v'  -> year='2020', candidates=['2001', '2005']
      '2023 2301-2304.m4v'      -> year='2023', candidates=['2301','2302','2303','2304']
      '2023 2301-2304 2306.m4v' -> year='2023', candidates=['2301','2302','2303','2304','2306']
    """
    stem = Path(filename).stem.strip()
    m = re.match(r'^(20\d{2})\s+(.+)$', stem)
    if not m:
        return "", [stem]

    year = m.group(1)
    rest = m.group(2).strip()

    rest = rest.replace(' and ', ' ')
    rest = re.sub(r',\s*', ' ', rest)

    tokens = re.split(r'\s+', rest)
    candidates: List[str] = []
    for token in tokens:
        range_m = re.match(r'^(\d+)-(\d+)$', token)
        if range_m:
            start_s, end_s = range_m.group(1), range_m.group(2)
            try:
                start_n, end_n = int(start_s), int(end_s)
                if end_n >= start_n and (end_n - start_n) <= 20:
                    for n in range(start_n, end_n + 1):
                        candidates.append(str(n))
                else:
                    candidates.extend([start_s, end_s])
            except ValueError:
                candidates.extend([start_s, end_s])
        else:
            candidates.append(token.strip())

    return year, candidates if candidates else [rest]


def make_case_id(year: str, candidate_id: str) -> str:
    if year:
        return f"{year}_{candidate_id}"
    return candidate_id


class AssessmentValidationError(Exception):
    def __init__(self, message: str, analysis: str, advice: List[str]):
        super().__init__(message)
        self.analysis = analysis
        self.advice = advice


def parse_json_response(response_text: str) -> Any:
    text = response_text.strip()

    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1).strip()

    if text.startswith('['):
        bracket_end = text.rfind(']')
        if bracket_end != -1:
            text = text[:bracket_end + 1]
        return json.loads(text)

    if not text.startswith('{'):
        bracket_start = text.find('[')
        brace_start = text.find('{')
        if bracket_start != -1 and (brace_start == -1 or bracket_start < brace_start):
            text = text[bracket_start:]
            bracket_end = text.rfind(']')
            if bracket_end != -1:
                text = text[:bracket_end + 1]
            return json.loads(text)
        elif brace_start != -1:
            text = text[brace_start:]

    if not text.startswith('['):
        if not text.endswith('}'):
            brace_end = text.rfind('}')
            if brace_end != -1:
                text = text[:brace_end + 1]

    return json.loads(text)


def normalize_to_candidate_list(parsed: Any) -> List[Dict[str, Any]]:
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    return []


# Rule-based coaching templates for the deterministic-scoring fallback.
# Coaching is only provided for items scored NO (an OBSERVED failure). NULL
# (not observed) items get no advice — you cannot coach on something unseen.
COACHING_TEMPLATES: Dict[int, str] = {
    1: "Extend the arteriotomy with an oblique/linear cut about twice the graft lumen; use a #11 blade and lengthen with fine scissors to avoid a transverse or jagged edge.",
    2: "Lift only the anterior wall and pass the needle under direct vision to avoid catching the back wall; use a nerve hook or forceps to tent the near wall away.",
    3: "Cut a longer spatulated bevel on the graft end (pointed toe, rounded heel) so it opens to match the arteriotomy without a blunt or square edge.",
    4: "Anchor the first suture at the heel and confirm alignment before running; if you start elsewhere, verify the heel seats flat with no purse-stringing before proceeding.",
    5: "Seat the heel knot on the outside of the anastomosis; pass the throw so the knot lands extraluminally, not inside the lumen.",
    6: "Use a right-angle to hand the suture cleanly to the opposite side of the heel; keep the two ends parallel so they do not twist, cross, or snag the tissue.",
    7: "Space back-wall bites evenly and take symmetric bites on graft and artery; keep edges everted and pull suture snug after each pass to close visible gaps.",
    8: "Trim the graft toe to fit the apex without redundancy; remove the dog-ear so the toe lies flat against the arteriotomy apex.",
    9: "Space front-wall bites evenly with symmetric graft/artery purchase; evert the edges and keep tension consistent to eliminate visible gaps or malapposition.",
    10: "Tie the final knot on the outside of the anastomosis and lay down enough throws so it seats securely without slipping.",
}


def _coaching_for_item(item_id: int, score: Any) -> str:
    """Return a rule-based coaching string for a checklist item scored NO.

    Coaching is only given for NO (an observed failure). Empty for YES
    (met criterion), NULL (not observed — cannot advise on the unseen), and
    for items without a template.
    """
    if score != "NO":
        return ""
    return COACHING_TEMPLATES.get(item_id, "")


def _make_item(
    item_id: int,
    score: Any,
    observability: str,
    evidence: Any,
    equivalence_pathway: str = "",
    coaching_tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "item_id": item_id,
        "label": ITEM_LABELS.get(item_id, f"Item {item_id}"),
        "score": score,
        "observability": observability,
        "evidence": evidence,
    }
    # Populate actionable coaching for below-perfect checklist items (1-10).
    if item_id in CHECKLIST_ITEMS:
        coaching = _coaching_for_item(item_id, score)
        if coaching:
            entry["coaching"] = coaching
    if equivalence_pathway and item_id in (4, 6):
        entry["equivalence_pathway"] = equivalence_pathway
    if coaching_tags is not None and item_id == COMMENTS_ITEM:
        entry["coaching_tags"] = coaching_tags
    return entry


def validate_for(data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Validate the Factual Observation Record (FOR) from Stage 1."""
    issues = []

    if not isinstance(data, dict):
        return False, {"issues": ["FOR is not a dict"], "valid_sections": 0}

    coverage = data.get("video_coverage")
    if not isinstance(coverage, dict):
        issues.append("Missing or invalid 'video_coverage' section")

    obs = data.get("observations")
    if not isinstance(obs, dict):
        issues.append("Missing or invalid 'observations' section")
    else:
        required_sections = [
            "initial_fixation", "progression_transfer", "posterior_wall",
            "anterior_wall", "toe_apex", "back_wall_injury_signs",
            "final_knot", "economy_markers",
        ]
        for section in required_sections:
            if section not in obs:
                issues.append(f"Missing observation section: {section}")

    valid_sections = 0
    if isinstance(coverage, dict):
        valid_sections += 1
    if isinstance(obs, dict):
        valid_sections += sum(1 for s in ["initial_fixation", "progression_transfer",
                                           "posterior_wall", "anterior_wall", "toe_apex",
                                           "back_wall_injury_signs", "final_knot",
                                           "economy_markers"] if s in obs)

    return len(issues) == 0, {"issues": issues, "valid_sections": valid_sections}


def derive_item_scores(for_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Stage 2: Derive all item scores deterministically from FOR observations.
    Uses tri-state scoring: YES/NO/NULL with OBSERVED/NOT_OBSERVED observability.
    """
    obs = for_data.get("observations", {})
    coverage = for_data.get("video_coverage", {})
    items: List[Dict[str, Any]] = []

    arteriotomy_obs = obs.get("arteriotomy", {})
    art_action = coverage.get("shows_arteriotomy_creation") is True or arteriotomy_obs.get("incision_observed") is True
    art_result = coverage.get("arteriotomy_result_visible") is True or arteriotomy_obs.get("incision_shape_visible_later") is True
    art_shape = arteriotomy_obs.get("shape", "unknown")

    if art_action or art_result:
        if art_shape in ("linear_oblique",):
            items.append(_make_item(1, "YES", "OBSERVED",
                                    f"Arteriotomy {'performed on camera' if art_action else 'result visible in subsequent frames'}: "
                                    f"linear/oblique shape. Instrument: {arteriotomy_obs.get('instrument', 'unknown')}."))
        elif art_shape in ("transverse", "jagged"):
            items.append(_make_item(1, "NO", "OBSERVED",
                                    f"Arteriotomy shape is {art_shape}, not a clean oblique incision."))
        elif art_action:
            items.append(_make_item(1, "YES", "OBSERVED",
                                    "Arteriotomy creation observed on camera."))
        else:
            items.append(_make_item(1, "NULL", "NOT_OBSERVED",
                                    "Arteriotomy result visible but shape could not be determined."))
    else:
        items.append(_make_item(1, "NULL", "NOT_OBSERVED",
                                "Neither arteriotomy creation nor resulting shape visible in video."))

    injury = obs.get("back_wall_injury_signs", {})
    catch = injury.get("posterior_wall_catch_observed")
    through = injury.get("through_and_through_observed")
    if catch is True or through is True:
        parts = []
        if catch is True:
            parts.append("posterior wall catch observed")
        if through is True:
            parts.append("through-and-through observed")
        items.append(_make_item(2, "NO", "OBSERVED",
                                f"Back wall injury: {', '.join(parts)}."))
    elif catch is False and through is False:
        items.append(_make_item(2, "YES", "OBSERVED",
                                "No posterior wall catch or through-and-through observed."))
    else:
        items.append(_make_item(2, "NULL", "NOT_OBSERVED",
                                "Back wall injury signs not clearly visible."))

    graft_obs = obs.get("graft_preparation", {})
    spat_action = coverage.get("shows_graft_spatulation") is True or graft_obs.get("spatulation_observed") is True
    spat_result = coverage.get("graft_shape_visible") is True or graft_obs.get("graft_shape_visible_later") is True
    graft_shape = graft_obs.get("shape", "unknown")

    if spat_action or spat_result:
        if graft_shape in ("spatulated_pointed_toe_rounded_heel",):
            items.append(_make_item(3, "YES", "OBSERVED",
                                    f"Graft {'spatulation performed on camera' if spat_action else 'shape visible in subsequent frames'}: "
                                    f"spatulated with pointed toe and rounded heel. Instrument: {graft_obs.get('instrument', 'unknown')}."))
        elif graft_shape in ("blunt", "irregular"):
            items.append(_make_item(3, "NO", "OBSERVED",
                                    f"Graft shape is {graft_shape}, not properly spatulated."))
        elif spat_action:
            items.append(_make_item(3, "YES", "OBSERVED",
                                    "Graft spatulation observed on camera."))
        else:
            items.append(_make_item(3, "NULL", "NOT_OBSERVED",
                                    "Graft shape visible but could not determine if properly spatulated."))
    else:
        items.append(_make_item(3, "NULL", "NOT_OBSERVED",
                                "Neither spatulation action nor resulting graft shape visible in video."))

    # Item 4 — heel establishment. Technique-equivalence + prompt rule:
    # NO only if an OBSERVED persistent heel defect OR toe anchoring that causes
    # a demonstrated heel defect. Toe/other start alone is NOT a failure.
    fixation = obs.get("initial_fixation", {})
    start = fixation.get("start_location")
    config = fixation.get("suture_configuration")
    heel_defect = fixation.get("heel_defect_observed")

    if heel_defect is True:
        # Demonstrated heel defect (persistent, or caused by toe anchoring) -> NO.
        anchor_note = " (toe anchoring)" if start == "toe" else ""
        items.append(_make_item(4, "NO", "OBSERVED",
                                f"Observed persistent heel defect{anchor_note}: heel not correctly established."))
    elif start == "heel":
        items.append(_make_item(4, "YES", "OBSERVED",
                                f"Heel correctly established. Suture config: {config or 'unknown'}."))
    elif start in ("toe", "other"):
        # Equivalent progression without a demonstrated heel defect -> YES.
        items.append(_make_item(4, "YES", "OBSERVED",
                                f"Started at '{start}' but no demonstrated heel defect; "
                                f"equivalent progression established a stable heel."))
    elif start == "unknown" and heel_defect is False:
        # Start not seen, but downstream confirms no heel defect -> YES.
        items.append(_make_item(4, "YES", "OBSERVED",
                                "Initial fixation not clearly seen, but no heel defect observed downstream."))
    else:
        items.append(_make_item(4, "NULL", "NOT_OBSERVED",
                                "Initial fixation and downstream heel alignment not visible enough to judge."))

    knot_loc = fixation.get("heel_knot_location")
    if knot_loc == "outside":
        items.append(_make_item(5, "YES", "OBSERVED",
                                "Heel knot tied outside (extraluminal)."))
    elif knot_loc == "inside":
        items.append(_make_item(5, "NO", "OBSERVED",
                                "Heel knot tied inside (intraluminal)."))
    else:
        items.append(_make_item(5, "NULL", "NOT_OBSERVED",
                                "Heel knot location not clearly visible."))

    transfer = obs.get("progression_transfer", {})
    sep_transfer = transfer.get("separate_transfer_maneuver_observed")
    instrument = transfer.get("instrument_used")
    twist = transfer.get("twist_or_crossing_observed")
    snag = transfer.get("tissue_snag_or_tear_observed")

    sub_6a = {"score": "NULL", "evidence": "Right-angle method not visible."}
    sub_6b = {"score": "NULL", "evidence": "Transfer outcome not evaluable."}

    if instrument == "right_angle":
        sub_6a = {"score": "YES", "evidence": "Right-angle instrument use clearly observed."}
    elif instrument in ("forceps", "needle_driver", "hands"):
        sub_6a = {"score": "NO", "evidence": f"Transfer used {instrument}, not right-angle."}
    elif sep_transfer is False:
        sub_6a = {"score": "NULL", "evidence": "No separate transfer maneuver; method N/A."}

    if sep_transfer is True or instrument == "right_angle":
        if twist is True or snag is True:
            issue_parts = []
            if twist is True:
                issue_parts.append("twist/crossing")
            if snag is True:
                issue_parts.append("tissue snag/tear")
            sub_6b = {"score": "NO", "evidence": f"Transfer issues: {', '.join(issue_parts)}."}
        elif twist is False or snag is False:
            sub_6b = {"score": "YES", "evidence": "Safe transfer outcome, no twist/crossing or snag/tear."}
    elif sep_transfer is False:
        sub_6b = {"score": "YES", "evidence": "No separate transfer needed; continuous technique, no twist observed."}

    item_6_score = sub_6b["score"]
    item_6_obs = "OBSERVED" if item_6_score != "NULL" else "NOT_OBSERVED"
    item_6 = _make_item(6, item_6_score, item_6_obs,
                        f"6a: {sub_6a['evidence']} 6b: {sub_6b['evidence']}")
    item_6["subitems"] = {
        "6a_right_angle_method": sub_6a,
        "6b_safe_transfer_outcome": sub_6b,
    }
    items.append(item_6)

    def _score_wall(wall_data, wall_name):
        completed = wall_data.get("completed")
        spacing = wall_data.get("spacing_uniformity")
        gaps = wall_data.get("visible_gaps")
        gap_conf = wall_data.get("gap_confidence", "unknown")
        gap_notes = wall_data.get("gap_notes", "")

        if completed is None or completed == "unknown":
            return "NULL", "NOT_OBSERVED", f"{wall_name} completion not visible."
        if spacing is None or spacing == "unknown":
            return "NULL", "NOT_OBSERVED", f"{wall_name} spacing not visible."
        if gaps is None or gaps == "unknown":
            return "NULL", "NOT_OBSERVED", f"{wall_name} gaps not assessable."

        if completed is False:
            return "NO", "OBSERVED", f"{wall_name} not completed."

        if gaps is True:
            if gap_conf == "high":
                return "NO", "OBSERVED", f"{wall_name} has high-confidence visible gaps/malapposition. {gap_notes}".strip()
            else:
                return "NULL", "NOT_OBSERVED", f"{wall_name} possible/subtle gaps (low confidence). {gap_notes}".strip()

        if spacing == "irregular":
            return "NULL", "NOT_OBSERVED", f"{wall_name} spacing irregular but no high-confidence malapposition evidence."

        gap_text = "none visible" if gaps is False else "unknown"
        return "YES", "OBSERVED", f"{wall_name} completed. Spacing: {spacing}. Gaps: {gap_text}."

    post = obs.get("posterior_wall", {})
    s7, o7, e7 = _score_wall(post, "Back wall")
    items.append(_make_item(7, s7, o7, e7))

    toe = obs.get("toe_apex", {})
    fit = toe.get("fit_appears_appropriate")
    dog_ear = toe.get("dog_ear_or_redundancy")
    if fit is True and dog_ear is not True:
        items.append(_make_item(8, "YES", "OBSERVED",
                                "Toe/apex fit appears appropriate, no dog-ear or redundancy."))
    elif fit is False or dog_ear is True:
        items.append(_make_item(8, "NO", "OBSERVED",
                                f"Toe/apex fit issue: fit_appropriate={fit}, dog_ear_or_redundancy={dog_ear}."))
    else:
        items.append(_make_item(8, "NULL", "NOT_OBSERVED",
                                "Toe/apex fit not clearly visible."))

    ant = obs.get("anterior_wall", {})
    s9, o9, e9 = _score_wall(ant, "Front wall")
    items.append(_make_item(9, s9, o9, e9))

    knot = obs.get("final_knot", {})
    final_loc = knot.get("knot_location")
    final_seated = knot.get("appears_seated")
    if final_loc == "outside" and final_seated is True:
        items.append(_make_item(10, "YES", "OBSERVED",
                                "Final knot outside and appears seated/secure."))
    elif final_loc == "inside":
        items.append(_make_item(10, "NO", "OBSERVED",
                                "Final knot inside (intraluminal)."))
    elif final_seated is False:
        items.append(_make_item(10, "NO", "OBSERVED",
                                f"Final knot at {final_loc or 'unknown'}, does not appear seated."))
    elif final_loc == "outside":
        items.append(_make_item(10, "YES", "OBSERVED",
                                f"Final knot outside, seated status: {final_seated}."))
    else:
        items.append(_make_item(10, "NULL", "NOT_OBSERVED",
                                "Final knot details not clearly visible."))

    econ_markers = obs.get("economy_markers", {})
    econ_score, econ_evidence, econ_obs = derive_economy_score(econ_markers)
    econ_item: Dict[str, Any] = {
        "item_id": 11,
        "label": ITEM_LABELS[11],
        "score": econ_score,
        "observability": econ_obs,
        "evidence": econ_evidence,
    }
    econ_coaching = _economy_coaching(econ_score, econ_evidence)
    if econ_coaching:
        econ_item["coaching"] = econ_coaching
    items.append(econ_item)

    return items


ECONOMY_MASTERY_TEXT = "Mastery achieved — maximum economy of movement and efficiency."


def _economy_coaching(score: Any, evidence: Dict[str, Any]) -> str:
    """Rule-based economy coaching for the deterministic fallback.

    score == 5 -> exact mastery statement.
    score  < 5 -> concrete recommendation tied to the dominant wasted-motion event type.
    NULL       -> empty.
    """
    if score == 5:
        return ECONOMY_MASTERY_TEXT
    if not isinstance(score, int):
        return ""
    events = evidence.get("wasted_motion_events", []) if isinstance(evidence, dict) else []
    # Find the highest-count event type to target the advice.
    dominant = ""
    max_count = 0
    for e in events:
        if isinstance(e, dict):
            c = e.get("count_estimate", 0) or 0
            if isinstance(c, (int, float)) and c > max_count:
                max_count = c
                dominant = e.get("type", "")
    tips = {
        "instrument_search": "Lay out and preload every instrument before you start each phase so you never pause to search for a tool mid-suture.",
        "pause_reset": "Plan each suture line before starting so you don't stop to reset position; rehearse the sequence of bites to keep a continuous rhythm.",
        "failed_pass_sequence": "Set the needle in the driver at the correct angle and follow its curve through the tissue to pass cleanly on the first attempt.",
        "excess_regrasp_cluster": "Grasp the needle once at the right spot and commit; avoid repeated regrips by loading it correctly before each pass.",
        "other": "Tighten your movement economy by planning each step so there is no unnecessary motion between bites.",
    }
    if dominant and dominant in tips:
        return tips[dominant]
    return "Reduce wasted motion by preloading instruments and planning each suture line so movements flow without pauses or fumbling."


def derive_economy_score(economy_markers: Dict[str, Any]) -> Tuple[Any, Dict[str, Any], str]:
    """Derive Economy Index and score from economy markers.
    Returns (score, evidence_dict, observability).
    Score is int 1-5 or "NULL" (2023 rubric, out of 5).
    No clip-length dependency — EI scored directly from event weights.
    """
    flow = economy_markers.get("flow_organization", "unknown")
    raw_events = economy_markers.get("wasted_motion_events", [])

    has_any_info = flow != "unknown" or len(raw_events) > 0
    if not has_any_info:
        return "NULL", {
            "flow_organization": "unknown",
            "wasted_motion_events": [],
            "economy_index": None,
        }, "NOT_OBSERVED"

    structured_events = []
    needle_micro_count = 0
    economy_index = 0.0
    for evt in raw_events:
        if isinstance(evt, dict):
            evt_type = evt.get("type", "other")
            count = evt.get("count_estimate", 1)
            if not isinstance(count, (int, float)) or count < 0:
                count = 1
            note = evt.get("note", "")

            if evt_type == "needle_reload_adjustment":
                needle_micro_count += count
                continue

            weight = ECONOMY_EVENT_WEIGHTS.get(evt_type, 1)
            economy_index += weight * count
            structured_events.append({
                "type": evt_type,
                "count_estimate": count,
                "note": note,
            })
        elif isinstance(evt, str):
            economy_index += 1
            structured_events.append({
                "type": "other",
                "count_estimate": 1,
                "note": evt,
            })

    if needle_micro_count > 0:
        economy_index += ECONOMY_EVENT_WEIGHTS.get("excess_regrasp_cluster", 1) * 1
        structured_events.append({
            "type": "excess_regrasp_cluster",
            "count_estimate": 1,
            "note": f"Collapsed from {int(needle_micro_count)} needle micro-adjustments.",
        })

    if economy_index == 0 and flow == "organized":
        score = 5
    elif economy_index <= 1:
        score = 4
    elif economy_index <= 3:
        score = 3
    elif economy_index <= 6:
        score = 2
    else:
        score = 1

    evidence = {
        "flow_organization": flow if flow != "unknown" else "mixed",
        "wasted_motion_events": structured_events,
        "economy_index": economy_index,
    }
    return score, evidence, "OBSERVED"


def derive_coaching_tags(for_data: Dict[str, Any], items: List[Dict[str, Any]]) -> List[str]:
    """Derive coaching tags from FOR observations and scored items."""
    tags: List[str] = []
    obs = for_data.get("observations", {})

    fixation = obs.get("initial_fixation", {})
    if fixation.get("start_location") == "toe":
        tags.append("TIE_DIRECTLY_ON_TOE")

    transfer = obs.get("progression_transfer", {})
    if transfer.get("instrument_used") == "forceps":
        tags.append("SUTURE_GRASPED_WITH_FORCEPS_OR_PICKUPS")
    if transfer.get("twist_or_crossing_observed") is True:
        tags.append("NEEDLE_DIRECTION_REVERSED")

    post = obs.get("posterior_wall", {})
    ant = obs.get("anterior_wall", {})
    for wall in [post, ant]:
        spacing = wall.get("spacing_uniformity")
        if spacing == "irregular":
            if "SPACING_CAN_BE_WIDER" not in tags:
                tags.append("SPACING_CAN_BE_WIDER")

    toe = obs.get("toe_apex", {})
    if toe.get("dog_ear_or_redundancy") is True:
        if "SPATULATION_CONCAVE_GEOMETRY" not in tags:
            tags.append("SPATULATION_CONCAVE_GEOMETRY")

    return [t for t in tags if t in COACHING_TAGS_TAXONOMY]


def compute_coverage(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute coverage metrics over items 1-10."""
    observed_count = 0
    core_observed: Dict[str, bool] = {}
    for entry in items:
        if not isinstance(entry, dict):
            continue
        iid = entry.get("item_id")
        if iid is None or iid > 10:
            continue
        score = entry.get("score")
        is_observed = score != "NULL"
        if is_observed:
            observed_count += 1
        if iid in CORE_PROFICIENCY_ITEMS:
            core_observed[str(iid)] = is_observed

    return {
        "observed_count": observed_count,
        "total_count": 10,
        "observed_percent": round(observed_count / 10 * 100, 1),
        "core_observed": core_observed,
    }


def derive_proficiency_redlines(
    items: List[Dict[str, Any]], for_data: Dict[str, Any]
) -> Tuple[str, str, List[str], List[str]]:
    """Derive proficiency using 2023 rubric red-line rules.

    Returns (proficiency, rationale, red_lines_triggered, missing_core_domains).

    Proficiency states:
    - NOT_PROFICIENT: any core domain (2,7,9,10) is observed NO
    - PROFICIENT: all core domains observed YES, no RL-E trigger
    - INSUFFICIENT_EVIDENCE: any core domain is NULL (not observed)
    """
    item_map: Dict[int, Dict[str, Any]] = {}
    for entry in items:
        if isinstance(entry, dict) and "item_id" in entry:
            item_map[entry["item_id"]] = entry

    red_lines: List[str] = []
    missing_core: List[str] = []

    core_items_config = [
        (2, "RL-C", "back wall injury"),
        (7, "RL-A", "back wall gaping"),
        (9, "RL-B", "front wall gaping"),
        (10, "RL-D", "insecure knot"),
    ]

    for iid, rl_code, desc in core_items_config:
        entry = item_map.get(iid, {})
        score = entry.get("score")
        if score == "NO":
            red_lines.append(f"{rl_code}: item {iid} = NO ({desc})")
        elif score == "NULL" or score is None:
            missing_core.append(f"Item {iid} ({desc})")

    item_4 = item_map.get(4, {})
    if item_4.get("score") == "NO" and item_4.get("observability") == "OBSERVED":
        red_lines.append("RL-E: item 4 = NO (heel establishment failure)")

    if red_lines:
        rationale = "Not proficient — " + "; ".join(red_lines)
        return "NOT_PROFICIENT", rationale, red_lines, missing_core

    if missing_core:
        rationale = "Insufficient evidence — core domains not observed: " + "; ".join(missing_core)
        return "INSUFFICIENT_EVIDENCE", rationale, [], missing_core

    notes = []
    item_6 = item_map.get(6, {})
    if item_6.get("score") == "NO":
        notes.append("Item 6 NO but non-gating")

    null_items = [
        str(e["item_id"]) for e in items
        if isinstance(e, dict) and e.get("item_id") in CHECKLIST_ITEMS
        and e.get("score") == "NULL"
    ]
    if null_items:
        notes.append(f"Items {', '.join(null_items)} not observed (non-core)")

    rationale = "Proficient. All core domains observed YES, no red-line failures."
    if notes:
        rationale += " " + "; ".join(notes) + "."
    return "PROFICIENT", rationale, [], []


def generate_item_13(items: List[Dict[str, Any]], coaching_tags: List[str], red_lines: List[str]) -> str:
    """Generate Item 13 summative comment from scored items and coaching tags."""
    item_map = {e["item_id"]: e for e in items if isinstance(e, dict) and "item_id" in e}

    deficiencies = []
    null_items = []
    for i in CHECKLIST_ITEMS:
        entry = item_map.get(i, {})
        if entry.get("score") == "NO":
            deficiencies.append(f"Item {i} ({ITEM_LABELS.get(i, '')})")
        elif entry.get("score") == "NULL":
            null_items.append(str(i))

    summary_parts = []
    if deficiencies:
        summary_parts.append(f"Deficiencies: {'; '.join(deficiencies[:3])}.")
    else:
        summary_parts.append("No observed deficiencies in procedural checklist.")

    if null_items:
        summary_parts.append(f"Items {', '.join(null_items)} not observed.")

    econ = item_map.get(11, {})
    econ_score = econ.get("score")
    if isinstance(econ_score, int):
        summary_parts.append(f"Economy score: {econ_score}/5.")

    if coaching_tags:
        tag_text = ", ".join(coaching_tags)
        summary_parts.append(f"Coaching: {tag_text}.")

    return " ".join(summary_parts)


def project_to_docx(items: List[Dict[str, Any]], mode: str = "construct_mode") -> Dict[str, Any]:
    """Project evidence-based tri-state scores to DOCX-compatible binary output.

    Modes:
    - construct_mode (default): NULL on items 1-3 -> YES; core NULLs -> uncertain
    - hawk_mode: Item 6 uses 6a (method); 6a NULL -> NO
    - dove_mode: Item 6 uses 6b; 6b NULL -> YES; core NULLs -> YES
    """
    item_map = {e["item_id"]: e for e in items if isinstance(e, dict) and "item_id" in e}
    projection_notes: List[str] = []
    projected: Dict[str, str] = {}

    for i in range(1, 11):
        entry = item_map.get(i, {})
        score = entry.get("score")

        if i == 6:
            subitems = entry.get("subitems", {})
            sub_6a = subitems.get("6a_right_angle_method", {})
            sub_6b = subitems.get("6b_safe_transfer_outcome", {})

            if mode == "hawk_mode":
                if sub_6a.get("score") == "YES":
                    projected["6"] = "YES"
                elif sub_6a.get("score") == "NO":
                    projected["6"] = "NO"
                else:
                    projected["6"] = "NO"
                    projection_notes.append("Item 6: hawk_mode projects NULL 6a to NO")
            elif mode == "dove_mode":
                if sub_6b.get("score") == "YES":
                    projected["6"] = "YES"
                elif sub_6b.get("score") == "NO":
                    projected["6"] = "NO"
                else:
                    projected["6"] = "YES"
                    projection_notes.append("Item 6: dove_mode projects NULL 6b to YES")
            else:
                if sub_6b.get("score") in ("YES", "NO"):
                    projected["6"] = sub_6b["score"]
                else:
                    projected["6"] = "YES"
                    projection_notes.append("Item 6: construct_mode projects NULL 6b to YES (outcome)")
            continue

        if score in ("YES", "NO"):
            projected[str(i)] = str(score)
            continue

        if i in (1, 2, 3):
            if mode == "dove_mode":
                projected[str(i)] = "YES"
                projection_notes.append(f"Item {i}: dove_mode projects NULL to YES (setup)")
            elif mode == "construct_mode":
                projected[str(i)] = "YES"
                projection_notes.append(f"Item {i}: construct_mode projects NULL to YES (setup)")
            else:
                projected[str(i)] = "YES"
                projection_notes.append(f"Item {i}: hawk_mode projects NULL to YES (setup)")
        elif i in CORE_PROFICIENCY_ITEMS:
            if mode == "dove_mode":
                projected[str(i)] = "YES"
                projection_notes.append(f"Item {i}: dove_mode projects NULL core to YES")
            else:
                projected[str(i)] = "YES"
                projection_notes.append(f"Item {i}: core domain not observed; projected with uncertainty")
        else:
            if mode == "dove_mode":
                projected[str(i)] = "YES"
                projection_notes.append(f"Item {i}: dove_mode projects NULL to YES")
            else:
                projected[str(i)] = "YES"
                projection_notes.append(f"Item {i}: projected NULL to YES")

    item_11 = item_map.get(11, {})
    econ_score = item_11.get("score")
    if isinstance(econ_score, int):
        economy_projected = min(econ_score, 5)
    else:
        economy_projected = 3
        projection_notes.append("Economy not observed; neutral projection (3)")

    has_observed_no = any(
        item_map.get(iid, {}).get("score") == "NO"
        for iid in CORE_PROFICIENCY_ITEMS
    )
    item_4 = item_map.get(4, {})
    has_rl_e = item_4.get("score") == "NO" and item_4.get("observability") == "OBSERVED"

    if has_observed_no or has_rl_e:
        prof_projected = "NO"
    else:
        prof_projected = "YES"

    return {
        "mode": mode,
        "items_1_10": projected,
        "economy_1_5": economy_projected,
        "proficiency_yes_no": prof_projected,
        "projection_notes": projection_notes,
    }

ITEM_COACHING_FALLBACKS = {
    1: "Angle the #11 blade to make an oblique arteriotomy roughly twice the graft lumen diameter.",
    2: "Keep the blade tip and needle away from the back wall — lift the front wall or use a guarded pass to avoid back-wall injury.",
    3: "Trim the graft end into a spatula shape so it matches the oblique arteriotomy without tension or bunching.",
    4: "Place the double-ended suture precisely at the heel so both limbs run symmetrically along each wall.",
    5: "Tie the heel knot securely outside the anastomosis lumen; confirm it seats flat before continuing.",
    6: "Use a right-angle transfer to pass the suture to the opposite side of the heel without twisting or catching tissue.",
    7: "Space back-wall sutures ~1 mm apart with consistent bites and no gaping between throws.",
    8: "Trim the toe of the graft to size at the apex so it lies flat without redundancy or tension.",
    9: "Space front-wall sutures ~1 mm apart with consistent bites and no gaping between throws.",
    10: "Tie the finishing knot securely outside the anastomosis; confirm adequate throws and a flat, locked knot.",
}


def enforce_item_coaching(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Enforce the per-item coaching contract on checklist items 1-10.

    - YES items never carry a coaching tip (stripped if present).
    - NO items always carry a tip: keep a valid LLM-supplied string, otherwise
      synthesize a rubric-grounded fallback tip for that item.
    - NULL items keep a tip only when the scorer supplied one (caveat case);
      none is synthesized.
    """
    for entry in items:
        if not isinstance(entry, dict):
            continue
        item_id = entry.get("item_id")
        if item_id == ECONOMY_ITEM:
            # Economy carries its own coaching contract: exact mastery text at
            # 5, a recommendation below 5, no coaching when NULL/unscored.
            sc = entry.get("score")
            if sc == 5:
                entry["coaching"] = ECONOMY_MASTERY_TEXT
            elif isinstance(sc, int):
                existing = entry.get("coaching")
                if isinstance(existing, str) and existing.strip():
                    entry["coaching"] = existing.strip()
                else:
                    entry["coaching"] = _economy_coaching(sc, entry.get("evidence", {}))
            else:
                entry.pop("coaching", None)
            continue
        if item_id not in CHECKLIST_ITEMS:
            entry.pop("coaching", None)
            continue
        score = entry.get("score")
        coaching = entry.get("coaching")
        valid_tip = isinstance(coaching, str) and coaching.strip()
        if score == "YES":
            entry.pop("coaching", None)
        elif score == "NO":
            if not valid_tip:
                entry["coaching"] = ITEM_COACHING_FALLBACKS.get(item_id, "Review this step against the rubric criteria and correct the observed deficiency.")
            else:
                entry["coaching"] = coaching.strip()
        elif score == "NULL":
            if valid_tip:
                entry["coaching"] = coaching.strip()
            else:
                entry.pop("coaching", None)
        else:
            entry.pop("coaching", None)
    return items
def _extract_scored_items(scored_output: Optional[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """Pull the items list out of a scored_output (either evidence_based.items or items)."""
    if not scored_output or not isinstance(scored_output, dict):
        return None
    if scored_output.get("evidence_based"):
        eb = scored_output["evidence_based"]
        if isinstance(eb, dict):
            return eb.get("items", [])
    if scored_output.get("items"):
        return scored_output.get("items", [])
    return None


def is_current_rubric_scored_output(scored_output: Optional[Dict[str, Any]]) -> bool:
    """True only if scored_output is vop_2023_v1 with EXACTLY item ids 1..13.

    Focused Items 1&3 outputs (ids {1,3}) and legacy 14-item outputs (ids 1..14)
    both return False, so they never get relabeled/reinterpreted as vop_2023_v1.
    """
    items = _extract_scored_items(scored_output)
    if not items or not isinstance(items, list):
        return False
    ids = {i.get("item_id") for i in items if isinstance(i, dict) and "item_id" in i}
    if ids != CURRENT_RUBRIC_ITEM_IDS:
        return False
    rubric = scored_output.get("rubric_version") if isinstance(scored_output, dict) else None
    # rubric_version may be absent on some validated payloads; only reject explicit mismatches.
    if rubric is not None and rubric != RUBRIC_VERSION:
        return False
    return True


def is_legacy_full_scored_output(scored_output: Optional[Dict[str, Any]]) -> bool:
    """True if scored_output is a FULL-form record from a pre-2023 rubric.

    Detects the old 14-item numbering (or any full record that is not exactly
    ids 1..13). The Focused Items 1&3 path (small id set {1,3}) is NOT flagged
    legacy here — it is handled by its own display path.
    """
    if is_current_rubric_scored_output(scored_output):
        return False
    items = _extract_scored_items(scored_output)
    if not items or not isinstance(items, list):
        return False
    ids = {i.get("item_id") for i in items if isinstance(i, dict) and "item_id" in i}
    # A full-form record has the trailing economy/proficiency/comments block.
    # Treat anything that is not the small focused set and not exactly 1..13 as legacy.
    focused_ids = {1, 3}
    if ids and ids != focused_ids and ids != CURRENT_RUBRIC_ITEM_IDS:
        return True
    return False


def _build_legacy_record(
    scored_output: Dict[str, Any],
    for_data: Dict[str, Any],
    video_id: str,
    year: str = "",
    candidate_id: str = "",
    segment_start: str = "",
    timestamp: str = "",
) -> Dict[str, Any]:
    """Build a passthrough record for a legacy (pre-2023) scored_output.

    Does NOT renumber or relabel. Preserves the original items and
    rubric_version verbatim, and marks the record is_legacy=True so display/CSV
    paths can skip the new-numbering logic and show a clear warning instead.
    """
    items = _extract_scored_items(scored_output) or []
    legacy_rubric = scored_output.get("rubric_version", "legacy_pre_2023")

    case_id = scored_output.get("case_id") or for_data.get("case_id", "")
    if not case_id and candidate_id:
        case_id = make_case_id(year, candidate_id)

    record: Dict[str, Any] = {
        "case_id": case_id,
        "video_id": video_id,
        "candidate_id": candidate_id or video_id,
        "segment_start": segment_start,
        "is_legacy": True,
        "rubric_version": legacy_rubric,
        "scoring_path": "legacy_passthrough",
        "items": items,
        "legacy_scored_output": scored_output,
        "diagnostics": {
            "validation_passed": False,
            "notes": [
                "Legacy rubric (pre-2023 refactor) — stored under the old 14-item numbering. "
                "Not reinterpreted under vop_2023_v1."
            ],
        },
        "for_data": for_data,
    }
    if timestamp:
        record["timestamp"] = timestamp
    return record


CURRENT_RUBRIC_ITEM_IDS = set(range(1, 14))  # ids 1..13 for vop_2023_v1


def _extract_scored_items(scored_output: Optional[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """Pull the items list out of a scored_output (either evidence_based.items or items)."""
    if not scored_output or not isinstance(scored_output, dict):
        return None
    if scored_output.get("evidence_based"):
        eb = scored_output["evidence_based"]
        if isinstance(eb, dict):
            return eb.get("items", [])
    if scored_output.get("items"):
        return scored_output.get("items", [])
    return None


def is_current_rubric_scored_output(scored_output: Optional[Dict[str, Any]]) -> bool:
    """True only if scored_output is vop_2023_v1 with EXACTLY item ids 1..13.

    Focused Items 1&3 outputs (ids {1,3}) and legacy 14-item outputs (ids 1..14)
    both return False, so they never get relabeled/reinterpreted as vop_2023_v1.
    """
    items = _extract_scored_items(scored_output)
    if not items or not isinstance(items, list):
        return False
    ids = {i.get("item_id") for i in items if isinstance(i, dict) and "item_id" in i}
    if ids != CURRENT_RUBRIC_ITEM_IDS:
        return False
    rubric = scored_output.get("rubric_version") if isinstance(scored_output, dict) else None
    # rubric_version may be absent on some validated payloads; only reject explicit mismatches.
    if rubric is not None and rubric != RUBRIC_VERSION:
        return False
    return True


def is_legacy_full_scored_output(scored_output: Optional[Dict[str, Any]]) -> bool:
    """True if scored_output is a FULL-form record from a pre-2023 rubric.

    Detects the old 14-item numbering (or any full record that is not exactly
    ids 1..13). The Focused Items 1&3 path (small id set {1,3}) is NOT flagged
    legacy here — it is handled by its own display path.
    """
    if is_current_rubric_scored_output(scored_output):
        return False
    items = _extract_scored_items(scored_output)
    if not items or not isinstance(items, list):
        return False
    ids = {i.get("item_id") for i in items if isinstance(i, dict) and "item_id" in i}
    # A full-form record has the trailing economy/proficiency/comments block.
    # Treat anything that is not the small focused set and not exactly 1..13 as legacy.
    focused_ids = {1, 3}
    if ids and ids != focused_ids and ids != CURRENT_RUBRIC_ITEM_IDS:
        return True
    return False


def _build_legacy_record(
    scored_output: Dict[str, Any],
    for_data: Dict[str, Any],
    video_id: str,
    year: str = "",
    candidate_id: str = "",
    segment_start: str = "",
    timestamp: str = "",
) -> Dict[str, Any]:
    """Build a passthrough record for a legacy (pre-2023) scored_output.

    Does NOT renumber or relabel. Preserves the original items and
    rubric_version verbatim, and marks the record is_legacy=True so display/CSV
    paths can skip the new-numbering logic and show a clear warning instead.
    """
    items = _extract_scored_items(scored_output) or []
    legacy_rubric = scored_output.get("rubric_version", "legacy_pre_2023")

    case_id = scored_output.get("case_id") or for_data.get("case_id", "")
    if not case_id and candidate_id:
        case_id = make_case_id(year, candidate_id)

    record: Dict[str, Any] = {
        "case_id": case_id,
        "video_id": video_id,
        "candidate_id": candidate_id or video_id,
        "segment_start": segment_start,
        "is_legacy": True,
        "rubric_version": legacy_rubric,
        "scoring_path": "legacy_passthrough",
        "items": items,
        "legacy_scored_output": scored_output,
        "diagnostics": {
            "validation_passed": False,
            "notes": [
                "Legacy rubric (pre-2023 refactor) — stored under the old 14-item numbering. "
                "Not reinterpreted under vop_2023_v1."
            ],
        },
        "for_data": for_data,
    }
    if timestamp:
        record["timestamp"] = timestamp
    return record


def _build_v4_record(
    for_data: Dict[str, Any],
    video_id: str,
    year: str = "",
    candidate_id: str = "",
    segment_start: str = "",
    timestamp: str = "",
    scored_output: Optional[Dict[str, Any]] = None,
    scoring_warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build a complete assessment record from FOR data.

    If scored_output is provided (LLM path), builds from pre-scored/validated
    output. Otherwise falls back to deterministic scoring.
    Output uses vop 2023 v1 format with evidence_based + coverage.

    Legacy (pre-2023) scored_outputs are NOT relabeled: they are returned as a
    passthrough legacy record (is_legacy=True) so old 14-item numbering is never
    misinterpreted under the new schema.
    """
    # Version-gate: a full-form scored_output that is not exactly ids 1..13 is
    # a legacy record. Pass it through verbatim instead of relabeling.
    if is_legacy_full_scored_output(scored_output):
        return _build_legacy_record(
            scored_output, for_data, video_id,
            year=year, candidate_id=candidate_id,
            segment_start=segment_start, timestamp=timestamp,
        )

    scoring_path = "deterministic"

    eb_items = None
    # Only consume a scored_output as vop_2023_v1 if it passes the version gate.
    if is_current_rubric_scored_output(scored_output):
        if scored_output.get("evidence_based"):
            eb = scored_output["evidence_based"]
            eb_items = eb.get("items", [])
        elif scored_output.get("items"):
            eb_items = scored_output.get("items", [])

    if eb_items and isinstance(eb_items, list) and len(eb_items) > 0:
        scoring_path = "llm_vop_2023_v1"
        items = [i if isinstance(i, dict) else i for i in eb_items]

        item_map = {i["item_id"]: i for i in items if isinstance(i, dict) and "item_id" in i}
        item_12 = item_map.get(12, {})
        proficiency = item_12.get("score", "INSUFFICIENT_EVIDENCE")
        prof_evidence = item_12.get("evidence", {})
        if isinstance(prof_evidence, dict):
            red_lines = prof_evidence.get("red_lines_triggered", [])
            missing_core = prof_evidence.get("missing_core_domains", [])
        else:
            red_lines = []
            missing_core = []
        prof_rationale = f"Derived by LLM scoring. Red lines: {red_lines}" if red_lines else (
            f"Insufficient evidence: {missing_core}" if missing_core else "Proficient per LLM scoring.")

        item_13 = item_map.get(13, {})
        item_13_evidence = item_13.get("evidence", {})
        if isinstance(item_13_evidence, dict):
            coaching_tags = item_13_evidence.get("coaching_tags", [])
        else:
            coaching_tags = []
    else:
        items = derive_item_scores(for_data)

        coaching_tags = derive_coaching_tags(for_data, items)

        proficiency, prof_rationale, red_lines, missing_core = derive_proficiency_redlines(items, for_data)

        items.append(_make_item(
            12, proficiency, "DERIVED",
            {"red_lines_triggered": red_lines, "missing_core_domains": missing_core}
        ))

        comment_text = generate_item_13(items, coaching_tags, red_lines)
        items.append(_make_item(
            13, comment_text, "OBSERVED",
            {"coaching_tags": coaching_tags},
        ))

    # Enforce the per-item coaching contract on both the LLM and the
    # deterministic-fallback scoring paths: every NO gets a tip, YES never
    # does, NULL only keeps an explicitly supplied caveat tip.
    items = enforce_item_coaching(items)

    case_id = for_data.get("case_id", "")
    if not case_id and candidate_id:
        case_id = make_case_id(year, candidate_id)

    coverage = compute_coverage(items)

    checklist_yes = sum(
        1 for e in items
        if isinstance(e, dict) and e.get("item_id") in CHECKLIST_ITEMS and e.get("score") == "YES"
    )
    null_count = sum(
        1 for e in items
        if isinstance(e, dict) and e.get("item_id") in CHECKLIST_ITEMS and e.get("score") == "NULL"
    )
    economy_entry = next((e for e in items if isinstance(e, dict) and e.get("item_id") == 11), {})
    economy_score = economy_entry.get("score")

    econ_evidence = economy_entry.get("evidence", {})
    econ_events = econ_evidence.get("wasted_motion_events", []) if isinstance(econ_evidence, dict) else []
    economy_warnings = _validate_economy_tally(len(econ_events), economy_score)

    projected = project_to_docx(items, "construct_mode")

    notes = []
    if scoring_path == "llm_vop_2023_v1":
        notes.append("Scored via Stage 2 LLM (vop 2023 v1)")
    else:
        notes.append("Scored via deterministic engine (fallback)")

    record: Dict[str, Any] = {
        "case_id": case_id,
        "video_id": video_id,
        "candidate_id": candidate_id or video_id,
        "segment_start": segment_start,
        "rubric_version": RUBRIC_VERSION,
        "model_version": MODEL_VERSION,
        "scoring_path": scoring_path,
        "items": items,
        "coverage": coverage,
        "checklist_yes_count": checklist_yes,
        "checklist_total": len(CHECKLIST_ITEMS),
        "null_count": null_count,
        "economy_score": economy_score,
        "proficiency": proficiency,
        "proficiency_rationale": prof_rationale,
        "red_lines_triggered": red_lines,
        "missing_core_domains": missing_core,
        "coaching_tags": coaching_tags,
        "projected_docx": projected,
        "diagnostics": {
            "validation_passed": True,
            "economy_tally_warnings": economy_warnings,
            "scoring_warnings": scoring_warnings or [],
            "for_available": True,
            "notes": notes,
        },
        "for_data": for_data,
    }
    if timestamp:
        record["timestamp"] = timestamp
    return record


def _validate_economy_tally(n_events: int, score: Any) -> List[str]:
    """Check economy tally-to-score constraint. Returns warnings."""
    warnings = []
    if not isinstance(score, int):
        return warnings
    if n_events <= 1 and score not in (4, 5):
        warnings.append(f"Economy: {n_events} events but score {score} (expected 4-5)")
    elif 2 <= n_events <= 3 and score != 3:
        warnings.append(f"Economy: {n_events} events but score {score} (expected 3)")
    elif n_events >= 4 and score not in (1, 2):
        warnings.append(f"Economy: {n_events} events but score {score} (expected 1-2)")
    return warnings


def validate_scoring_output(raw_text: str, for_data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Validate and fix Stage 2 scoring output. Returns (validated_output, warnings).
    Handles both new vop 2023 v1 format (evidence_based wrapper) and legacy flat format.
    Recomputes economy from FOR data when LLM returns empty/missing economy events.
    """
    warnings = []

    parsed = parse_json_response(raw_text)

    if isinstance(parsed, dict) and "evidence_based" in parsed:
        eb = parsed.get("evidence_based", {})
        items = eb.get("items", [])
    else:
        items = parsed.get("items", []) if isinstance(parsed, dict) else []

    output_str = json.dumps(parsed)
    if re.search(r'\d+\s*mm\b', output_str, re.IGNORECASE):
        warnings.append("Output contains numeric spacing claims (mm)")

    item_11 = next((i for i in items if isinstance(i, dict) and i.get("item_id") == 11), None)
    if item_11:
        evidence = item_11.get("evidence", {})
        if isinstance(evidence, dict):
            evidence.pop("economy_index_per_min", None)
            evidence.pop("clip_minutes", None)

            events = evidence.get("wasted_motion_events", [])
            score = item_11.get("score")

            for_econ = for_data.get("observations", {}).get("economy_markers", {}) if isinstance(for_data, dict) else {}
            if not isinstance(for_econ, dict):
                for_econ = {}
            for_events = for_econ.get("wasted_motion_events", [])
            if not isinstance(for_events, list):
                for_events = []
            for_flow = for_econ.get("flow_organization", "unknown")

            if (not events or len(events) == 0) and len(for_events) > 0:
                warnings.append(f"Economy: LLM returned 0 events but FOR has {len(for_events)} — recomputing from FOR")
                recomputed_score, recomputed_evidence, recomputed_obs = derive_economy_score(for_econ)
                item_11["score"] = recomputed_score
                item_11["evidence"] = recomputed_evidence
                item_11["observability"] = recomputed_obs
                score = recomputed_score
                evidence = recomputed_evidence
                events = recomputed_evidence.get("wasted_motion_events", [])

            if not evidence.get("flow_organization") or evidence.get("flow_organization") == "unknown":
                if for_flow != "unknown":
                    evidence["flow_organization"] = for_flow

            if score != "NULL" and isinstance(score, int):
                ei = evidence.get("economy_index")
                if ei is None:
                    ei_val = 0.0
                    for evt in events:
                        if isinstance(evt, dict):
                            evt_type = evt.get("type", "other")
                            count = evt.get("count_estimate", 1)
                            weight = ECONOMY_EVENT_WEIGHTS.get(evt_type, 1)
                            ei_val += weight * (count if isinstance(count, (int, float)) else 1)
                    evidence["economy_index"] = ei_val
                    ei = ei_val

                flow = evidence.get("flow_organization", "mixed")
                expected = _economy_guardrail_check(ei, score, flow)
                if expected is not None:
                    warnings.append(f"Economy guardrail: EI={ei}, score {score} -> expected {expected}")
                    item_11["score"] = expected

    item_map = {i["item_id"]: i for i in items if isinstance(i, dict) and "item_id" in i}
    computed_prof, computed_rls, computed_missing = _recompute_proficiency(item_map)
    item_12 = item_map.get(12)
    if item_12:
        model_prof = item_12.get("score")
        if model_prof != computed_prof:
            warnings.append(f"Proficiency override: model said {model_prof}, rules say {computed_prof}")
            item_12["score"] = computed_prof
            item_12["evidence"] = {
                "red_lines_triggered": computed_rls,
                "missing_core_domains": computed_missing,
            }

    # Ensure actionable coaching is present. Coaching is ONLY for checklist
    # items scored NO (an observed failure); backfill from rule-based templates
    # when missing. NULL (not observed) and YES items must carry no coaching.
    # For economy (item 11), enforce exact mastery text at 5 and a
    # recommendation below 5; NULL economy carries no coaching.
    for it in items:
        if not isinstance(it, dict):
            continue
        iid = it.get("item_id")
        if iid in CHECKLIST_ITEMS:
            sc = it.get("score")
            if sc == "NO":
                if not (it.get("coaching") or "").strip():
                    fallback = _coaching_for_item(iid, sc)
                    if fallback:
                        it["coaching"] = fallback
            else:
                # NULL (not observed), YES, or other: no coaching.
                it.pop("coaching", None)
        elif iid == 11:
            sc = it.get("score")
            if sc == 5:
                it["coaching"] = ECONOMY_MASTERY_TEXT
            elif isinstance(sc, int):
                if not (it.get("coaching") or "").strip():
                    it["coaching"] = _economy_coaching(sc, it.get("evidence", {}))
            else:
                it.pop("coaching", None)

    coverage = compute_coverage(items)
    if isinstance(parsed, dict) and "evidence_based" in parsed:
        parsed["evidence_based"]["coverage"] = coverage
    elif isinstance(parsed, dict):
        parsed["coverage"] = coverage

    return parsed, warnings


def _economy_guardrail_check(ei: float, score: Any, flow: str, ei_per_min: Optional[float] = None) -> Optional[int]:
    """Check economy score vs Economy Index (2023 rubric, out of 5). Returns corrected
    score or None if OK. No per-minute normalization used.
    EI mapping: EI 0 & organized→5, 0-1→4, 2-3→3, 4-6→2, ≥7→1.
    """
    if not isinstance(score, int):
        return None

    if ei == 0 and flow == "organized":
        return None if score == 5 else 5

    if ei <= 1 and score == 4:
        return None
    if 2 <= ei <= 3 and score == 3:
        return None
    if 4 <= ei <= 6 and score == 2:
        return None
    if ei >= 7 and score == 1:
        return None

    if ei <= 1:
        return 4
    elif ei <= 3:
        return 3
    elif ei <= 6:
        return 2
    else:
        return 1


def _recompute_proficiency(item_map: Dict[int, Dict]) -> Tuple[str, List[str], List[str]]:
    """Recompute item 12 proficiency from core domains using 2023 rubric red-line rules.
    Returns (proficiency_score, red_lines, missing_core_domains).
    """
    red_lines = []
    missing_core = []

    core_items_config = [
        (2, "RL-C", "back wall injury"),
        (7, "RL-A", "back wall gaping"),
        (9, "RL-B", "front wall gaping"),
        (10, "RL-D", "insecure knot"),
    ]

    for iid, rl_code, desc in core_items_config:
        entry = item_map.get(iid, {})
        score = entry.get("score")
        if score == "NO":
            red_lines.append(f"{rl_code}: item {iid} = NO ({desc})")
        elif score == "NULL" or score is None:
            missing_core.append(f"Item {iid} ({desc})")

    item_4 = item_map.get(4, {})
    if item_4.get("score") == "NO" and item_4.get("observability") == "OBSERVED":
        red_lines.append("RL-E: item 4 = NO (heel establishment failure)")

    if red_lines:
        return "NOT_PROFICIENT", red_lines, missing_core
    if missing_core:
        return "INSUFFICIENT_EVIDENCE", [], missing_core
    return "PROFICIENT", [], []


def build_result_json(filename: str, response_text: str, timestamp: str = "",
                      scored_outputs: Optional[List[Dict[str, Any]]] = None,
                      scoring_warnings: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Build records from Stage 1 FOR response.

    If scored_outputs is provided (v4.1 LLM path), uses pre-scored/validated
    outputs. Otherwise falls back to deterministic scoring (v4 path).
    """
    video_id = extract_video_id(filename)
    year, expected_ids = parse_candidate_ids_from_filename(filename)

    try:
        parsed = parse_json_response(response_text)
    except (json.JSONDecodeError, Exception):
        parsed = {"observations": {}, "video_coverage": {}, "parse_error": True}

    candidates = normalize_to_candidate_list(parsed)
    if not candidates:
        candidates = [{"observations": {}, "video_coverage": {}}]

    if len(expected_ids) > 1 and len(candidates) != len(expected_ids):
        print(f"WARNING: Expected {len(expected_ids)} candidates from filename but AI returned {len(candidates)}")

    records = []
    for idx, for_data in enumerate(candidates):
        cid = expected_ids[idx] if idx < len(expected_ids) else str(idx + 1) if len(candidates) > 1 else video_id
        if not for_data.get("case_id"):
            for_data["case_id"] = make_case_id(year, cid)

        segment_start = for_data.pop("segment_start", "") if "segment_start" in for_data else ""

        scored = scored_outputs[idx] if scored_outputs and idx < len(scored_outputs) else None

        record = _build_v4_record(
            for_data, video_id,
            year=year,
            candidate_id=cid,
            segment_start=segment_start,
            timestamp=timestamp,
            scored_output=scored,
            scoring_warnings=scoring_warnings,
        )
        records.append(record)

    return records


def get_canonical_output(record: Dict[str, Any], projection_mode: str = "construct_mode") -> Dict[str, Any]:
    """Extract the canonical vop 2023 v1 output schema.

    Returns both evidence_based and projected_docx.
    """
    canonical_items = []
    for item in record.get("items", []):
        if not isinstance(item, dict):
            continue
        item_id = item.get("item_id")
        ci: Dict[str, Any] = {
            "item_id": item_id,
            "score": item.get("score"),
            "observability": item.get("observability"),
            "evidence": item.get("evidence"),
        }
        if item.get("coaching"):
            ci["coaching"] = item["coaching"]
        if item_id == 6 and "subitems" in item:
            ci["subitems"] = item["subitems"]
        if item_id == PROFICIENCY_ITEM:
            evidence = item.get("evidence")
            if isinstance(evidence, str):
                ci["evidence"] = {
                    "red_lines_triggered": record.get("red_lines_triggered", []),
                    "missing_core_domains": record.get("missing_core_domains", []),
                }
        if item_id == COMMENTS_ITEM:
            evidence = item.get("evidence")
            if isinstance(evidence, str):
                coaching_tags = item.get("coaching_tags", [])
                ci["evidence"] = {"coaching_tags": coaching_tags}
        canonical_items.append(ci)

    projected = record.get("projected_docx")
    if not projected or projected.get("mode") != projection_mode:
        projected = project_to_docx(record.get("items", []), projection_mode)

    return {
        "case_id": record.get("case_id", ""),
        "rubric_version": record.get("rubric_version", RUBRIC_VERSION),
        "evidence_based": {
            "items": canonical_items,
            "coverage": record.get("coverage", compute_coverage(record.get("items", []))),
        },
        "projected_docx": projected,
    }


class GeminiVisionClient:
    """Client for Stage 1: Factual Observation Record extraction via Gemini."""

    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found. Please add it in Secrets.")
        # Client-wide HTTP timeout (ms): aborts stalled requests (notably large
        # video uploads) at the HTTP layer instead of hanging indefinitely.
        self.client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options=types.HttpOptions(timeout=UPLOAD_TIMEOUT_SECONDS * 1000),
        )
        self.last_response = ""

    def _build_for_prompt(self, candidate_ids: Optional[List[str]] = None, case_id: str = "", num_parts: int = 0) -> str:
        parts_block = ""
        if num_parts > 1:
            parts_block = f"""**MULTI-PART VIDEO — {num_parts} CONSECUTIVE PARTS**
This video has been split into {num_parts} parts for upload. They are sequential segments of ONE continuous recording.
Treat all {num_parts} parts as a single uninterrupted video. A candidate's procedure may span across part boundaries — do NOT treat a part boundary as a new candidate or new procedure. Stitch your observations across parts seamlessly.

"""

        if candidate_ids and len(candidate_ids) > 1:
            candidate_list = ", ".join(candidate_ids)
            multi_block = f"""**MULTI-CANDIDATE VIDEO — {len(candidate_ids)} CANDIDATES EXPECTED**
This video contains {len(candidate_ids)} candidates performing separate VOP segments sequentially.
The candidate IDs are: {candidate_list}

INSTRUCTIONS:
1. Return a JSON **array** of exactly {len(candidate_ids)} objects — each following the FOR schema below.
2. Look for visual cues of a NEW candidate segment: new hands/gloves, station reset, title card, abrupt scene change.
3. Set each object's "case_id" to the candidate ID in order: first segment = {candidate_ids[0]}, second = {candidate_ids[1]}, etc.
4. Add "segment_start" field to each object with approximate timestamp (e.g., "0:00", "5:23").
5. Observe each candidate's procedure independently.
6. A part boundary (transition between uploaded video files) is NOT a candidate boundary — only visual cues indicate a new candidate.

"""
        else:
            cid = candidate_ids[0] if candidate_ids else (case_id or "unknown")
            multi_block = f"""**SINGLE-CANDIDATE VIDEO**
Set "case_id" to "{cid}".

"""
        return parts_block + multi_block + FOR_PROMPT

    def _validate_for_response(self, response_text: str) -> Tuple[bool, Dict[str, Any]]:
        try:
            parsed = parse_json_response(response_text)
            candidates = normalize_to_candidate_list(parsed)
            if not candidates:
                return False, {
                    "json_parsed": True,
                    "response_length": len(response_text),
                    "candidate_count": 0,
                    "issues": ["No FOR objects found in response"],
                }

            all_issues: List[str] = []
            for idx, cand in enumerate(candidates):
                is_valid, details = validate_for(cand)
                if not is_valid:
                    cid = cand.get("case_id", idx + 1)
                    for issue in details.get("issues", []):
                        all_issues.append(f"Candidate {cid}: {issue}")

            return len(all_issues) == 0, {
                "json_parsed": True,
                "response_length": len(response_text),
                "candidate_count": len(candidates),
                "issues": all_issues,
            }
        except json.JSONDecodeError as e:
            return False, {
                "json_parsed": False,
                "json_error": str(e),
                "response_length": len(response_text),
                "candidate_count": 0,
                "issues": [f"Failed to parse JSON: {e}"],
            }

    def _analyze_failure(
        self, response_text: str, validation_details: Dict[str, Any]
    ) -> Tuple[str, List[str]]:
        analysis_parts = []
        advice = []

        if not validation_details.get("json_parsed", False):
            analysis_parts.append("The AI did not return valid JSON.")
            advice.append("Retry the assessment — this may be a transient formatting issue.")
        else:
            issues = validation_details.get("issues", [])
            if issues:
                analysis_parts.append(f"FOR validation issues: {'; '.join(issues[:5])}")
                advice.append("The FOR schema may be incomplete. Check video quality and duration.")

        if not analysis_parts:
            analysis_parts.append("Unknown validation failure.")
            advice.append("Try again with a different video or check video format/quality.")

        return " ".join(analysis_parts), advice

    def _get_video_duration(self, video_path: str) -> float:
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
                capture_output=True, text=True, timeout=30
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0

    def _convert_to_mp4(self, video_path: str) -> str:
        ext = Path(video_path).suffix.lower()
        unsupported = {'.m4v', '.avi', '.mkv', '.flv', '.wmv'}
        if ext not in unsupported:
            return video_path

        output_path = tempfile.mktemp(suffix='.mp4')
        try:
            print(f"Trying fast remux (stream copy) for {ext} → .mp4...")
            subprocess.run(
                ['ffmpeg', '-i', video_path, '-c', 'copy',
                 '-movflags', '+faststart', '-y', output_path],
                capture_output=True, timeout=120, check=True
            )
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print("Fast remux succeeded")
                return output_path
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            print("Fast remux failed, falling back to full re-encode...")
            if os.path.exists(output_path):
                os.unlink(output_path)

        duration = self._get_video_duration(video_path)
        encode_timeout = max(600, int(duration * 0.5)) if duration > 0 else 1800
        try:
            subprocess.run(
                ['ffmpeg', '-i', video_path, '-c:v', 'libx264', '-c:a', 'aac',
                 '-movflags', '+faststart', '-y', output_path],
                capture_output=True, timeout=encode_timeout, check=True
            )
            return output_path
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"FFmpeg re-encode also failed: {e}")
            if os.path.exists(output_path):
                os.unlink(output_path)
            return video_path

    def _split_video(self, video_path: str, num_candidates: int = 1) -> List[str]:
        duration = self._get_video_duration(video_path)

        if num_candidates > 1 and duration > 0:
            per_candidate = duration / num_candidates
            segment_len = min(per_candidate + 120, SEGMENT_DURATION_SECONDS)
            segment_len = max(segment_len, 300)
            num_parts = int(duration // segment_len) + (1 if duration % segment_len > 0 else 0)
            num_parts = max(num_parts, num_candidates)
            segment_len = duration / num_parts
            print(f"Multi-candidate video ({num_candidates} candidates, {duration:.0f}s / {duration/60:.1f} min), "
                  f"splitting into {num_parts} segments of ~{segment_len/60:.1f} min each")
        elif duration <= 0 or duration <= MAX_VIDEO_DURATION_SECONDS:
            return [video_path]
        else:
            segment_len = SEGMENT_DURATION_SECONDS
            num_parts = int(duration // segment_len) + (1 if duration % segment_len > 0 else 0)
            print(f"Video is {duration:.0f}s ({duration/60:.1f} min), splitting into {num_parts} parts of ~{segment_len/60:.0f} min each")

        is_mp4 = Path(video_path).suffix.lower() in ('.mp4', '.mov')

        seg_int = int(segment_len)
        segment_paths: List[str] = []
        for i in range(num_parts):
            start = int(i * segment_len)
            seg_path = tempfile.mktemp(suffix=f'_part{i+1}.mp4')
            success = False

            if is_mp4:
                try:
                    cmd = [
                        'ffmpeg', '-ss', str(start),
                        '-i', video_path,
                        '-t', str(seg_int),
                        '-c', 'copy',
                        '-movflags', '+faststart',
                        '-y', seg_path,
                    ]
                    subprocess.run(cmd, capture_output=True, timeout=120, check=True)
                    if os.path.exists(seg_path) and os.path.getsize(seg_path) > 0:
                        success = True
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
                    if os.path.exists(seg_path):
                        os.unlink(seg_path)

            if not success:
                try:
                    cmd = [
                        'ffmpeg', '-i', video_path,
                        '-ss', str(start),
                        '-t', str(seg_int),
                        '-c:v', 'libx264', '-c:a', 'aac',
                        '-movflags', '+faststart',
                        '-y', seg_path,
                    ]
                    subprocess.run(cmd, capture_output=True, timeout=900, check=True)
                    success = True
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
                    print(f"FFmpeg split failed for part {i+1}: {e}")
                    for p in segment_paths:
                        if os.path.exists(p):
                            os.unlink(p)
                    return [video_path]

            segment_paths.append(seg_path)
            end_s = min(start + seg_int, int(duration))
            print(f"  Part {i+1}/{num_parts}: {start//60}:{start%60:02d} - {end_s//60}:{end_s%60:02d}")

        return segment_paths

    @retry(
        stop=stop_after_attempt(7),
        wait=wait_exponential(multiplier=30, min=30, max=300),
        retry=retry_if_exception(is_retryable_error),
        before_sleep=_before_retry_log,
        reraise=True,
    )
    def _score_from_for(self, for_json: Dict[str, Any]) -> str:
        """Stage 2: Send FOR JSON to Gemini for scoring."""
        prompt = SCORING_PROMPT + "\n\nFOR INPUT:\n" + json.dumps(for_json, indent=2)
        config = types.GenerateContentConfig(
            temperature=0.1,
            thinking_config=types.ThinkingConfig(thinking_budget=8000),
        )
        response = self.client.models.generate_content(
            model=MODEL_VERSION,
            contents=[prompt],
            config=config,
        )
        result_text = ""
        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                        result_text += part.text
        return result_text

    @staticmethod
    def _is_token_or_size_error(err_str: str) -> bool:
        err_lower = err_str.lower() if not err_str.islower() else err_str
        token_patterns = [
            "token", "too long", "too large", "exceeds", "max_tokens",
            "content_too_large", "payload", "request too large",
            "400", "invalid_argument",
            "500", "internal", "failed to convert",
        ]
        return any(p in err_lower for p in token_patterns)

    def _split_video_forced(self, video_path: str, num_parts: int) -> List[str]:
        duration = self._get_video_duration(video_path)
        if duration <= 0 or num_parts <= 1:
            return [video_path]
        segment_len = duration / num_parts
        print(f"Force-splitting {duration:.0f}s video into {num_parts} segments of ~{segment_len/60:.1f} min")
        is_mp4 = Path(video_path).suffix.lower() in ('.mp4', '.mov')
        seg_int = int(segment_len)
        segments: List[str] = []
        for i in range(num_parts):
            start = int(i * segment_len)
            seg_path = tempfile.mktemp(suffix=f'_fsplit{i+1}.mp4')
            try:
                if is_mp4:
                    cmd = ['ffmpeg', '-ss', str(start), '-i', video_path,
                           '-t', str(seg_int), '-c', 'copy', '-movflags', '+faststart', '-y', seg_path]
                else:
                    cmd = ['ffmpeg', '-i', video_path, '-ss', str(start),
                           '-t', str(seg_int), '-c:v', 'libx264', '-c:a', 'aac',
                           '-movflags', '+faststart', '-y', seg_path]
                subprocess.run(cmd, capture_output=True, timeout=300, check=True)
                if os.path.exists(seg_path) and os.path.getsize(seg_path) > 0:
                    segments.append(seg_path)
                    end_s = min(start + seg_int, int(duration))
                    print(f"  Force split {i+1}/{num_parts}: {start//60}:{start%60:02d} - {end_s//60}:{end_s%60:02d}")
            except Exception as e:
                print(f"  Force split failed for part {i+1}: {e}")
                for p in segments:
                    if os.path.exists(p):
                        os.unlink(p)
                return [video_path]
        return segments if segments else [video_path]

    @staticmethod
    def _as_video_part(video_file: Any) -> Any:
        """Wrap an uploaded video File in a Part carrying fps sampling metadata.

        Sets video_metadata=VideoMetadata(fps=VIDEO_SAMPLING_FPS) so the model
        samples frames at the configured rate (3 fps).
        """
        try:
            file_uri = getattr(video_file, "uri", None)
            mime_type = getattr(video_file, "mime_type", None) or "video/mp4"
            if file_uri:
                return types.Part(
                    file_data=types.FileData(file_uri=file_uri, mime_type=mime_type),
                    video_metadata=types.VideoMetadata(fps=VIDEO_SAMPLING_FPS),
                )
        except Exception:
            pass
        # Fallback: pass the raw file object if we can't build a Part.
        return video_file

    @retry(
        stop=stop_after_attempt(7),
        wait=wait_exponential(multiplier=30, min=30, max=300),
        retry=retry_if_exception(is_retryable_error),
        before_sleep=_before_retry_log,
        reraise=True,
    )
    def _call_gemini(self, video_files: Any, prompt: str) -> str:
        config = types.GenerateContentConfig(
            temperature=0.1,
            thinking_config=types.ThinkingConfig(thinking_budget=8000),
            media_resolution=types.MediaResolution("MEDIA_RESOLUTION_HIGH"),
        )
        if isinstance(video_files, list):
            parts = [self._as_video_part(vf) for vf in video_files]
            contents = parts + [prompt]
        else:
            contents = [self._as_video_part(video_files), prompt]
        response = self.client.models.generate_content(
            model=MODEL_VERSION,
            contents=contents,
            config=config,
        )
        result_text = ""
        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                        result_text += part.text
        return result_text

    def _compress_for_upload(self, file_path: str) -> str:
        """Re-encode oversized videos to a lower bitrate before upload.

        Returns a path to a compressed temp file, or the original path if the
        file is small enough or compression fails/doesn't help. Caller is
        responsible for deleting the returned file if it differs from input.
        """
        try:
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
        except OSError:
            return file_path
        if size_mb <= UPLOAD_COMPRESS_THRESHOLD_MB:
            return file_path

        duration = self._get_video_duration(file_path)
        encode_timeout = max(600, int(duration * 1.0)) if duration > 0 else 1800
        output_path = tempfile.mktemp(suffix='_upload.mp4')
        print(f"  Video is {size_mb:.0f}MB (> {UPLOAD_COMPRESS_THRESHOLD_MB}MB), "
              f"compressing before upload (720p, CRF 28)...")
        t0 = time.time()
        try:
            subprocess.run(
                ['ffmpeg', '-i', file_path,
                 '-vf', "scale='min(1280,iw)':-2",
                 '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '28',
                 '-c:a', 'aac', '-b:a', '64k',
                 '-movflags', '+faststart', '-y', output_path],
                capture_output=True, timeout=encode_timeout, check=True,
            )
            new_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            if new_size_mb > 0 and new_size_mb < size_mb:
                print(f"  Compressed {size_mb:.0f}MB → {new_size_mb:.0f}MB "
                      f"in {time.time() - t0:.0f}s")
                return output_path
            # Compression didn't help; use original.
            os.unlink(output_path)
            return file_path
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                FileNotFoundError, OSError) as e:
            print(f"  Pre-upload compression failed ({e}); uploading original file")
            if os.path.exists(output_path):
                try:
                    os.unlink(output_path)
                except OSError:
                    pass
            return file_path

    def _delete_remote_file(self, video_file: Any) -> None:
        """Best-effort deletion of a remote Gemini file (data-retention hygiene)."""
        name = getattr(video_file, "name", None)
        if name:
            try:
                self.client.files.delete(name=name)
                print(f"  Deleted remote file {name}")
            except Exception as e:
                print(f"  Warning: could not delete remote file {name}: {e}")

    def _upload_once(self, file_path: str, mime_type: str) -> Any:
        """Single upload attempt with an HTTP-level timeout, then wait for processing.

        The timeout is enforced at the Gemini HTTP request layer (via
        HttpOptions.timeout), so a stalled upload is aborted client-side rather
        than left running in the background. If the upload succeeds but
        Gemini-side processing fails or times out, the remote file is deleted
        before the error is raised so retries never leave orphaned trainee
        videos on Google's servers.
        """
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"  Uploading {Path(file_path).name} ({size_mb:.0f}MB, "
              f"timeout {UPLOAD_TIMEOUT_SECONDS}s)...")
        t0 = time.time()

        video_file = self.client.files.upload(
            file=file_path,
            config=types.UploadFileConfig(mime_type=mime_type),
        )

        print(f"  Upload finished in {time.time() - t0:.0f}s; waiting for Gemini processing...")

        try:
            proc_start = time.time()
            while video_file.state and video_file.state.name == "PROCESSING":
                if time.time() - proc_start > FILE_PROCESSING_TIMEOUT_SECONDS:
                    raise TimeoutError(
                        f"Gemini file processing for {Path(file_path).name} timed out "
                        f"after {FILE_PROCESSING_TIMEOUT_SECONDS}s"
                    )
                time.sleep(5)
                if video_file.name:
                    video_file = self.client.files.get(name=video_file.name)

            if video_file.state and video_file.state.name == "FAILED":
                raise RuntimeError(
                    f"Gemini reported FAILED state for uploaded file {Path(file_path).name}"
                )
        except Exception:
            # Don't leave the uploaded video orphaned on Gemini before retrying.
            self._delete_remote_file(video_file)
            raise

        print(f"  File ready ({video_file.state.name if video_file.state else 'unknown state'}) "
              f"after {time.time() - t0:.0f}s total")
        return video_file

    def _upload_and_wait(self, file_path: str, mime_type: str = "video/mp4") -> Any:
        """Upload a video with pre-compression, timeout, and retries."""
        upload_path = self._compress_for_upload(file_path)
        if upload_path != file_path:
            # Compression always produces an MP4 container regardless of source.
            mime_type = "video/mp4"
        try:
            last_exc: Optional[BaseException] = None
            for attempt in range(1, UPLOAD_MAX_ATTEMPTS + 1):
                try:
                    return self._upload_once(upload_path, mime_type)
                except Exception as e:
                    last_exc = e
                    if attempt < UPLOAD_MAX_ATTEMPTS:
                        wait_s = 30 * attempt if is_rate_limit_error(e) else 10 * attempt
                        print(f"  Upload attempt {attempt}/{UPLOAD_MAX_ATTEMPTS} failed: {e}. "
                              f"Retrying in {wait_s}s...")
                        time.sleep(wait_s)
                    else:
                        print(f"  Upload failed after {UPLOAD_MAX_ATTEMPTS} attempts: {e}")
            assert last_exc is not None
            raise last_exc
        finally:
            if upload_path != file_path and os.path.exists(upload_path):
                try:
                    os.unlink(upload_path)
                except OSError:
                    pass

    def _cleanup_uploaded(self, uploaded_files: List[Any]) -> None:
        for vf in uploaded_files:
            if vf.name:
                try:
                    self.client.files.delete(name=vf.name)
                except Exception:
                    pass

    @staticmethod
    def _merge_for_data(existing: Dict[str, Any], new_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deep-merge two FOR dicts for the same candidate across segments.

        Strategy: for each top-level key, merge observations additively.
        - 'observations' dict: merge sub-dicts, preferring non-empty/non-default values
        - Lists: concatenate and deduplicate
        - Scalars: prefer the value that has more information (non-empty, non-null)
        - 'video_coverage': merge segment info
        """
        merged = dict(existing)

        for key, new_val in new_data.items():
            if key == "case_id":
                continue

            old_val = merged.get(key)

            if key == "observations" and isinstance(old_val, dict) and isinstance(new_val, dict):
                merged_obs = dict(old_val)
                for obs_key, obs_val in new_val.items():
                    if obs_key not in merged_obs:
                        merged_obs[obs_key] = obs_val
                    elif isinstance(obs_val, dict) and isinstance(merged_obs[obs_key], dict):
                        sub_merged = dict(merged_obs[obs_key])
                        for sk, sv in obs_val.items():
                            existing_sv = sub_merged.get(sk)
                            if existing_sv is None or existing_sv == "" or existing_sv == "unknown" or existing_sv == "not_visible":
                                sub_merged[sk] = sv
                            elif isinstance(sv, list) and isinstance(existing_sv, list):
                                combined = existing_sv + [x for x in sv if x not in existing_sv]
                                sub_merged[sk] = combined
                        merged_obs[obs_key] = sub_merged
                    elif isinstance(obs_val, list) and isinstance(merged_obs[obs_key], list):
                        combined = merged_obs[obs_key] + [x for x in obs_val if x not in merged_obs[obs_key]]
                        merged_obs[obs_key] = combined
                    elif merged_obs[obs_key] is None or merged_obs[obs_key] == "" or merged_obs[obs_key] == "unknown":
                        merged_obs[obs_key] = obs_val
                merged["observations"] = merged_obs

            elif key == "economy_markers" and isinstance(old_val, dict) and isinstance(new_val, dict):
                merged_econ = dict(old_val)
                for ek, ev in new_val.items():
                    if ek == "events" and isinstance(ev, list):
                        existing_events = merged_econ.get("events", [])
                        merged_econ["events"] = existing_events + ev
                    elif ek not in merged_econ or merged_econ[ek] is None:
                        merged_econ[ek] = ev
                merged["economy_markers"] = merged_econ

            elif key == "video_coverage" and isinstance(old_val, dict) and isinstance(new_val, dict):
                merged_vc = dict(old_val)
                for vk, vv in new_val.items():
                    if vk not in merged_vc or merged_vc[vk] is None:
                        merged_vc[vk] = vv
                merged["video_coverage"] = merged_vc

            elif isinstance(new_val, list) and isinstance(old_val, list):
                merged[key] = old_val + [x for x in new_val if x not in old_val]

            elif old_val is None or old_val == "" or old_val == "unknown":
                merged[key] = new_val

        merged["_merged_from_segments"] = True
        return merged

    def _process_segments_individually(
        self, segment_paths: List[str], candidate_ids: List[str],
        year: str, case_id: str, mime_type: str
    ) -> str:
        """Fallback: process each video segment individually and merge FOR results by case_id."""
        fors_by_case_id: Dict[str, Dict[str, Any]] = {}
        segment_failures: List[str] = []

        for seg_idx, seg_path in enumerate(segment_paths):
            print(f"  Sequential processing: segment {seg_idx + 1}/{len(segment_paths)}")
            vf = None
            try:
                vf = self._upload_and_wait(seg_path, mime_type)
            except Exception as e:
                print(f"  Upload failed for segment {seg_idx + 1}: {e}")
                segment_failures.append(f"Segment {seg_idx + 1} upload failed: {e}")
                continue

            try:
                seg_prompt = self._build_for_prompt(
                    candidate_ids=candidate_ids,
                    case_id=case_id,
                    num_parts=0,
                )
                seg_prompt = (
                    f"**VIDEO SEGMENT {seg_idx + 1} of {len(segment_paths)}**\n"
                    f"This is one part of a longer recording split for processing. "
                    f"Observe ALL candidates visible in this segment. A candidate's "
                    f"procedure may span across segments — report everything you observe "
                    f"in THIS segment even if the candidate also appeared in another. "
                    f"Return FOR(s) only for candidates you actually observe performing "
                    f"a procedure in this segment.\n\n"
                    + seg_prompt
                )

                response = self._call_gemini(vf, seg_prompt)

                try:
                    parsed = parse_json_response(response)
                    seg_candidates = normalize_to_candidate_list(parsed)
                    for c in seg_candidates:
                        cid = c.get("case_id", "")
                        if cid and cid not in fors_by_case_id:
                            fors_by_case_id[cid] = c
                        elif cid and cid in fors_by_case_id:
                            print(f"  Merging case_id {cid} from segment {seg_idx + 1} into existing FOR")
                            fors_by_case_id[cid] = self._merge_for_data(fors_by_case_id[cid], c)
                        else:
                            fallback_id = f"segment_{seg_idx + 1}_candidate_{len(fors_by_case_id) + 1}"
                            c["case_id"] = fallback_id
                            fors_by_case_id[fallback_id] = c
                except Exception as parse_err:
                    print(f"  Failed to parse segment {seg_idx + 1} response: {parse_err}")
                    segment_failures.append(f"Segment {seg_idx + 1} parse failed: {parse_err}")

            except Exception as e:
                print(f"  Gemini call failed for segment {seg_idx + 1}: {e}")
                segment_failures.append(f"Segment {seg_idx + 1} API failed: {e}")
            finally:
                if vf:
                    try:
                        self._cleanup_uploaded([vf])
                    except Exception:
                        pass

        all_fors = list(fors_by_case_id.values())

        if segment_failures:
            print(f"  Segment failures: {segment_failures}")

        if len(all_fors) == 0:
            failure_detail = "; ".join(segment_failures) if segment_failures else "Unknown"
            print(f"Sequential processing failed: no FORs extracted. Failures: {failure_detail}")
            return json.dumps({"observations": {}, "video_coverage": {}, "parse_error": True,
                               "error": f"Sequential fallback failed: {failure_detail}"})

        print(f"Sequential processing complete: {len(all_fors)} FOR(s) from {len(segment_paths)} segments"
              + (f" ({len(segment_failures)} segment failures)" if segment_failures else ""))

        if len(candidate_ids) > 1 and len(all_fors) != len(candidate_ids):
            print(f"  WARNING: Expected {len(candidate_ids)} candidates but got {len(all_fors)} FORs")

        if len(all_fors) == 1:
            return json.dumps(all_fors[0])
        else:
            return json.dumps(all_fors)

    def _process_focused_segments_individually(
        self, segment_paths: List[str], candidate_ids: List[str],
        year: str, case_id: str, mime_type: str, prompt: str
    ) -> str:
        """Process focused Items 1&3 per segment, merging observations by case_id."""
        obs_by_case_id: Dict[str, Dict[str, Any]] = {}
        failures: List[str] = []

        for seg_idx, seg_path in enumerate(segment_paths):
            print(f"  [Items 1&3] Sequential: segment {seg_idx + 1}/{len(segment_paths)}")
            vf = None
            try:
                vf = self._upload_and_wait(seg_path, mime_type)
            except Exception as e:
                print(f"  Upload failed for segment {seg_idx + 1}: {e}")
                failures.append(f"Segment {seg_idx + 1} upload failed: {e}")
                continue

            try:
                seg_prompt = (
                    f"**VIDEO SEGMENT {seg_idx + 1} of {len(segment_paths)}**\n"
                    f"This is one part of a longer recording. Observe ALL candidates "
                    f"visible in this segment. A candidate's procedure may span across "
                    f"segments — report everything you observe in THIS segment even if "
                    f"the candidate also appeared in another. Return observations only "
                    f"for candidates you actually see performing in this segment.\n\n"
                    + prompt
                )
                response = self._call_gemini(vf, seg_prompt)

                try:
                    parsed = parse_json_response(response)
                    if isinstance(parsed, list):
                        seg_obs = parsed
                    else:
                        seg_obs = [parsed]

                    for obs in seg_obs:
                        if not isinstance(obs, dict):
                            continue
                        cid = obs.get("case_id", "")
                        if cid and cid not in obs_by_case_id:
                            obs_by_case_id[cid] = obs
                        elif cid and cid in obs_by_case_id:
                            print(f"  [Items 1&3] Merging case_id {cid} from segment {seg_idx + 1}")
                            obs_by_case_id[cid] = self._merge_for_data(obs_by_case_id[cid], obs)
                        elif not cid:
                            fallback_id = f"{year}_{seg_idx + 1}_unknown"
                            obs["case_id"] = fallback_id
                            obs_by_case_id[fallback_id] = obs
                except Exception as parse_err:
                    print(f"  Parse failed for segment {seg_idx + 1}: {parse_err}")
                    failures.append(f"Segment {seg_idx + 1} parse failed: {parse_err}")

            except Exception as e:
                print(f"  Gemini call failed for segment {seg_idx + 1}: {e}")
                failures.append(f"Segment {seg_idx + 1} API failed: {e}")
            finally:
                if vf:
                    try:
                        self._cleanup_uploaded([vf])
                    except Exception:
                        pass

        all_obs = list(obs_by_case_id.values())

        if failures:
            print(f"  [Items 1&3] Segment failures: {failures}")

        if len(all_obs) == 0:
            failure_detail = "; ".join(failures) if failures else "Unknown"
            return json.dumps({"error": f"Sequential focused fallback failed: {failure_detail}"})

        print(f"  [Items 1&3] Sequential complete: {len(all_obs)} observation(s) from {len(segment_paths)} segments")

        if len(all_obs) == 1:
            return json.dumps(all_obs[0])
        else:
            return json.dumps(all_obs)

    def assess_video(self, video_path: str, filename: str = "") -> Dict[str, Any]:
        """Stage 1: Extract FOR from video, then Stage 2: LLM scoring with validators."""
        if not filename:
            filename = Path(video_path).name

        year, candidate_ids = parse_candidate_ids_from_filename(filename)
        case_id = make_case_id(year, candidate_ids[0]) if candidate_ids else ""

        converted_path = self._convert_to_mp4(video_path)
        converted = converted_path != video_path

        segment_paths = self._split_video(converted_path, num_candidates=len(candidate_ids))
        is_multipart = len(segment_paths) > 1

        uploaded_files: List[Any] = []
        temp_segments: List[str] = []

        try:
            mime_type = "video/mp4"
            ext = Path(converted_path).suffix.lower()
            mime_map = {'.mp4': 'video/mp4', '.mov': 'video/quicktime', '.webm': 'video/webm'}
            mime_type = mime_map.get(ext, 'video/mp4')

            if is_multipart:
                temp_segments = [p for p in segment_paths if p != converted_path]
                print(f"Uploading {len(segment_paths)} video parts for: {filename}")
                for i, seg_path in enumerate(segment_paths):
                    print(f"  Uploading part {i+1}/{len(segment_paths)}...")
                    vf = self._upload_and_wait(seg_path, mime_type)
                    uploaded_files.append(vf)
            else:
                print(f"Uploading video: {filename} ({mime_type})")
                vf = self._upload_and_wait(converted_path, mime_type)
                uploaded_files.append(vf)

            prompt = self._build_for_prompt(
                candidate_ids=candidate_ids,
                case_id=case_id,
                num_parts=len(segment_paths) if is_multipart else 0,
            )
            print(f"Sending Stage 1 FOR prompt for: {filename}")

            response_text = ""
            used_sequential_fallback = False

            if is_multipart:
                try:
                    response_text = self._call_gemini(uploaded_files, prompt)
                except Exception as multi_err:
                    err_str = str(multi_err).lower()
                    if self._is_token_or_size_error(err_str):
                        print(f"Multi-part API call failed ({multi_err}), falling back to sequential segment processing...")
                        self._cleanup_uploaded(uploaded_files)
                        uploaded_files = []
                        response_text = self._process_segments_individually(
                            segment_paths, candidate_ids, year, case_id, mime_type
                        )
                        used_sequential_fallback = True
                    else:
                        raise
            else:
                try:
                    response_text = self._call_gemini(uploaded_files[0], prompt)
                except Exception as single_err:
                    err_str = str(single_err).lower()
                    if self._is_token_or_size_error(err_str) and len(candidate_ids) > 1:
                        print(f"Single-file API call hit token limit ({single_err}), splitting and retrying sequentially...")
                        self._cleanup_uploaded(uploaded_files)
                        uploaded_files = []
                        force_segments = self._split_video_forced(converted_path, len(candidate_ids))
                        if len(force_segments) > 1:
                            temp_segments.extend([p for p in force_segments if p != converted_path])
                            response_text = self._process_segments_individually(
                                force_segments, candidate_ids, year, case_id, mime_type
                            )
                            used_sequential_fallback = True
                        else:
                            raise
                    else:
                        raise

            self.last_response = response_text
            if uploaded_files:
                self._cleanup_uploaded(uploaded_files)
                uploaded_files = []

            is_valid, validation_details = self._validate_for_response(response_text)
            if used_sequential_fallback:
                validation_details["sequential_fallback"] = True

            if not is_valid:
                issues = validation_details.get("issues", [])
                if validation_details.get("json_parsed") and validation_details.get("candidate_count", 0) > 0:
                    print(f"FOR validation warnings (proceeding): {issues}")
                else:
                    analysis, advice = self._analyze_failure(response_text, validation_details)
                    return {
                        "success": False,
                        "validation_failed": True,
                        "analysis": analysis,
                        "advice": advice,
                        "validation_details": validation_details,
                        "response": response_text,
                    }

            try:
                parsed_for = parse_json_response(response_text)
                candidates = normalize_to_candidate_list(parsed_for)
            except Exception:
                candidates = []

            scored_outputs: List[Dict[str, Any]] = []
            all_scoring_warnings: List[str] = []
            stage2_success = False

            if candidates:
                print(f"Starting Stage 2 LLM scoring for {len(candidates)} candidate(s)...")
                for idx, for_data in enumerate(candidates):
                    try:
                        print(f"  Stage 2 scoring candidate {idx + 1}/{len(candidates)}...")
                        raw_scored = self._score_from_for(for_data)
                        validated_output, warnings = validate_scoring_output(raw_scored, for_data)
                        scored_outputs.append(validated_output)
                        if warnings:
                            all_scoring_warnings.extend(warnings)
                            print(f"  Stage 2 warnings for candidate {idx + 1}: {warnings}")
                        stage2_success = True
                    except Exception as e:
                        print(f"  Stage 2 scoring failed for candidate {idx + 1}: {e}")
                        all_scoring_warnings.append(f"Stage 2 failed for candidate {idx + 1}: {e}")
                        scored_outputs.append({})

            result: Dict[str, Any] = {
                "success": True,
                "response": response_text,
                "validation_details": validation_details,
            }

            if stage2_success and scored_outputs:
                result["scored_outputs"] = scored_outputs
                result["scoring_warnings"] = all_scoring_warnings

            return result

        finally:
            self._cleanup_uploaded(uploaded_files)
            if converted and os.path.exists(converted_path):
                os.unlink(converted_path)
            for seg in temp_segments:
                if os.path.exists(seg):
                    os.unlink(seg)

    def assess_video_items_1_3(self, video_path: str, filename: str = "") -> Dict[str, Any]:
        """Focused assessment: only Items 1 (arteriotomy) and 3 (spatulation).
        Uses a shorter prompt for faster processing and lower quota usage.
        """
        if not filename:
            filename = Path(video_path).name

        year, candidate_ids = parse_candidate_ids_from_filename(filename)
        case_id = make_case_id(year, candidate_ids[0]) if candidate_ids else ""

        converted_path = self._convert_to_mp4(video_path)
        converted = converted_path != video_path

        segment_paths = self._split_video(converted_path, num_candidates=len(candidate_ids))
        is_multipart = len(segment_paths) > 1

        uploaded_files: List[Any] = []
        temp_segments: List[str] = []

        try:
            mime_type = "video/mp4"
            ext = Path(converted_path).suffix.lower()
            mime_map = {'.mp4': 'video/mp4', '.mov': 'video/quicktime', '.webm': 'video/webm'}
            mime_type = mime_map.get(ext, 'video/mp4')

            if is_multipart:
                temp_segments = [p for p in segment_paths if p != converted_path]

            print(f"[Items 1&3] Uploading video: {filename}")
            for seg_path in segment_paths:
                vf = self._upload_and_wait(seg_path, mime_type)
                uploaded_files.append(vf)

            multi_block = ""
            if len(candidate_ids) > 1:
                cids = ", ".join(candidate_ids)
                multi_block = (
                    f"\nThis video contains MULTIPLE candidates: {cids}. "
                    f"Year: {year}. Return a JSON ARRAY with one object per candidate, "
                    f"each with case_id in format '{year}_<candidate_id>'.\n\n"
                )
            else:
                multi_block = f"\ncase_id: \"{case_id}\"\n\n"

            prompt = multi_block + FOCUSED_ITEMS_1_3_PROMPT

            print(f"[Items 1&3] Sending focused observation prompt for: {filename}")
            try:
                if len(uploaded_files) == 1:
                    obs_text = self._call_gemini(uploaded_files[0], prompt)
                else:
                    obs_text = self._call_gemini(uploaded_files, prompt)
            except Exception as api_err:
                err_str = str(api_err).lower()
                if self._is_token_or_size_error(err_str) and is_multipart:
                    print(f"[Items 1&3] Multi-part call hit token limit, processing segments individually...")
                    self._cleanup_uploaded(uploaded_files)
                    uploaded_files = []
                    obs_text = self._process_focused_segments_individually(
                        segment_paths, candidate_ids, year, case_id, mime_type, prompt
                    )
                elif self._is_token_or_size_error(err_str) and len(candidate_ids) > 1:
                    print(f"[Items 1&3] Single-file call hit token limit, force-splitting...")
                    self._cleanup_uploaded(uploaded_files)
                    uploaded_files = []
                    force_segments = self._split_video_forced(converted_path, len(candidate_ids))
                    if len(force_segments) > 1:
                        temp_segments.extend([p for p in force_segments if p != converted_path])
                        obs_text = self._process_focused_segments_individually(
                            force_segments, candidate_ids, year, case_id, mime_type, prompt
                        )
                    else:
                        raise
                else:
                    raise

            self._cleanup_uploaded(uploaded_files)
            uploaded_files = []

            try:
                parsed_obs = parse_json_response(obs_text)
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to parse Items 1&3 observations: {e}",
                    "response": obs_text,
                }

            if isinstance(parsed_obs, list):
                obs_list = parsed_obs
            else:
                obs_list = [parsed_obs]

            scored_outputs = []
            all_warnings: List[str] = []

            for obs_data in obs_list:
                if not isinstance(obs_data, dict):
                    continue

                if not obs_data.get("case_id") and case_id:
                    obs_data["case_id"] = case_id

                try:
                    scoring_prompt = FOCUSED_ITEMS_1_3_SCORING_PROMPT + json.dumps(obs_data, indent=2)
                    config = types.GenerateContentConfig(
                        temperature=0.1,
                        thinking_config=types.ThinkingConfig(thinking_budget=4000),
                    )
                    scoring_response = self.client.models.generate_content(
                        model=MODEL_VERSION,
                        contents=[scoring_prompt],
                        config=config,
                    )
                    scored_text = ""
                    if scoring_response.candidates:
                        for part in scoring_response.candidates[0].content.parts:
                            if hasattr(part, 'text') and part.text:
                                scored_text += part.text

                    scored = parse_json_response(scored_text)
                    if isinstance(scored, dict):
                        scored_outputs.append(scored)
                    else:
                        fallback = score_items_1_3_from_observations(obs_data)
                        scored_outputs.append(fallback)
                        all_warnings.append("LLM scoring returned unexpected format, used deterministic fallback")
                except Exception as e:
                    print(f"[Items 1&3] Stage 2 scoring failed: {e}, using deterministic fallback")
                    fallback = score_items_1_3_from_observations(obs_data)
                    scored_outputs.append(fallback)
                    all_warnings.append(f"LLM scoring failed ({e}), used deterministic fallback")

            return {
                "success": True,
                "response": obs_text,
                "scored_outputs": scored_outputs,
                "scoring_warnings": all_warnings,
                "focused_mode": "items_1_3",
                "filename": filename,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }

        finally:
            self._cleanup_uploaded(uploaded_files)
            if converted and os.path.exists(converted_path):
                os.unlink(converted_path)
            for seg in temp_segments:
                if os.path.exists(seg):
                    os.unlink(seg)
