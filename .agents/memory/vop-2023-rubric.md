---
name: 2023 VOP rubric is the only standard
description: The assessment app must apply only the official 2023 13-item VOP rubric; legacy 14-item results are display-only.
---
The user's uploaded official 2023 rubric (13 items: 10 checklist, 11=Economy out of 5, 12=Proficiency, 13=Comments) is the ONLY scoring standard. The old 14-item rubric's "correct suture (5-0 Prolene)" item was removed; all later items shifted down one. rubric_version is `vop_2023_v1`; core proficiency items are 2/7/9/10.

**Why:** User explicitly stated the 2023 docx "is the only standard I want applied."

**How to apply:** Any rubric/prompt/schema change must keep the 13-item 2023 numbering. Saved sessions predating the refactor use old 14-item numbering — they are version-gated and rendered as legacy passthrough only; never reinterpret them under new numbering. Spacing is still scored functionally (uniformity/no gaping), not literal millimeters, despite the docx's "~1 mm" wording — a deliberate calibration decision.
