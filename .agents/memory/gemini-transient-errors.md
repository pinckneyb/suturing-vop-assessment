---
name: Gemini transient server errors
description: 503/504 spikes on preview models must be retried, not just rate limits
---

Gemini preview models intermittently return 503 UNAVAILABLE ("high demand") and 504 DEADLINE_EXCEEDED on generate_content for long-video requests. These are transient but can persist for several minutes.

**Why:** A retry predicate that only matches rate-limit (429/quota) errors lets a single 503/504 kill a whole assessment even though the upload succeeded.

**How to apply:** Retry predicates around Gemini calls must include 500-internal/503/504/deadline/unavailable/overloaded strings, with exponential backoff up to ~5 min and ~7 attempts for long-video scoring calls.
