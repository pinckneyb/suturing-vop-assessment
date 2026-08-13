---
name: Gemini model availability
description: Durable decision on which Gemini vision model version to use and how to keep the ID valid.
---
- Pin the scoring model to a stable (non-preview) Gemini vision model ID; verify the ID against the live `models.list` endpoint rather than assuming a hardcoded ID still exists.
- **Why:** preview model IDs are periodically retired and return 404, silently breaking scoring.
- **How to apply:** when changing the model or fps, confirm the model ID via models.list; higher fps requires proportionally shorter video segments to stay within input limits.
