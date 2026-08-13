---
name: Gemini upload reliability
description: How to enforce timeouts on google-genai file uploads without breaking them
---

**Rule:** Set HTTP timeouts client-wide via `genai.Client(http_options=types.HttpOptions(timeout=ms))`, NOT per-request in `UploadFileConfig(http_options=...)`.

**Why:** Passing per-request `HttpOptions` (even timeout-only) to `files.upload` overrides SDK defaults and breaks upload URL resolution — the initial create request 404s. A thread-based timeout (`future.result(timeout=)`) is also wrong: it can't cancel the running request and leaks duplicate remote files.

**How to apply:** Client-wide timeout aborts stalled uploads at the HTTP layer. On processing failure/timeout after a successful upload, delete the remote file before retrying (trainee-video data retention). Compression output is always MP4 — upload it as `video/mp4` regardless of source mime.
