# Vascular Anastomosis VOP Retrospective Assessment

## Overview

This project is a retrospective proficiency assessment system designed to analyze videos of simulated vascular anastomosis performed by first-year medical residents. It utilizes a Moderated v1 architecture, employing a two-stage process: Stage 1 extracts factual observations from video using Gemini 3 Pro, and Stage 2 performs LLM-based scoring with integrated app-side validators. The system supports multi-candidate videos, parsing candidate IDs from filenames, including range expansion. A key feature is the inclusion of a structured coaching tags taxonomy. The goal is to provide objective, evidence-based feedback to medical residents, improve surgical training, and standardize assessment, ultimately enhancing patient safety and surgical outcomes.

## User Preferences

I prefer iterative development, with clear communication at each major step. Please ask for confirmation before making significant architectural changes or implementing complex features. I value detailed explanations of the choices made, especially concerning AI model interactions and scoring logic. I expect the agent to maintain a high standard of code quality, readability, and adherence to Python best practices. I want to be informed about any potential edge cases or limitations identified during development.

## System Architecture

The system operates on a Moderated v1 architecture with a two-stage processing pipeline:

1.  **Stage 1 (FOR Extraction)**: Gemini 3 Pro extracts a Factual Observation Record (FOR) from the uploaded video. This stage focuses on purely descriptive observations, avoiding any scoring or interpretation.
2.  **Stage 2 (LLM Scoring)**: Gemini scores the performance based on the FOR JSON. This stage incorporates app-side validators that enforce schema correctness, apply economy guardrails, and implement a proficiency override using a red-line model (RL-A through RL-E).

**Key Architectural Decisions and Features:**

*   **Tri-state Scoring**: Items are scored as YES, NO, or NULL (not visible/observable), replacing binary YES/NO.
*   **Observability**: Each scored item includes an OBSERVED/NOT\_OBSERVED state, eliminating assumed defaults.
*   **Coverage Metrics**: Tracks `observed_count`, `observed_percent`, and `core_observed` status for critical items.
*   **3-State Proficiency**: Proficiency is categorized as PROFICIENT, NOT\_PROFICIENT, or INSUFFICIENT\_EVIDENCE, derived from red-line rules and core domain coverage.
*   **Economy Index (EI)**: Calculated based on weighted wasted-motion events (`instrument_search`, `pause_reset`, `failed_pass_sequence`, `excess_regrasp_cluster`), with a score mapping from 1 to 4.
*   **Projection Layer**: Converts tri-state scores into DOCX-compatible binary outputs using `hawk_mode`, `construct_mode`, or `dove_mode` for varied reporting perspectives.
*   **Video Handling**: Videos over 1 hour are automatically split into ~55-minute segments using FFmpeg and processed as continuous parts. Multi-candidate videos (e.g., "2020 2001 and 2005") are proactively split into segments proportional to the number of candidates to prevent token overage. If a Gemini API call hits token/size limits, the system falls back to sequential per-segment processing automatically. Candidate IDs are parsed from filenames, including range expansion (e.g., "2301-2304").
*   **App-Side Validators**: Critical validation includes schema validation (Pydantic models), prevention of numeric spacing claims, economy guardrail checks, and proficiency recomputation/override.
*   **Robust Batch Processing (Inbox Folder Mode)**: A "Video Inbox" folder-based input mode (`videos_inbox/`) allows processing large batches of videos from disk without browser memory limitations. Videos are processed one at a time sequentially, results saved to disk after each video completes, and a persistent queue file (`_queue.json`) tracks pending/completed/failed status for automatic crash recovery. On restart, completed videos are skipped and processing resumes from where it left off. New videos added to the inbox are detected automatically.
*   **UI/UX**: The application features a Streamlit UI with two input modes (Upload for 1-2 videos, Inbox Folder for large batches), displaying tri-state scores, coverage bars, core domain badges, NULL indicators, economy index metrics, and projection views.
*   **Surgeon vs. AI Comparison**: Allows uploading surgeon assessments (CSV/DOCX) to generate an inline comparison report, detailing item-level agreement, economy delta, and proficiency agreement.
*   **Focused Items 1 & 3 Mode**: A toggle in the sidebar enables scoring only Item 1 (arteriotomy incision) and Item 3 (graft spatulation) for faster processing and lower API quota usage. Uses dedicated prompts (`FOCUSED_ITEMS_1_3_PROMPT`, `FOCUSED_ITEMS_1_3_SCORING_PROMPT`) and a deterministic fallback scorer (`score_items_1_3_from_observations`). Results display in a compact card format with a dedicated CSV export.
*   **Canonical Output**: Generates a standardized JSON output (moderated v1 schema) containing `evidence_based` results and `projected_docx` structures.

## External Dependencies

*   **AI Model**: Gemini 3 Flash (`gemini-3-flash-preview`) is used for both Stage 1 FOR extraction and Stage 2 LLM scoring.
*   **Video Processing**: FFmpeg is utilized for video format conversion (e.g., .m4v to .mp4) and splitting long videos into manageable segments.
*   **Google Gemini API**: Used for uploading video files (Gemini Files API) and interacting with the Gemini AI model.
*   **Pydantic**: Employed for defining and validating data schemas within the application, ensuring data integrity and consistency.