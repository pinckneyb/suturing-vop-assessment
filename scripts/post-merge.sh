#!/bin/bash
set -e

# Post-merge setup for the surgical VOP assessment app.
# Idempotent: installs Python dependencies needed by the Streamlit app.
cd surgical-vop-assessment
uv pip install -r requirements.txt 2>/dev/null || pip install -r requirements.txt
