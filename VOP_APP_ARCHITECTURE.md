# Surgical VOP Assessment App - Complete Architecture Documentation

## 🏥 **What This App Does**

This is a **Surgical Verification of Proficiency (VOP) Assessment System** that analyzes video recordings of suturing procedures and generates professional assessment reports. It uses AI to evaluate surgical technique across 7 rubric points and produces detailed feedback for learners.

## 🔄 **Core Workflow (DO NOT BREAK THIS)**

### **1. Video Processing Pipeline**
```
Video Upload → Frame Extraction → GPT-4o Analysis → GPT-5 Synthesis → PDF Report
```

**Step 1: Video Upload & Frame Extraction**
- Supports MP4, MOV, M4V formats up to 2GB
- Extracts frames at configurable FPS (default: 2.0 fps)
- Uses FFmpeg for video processing

**Step 2: GPT-4o Frame Analysis**
- Analyzes video in chunks/batches for memory efficiency
- Each chunk gets analyzed by GPT-4o vision model
- Maintains context across chunks for narrative continuity
- Output: Raw frame-by-frame observations with timestamps

**Step 3: GPT-5 Narrative Enhancement**
- Takes ALL GPT-4o chunk outputs
- Synthesizes into coherent assessment covering ENTIRE video
- Generates specific feedback for each rubric point
- Output: Enhanced narrative with rubric scores

**Step 4: PDF Report Generation**
- Uses GPT-5 enhanced narrative (NOT raw GPT-4o output)
- Extracts rubric-specific content for each assessment point
- Generates professional VOP report with scores and feedback

## 🎯 **Critical Architecture Points (NEVER CHANGE)**

### **Video Coverage: ENTIRE VIDEO, NOT FRAGMENTS**
- GPT-4o analyzes in chunks but maintains full context
- GPT-5 receives ALL chunk outputs for complete synthesis
- Each rubric assessment covers performance across entire procedure
- Reports must reflect complete video analysis, not just first few seconds

### **Content Flow: GPT-4o → GPT-5 → PDF**
```
GPT-4o Chunks → Full Transcript → GPT-5 Enhancement → PDF Extraction
```
- **NEVER** extract rubric feedback from raw GPT-4o output
- **ALWAYS** use GPT-5 enhanced narrative for PDF generation
- GPT-5 narrative contains the complete, coherent assessment

### **Rubric Assessment Structure**
Each rubric point gets:
1. **Score (1-5) + VOP Adjective** (e.g., "4 - Proficient")
2. **Specific observation** from complete video analysis
3. **Actionable advice** based on what was actually observed

## 📊 **VOP Scoring System**

### **1-5 Likert Scale (DO NOT CHANGE)**
- **1 — Remediation / Unsafe**: Incorrect or unsafe. Closure unreliable. Major errors. Fails.
- **2 — Minimal Pass / Basic Competent**: Safe and complete but inefficient or inconsistent. Technique rough. Minor, non-dangerous errors. Closure holds.
- **3 — Developing Pass / Generally Reliable**: Mostly correct and independent. Occasional flaws in spacing, tension, or handling. Closure reliable.
- **4 — Proficient**: Consistently correct, independent, and safe. Perpendicular passes, gentle handling, secure knots, proper eversion, even spacing. Efficient.
- **5 — Exemplary / Model**: Near-ideal execution. Precise, smooth, economical. Teachable example.

### **Pass Rule (DO NOT CHANGE)**
- **PASS**: Every rubric point ≥ 2
- **REMEDIATION**: Any rubric point < 2

## 🔧 **Key Files & Their Purpose**

### **Core Application**
- `surgical_vop_app.py` - Main Streamlit app, handles video upload and analysis orchestration
- `gpt4o_client.py` - GPT-4o API client, manages chunk analysis and context
- `video_processor.py` - Video loading, frame extraction, metadata management

### **Assessment Engine**
- `surgical_report_generator.py` - PDF report generation using GPT-5 enhanced narrative
- `unified_rubric.JSON` - VOP assessment criteria and scoring rules
- `profiles.py` - AI personality profiles for different assessment styles

### **Narrative Guides**
- `simple_interrupted_narrative.txt` - Ideal reference for simple interrupted suturing
- `vertical_mattress_narrative.txt` - Ideal reference for vertical mattress suturing  
- `subcuticular_narrative.txt` - Ideal reference for subcuticular suturing

### **Gold Standard Images**
- `Simple_Interrupted_Suture_example.png` - Reference image for comparison
- `Vertical_Mattress_Suture_example.png` - Reference image for comparison
- `subcuticular_example.png` - Reference image for comparison

## 🚨 **Common Breaking Points (AVOID THESE)**

### **1. Rubric Feedback Extraction**
**WRONG**: Using generic templates or keyword matching that returns same content
**RIGHT**: Extract specific content from GPT-5 enhanced narrative for each rubric point

**Code Location**: `surgical_report_generator.py` → `_extract_rubric_feedback()`

### **2. Narrative Processing**
**WRONG**: Only using first few seconds or fragments of video analysis
**RIGHT**: Ensure GPT-5 receives and synthesizes ALL GPT-4o chunk outputs

**Code Location**: `surgical_vop_app.py` → `create_surgical_vop_narrative()`

### **3. Content Flow**
**WRONG**: Bypassing GPT-5 enhanced narrative for PDF generation
**RIGHT**: Always use GPT-5 output as source for rubric assessments

**Code Location**: `surgical_report_generator.py` → `_create_rubric_assessment_section()`

## 🧪 **Testing the System**

### **What to Verify**
1. **Video Processing**: Upload video, check frame extraction
2. **GPT-4o Analysis**: Verify chunks are processed and context maintained
3. **GPT-5 Enhancement**: Confirm complete narrative generation
4. **PDF Generation**: Check unique rubric assessments (not identical content)
5. **Scoring**: Verify VOP-aligned 1-5 scale with proper adjectives

### **Red Flags (System Broken)**
- All rubric points show identical feedback
- Feedback only covers first few seconds of video
- Generic template language instead of specific observations
- Missing or corrupted GPT-5 enhanced narrative

## 🔄 **Recovery Procedures**

### **If Rubric Feedback is Identical**
1. Check `_extract_rubric_feedback()` function in `surgical_report_generator.py`
2. Verify it's parsing GPT-5 enhanced narrative, not raw GPT-4o output
3. Ensure proper content extraction strategies are working

### **If Only Partial Video Coverage**
1. Check GPT-4o chunk processing in `gpt4o_client.py`
2. Verify context continuity across chunks
3. Ensure GPT-5 receives complete transcript

### **If PDF Generation Fails**
1. Check `surgical_report_generator.py` for proper narrative parsing
2. Verify GPT-5 enhanced narrative exists in session state
3. Check rubric extraction logic

## 📝 **Configuration Files**

### **Environment Variables**
- `.env` - Contains `OPENAI_API_KEY` (gitignored)
- `.streamlit/config.toml` - Streamlit server configuration (maxUploadSize = 2048)

### **Dependencies**
- `surgical_requirements.txt` - Python packages for surgical app
- `requirements.txt` - General app dependencies

## 🎯 **Success Criteria**

The system is working correctly when:
1. ✅ Video uploads and processes completely
2. ✅ GPT-4o analyzes all frames with context continuity
3. ✅ GPT-5 generates coherent narrative covering entire video
4. ✅ Each rubric point gets unique, specific feedback
5. ✅ PDF report shows complete video assessment
6. ✅ VOP scoring follows 1-5 scale with proper adjectives
7. ✅ Pass/fail determination follows VOP rules

## 🚫 **What NOT to Do**

- **NEVER** replace AI-generated content with generic templates
- **NEVER** break the GPT-4o → GPT-5 → PDF content flow
- **NEVER** change the VOP scoring system or pass rules
- **NEVER** modify the video processing pipeline without testing
- **NEVER** assume rubric feedback extraction is working without verification

## 🔍 **Debugging Checklist**

When the system breaks:
1. Check video processing logs for frame extraction
2. Verify GPT-4o chunk analysis completion
3. Confirm GPT-5 enhanced narrative generation
4. Test rubric feedback extraction with sample narrative
5. Validate PDF generation with proper content
6. Check all configuration files and dependencies

---

**Remember**: This system was built to work. If it breaks, the problem is likely in the content flow or extraction logic, not the core architecture. Always verify the GPT-4o → GPT-5 → PDF pipeline is intact.
