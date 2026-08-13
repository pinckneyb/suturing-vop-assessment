# AI Video Watcher - Project Overview

## 🎯 **What the App Does**
The AI Video Watcher is a sophisticated video analysis application that uses GPT-4o to "watch" videos frame-by-frame and generate intelligent, contextual narratives. It's designed for detailed video analysis with applications in medical training (surgical technique assessment), content creation, and general video understanding.

## 🚀 **Core Functionality**
1. **Frame-by-frame Analysis**: Extracts frames at configurable FPS (0.5-5.0) and analyzes them using GPT-4o
2. **Intelligent Batching**: Groups frames into batches (3-15 frames) for efficient API usage and cost optimization
3. **Contextual Continuity**: Maintains narrative flow across batches using sophisticated context management
4. **Profile-based Analysis**: Different AI personalities for different use cases (surgical, social media, movie analysis)
5. **Real-time Processing**: Streamlit interface with live progress tracking and batch status monitoring

## 🏗️ **Architecture & Components**

### **Main Application (`app.py`)**
- **Streamlit UI**: Main interface with video upload, settings, and analysis controls
- **Session Management**: Handles user preferences, analysis state, and progress tracking
- **Concurrency Control**: Manages batch processing with user-defined safety levels
- **Progress Visualization**: Real-time charts showing batch completion and processing speed
- **Error Handling**: Graceful failure recovery and user feedback

### **AI Client (`gpt4o_client.py`)**
- **OpenAI Integration**: Handles GPT-4o API calls with proper error handling
- **Rate Limiting**: Built-in retry logic and exponential backoff
- **Response Processing**: Parses and validates AI responses
- **Cost Tracking**: Monitors API usage and estimates costs

### **Video Processing (`video_processor.py`)**
- **Frame Extraction**: Uses ffmpeg-python to extract frames at specified intervals
- **Batch Creation**: Groups frames into optimal batches for analysis
- **Format Handling**: Supports multiple video formats and resolutions
- **Memory Management**: Efficient frame storage and cleanup

### **Profile System (`profiles.py`)**
- **Surgical Profile**: Clinical, critical analysis for medical training videos
  - Matter-of-fact tone, no flattery
  - Focuses on technique assessment and safety protocols
  - Uses surgical terminology and step-by-step evaluation
- **Social Media Profile**: Engaging, narrative-driven analysis for content creation
- **Movie Lover Profile**: Entertainment-focused analysis for film appreciation
- **Customizable Prompts**: Each profile has base, rescan, and context condensation prompts

### **Utility Functions (`utils.py`)**
- **File Operations**: Video validation, temporary file management
- **Data Processing**: Frame analysis, batch optimization
- **Helper Functions**: Common operations used across modules

### **Configuration (`config.example`)**
- **Environment Variables**: API keys, model settings, default parameters
- **User Preferences**: Default FPS, batch sizes, concurrency levels

## 🔧 **Key Features**

### **Concurrency System**
- **Proven Safe Level**: 10 concurrent batches (100% reliable based on real-world testing)
- **Testing Levels**: 12+ batches for performance optimization
- **FPS-aware**: Adjusts recommendations based on video frame rate
- **User-defined Limits**: Slider control from 1-20 concurrent batches

### **Context Management**
- **Running Summary**: Maintains narrative continuity across batches
- **Context Condensation**: Compresses analysis state for efficient processing
- **Rescan Capability**: Re-analyzes specific time segments with enhanced detail
- **Memory Optimization**: Prevents context overflow in long videos

### **Analysis Profiles**
- **Surgical**: Clinical assessment with rubric-based evaluation
- **Social Media**: Engaging narratives for content creation
- **Movie Lover**: Entertainment and artistic analysis
- **Extensible**: Easy to add new profiles with custom prompts

## 📊 **Data Flow**
1. **Video Upload** → Frame extraction at specified FPS
2. **Batch Creation** → Frames grouped into optimal batches
3. **AI Analysis** → GPT-4o processes each batch with context
4. **Context Update** → Running summary maintained for continuity
5. **Progress Tracking** → Real-time updates on processing status
6. **Results Display** → Final analysis with timestamps and insights

## 🎛️ **User Controls**
- **FPS Selection**: 0.5-5.0 frames per second
- **Batch Size**: 3-15 frames per batch
- **Concurrency**: 1-20 concurrent API calls
- **Profile Selection**: Choose analysis personality
- **Quality Settings**: Balance between speed and detail

## 💰 **Cost Optimization**
- **Batch Processing**: Groups frames to minimize API calls
- **Context Efficiency**: Maintains narrative without redundant analysis
- **Concurrency Control**: Balances speed with API rate limits
- **Progress Monitoring**: Real-time cost tracking and estimates

## 🔍 **Use Cases**
1. **Medical Training**: Surgical technique assessment and feedback
2. **Content Creation**: Video analysis for social media and marketing
3. **Educational Content**: Detailed video breakdowns and explanations
4. **Quality Assurance**: Video content review and analysis
5. **Research**: Frame-by-frame analysis for academic studies

## 🚨 **Important Notes**
- **API Requirements**: Requires OpenAI Tier 4+ for high concurrency
- **Video Formats**: Supports common formats via ffmpeg
- **Memory Usage**: Frame storage scales with video length and FPS
- **Processing Time**: Depends on video length, FPS, and concurrency settings
- **Cost**: GPT-4o API costs scale with frame count and analysis detail

## 🔄 **Recent Updates**
- Surgical profile made more critical and analytical (removed flattery)
- Concurrency system updated based on real-world testing (10 batches = 100% safe)
- Obsolete warning messages removed
- Unified rubric system added for surgical assessment

## 📁 **File Structure**
```
AI_video_watcher/
├── app.py                          # Main Streamlit application (general video analysis)
├── surgical_vop_app.py            # Specialized surgical VOP assessment app
├── surgical_report_generator.py    # PDF report generation for surgical assessments
├── gpt4o_client.py                # OpenAI API client and handling
├── video_processor.py             # Video processing and frame extraction
├── profiles.py                    # AI personality profiles and prompts
├── utils.py                       # Utility functions and helpers
├── config.example                 # Configuration template
├── requirements.txt               # Python dependencies (general app)
├── surgical_requirements.txt      # Python dependencies (surgical app)
├── unified_rubric.JSON            # Surgical assessment rubric
├── PROJECT_OVERVIEW.md            # This file
└── README.md                     # User documentation
```

## 🏥 **Surgical VOP Assessment App**
A specialized application built on the same video analysis foundation, focused specifically on Verification of Proficiency (VOP) assessments for surgical residents:

### **Key Features**
- **Pattern Detection**: Automatically detects suture patterns from folder/file names
- **Structured Assessment**: Uses unified_rubric.JSON for standardized evaluation
- **Clinical Interface**: Clean, medical-focused UI designed for surgical education
- **Timestamp-Specific Feedback**: All assessments reference specific video moments
- **Professional Reporting**: Generates formal PDF reports for resident evaluation
- **Rubric-Based Scoring**: 1-5 Likert scale scoring for each assessment criterion

### **Supported Suture Patterns**
- Simple Interrupted
- Vertical Mattress  
- Subcuticular

### **Assessment Process**
1. **Upload Video**: Surgical technique video (MP4, AVI, MOV, MKV)
2. **Pattern Recognition**: Auto-detect or manually select suture pattern
3. **AI Analysis**: GPT-4o analyzes video using surgical assessment prompts
4. **Manual Scoring**: Instructor scores each rubric point (1-5 scale)
5. **Generate Report**: Professional PDF report with detailed feedback

This overview should provide sufficient context for future development sessions without needing to re-explore the entire codebase.

