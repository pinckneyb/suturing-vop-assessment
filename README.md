# 🎥 AI Video Watcher

A powerful video analysis application that uses GPT-4o to "watch" and narrate video content, with GPT-5 enhancement for coherent storytelling. Built with Streamlit, OpenCV, and OpenAI's latest models.

## ✨ Features

### 🧠 **Core Video Analysis**
- **Frame-by-frame analysis** using GPT-4o at configurable FPS (0.5-5.0)
- **Intelligent batching** with customizable batch sizes (3-15 frames)
- **Concurrent processing** for faster analysis (1-10 concurrent batches)
- **Context awareness** with rolling narrative continuity

### 🎭 **Profile System**
- **Generic Profile**: Human-like narrative description
- **Surgical Profile**: Medical assessment with rubric awareness
- **Sports Profile**: Play-by-play sports commentary
- **Customizable prompts** for different use cases

### 🔍 **Advanced Features**
- **Rescan segments** at higher FPS (5-20) for detailed analysis
- **GPT-5 enhancement** for coherent, continuous narrative
- **Absolute specificity** - no generic terms, concrete visual details
- **Text transcription** of overlaid or scene text
- **Event timeline** with structured JSON output

### ⚡ **Performance & UX**
- **Concurrent processing** leveraging OpenAI Tier 4+ limits
- **API key persistence** across sessions
- **Real-time progress** tracking and status updates
- **Download options** for transcripts and timelines
- **Streamlined workflow** - video upload to analysis in one click

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key (GPT-4o and GPT-5 access)
- FFmpeg installed on your system

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/pinckneyb/ai-video-watcher.git
   cd ai-video-watcher
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API key**
   - Copy `config.example` to `config.json`
   - Add your OpenAI API key
   - Or enter it directly in the app

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** at `http://localhost:8501`

## 📖 Usage

### 1. **Video Input**
- Upload video files (MP4, AVI, MOV, MKV, WMV, M4V)
- Or provide direct video URLs
- Supported formats: MP4, AVI, MOV, MKV, WMV, M4V

### 2. **Configuration**
- **Select Profile**: Choose analysis style (Generic, Surgical, Sports)
- **Video Settings**: Adjust FPS (0.5-5.0) and batch size (3-15)
- **Concurrency**: Set concurrent batches (1-10) for speed
- **Rescan Settings**: Configure high-FPS rescan (5-20 FPS)

### 3. **Analysis**
- **Load Video**: Upload/URL → automatic analysis start
- **Watch Progress**: Real-time batch processing updates
- **View Results**: Transcript, events timeline, enhanced narrative

### 4. **Enhanced Narrative**
- **GPT-5 Processing**: Transform raw analysis into coherent story
- **Specificity**: Concrete details, no generic terms
- **Continuity**: Smooth narrative flow with logical connections
- **Download**: Save enhanced narrative as Markdown

### 5. **Rescan Segments**
- **Select Time Range**: Choose specific video segments
- **Higher Detail**: Extract frames at 5-20 FPS
- **Deep Analysis**: Detailed examination of critical moments

## 🏗️ Architecture

### **Core Modules**
- **`app.py`**: Main Streamlit application and UI
- **`video_processor.py`**: Video loading and frame extraction
- **`gpt4o_client.py`**: OpenAI API integration and context management
- **`profiles.py`**: Configurable analysis profiles and prompts
- **`utils.py`**: Utility functions and file operations

### **Key Components**
- **VideoProcessor**: Handles video loading, properties, frame extraction
- **FrameBatchProcessor**: Creates and manages frame batches
- **GPT4oClient**: Manages API calls, context, and narrative building
- **ProfileManager**: Provides different analysis styles and prompts

### **Data Flow**
1. **Video Input** → Frame extraction at specified FPS
2. **Batch Creation** → Group frames for efficient processing
3. **GPT-4o Analysis** → Narrative generation with context
4. **Context Update** → Rolling summary for continuity
5. **GPT-5 Enhancement** → Coherent story creation
6. **Output Generation** → Transcript, timeline, enhanced narrative

## ⚙️ Configuration

### **Environment Variables**
```bash
OPENAI_API_KEY=your_api_key_here
```

### **App Settings**
- **FPS Range**: 0.5 - 5.0 frames per second
- **Batch Size**: 3 - 15 frames per batch
- **Concurrency**: 1 - 10 concurrent batches
- **Rescan FPS**: 5.0 - 20.0 for detailed analysis

### **Profile Customization**
Each profile provides:
- **Base Prompt**: Main analysis instructions
- **Rescan Prompt**: Detailed segment analysis
- **Context Condensation**: Rolling summary generation

## 📊 Output Formats

### **Transcript (Markdown)**
- Chronological narrative with timestamps
- Rich visual descriptions and actions
- Context-aware continuity

### **Events Timeline (JSON)**
- Structured event data
- Timestamps and confidence scores
- Profile-specific metadata

### **Enhanced Narrative (Markdown)**
- GPT-5 processed coherent story
- No time markers, natural flow
- Absolute specificity and detail

## 🔧 Troubleshooting

### **Common Issues**
- **FFmpeg not found**: Install FFmpeg and add to PATH
- **API key errors**: Verify OpenAI API key and model access
- **Memory issues**: Reduce batch size or FPS
- **Encoding errors**: Check video format compatibility

### **Performance Tips**
- **Concurrent processing**: Use 3-5 batches for optimal speed
- **Batch size**: Balance between detail and processing time
- **FPS selection**: Lower FPS for longer videos, higher for detail

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### **Development Setup**
```bash
git clone https://github.com/pinckneyb/ai-video-watcher.git
cd ai-video-watcher
pip install -r requirements.txt
streamlit run app.py
```

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **OpenAI** for GPT-4o and GPT-5 models
- **Streamlit** for the web framework
- **OpenCV** for video processing
- **FFmpeg** for robust video handling

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/pinckneyb/ai-video-watcher/issues)
- **Discussions**: [GitHub Discussions](https://github.com/pinckneyb/ai-video-watcher/discussions)

---

**AI Video Watcher** - Transform your videos into intelligent, coherent narratives with the power of GPT-4o and GPT-5.
