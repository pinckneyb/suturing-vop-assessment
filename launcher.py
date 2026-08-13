"""
Application Launcher - Choose between AI Video Watcher and Surgical VOP Assessment
"""

import streamlit as st
import subprocess
import sys
import os

# Page configuration
st.set_page_config(
    page_title="AI Video Analysis Suite",
    page_icon="🎥",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .app-card {
        background-color: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 1rem;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
        transition: all 0.3s ease;
    }
    .app-card:hover {
        border-color: #1f77b4;
        background-color: #f0f8ff;
    }
    .app-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .app-description {
        color: #6c757d;
        margin-bottom: 1.5rem;
        line-height: 1.6;
    }
    .feature-list {
        text-align: left;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main launcher interface."""
    
    # Header
    st.markdown('<h1 class="main-header">🎥 AI Video Analysis Suite</h1>', unsafe_allow_html=True)
    st.markdown("### Choose your application")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="app-card">
            <div class="app-title">🎬 AI Video Watcher</div>
            <div class="app-description">
                General-purpose video analysis with GPT-4o. Perfect for content creation, 
                education, and detailed video understanding.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📋 Features"):
            st.markdown("""
            - **Frame-by-frame analysis** at configurable FPS
            - **Multiple analysis profiles** (Generic, Sports, Social Media)
            - **Concurrent processing** for faster results
            - **Rescan capability** for detailed segments
            - **GPT-5 enhancement** for coherent narratives
            - **Text transcription** and event timelines
            """)
        
        if st.button("🚀 Launch AI Video Watcher", type="primary", use_container_width=True, key="launch_watcher"):
            # Set launching state to prevent multiple clicks
            if 'launching_watcher' not in st.session_state:
                st.session_state.launching_watcher = True
                st.success("🚀 Starting AI Video Watcher...")
                st.info("⏳ Please wait while the application loads...")
                
                # Add a brief delay and then replace process
                import time
                time.sleep(0.5)
                
                # Replace current process with ai-video-watcher/app.py
                os.execl(sys.executable, sys.executable, "-m", "streamlit", "run", "ai-video-watcher/app.py")
            else:
                st.warning("🔄 Application is already starting... Please wait.")
    
    with col2:
        st.markdown("""
        <div class="app-card">
            <div class="app-title">🏥 Surgical VOP Assessment</div>
            <div class="app-description">
                Specialized surgical assessment tool for medical education. 
                Evaluate suturing techniques with structured rubrics.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📋 Features"):
            st.markdown("""
            - **Surgical technique evaluation** with clinical rubrics
            - **Pattern detection** (Simple Interrupted, Subcuticular, Vertical Mattress)
            - **Professional PDF reports** for resident assessment
            - **Structured scoring** with 1-5 point scales
            - **Clinical terminology** and assessment language
            - **Medical education focused** interface
            """)
        
        if st.button("🏥 Launch Surgical VOP Assessment", type="primary", use_container_width=True, key="launch_surgical"):
            # Set launching state to prevent multiple clicks
            if 'launching_surgical' not in st.session_state:
                st.session_state.launching_surgical = True
                st.success("🚀 Starting Surgical VOP Assessment...")
                st.info("⏳ Please wait while the application loads...")
                
                # Add a brief delay and then replace process
                import time
                time.sleep(0.5)
                
                # Replace current process with surgical-vop-assessment/surgical_vop_app.py
                os.execl(sys.executable, sys.executable, "-m", "streamlit", "run", "surgical-vop-assessment/surgical_vop_app.py")
            else:
                st.warning("🔄 Application is already starting... Please wait.")
    
    # Information section
    st.markdown("---")
    st.markdown("### ℹ️ About")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **AI Video Watcher** is ideal for:
        - Content creators analyzing videos
        - Educators creating video summaries
        - Researchers studying video content
        - General video understanding tasks
        """)
    
    with col2:
        st.info("""
        **Surgical VOP Assessment** is designed for:
        - Medical education programs
        - Surgical resident training
        - Clinical skills assessment
        - Structured competency evaluation
        """)

if __name__ == "__main__":
    main()