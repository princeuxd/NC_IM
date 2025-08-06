import streamlit as st

# UI modules
from src.ui.onboarding import render_onboarding
from src.ui.video_analytics import render_video_analytics
from src.ui.channel_analytics import render_channel_analytics
from src.ui.talk_with_ai import render_talk_with_ai


def main():
    st.set_page_config(
        page_title="YouTube Analytics NC", 
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Clean header styling
    st.markdown("""
    <style>
    .main-header {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border-left: 4px solid #667eea;
    }
    .main-header h1 {
        color: #333;
        margin: 0;
        font-size: 2rem;
        font-weight: 600;
    }
    .main-header p {
        color: #666;
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
    }
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Clean main header
    st.markdown("""
    <div class="main-header">
        <h1>📊 YouTube Analytics NC</h1>
        <p>YouTube analytics with AI insights</p>
    </div>
    """, unsafe_allow_html=True)

    # Clean sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-section">
            <h3 style="margin-top: 0; color: #333;">Analytics</h3>
            <p style="margin-bottom: 0; color: #666; font-size: 0.9rem;">Choose analysis type</p>
        </div>
        """, unsafe_allow_html=True)
        
        section = st.selectbox(
            "Select Analysis Type",
            (
                "Creator Onboarding",
                "Video Analytics", 
                "Channel Analytics",
                "Talk with AI",
            ),
            help="Choose analysis type"
        )

    # Route to appropriate section
    if section == "Creator Onboarding":
        render_onboarding()
    elif section == "Video Analytics":
        render_video_analytics()
    elif section == "Channel Analytics":
        render_channel_analytics()
    elif section == "Talk with AI":
        render_talk_with_ai()


if __name__ == "__main__":
    main()
