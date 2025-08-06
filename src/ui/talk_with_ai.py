"""Streamlit UI for Talk with AI - Interactive chat interface."""
from __future__ import annotations

import streamlit as st
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from src.llms import get_smart_client
from src.config.settings import SETTINGS


def _initialize_chat_session():
    """Initialize chat session state."""
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_context" not in st.session_state:
        st.session_state.chat_context = None
    if "available_analyses" not in st.session_state:
        st.session_state.available_analyses = _get_available_analyses()


def _get_available_analyses() -> List[Dict[str, Any]]:
    """Get list of available analysis files."""
    analyses = []
    
    # Check for video analyses
    video_reports_dir = Path("data/reports/video_analysis")
    if video_reports_dir.exists():
        for video_dir in video_reports_dir.iterdir():
            if video_dir.is_dir():
                analysis_file = video_dir / f"{video_dir.name}_analysis.json"
                summary_file = video_dir / f"{video_dir.name}_summary.md"
                
                if analysis_file.exists():
                    try:
                        analysis_data = json.loads(analysis_file.read_text())
                        analyses.append({
                            "type": "video",
                            "id": video_dir.name,
                            "title": analysis_data.get("title", "Unknown Video"),
                            "analysis_file": analysis_file,
                            "summary_file": summary_file if summary_file.exists() else None,
                            "data": analysis_data
                        })
                    except Exception:
                        continue
    
    # Check for channel analyses (look for COLLECTIVE_ANALYSIS files or video directories)
    channel_reports_dir = Path("data/reports/channel_analysis")
    if channel_reports_dir.exists():
        for channel_dir in channel_reports_dir.iterdir():
            if channel_dir.is_dir():
                # Look for the collective analysis markdown file first
                collective_analysis_file = None
                for file in channel_dir.glob("COLLECTIVE_ANALYSIS_*.md"):
                    collective_analysis_file = file
                    break
                
                channel_title = "Unknown Channel"
                videos_analyzed = 0
                total_duration = 0
                authenticity_score = 0.0
                analysis_content = ""
                
                if collective_analysis_file:
                    # Process collective analysis file
                    try:
                        content = collective_analysis_file.read_text()
                        analysis_content = content
                        
                        # Extract channel name from the markdown content
                        for line in content.split('\n')[:20]:  # Check first 20 lines
                            if line.startswith("## ") and line != "## Astro K Joseph: Strategic Channel Assessment":
                                channel_title = line.replace("## ", "").strip()
                                break
                        
                        # Get video count and other stats from the content
                        for line in content.split('\n'):
                            if "- **Videos Analyzed:**" in line:
                                try:
                                    videos_analyzed = int(line.split("**")[-1].strip())
                                except:
                                    pass
                            elif "- **Total Content Duration:**" in line:
                                try:
                                    duration_str = line.split("**")[-1].strip()
                                    if "minutes" in duration_str:
                                        total_duration = int(duration_str.split()[0])
                                except:
                                    pass
                            elif "- **Average Authenticity Score:**" in line:
                                try:
                                    score_str = line.split("**")[-1].strip()
                                    authenticity_score = float(score_str.split("/")[0])
                                except:
                                    pass
                        
                    except Exception as e:
                        print(f"Error processing collective analysis for {channel_dir.name}: {e}")
                else:
                    # No collective analysis file, try to get channel info from individual videos
                    video_dirs = [d for d in channel_dir.iterdir() if d.is_dir()]
                    if video_dirs:
                        videos_analyzed = len(video_dirs)
                        
                        # Try to get channel title from first video's stats
                        first_video_dir = video_dirs[0]
                        stats_file = first_video_dir / f"{first_video_dir.name}_stats.json"
                        if stats_file.exists():
                            try:
                                stats_data = json.loads(stats_file.read_text())
                                snippet = stats_data.get('snippet', {})
                                channel_title = snippet.get('channelTitle', f'Channel {channel_dir.name[:8]}...')
                            except Exception:
                                channel_title = f'Channel {channel_dir.name[:8]}...'
                        
                        # Calculate total duration from individual videos
                        for video_dir in video_dirs:
                            data_file = video_dir / f"{video_dir.name}_data.json"
                            if data_file.exists():
                                try:
                                    video_data = json.loads(data_file.read_text())
                                    duration = video_data.get('duration_minutes', 0)
                                    total_duration += duration
                                except Exception:
                                    pass
                        
                        analysis_content = f"Channel analysis available for {videos_analyzed} individual videos. No collective summary report found."
                
                # Only add if we have some data
                if videos_analyzed > 0 or collective_analysis_file:
                    analyses.append({
                        "type": "channel",
                        "id": channel_dir.name,
                        "title": channel_title,
                        "analysis_file": collective_analysis_file,
                        "summary_file": collective_analysis_file,
                        "data": {
                            "channel_id": channel_dir.name,
                            "channel_title": channel_title,
                            "videos_analyzed": videos_analyzed,
                            "total_duration": total_duration,
                            "authenticity_score": authenticity_score,
                            "analysis_content": analysis_content
                        }
                    })
    
    return sorted(analyses, key=lambda x: x["title"])


def _load_analysis_context(analysis: Dict[str, Any]) -> str:
    """Load analysis data as context for the AI."""
    if analysis["type"] == "video":
        context = f"""Video Analysis Context:
Title: {analysis['title']}
Video ID: {analysis['id']}

Analysis Data:
- Audio Analysis: {analysis['data'].get('audio_analysis', 'Not available')}
- Video Analysis: {analysis['data'].get('video_analysis', 'Not available')}
- Comments Analysis: {analysis['data'].get('comments_analysis', 'Not available')}
- Statistics: {json.dumps(analysis['data'].get('statistics', {}), indent=2)}
- OAuth Analytics: {json.dumps(analysis['data'].get('oauth_analytics', {}), indent=2) if analysis['data'].get('oauth_analytics') else 'Not available'}

Frames Count: {analysis['data'].get('frames_count', 0)}
Comments Count: {analysis['data'].get('comments_count', 0)}
"""
        
        # Add summary if available
        if analysis['summary_file'] and analysis['summary_file'].exists():
            summary_content = analysis['summary_file'].read_text()
            context += f"\nGenerated Summary:\n{summary_content}"
            
    elif analysis["type"] == "channel":
        data = analysis['data']
        
        # Start with basic channel info
        context = f"""Channel Analysis Context:
Channel: {analysis['title']}
Channel ID: {analysis['id']}

Channel Overview:
- Videos Analyzed: {data.get('videos_analyzed', 'N/A')}
- Total Content Duration: {data.get('total_duration', 'N/A')} minutes
- Average Authenticity Score: {data.get('authenticity_score', 'N/A')}/10

===== COMPLETE CHANNEL ANALYSIS REPORT =====
{data.get('analysis_content', 'Analysis content not available')}

===== INDIVIDUAL VIDEO SUMMARIES AND DATA =====
"""
        
        # Load all individual video data from the channel directory
        channel_dir = Path("data/reports/channel_analysis") / analysis['id']
        if channel_dir.exists():
            video_dirs = [d for d in channel_dir.iterdir() if d.is_dir()]
            
            for i, video_dir in enumerate(sorted(video_dirs), 1):
                video_id = video_dir.name
                context += f"\n--- VIDEO {i}: {video_id} ---\n"
                
                # Load video data JSON
                data_file = video_dir / f"{video_id}_data.json"
                if data_file.exists():
                    try:
                        video_data = json.loads(data_file.read_text())
                        context += f"Title: {video_data.get('title', 'N/A')}\n"
                        context += f"Duration: {video_data.get('duration_minutes', 'N/A')} minutes\n"
                        context += f"URL: {video_data.get('url', 'N/A')}\n"
                        
                        # Add analysis data
                        analysis_data = video_data.get('analysis', {})
                        if analysis_data:
                            content_type = analysis_data.get('content_type', {})
                            context += f"Content Type: {content_type.get('primary', 'N/A')}\n"
                            
                            voice_style = analysis_data.get('voice_style', {})
                            context += f"Creator Style: {voice_style.get('tone', 'N/A')}\n"
                            
                            authenticity = analysis_data.get('authenticity', {})
                            if authenticity:
                                context += f"Authenticity Score: {authenticity.get('score', 'N/A')}/10\n"
                                context += f"Authenticity Reasoning: {authenticity.get('reasoning', 'N/A')}\n"
                        
                    except Exception as e:
                        context += f"Error loading video data: {e}\n"
                
                # Load video statistics JSON
                stats_file = video_dir / f"{video_id}_stats.json"
                if stats_file.exists():
                    try:
                        stats_data = json.loads(stats_file.read_text())
                        snippet = stats_data.get('snippet', {})
                        statistics = stats_data.get('statistics', {})
                        
                        context += f"Published: {snippet.get('publishedAt', 'N/A')}\n"
                        context += f"Views: {statistics.get('viewCount', 'N/A')}\n"
                        context += f"Likes: {statistics.get('likeCount', 'N/A')}\n"
                        context += f"Comments: {statistics.get('commentCount', 'N/A')}\n"
                        
                        # Add description (first 200 chars)
                        description = snippet.get('description', '')
                        if description:
                            context += f"Description: {description[:200]}{'...' if len(description) > 200 else ''}\n"
                        
                    except Exception as e:
                        context += f"Error loading video stats: {e}\n"
                
                # Load video summary markdown
                summary_file = video_dir / f"{video_id}_summary.md"
                if summary_file.exists():
                    try:
                        summary_content = summary_file.read_text()
                        context += f"\nCOMPLETE VIDEO ANALYSIS:\n{summary_content}\n"
                    except Exception as e:
                        context += f"Error loading video summary: {e}\n"
                
                context += "\n" + "="*50 + "\n"
        
        context += "\n===== END OF COMPREHENSIVE CHANNEL DATA =====\n"
    
    return context


def _get_system_prompt() -> str:
    """Get the system prompt for the AI assistant focused on brand management and sponsorship evaluation."""
    return """You are a specialized YouTube analytics consultant for BRAND MANAGERS and CAMPAIGN MANAGERS evaluating creators for sponsorship opportunities and brand partnerships.

Your expertise focuses on:
🎯 SPONSORSHIP FIT ANALYSIS
- Creator-brand alignment assessment
- Audience demographics vs target market match
- Content style compatibility with brand values
- Authenticity and trust indicators for brand safety

📊 PERFORMANCE ANALYTICS  
- Engagement rate analysis and benchmarking
- Audience sentiment analysis and brand safety
- Content performance patterns and consistency
- ROI potential based on reach and engagement

🔍 CREATOR EVALUATION
- Content authenticity scores and reasoning
- Creator communication style and brand voice fit
- Production quality and professionalism assessment
- Previous brand integration effectiveness

📈 CAMPAIGN INSIGHTS
- Optimal content types for product placement
- Audience response patterns to sponsored content
- Risk assessment for brand reputation
- Budget allocation recommendations based on performance

ANALYSIS APPROACH:
- Always provide specific metrics and percentages
- Reference actual data points from analytics
- Compare performance against industry benchmarks
- Highlight potential red flags or opportunities
- Assess audience sentiment toward branded content
- Evaluate creator's track record with sponsorships

BRAND SAFETY FOCUS:
- Flag any controversial content or negative sentiment spikes
- Assess comment quality and audience maturity
- Evaluate creator's professionalism and reliability
- Identify potential reputation risks

When analyzing creators or channels, think from a brand manager's perspective:
- "Is this creator a good fit for our brand values?"
- "Will their audience convert to customers?"
- "What's the risk/reward ratio for this partnership?"
- "How authentic will our product integration feel?"

Be analytical, data-driven, and focus on ROI and brand safety considerations."""


def _format_message_for_display(message: Dict[str, str]) -> None:
    """Format and display a chat message."""
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write(message["content"])




def _handle_chat_input(user_input: str, context: str = None):
    """Handle user chat input and get AI response."""
    # Add user message
    st.session_state.chat_messages.append({
        "role": "user", 
        "content": user_input,
        "timestamp": datetime.now().isoformat()
    })
    
    # Prepare messages for AI
    messages = [{"role": "system", "content": _get_system_prompt()}]
    
    # Add context if available
    if context:
        messages.append({
            "role": "system", 
            "content": f"Here is the analytics data to reference:\n\n{context}"
        })
    
    # Add conversation history (last 10 messages to manage token limits)
    recent_messages = st.session_state.chat_messages[-10:]
    for msg in recent_messages:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    # Get AI response
    try:
        client = get_smart_client()
        response = client.chat(messages, temperature=0.7, max_tokens=1000)
        
        # Add AI response
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        st.error(f"AI response failed: {str(e)}")
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": "I apologize, but I'm having trouble processing your request right now. Please try again or check your API configuration.",
            "timestamp": datetime.now().isoformat()
        })


def render_talk_with_ai():
    """Render the Talk with AI interface."""
    _initialize_chat_session()
    
    st.header("🤖 Talk with AI")
    st.markdown("**For Brand Managers & Campaign Teams:** Evaluate YouTube creators for sponsorship opportunities using comprehensive analytics, sentiment analysis, and brand safety assessment.")
    
    # Check if LLM is configured
    if not (SETTINGS.openrouter_api_keys or SETTINGS.groq_api_keys or SETTINGS.gemini_api_keys):
        st.error("⚠️ No LLM API keys configured. Please set up your API keys in settings to use Talk with AI.")
        st.info("""
        **To enable Talk with AI:**
        1. Add your API keys to environment variables or config
        2. Supported providers: OpenRouter, Groq, Gemini
        3. Check the settings configuration
        """)
        return
    
    # Context Selection
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 📊 Analysis Context (Optional)")
        context_option = st.selectbox(
            "Select analysis to discuss:",
            ["None - Select an analysis"] + [f"{a['type'].title()}: {a['title']}" for a in st.session_state.available_analyses],
            help="Choose a specific analysis to discuss"
        )
    
    with col2:
        if st.button("🔄 Refresh Analyses", help="Refresh the list of available analyses"):
            st.session_state.available_analyses = _get_available_analyses()
            st.rerun()
    
    # Load selected context
    selected_analysis = None
    context_text = None
    
    if context_option != "None - Select an analysis":
        # Find the selected analysis
        for analysis in st.session_state.available_analyses:
            if f"{analysis['type'].title()}: {analysis['title']}" == context_option:
                selected_analysis = analysis
                context_text = _load_analysis_context(analysis)
                st.session_state.chat_context = context_text
                break
        
        if selected_analysis:
            st.success(f"✅ Loaded context: {selected_analysis['title']}")
            
            # Show quick stats for the loaded analysis
            col1, col2, col3, col4 = st.columns(4)
            if selected_analysis["type"] == "video":
                data = selected_analysis["data"]
                stats = data.get("statistics", {}).get("statistics", {})
                
                col1.metric("Views", stats.get("viewCount", "N/A"))
                col2.metric("Likes", stats.get("likeCount", "N/A"))
                col3.metric("Comments", stats.get("commentCount", "N/A"))
                col4.metric("Analysis Date", data.get("timestamp", "N/A")[:10] if data.get("timestamp") else "N/A")
                
            elif selected_analysis["type"] == "channel":
                data = selected_analysis["data"]
                
                col1.metric("Videos Analyzed", data.get("videos_analyzed", "N/A"))
                col2.metric("Total Duration", f"{data.get('total_duration', 'N/A')} min")
                col3.metric("Authenticity Score", f"{data.get('authenticity_score', 'N/A')}/10")
                col4.metric("Channel ID", data.get("channel_id", "N/A")[:15] + "..." if len(data.get("channel_id", "")) > 15 else data.get("channel_id", "N/A"))
            
            with st.expander("📋 View Context Data"):
                st.text(context_text[:1000] + "..." if len(context_text) > 1000 else context_text)
    else:
        st.session_state.chat_context = None
    
    st.markdown("---")
    
    # Chat Interface
    st.markdown("### 💬 Chat")
    
    # Display chat history
    if st.session_state.chat_messages:
        for message in st.session_state.chat_messages:
            _format_message_for_display(message)
    else:
        # Welcome message
        with st.chat_message("assistant"):
            if selected_analysis:
                if selected_analysis['type'] == 'channel':
                    st.write(f"👋 Welcome! I've loaded comprehensive analytics for **{selected_analysis['title']}** channel. What would you like to know?")
                else:
                    st.write(f"👋 Hi! I've loaded the analysis for the video **{selected_analysis['title']}**. How can I assist your campaign evaluation?")
            else:
                st.write("👋 Welcome! Select an analysis above to get started with brand evaluation insights.")
    
    # Chat Input
    if user_input := st.chat_input("Ask about anything regarding creator or their content..."):
        with st.spinner("🤔 Analyzing..."):
            _handle_chat_input(user_input, st.session_state.chat_context)
        st.rerun()
    
    # Chat Controls
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🗑️ Clear Chat", help="Clear conversation history"):
            st.session_state.chat_messages = []
            st.rerun()
    
    with col2:
        if st.button("💾 Export Chat", help="Export conversation to file"):
            if st.session_state.chat_messages:
                chat_export = {
                    "timestamp": datetime.now().isoformat(),
                    "context": context_option,
                    "messages": st.session_state.chat_messages
                }
                
                st.download_button(
                    "📥 Download Chat",
                    json.dumps(chat_export, indent=2),
                    file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.info("No conversation to export")
    
    with col3:
        st.caption(f"💡 {len(st.session_state.available_analyses)} analysis files available • Powered by AI")