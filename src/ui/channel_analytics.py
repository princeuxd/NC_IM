"""Streamlit UI for multi-video Channel Analytics."""
from __future__ import annotations

import streamlit as st
import json
import pandas as pd
from pathlib import Path

from src.helpers import channel_analytics as ca
from src.analysis.video_frames import parse_iso_duration_to_minutes


def _safe_read_file(file_path: Path, encoding: str = "utf-8") -> str:
    """Safely read file content, return empty string if file doesn't exist."""
    try:
        if file_path.exists():
            return file_path.read_text(encoding=encoding)
        return ""
    except Exception:
        return ""


def _safe_read_json(file_path: Path) -> dict:
    """Safely read JSON file, return empty dict if file doesn't exist."""
    try:
        if file_path.exists():
            return json.loads(file_path.read_text(encoding="utf-8"))
        return {}
    except Exception:
        return {}


def render_channel_analytics():
    # Minimal CSS styling
    st.markdown("""
    <style>
    .analysis-header {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border-left: 4px solid #667eea;
    }
    .input-section {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border: 1px solid #e9ecef;
    }
    .success-banner {
        background: #d4edda;
        color: #155724;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin: 1rem 0;
        border: 1px solid #c3e6cb;
    }
    .section-header {
        background: #f8f9fa;
        color: #495057;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin: 1.5rem 0 1rem 0;
        font-size: 1.1rem;
        font-weight: 600;
        border-left: 3px solid #667eea;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Clean header section
    st.markdown("""
    <div class="analysis-header">
        <h2 style="margin: 0; color: #333;">📊 Channel Analytics</h2>
        <p style="margin: 0.5rem 0 0 0; color: #666; font-size: 0.95rem;">
            Multi-video analysis with AI insights
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Clean input section
    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            channel_id = st.text_input(
                "Channel ID", 
                key="ca_channel", 
                placeholder="UC...",
                help="Enter YouTube channel ID"
            )
        
        with col2:
            num_videos = st.number_input(
                "Videos", 
                min_value=1, 
                max_value=50, 
                value=10, 
                step=1, 
                key="ca_num",
                help="Number of videos to analyze"
            )
        
        with col3:
            st.write("")  # Spacing
            run_analysis = st.button(
                "Run Analysis", 
                key="ca_run",
                help="Start analysis",
                use_container_width=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)

    if not channel_id:
        st.info("Enter a channel ID to begin analysis.")
        return

    if run_analysis:
        with st.spinner("Running channel analysis..."):
            try:
                result = ca.analyze_channel(channel_id, num_videos=int(num_videos))
                st.session_state["ca_result"] = result
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                return

    result = st.session_state.get("ca_result")
    if not result:
        return

    # Simple success message
    st.success("✅ Analysis complete!")

    # Channel Statistics
    _display_channel_statistics(channel_id)

    # OAuth Channel Analytics with Time Period Selector
    _display_oauth_channel_analytics(channel_id)

    # Simple analysis summary
    st.markdown("""
    <div class="section-header">
        📊 Analysis Summary
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Videos Processed", result.get("videos_processed", 0))
    with col2:
        st.metric("Successful", result.get("successful_analyses", 0))
    with col3:
        st.metric("Failed", result.get("failed_analyses", 0))
    with col4:
        st.metric("Skipped", result.get("skipped_analyses", 0))

    # Simple LLM Analysis Section
    if collective := result.get("collective"):
        if collective.get("analysis"):
            st.markdown("""
            <div class="section-header">
                🎯 AI Channel Insights
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(collective["analysis"])
            
            if path := collective.get("file_path"):
                st.download_button(
                    "Download Report", 
                    open(path, "rb").read(), 
                    file_name=path.name, 
                    mime="text/markdown"
                )

    # Simple Individual Video Analysis Results Section
    st.markdown("""
    <div class="section-header">
        📹 Video Analysis Results
    </div>
    """, unsafe_allow_html=True)
    
    output_dir = result.get("output_dir")
    if not output_dir:
        st.warning("⚠️ No output directory found.")
        return

    for vid_res in result.get("results", []):
        video_id = vid_res.get("video_id", "<unknown>")
        video_title = vid_res.get("title", "Unknown Title")
        success = vid_res.get("success", False)
        skipped = vid_res.get("skipped", False)
        
        # Modern video section with color-coded status
        status_config = {
            "skipped": {"icon": "⏭️", "color": "#FF9800", "bg": "#FFF3E0"},
            "success": {"icon": "✅", "color": "#4CAF50", "bg": "#E8F5E8"},
            "failed": {"icon": "❌", "color": "#F44336", "bg": "#FFEBEE"}
        }
        
        status_key = "skipped" if skipped else ("success" if success else "failed")
        config = status_config[status_key]
        
        # Create a modern expandable section
        with st.expander(
            f"{config['icon']} {video_title[:60]}{'...' if len(video_title) > 60 else ''}", 
            expanded=False
        ):
            
            if not success:
                st.error(f"Analysis failed: {vid_res.get('error', 'Unknown error')}")
                continue
                
            if skipped:
                st.info("This video was already processed in a previous run.")
            
            video_dir = Path(output_dir) / video_id
            
            # a. Video Summary File
            summary_file = video_dir / f"{video_id}_summary.md"
            summary_content = _safe_read_file(summary_file)
            if summary_content:
                st.markdown("### 📄 Complete Video Analysis Summary")
                with st.expander("View Full Summary", expanded=False):
                    st.markdown(summary_content)
                st.download_button(
                    "⬇️ Download Summary", 
                    summary_content.encode('utf-8'), 
                    file_name=f"{video_id}_summary.md",
                    mime="text/markdown",
                    key=f"summary_{video_id}"
                )
            
            # b. Audio Analysis
            audio_file = video_dir / f"{video_id}_audio.json"
            audio_data = _safe_read_json(audio_file)
            if audio_data:
                st.markdown("### 🎤 Audio Analysis")
                st.markdown("**Transcript segments with sentiment analysis**")
                
                # Show sample segments
                segments = audio_data if isinstance(audio_data, list) else []
                if segments:
                    sample_segments = segments[:3]  # Show first 3 segments
                    for i, seg in enumerate(sample_segments):
                        sentiment = seg.get('sentiment', 'N/A')
                        text = seg.get('text', '').strip()
                        if text:
                            st.markdown(f"**Segment {i+1}** (Sentiment: {sentiment}): {text}")
                    
                    if len(segments) > 3:
                        st.markdown(f"... and {len(segments) - 3} more segments")
                
                st.download_button(
                    "⬇️ Download Audio Analysis JSON", 
                    json.dumps(audio_data, indent=2).encode('utf-8'), 
                    file_name=f"{video_id}_audio.json",
                    mime="application/json",
                    key=f"audio_{video_id}"
                )
            
            # c. Video Frame Analysis
            frames_file = video_dir / f"{video_id}_frames.json"
            frames_data = _safe_read_json(frames_file)
            vision_summary_file = video_dir / f"{video_id}_vision_summary.md"
            vision_content = _safe_read_file(vision_summary_file)
            
            if frames_data or vision_content:
                st.markdown("### 🎬 Video Frame Analysis")
                
                if vision_content:
                    st.markdown("**LLM Vision Analysis:**")
                    st.markdown(vision_content)
                
                if frames_data:
                    frame_count = len(frames_data) if isinstance(frames_data, list) else 0
                    st.markdown(f"**Extracted {frame_count} frames for analysis**")
                    
                    st.download_button(
                        "⬇️ Download Frames JSON", 
                        json.dumps(frames_data, indent=2).encode('utf-8'), 
                        file_name=f"{video_id}_frames.json",
                        mime="application/json",
                        key=f"frames_{video_id}"
                    )
            
            # d. Comments Analysis
            comments_file = video_dir / f"{video_id}_comments.json"
            comments_data = _safe_read_json(comments_file)
            if comments_data:
                st.markdown("### 💬 Comments Analysis")
                
                comments_list = comments_data if isinstance(comments_data, list) else []
                if comments_list:
                    # Calculate sentiment stats
                    sentiments = [c.get('sentiment', 0) for c in comments_list if 'sentiment' in c]
                    if sentiments:
                        avg_sentiment = sum(sentiments) / len(sentiments)
                        positive_count = sum(1 for s in sentiments if s > 0.1)
                        negative_count = sum(1 for s in sentiments if s < -0.1)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total Comments", len(comments_list))
                        col2.metric("Avg Sentiment", f"{avg_sentiment:.2f}")
                        col3.metric("Positive", positive_count)
                        col4.metric("Negative", negative_count)
                    
                    # Show sample comments
                    st.markdown("**Sample Comments:**")
                    for i, comment in enumerate(comments_list[:3]):
                        author = comment.get('author', 'Unknown')
                        text = comment.get('text', comment.get('textDisplay', ''))
                        sentiment = comment.get('sentiment', 'N/A')
                        likes = comment.get('likeCount', 0)
                        
                        st.markdown(f"**{author}** (Sentiment: {sentiment}, Likes: {likes})")
                        st.markdown(f"> {text[:200]}{'...' if len(text) > 200 else ''}")
                
                st.download_button(
                    "⬇️ Download Comments Analysis JSON", 
                    json.dumps(comments_data, indent=2).encode('utf-8'), 
                    file_name=f"{video_id}_comments.json",
                    mime="application/json",
                    key=f"comments_{video_id}"
                )
            
            # e. Statistics
            stats_file = video_dir / f"{video_id}_stats.json"
            stats_data = _safe_read_json(stats_file)
            if stats_data:
                st.markdown("### 📊 Video Statistics")
                
                # Extract key metrics
                statistics = stats_data.get('statistics', {})
                snippet = stats_data.get('snippet', {})
                
                if statistics:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Views", statistics.get('viewCount', 0))
                    col2.metric("Likes", statistics.get('likeCount', 0))
                    col3.metric("Comments", statistics.get('commentCount', 0))
                    # Parse duration from ISO format to human readable
                    duration_iso = stats_data.get('contentDetails', {}).get('duration', 'PT0S')
                    duration_minutes = parse_iso_duration_to_minutes(duration_iso)
                    duration_formatted = f"{duration_minutes} min" if duration_minutes > 0 else "N/A"
                    col4.metric("Duration", duration_formatted)
                
                # Show additional info
                if snippet:
                    published_at = snippet.get('publishedAt', 'N/A')
                    description = snippet.get('description', '')[:200]
                    st.markdown(f"**Published:** {published_at}")
                    if description:
                        st.markdown(f"**Description:** {description}...")
                
                st.download_button(
                    "⬇️ Download Statistics JSON", 
                    json.dumps(stats_data, indent=2).encode('utf-8'), 
                    file_name=f"{video_id}_stats.json",
                    mime="application/json",
                    key=f"stats_{video_id}"
                )
            
            # OAuth Analytics (if available)
            oauth_file = video_dir / f"{video_id}_oauth_analytics.json"
            oauth_data = _safe_read_json(oauth_file)
            if oauth_data and (not isinstance(oauth_data, dict) or not oauth_data.get("error")):
                st.markdown("### 🔐 OAuth Available")
                
                # Display comprehensive OAuth analytics using the same functions as video analytics
                _display_enhanced_analytics(oauth_data, video_title, video_id)
                
                st.download_button(
                    "⬇️ Download OAuth Analytics JSON", 
                    json.dumps(oauth_data, indent=2).encode('utf-8'), 
                    file_name=f"{video_id}_oauth_analytics.json",
                    mime="application/json",
                    key=f"oauth_{video_id}"
                )
            elif oauth_data and isinstance(oauth_data, dict) and oauth_data.get("error"):
                st.markdown("### 🔐 OAuth Analytics")
                st.error(f"OAuth analytics failed: {oauth_data['error']}")
            else:
                st.markdown("### 🔐 OAuth Analytics")
                st.info("OAuth analytics not available for this video.")
            
            # f. Combined Analysis Data
            analysis_file = video_dir / f"{video_id}_analysis.json"
            analysis_data = _safe_read_json(analysis_file)
            if analysis_data:
                st.markdown("### 🔗 Combined Analysis Data")
                st.markdown("This file contains all analysis results combined for LLM processing.")
                st.download_button(
                    "⬇️ Download Combined Analysis JSON", 
                    json.dumps(analysis_data, indent=2).encode('utf-8'), 
                    file_name=f"{video_id}_analysis.json",
                    mime="application/json",
                    key=f"analysis_{video_id}"
                )

    # Simple bulk downloads section
    st.markdown("""
    <div class="section-header">
        📦 Downloads
    </div>
    """, unsafe_allow_html=True)
    
    if output_dir and Path(output_dir).exists():
        st.info(f"Analysis files saved to: `{output_dir}`")


def _display_enhanced_analytics(analytics_data, video_title, display_title="Video"):
    """Display comprehensive enhanced analytics data in a beautiful format."""
    if not analytics_data:
        return

    st.subheader("📊 Comprehensive Video Analytics (OAuth)")
    
    # Create tabs for different analytics sections
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Overview", "📊 Audience Retention", "🌍 Demographics", 
        "🗺️ Geography", "💰 Monetization", "📅 Time Series", "🚀 Engagement"
    ])
    
    with tab1:
        _display_overview_metrics(analytics_data)
    
    with tab2:
        _display_audience_retention(analytics_data)
    
    with tab3:
        _display_demographics_analytics(analytics_data)
    
    with tab4:
        _display_geography_analytics(analytics_data)
    
    with tab5:
        _display_monetization_analytics(analytics_data)
    
    with tab6:
        _display_time_series_analytics(analytics_data)
    
    with tab7:
        _display_engagement_analytics(analytics_data)


def _display_overview_metrics(analytics_data):
    """Display overview metrics and key performance indicators."""
    
    # Summary metrics
    summary_data = analytics_data.get("summary_metrics", {})
    engagement_data = analytics_data.get("engagement_metrics", {})
    impressions_data = analytics_data.get("impressions", {})
    
    if summary_data.get("rows"):
        row = summary_data["rows"][0]
        
        st.markdown("### 🎯 Key Performance Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            views = int(row[1]) if len(row) > 1 else 0
            st.metric("👀 Views", f"{views:,}")
        with col2:
            watch_time = int(row[2]) if len(row) > 2 else 0
            st.metric("⏱️ Watch Time", f"{watch_time:,} min")
        with col3:
            avg_duration = int(row[3]) if len(row) > 3 else 0
            st.metric("🎯 Avg Duration", f"{avg_duration:,} sec")
        with col4:
            likes = int(row[4]) if len(row) > 4 else 0
            st.metric("👍 Likes", f"{likes:,}")
        with col5:
            comments = int(row[5]) if len(row) > 5 else 0
            st.metric("💬 Comments", f"{comments:,}")
    
    # Enhanced engagement metrics
    if engagement_data.get("rows"):
        eng_row = engagement_data["rows"][0]
        st.markdown("### 📊 Engagement Breakdown")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            shares = int(eng_row[4]) if len(eng_row) > 4 else 0
            st.metric("🔄 Shares", f"{shares:,}")
        with col2:
            subs_gained = int(eng_row[5]) if len(eng_row) > 5 else 0
            st.metric("🔔 Subs Gained", f"+{subs_gained:,}")
        with col3:
            playlist_adds = int(eng_row[7]) if len(eng_row) > 7 else 0
            st.metric("📋 Playlist Adds", f"{playlist_adds:,}")
        with col4:
            saves = int(eng_row[8]) if len(eng_row) > 8 else 0
            st.metric("💾 Saves", f"{saves:,}")
        with col5:
            # Calculate engagement rate
            total_views = int(eng_row[0]) if len(eng_row) > 0 else 0
            total_likes = int(eng_row[1]) if len(eng_row) > 1 else 0
            total_comments = int(eng_row[3]) if len(eng_row) > 3 else 0
            
            if total_views > 0:
                engagement_rate = ((total_likes + total_comments + shares) / total_views) * 100
                st.metric("📈 Engagement Rate", f"{engagement_rate:.2f}%")
            else:
                st.metric("📈 Engagement Rate", "0.00%")
        with col6:
            # Like to view ratio
            if total_views > 0:
                like_ratio = (total_likes / total_views) * 100
                st.metric("👍 Like Rate", f"{like_ratio:.2f}%")
            else:
                st.metric("👍 Like Rate", "0.00%")
    
    # Impressions data
    if impressions_data.get("rows") and not impressions_data.get("error"):
        imp_row = impressions_data["rows"][0]
        st.markdown("### 👁️ Impressions & Discovery")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            impressions = int(imp_row[0]) if len(imp_row) > 0 else 0
            st.metric("👁️ Impressions", f"{impressions:,}")
        with col2:
            ctr = float(imp_row[1]) if len(imp_row) > 1 else 0
            st.metric("🎯 Click-through Rate", f"{ctr:.2f}%")
        with col3:
            unique_viewers = int(imp_row[2]) if len(imp_row) > 2 else 0
            st.metric("👤 Unique Viewers", f"{unique_viewers:,}")


def _display_audience_retention(analytics_data):
    """Display audience retention analytics with interactive charts."""
    
    retention_data = analytics_data.get("audience_retention", [])
    
    if retention_data and not isinstance(retention_data, dict):
        st.markdown("### 📊 Audience Retention Curve")
        
        # Convert retention data to chart format
        
        if retention_data:
            time_points = []
            retention_rates = []
            
            for row in retention_data:
                if len(row) >= 2:
                    time_points.append(float(row[0]) * 100)  # Convert to percentage
                    retention_rates.append(float(row[1]) * 100)  # Convert to percentage
            
            if time_points and retention_rates:
                df = pd.DataFrame({
                    'Video Progress (%)': time_points,
                    'Audience Retention (%)': retention_rates
                })
                
                st.line_chart(df.set_index('Video Progress (%)'))
                
                # Key insights
                st.markdown("### 🔍 Retention Insights")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_retention = sum(retention_rates) / len(retention_rates)
                    st.metric("📊 Average Retention", f"{avg_retention:.1f}%")
                
                with col2:
                    max_retention = max(retention_rates)
                    max_point = time_points[retention_rates.index(max_retention)]
                    st.metric("🎯 Peak Retention", f"{max_retention:.1f}% at {max_point:.0f}%")
                
                with col3:
                    min_retention = min(retention_rates)
                    min_point = time_points[retention_rates.index(min_retention)]
                    st.metric("📉 Lowest Retention", f"{min_retention:.1f}% at {min_point:.0f}%")
                
                # Identify key moments
                st.markdown("### 🎬 Key Moments Analysis")
                
                # Find spikes (increases)
                spikes = []
                dips = []
                
                for i in range(1, len(retention_rates)):
                    change = retention_rates[i] - retention_rates[i-1]
                    if change > 5:  # Significant spike
                        spikes.append((time_points[i], retention_rates[i], change))
                    elif change < -5:  # Significant dip
                        dips.append((time_points[i], retention_rates[i], abs(change)))
                
                col1, col2 = st.columns(2)
                with col1:
                    if spikes:
                        st.markdown("**📈 Retention Spikes (Rewatched/Shared moments):**")
                        for time_point, retention, change in spikes[:3]:
                            st.write(f"• {time_point:.0f}% mark: +{change:.1f}% retention boost")
                    else:
                        st.info("No significant retention spikes detected")
                
                with col2:
                    if dips:
                        st.markdown("**📉 Retention Dips (Drop-off points):**")
                        for time_point, retention, change in dips[:3]:
                            st.write(f"• {time_point:.0f}% mark: -{change:.1f}% drop")
                    else:
                        st.info("No significant retention dips detected")
        else:
            st.info("No retention data available for the selected period")
    else:
        st.info("Audience retention data not available - requires video ownership and sufficient views (100+ views)")


def _display_demographics_analytics(analytics_data):
    """Display demographic breakdown of the audience."""
    
    demographics_data = analytics_data.get("demographics", [])
    
    if demographics_data and not isinstance(demographics_data, dict):
        st.markdown("### 👥 Audience Demographics")
        
        # Process demographics data
        age_gender_data = []
        
        for row in demographics_data:
            if len(row) >= 3:
                age_group = row[0]
                gender = row[1]
                percentage = float(row[2])
                age_gender_data.append({
                    'Age Group': age_group,
                    'Gender': gender,
                    'Percentage': percentage
                })
        
        if age_gender_data:
            df = pd.DataFrame(age_gender_data)
            
            # Age distribution
            age_totals = df.groupby('Age Group')['Percentage'].sum().reset_index()
            age_totals = age_totals.sort_values('Percentage', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Age Distribution**")
                st.bar_chart(age_totals.set_index('Age Group'))
            
            with col2:
                st.markdown("**⚧ Gender Distribution**")
                gender_totals = df.groupby('Gender')['Percentage'].sum().reset_index()
                st.bar_chart(gender_totals.set_index('Gender'))
            
            # Top demographics
            st.markdown("### 🎯 Top Demographics")
            df_sorted = df.sort_values('Percentage', ascending=False)
            
            for i, row in df_sorted.head(5).iterrows():
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"**{row['Age Group']} - {row['Gender']}**")
                with col2:
                    st.write(f"{row['Percentage']:.1f}%")
                with col3:
                    # Create a simple progress bar
                    progress = row['Percentage'] / df['Percentage'].max()
                    st.progress(progress)
        else:
            st.info("No demographic data available")
    else:
        st.info("Demographics data not available - requires sufficient views and channel permissions")


def _display_geography_analytics(analytics_data):
    """Display geographic distribution of viewers."""
    
    geography_data = analytics_data.get("geography", [])
    
    if geography_data and not isinstance(geography_data, dict):
        st.markdown("### 🗺️ Geographic Distribution")
        
        # Process geography data
        country_data = []
        
        for row in geography_data:
            if len(row) >= 2:
                country_code = row[0]
                views = int(row[1])
                
                # Map country codes to names (basic mapping)
                country_names = {
                    'US': 'United States', 'GB': 'United Kingdom', 'CA': 'Canada',
                    'AU': 'Australia', 'DE': 'Germany', 'FR': 'France', 'IN': 'India',
                    'JP': 'Japan', 'BR': 'Brazil', 'MX': 'Mexico', 'IT': 'Italy',
                    'ES': 'Spain', 'RU': 'Russia', 'KR': 'South Korea', 'NL': 'Netherlands'
                }
                
                country_name = country_names.get(country_code, country_code)
                country_data.append({
                    'Country': country_name,
                    'Country Code': country_code,
                    'Views': views
                })
        
        if country_data:
            df = pd.DataFrame(country_data)
            df = df.sort_values('Views', ascending=False)
            
            # Calculate percentages
            total_views = df['Views'].sum()
            df['Percentage'] = (df['Views'] / total_views * 100).round(1)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🌍 Top 10 Countries by Views**")
                top_countries = df.head(10)
                st.bar_chart(top_countries.set_index('Country')['Views'])
            
            with col2:
                st.markdown("**📊 Geographic Breakdown**")
                for i, row in df.head(10).iterrows():
                    col_country, col_views, col_percent = st.columns([2, 1, 1])
                    with col_country:
                        st.write(f"🌍 **{row['Country']}**")
                    with col_views:
                        st.write(f"{row['Views']:,}")
                    with col_percent:
                        st.write(f"{row['Percentage']}%")
            
            # Geographic insights
            st.markdown("### 🌐 Geographic Insights")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                top_country = df.iloc[0]
                st.metric("🥇 Top Country", top_country['Country'])
                st.caption(f"{top_country['Views']:,} views ({top_country['Percentage']}%)")
            
            with col2:
                countries_count = len(df)
                st.metric("🌍 Countries Reached", f"{countries_count}")
            
            with col3:
                top_5_percentage = df.head(5)['Percentage'].sum()
                st.metric("🎯 Top 5 Countries", f"{top_5_percentage:.1f}%")
        else:
            st.info("No geographic data available")
    else:
        st.info("Geographic data not available - requires sufficient views")


def _display_monetization_analytics(analytics_data):
    """Display monetization and revenue analytics."""
    
    monetization_data = analytics_data.get("monetization", {})
    
    if monetization_data.get("rows") and not monetization_data.get("error"):
        st.markdown("### 💰 Monetization Analytics")
        
        row = monetization_data["rows"][0]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            estimated_revenue = float(row[0]) if len(row) > 0 else 0
            st.metric("💵 Estimated Revenue", f"${estimated_revenue:.2f}")
        
        with col2:
            ad_revenue = float(row[1]) if len(row) > 1 else 0
            st.metric("📺 Ad Revenue", f"${ad_revenue:.2f}")
        
        with col3:
            cpm = float(row[4]) if len(row) > 4 else 0
            st.metric("📊 CPM", f"${cpm:.2f}")
        
        with col4:
            playback_cpm = float(row[5]) if len(row) > 5 else 0
            st.metric("▶️ Playback CPM", f"${playback_cpm:.2f}")
        
        # Revenue breakdown
        if estimated_revenue > 0:
            st.markdown("### 💹 Revenue Breakdown")
            
            red_revenue = float(row[2]) if len(row) > 2 else 0
            gross_revenue = float(row[3]) if len(row) > 3 else 0
            
            revenue_data = {
                'Ad Revenue': ad_revenue,
                'YouTube Premium Revenue': red_revenue,
                'Other Revenue': max(0, gross_revenue - ad_revenue - red_revenue)
            }
            
            df = pd.DataFrame(list(revenue_data.items()), columns=['Revenue Type', 'Amount'])
            df = df[df['Amount'] > 0]  # Only show non-zero revenues
            
            if not df.empty:
                st.bar_chart(df.set_index('Revenue Type'))
            
            # Performance indicators
            st.markdown("### 📈 Performance Indicators")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Revenue per 1000 views
                summary_data = analytics_data.get("summary_metrics", {})
                if summary_data.get("rows"):
                    views = int(summary_data["rows"][0][1])
                    rpm = (estimated_revenue / views * 1000) if views > 0 else 0
                    st.metric("💰 RPM (Revenue per 1000 views)", f"${rpm:.2f}")
            
            with col2:
                impressions_data = analytics_data.get("impressions", {})
                if impressions_data.get("rows") and not impressions_data.get("error"):
                    impressions = int(impressions_data["rows"][0][0])
                    impression_cpm = float(row[6]) if len(row) > 6 else 0
                    st.metric("👁️ Impression CPM", f"${impression_cpm:.2f}")
            
            with col3:
                # Ad revenue percentage
                ad_percentage = (ad_revenue / estimated_revenue * 100) if estimated_revenue > 0 else 0
                st.metric("📺 Ad Revenue %", f"{ad_percentage:.1f}%")
    else:
        st.info("💰 Monetization data not available - requires monetized channel and sufficient revenue")


def _display_time_series_analytics(analytics_data):
    """Display time series data with interactive charts."""
    
    time_series_data = analytics_data.get("time_series", {})
    
    if time_series_data.get("rows"):
        st.markdown("### 📅 Performance Over Time")
        
        # Process time series data
        dates = []
        views = []
        likes = []
        subscribers = []
        watch_time = []
        shares = []
        comments = []
        
        for row in time_series_data["rows"]:
            if len(row) >= 6:
                dates.append(row[0])  # Date
                views.append(int(row[1]))  # Views
                likes.append(int(row[2]))  # Likes
                subscribers.append(int(row[3]))  # Subscribers gained
                watch_time.append(int(row[4]))  # Watch time
                shares.append(int(row[5]))  # Shares
                comments.append(int(row[6]) if len(row) > 6 else 0)  # Comments
        
        if dates:
            df = pd.DataFrame({
                'Date': pd.to_datetime(dates),
                'Views': views,
                'Likes': likes,
                'Subscribers Gained': subscribers,
                'Watch Time (min)': watch_time,
                'Shares': shares,
                'Comments': comments
            })
            df = df.set_index('Date')
            
            # Display charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📈 Views Over Time**")
                st.line_chart(df[['Views']])
                
                st.markdown("**👍 Likes Over Time**")
                st.line_chart(df[['Likes']])
            
            with col2:
                st.markdown("**🔔 Subscribers Gained Over Time**")
                st.line_chart(df[['Subscribers Gained']])
                
                st.markdown("**⏱️ Watch Time Over Time**")
                st.line_chart(df[['Watch Time (min)']])
            
            # Combined engagement chart
            st.markdown("**📊 Engagement Metrics Over Time**")
            engagement_df = df[['Likes', 'Shares', 'Comments']]
            st.line_chart(engagement_df)
            
            # Summary statistics
            st.markdown("### 📊 Time Series Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_views = sum(views)
                st.metric("📈 Total Views", f"{total_views:,}")
                
                peak_views_day = df.loc[df['Views'].idxmax()].name.strftime('%Y-%m-%d')
                st.caption(f"Peak: {peak_views_day}")
            
            with col2:
                total_likes = sum(likes)
                st.metric("👍 Total Likes", f"{total_likes:,}")
                
                avg_likes = total_likes / len(likes) if likes else 0
                st.caption(f"Avg/day: {avg_likes:.1f}")
            
            with col3:
                total_subs = sum(subscribers)
                st.metric("🔔 Subscribers Gained", f"+{total_subs:,}")
                
                best_sub_day = max(subscribers) if subscribers else 0
                st.caption(f"Best day: +{best_sub_day}")
            
            with col4:
                total_watch_time = sum(watch_time)
                st.metric("⏱️ Total Watch Time", f"{total_watch_time:,} min")
                
                hours = total_watch_time / 60
                st.caption(f"{hours:.1f} hours")
        else:
            st.info("No time series data available")
    else:
        st.info("Time series data not available - requires video ownership")


def _display_engagement_analytics(analytics_data):
    """Display detailed engagement analytics and metrics."""
    
    st.markdown("### 🚀 Engagement Analytics")
    
    # Traffic sources
    traffic_data = analytics_data.get("traffic_sources", [])
    engagement_data = analytics_data.get("engagement_metrics", {})
    
    if traffic_data and not isinstance(traffic_data, dict):
        st.markdown("### 🚦 Traffic Sources")
        
        traffic_list = []
        for row in traffic_data:
            if len(row) >= 2:
                source_type = row[0]
                views = int(row[1])
                
                # Friendly source names
                source_names = {
                    'PLAYLIST': '📋 Playlists',
                    'SEARCH': '🔍 YouTube Search',
                    'SUGGESTED_VIDEO': '💡 Suggested Videos',
                    'BROWSE': '🏠 Browse Features',
                    'CHANNEL': '📺 Channel Page',
                    'EXTERNAL': '🌐 External Sources',
                    'DIRECT': '🔗 Direct Links',
                    'NOTIFICATION': '🔔 Notifications'
                }
                
                friendly_name = source_names.get(source_type, source_type)
                traffic_list.append({'Source': friendly_name, 'Views': views})
        
        if traffic_list:
            df = pd.DataFrame(traffic_list)
            df = df.sort_values('Views', ascending=False)
            
            total_traffic_views = df['Views'].sum()
            df['Percentage'] = (df['Views'] / total_traffic_views * 100).round(1)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.bar_chart(df.set_index('Source')['Views'])
            
            with col2:
                st.markdown("**Traffic Source Breakdown:**")
                for i, row in df.iterrows():
                    st.write(f"**{row['Source']}**: {row['Views']:,} views ({row['Percentage']}%)")
    
    # Engagement summary
    if engagement_data.get("rows"):
        row = engagement_data["rows"][0]
        
        st.markdown("### 💫 Engagement Summary")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_views = int(row[0]) if len(row) > 0 else 0
            st.metric("👀 Total Views", f"{total_views:,}")
        
        with col2:
            total_likes = int(row[1]) if len(row) > 1 else 0
            dislikes = int(row[2]) if len(row) > 2 else 0
            like_ratio = (total_likes / (total_likes + dislikes) * 100) if (total_likes + dislikes) > 0 else 0
            st.metric("👍 Like Ratio", f"{like_ratio:.1f}%")
        
        with col3:
            comments = int(row[3]) if len(row) > 3 else 0
            comment_rate = (comments / total_views * 100) if total_views > 0 else 0
            st.metric("💬 Comment Rate", f"{comment_rate:.2f}%")
        
        with col4:
            shares = int(row[4]) if len(row) > 4 else 0
            share_rate = (shares / total_views * 100) if total_views > 0 else 0
            st.metric("🔄 Share Rate", f"{share_rate:.2f}%")
        
        with col5:
            subs_gained = int(row[5]) if len(row) > 5 else 0
            sub_rate = (subs_gained / total_views * 100) if total_views > 0 else 0
            st.metric("🔔 Sub Rate", f"{sub_rate:.2f}%")
        
        # Additional engagement metrics
        st.markdown("### 📊 Additional Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            playlist_adds = int(row[7]) if len(row) > 7 else 0
            st.metric("📋 Playlist Additions", f"{playlist_adds:,}")
        
        with col2:
            saves = int(row[8]) if len(row) > 8 else 0
            st.metric("💾 Saves", f"{saves:,}")
        
        with col3:
            # Calculate overall engagement score
            if total_views > 0:
                engagement_score = ((total_likes + comments + shares + playlist_adds + saves) / total_views * 100)
                st.metric("🌟 Engagement Score", f"{engagement_score:.2f}%")
            else:
                st.metric("🌟 Engagement Score", "0.00%")


def _display_channel_statistics(channel_id: str):
    """Display channel statistics similar to simple streamlit app."""
    try:
        # Get channel stats using public API
        import os
        from src.config.settings import SETTINGS
        from src.youtube.public import get_service as get_public_service
        
        api_key = os.getenv("YT_API_KEY") or SETTINGS.youtube_api_key
        if not api_key:
            st.warning("YouTube API key not configured - cannot display channel statistics")
            return
            
        service = get_public_service(api_key)
        channel_stats = _get_channel_stats(service, channel_id)
        
        if channel_stats:
            _display_channel_stats_ui(channel_stats)
        else:
            st.warning("Could not fetch channel statistics")
            
    except Exception as e:
        st.error(f"Failed to display channel statistics: {e}")


def _get_channel_stats(service, channel_id: str) -> dict:
    """Fetch channel statistics and snippet information."""
    try:
        response = (
            service.channels().list(part="snippet,statistics", id=channel_id).execute()
        )

        if response["items"]:
            channel = response["items"][0]

            # Get the best available thumbnail
            thumbnails = channel["snippet"].get("thumbnails", {})
            thumbnail_url = None

            # Try thumbnails in order of preference (highest quality first)
            for size in ["high", "medium", "default"]:
                if size in thumbnails:
                    raw_url = thumbnails[size]["url"]

                    # Clean up Google's channel thumbnail URL parameters that can cause issues
                    if "yt3.ggpht.com" in raw_url:
                        try:
                            # Remove problematic parameters and use a simpler format
                            base_url = raw_url.split("=")[0]
                            thumbnail_url = f"{base_url}=s240-c-k-c0x00ffffff-no-rj"
                        except Exception:
                            thumbnail_url = raw_url
                    else:
                        thumbnail_url = raw_url

                    break

            return {
                "title": channel["snippet"]["title"],
                "description": channel["snippet"]["description"],
                "thumbnail": thumbnail_url,
                "subscriber_count": int(
                    channel["statistics"].get("subscriberCount", 0)
                ),
                "video_count": int(channel["statistics"].get("videoCount", 0)),
                "view_count": int(channel["statistics"].get("viewCount", 0)),
                "published_at": channel["snippet"]["publishedAt"],
                "custom_url": channel["snippet"].get("customUrl", ""),
            }
    except Exception as e:
        st.error(f"Failed to fetch channel stats: {e}")
    return {}


def _display_channel_stats_ui(channel_stats: dict):
    """Display channel statistics in a modern, visually appealing format."""
    if not channel_stats:
        return

    # Simple channel info section
    st.markdown("""
    <div class="section-header">
        📺 Channel Information
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        col_thumb, col_info = st.columns([1, 3])

        with col_thumb:
            thumbnail_url = channel_stats.get("thumbnail")
            if thumbnail_url:
                try:
                    if thumbnail_url.startswith("http://"):
                        thumbnail_url = thumbnail_url.replace("http://", "https://", 1)
                    st.image(thumbnail_url, width=100)
                except Exception:
                    st.markdown("📺")
            else:
                st.markdown("📺")

        with col_info:
            st.markdown(f"**{channel_stats.get('title', 'Unknown Channel')}**")
            if channel_stats.get("custom_url"):
                st.markdown(f"@{channel_stats['custom_url']}")

            # Simple description preview
            if channel_stats.get("description"):
                description = channel_stats["description"]
                preview = description[:100] + "..." if len(description) > 100 else description
                st.caption(f"_{preview}_")

    # Simple statistics section
    st.markdown("""
    <div class="section-header">
        📊 Channel Statistics
    </div>
    """, unsafe_allow_html=True)

    # Format numbers nicely
    def format_number(num):
        if num >= 1_000_000:
            return f"{num/1_000_000:.1f}M"
        elif num >= 1_000:
            return f"{num/1_000:.1f}K"
        return str(num)

    subscribers = format_number(channel_stats.get("subscriber_count", 0))
    videos = format_number(channel_stats.get("video_count", 0))
    views = format_number(channel_stats.get("view_count", 0))

    # Simple metric display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Subscribers", subscribers)
    with col2:
        st.metric("Videos", videos)
    with col3:
        st.metric("Total Views", views)
    with col4:
        if channel_stats.get("video_count", 0) > 0:
            avg_views = channel_stats.get("view_count", 0) // channel_stats["video_count"]
            st.metric("Avg Views", format_number(avg_views))

    # Simple channel age
    from datetime import datetime
    try:
        published_at = channel_stats.get("published_at")
        if published_at:
            created_date = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            years_old = (datetime.now(created_date.tzinfo) - created_date).days // 365
            age_text = f"{years_old} years" if years_old > 0 else f"{(datetime.now(created_date.tzinfo) - created_date).days} days"
            st.caption(f"**Channel Age:** {age_text}")
    except Exception:
        pass


def _display_oauth_channel_analytics(channel_id: str):
    """Display OAuth channel analytics with time period selector."""
    try:
        # Check OAuth capabilities first
        import os
        from src.config.settings import SETTINGS
        from src.analysis.channel_analysis import ChannelAnalysisService
        
        api_key = os.getenv("YT_API_KEY") or SETTINGS.youtube_api_key
        if not api_key:
            st.info("🔒 OAuth analytics not available - YouTube API key not configured")
            return
            
        service = ChannelAnalysisService(api_key)
        oauth_info = service._detect_oauth_capabilities()
        
        if not oauth_info.get("available") or not oauth_info.get("analytics_access"):
            st.info("🔒 OAuth analytics not available - requires channel authentication")
            return
            
        # Check if this specific channel has OAuth access
        try:
            oauth_service, access_type = service.get_service_for_channel(channel_id)
            if access_type != "oauth":
                st.info("🔒 OAuth analytics not available for this channel")
                return
        except Exception as e:
            st.info(f"🔒 OAuth analytics not available for this channel: {e}")
            return
            
        st.markdown("---")
        st.subheader("🔐 Advanced Analytics (OAuth)")
        
        # Time period selector
        col1, col2 = st.columns([2, 1])
        
        with col1:
            period_options = {
                "Last 7 days": 7,
                "Last 14 days": 14,
                "Last 30 days": 30,
                "Last 60 days": 60,
                "Last 90 days": 90,
                "Last 6 months": 180,
                "Last year": 365,
                "All Time": None  # Will be set dynamically
            }
            
            selected_period = st.selectbox(
                "Select time period for analytics",
                options=list(period_options.keys()),
                index=2,  # Default to Last 30 days
                help="Choose the time range for your analytics data",
                key="oauth_channel_period"
            )
            
            days_back = period_options[selected_period]
            
            # If 'All Time' is selected, calculate days_back from channel creation
            if selected_period == "All Time":
                try:
                    # Get channel info to find published date
                    import os
                    from src.config.settings import SETTINGS
                    from src.youtube.public import get_service as get_public_service
                    
                    api_key = os.getenv("YT_API_KEY") or SETTINGS.youtube_api_key
                    if api_key:
                        public_service = get_public_service(api_key)
                        response = public_service.channels().list(part="snippet", id=channel_id).execute()
                        if response["items"]:
                            published_at = response["items"][0]["snippet"]["publishedAt"]
                            from datetime import datetime
                            created_date = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                            days_back = (datetime.now(created_date.tzinfo) - created_date).days
                        else:
                            days_back = 3650  # fallback to 10 years
                    else:
                        days_back = 3650  # fallback to 10 years
                except Exception:
                    days_back = 3650  # fallback to 10 years
        
        with col2:
            st.metric("📊 Analysis Period", selected_period)
        
        # Fetch comprehensive analytics data
        with st.spinner(f"🔍 Loading OAuth analytics for {selected_period.lower()}..."):
            try:
                from src.analytics_helpers import get_full_channel_analytics
                import os
                from src.config.settings import SETTINGS
                from src.youtube.public import get_service as get_public_service
                
                # Get public service for basic data
                api_key = os.getenv("YT_API_KEY") or SETTINGS.youtube_api_key
                public_service = get_public_service(api_key) if api_key else None
                
                analytics_data = get_full_channel_analytics(
                    oauth_service, public_service, channel_id, days_back=days_back
                )
                
                # OAuth data fetched successfully
                
                if analytics_data.get("error"):
                    st.error(f"❌ {analytics_data['error']}")
                    return
                
                # Display OAuth analytics in tabs
                _display_oauth_channel_tabs(analytics_data, selected_period)
                
            except Exception as e:
                st.error(f"Failed to fetch OAuth analytics: {e}")
                
    except Exception as e:
        st.error(f"Failed to display OAuth channel analytics: {e}")


def _display_oauth_channel_tabs(analytics_data: dict, period: str):
    """Display OAuth channel analytics in tabbed interface."""
    oauth_data = analytics_data.get("oauth", {})
    if not oauth_data or oauth_data.get("error"):
        st.warning("OAuth analytics data not available")
        return
    
    # Create tabs for different analytics sections - exactly like old streamlit
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📈 Engagement Analysis",
        "📅 Upload Patterns",
        "🏆 Top Content",
        "👥 Audience Insights (OAuth, All Time)",
        "🚦 Traffic Sources",
        "💰 Monetization",
        "📈 Growth Trends (OAuth, All Time)",
        "👥 Views by Subscriber Status"
    ])
    
    with tab1:
        _display_oauth_enhanced_engagement(analytics_data, period)
    with tab2:
        _display_public_upload_patterns(analytics_data)
    with tab3:
        _display_public_top_content(analytics_data)
    with tab4:
        _display_oauth_audience_insights(analytics_data, "All Time")
    with tab5:
        _display_oauth_traffic_sources(analytics_data)
    with tab6:
        _display_oauth_revenue_metrics(analytics_data, period)
    with tab7:
        _display_oauth_growth_trends(analytics_data, "All Time")
    with tab8:
        _display_oauth_subscriber_status(analytics_data, period)


def _display_oauth_enhanced_engagement(analytics_data: dict, period: str):
    """Enhanced engagement section with OAuth data - exactly like old streamlit."""
    
    # First show public engagement analysis (like the old code)
    _display_public_engagement_analysis(analytics_data, None)
    
    # Add OAuth enhancements
    oauth_data = analytics_data.get("oauth", {})
    if not oauth_data or oauth_data.get("error"):
        return
    
    # Only show OAuth metrics if we actually have data
    impressions = oauth_data.get("impressions", {})
    engagement = oauth_data.get("engagement_breakdown", {})
    
    # Check if we have any OAuth data to display
    has_impressions = impressions.get("rows")
    has_engagement = engagement.get("rows")
    
    if has_impressions or has_engagement:
        st.subheader(f"🔒 Enhanced Engagement Metrics (OAuth) - {period}")
        
        if has_impressions:
            col1, col2, col3 = st.columns(3)
            row = impressions["rows"][0]
            with col1:
                total_impressions = int(row[0]) if len(row) > 0 else 0
                st.metric("👁️ Impressions", f"{total_impressions:,}")
            with col2:
                ctr = float(row[1]) if len(row) > 1 else 0
                st.metric("🎯 Click-Through Rate", f"{ctr:.2f}%")
            with col3:
                unique_viewers = int(row[2]) if len(row) > 2 else 0
                st.metric("👤 Unique Viewers", f"{unique_viewers:,}")
    
        # Engagement breakdown (only if we have data)
        if has_engagement:
            row = engagement["rows"][0]
            st.markdown("**📊 Detailed Engagement Breakdown**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                likes = int(row[1]) if len(row) > 1 else 0
                st.metric("👍 Likes", f"{likes:,}")
            with col2:
                shares = int(row[4]) if len(row) > 4 else 0
                st.metric("🔄 Shares", f"{shares:,}")
            with col3:
                saves = int(row[5]) if len(row) > 5 else 0
                st.metric("💾 Saves", f"{saves:,}")
            with col4:
                playlist_adds = int(row[7]) if len(row) > 7 else 0
                st.metric("📝 Playlist Adds", f"{playlist_adds:,}")


def _display_public_engagement_analysis(analytics_data: dict, recent_performance):
    """Display simple engagement metrics and recent performance analysis."""
    
    st.markdown("""
    <div class="section-header">
        📊 Engagement
    </div>
    """, unsafe_allow_html=True)
    
    engagement_data = analytics_data.get("engagement_analysis", {})
    
    if not engagement_data:
        st.info("📊 No engagement data available")
        return
    
    # Overall Performance Section
    st.markdown("**🎯 Overall Performance**")
    
    total_videos = engagement_data.get("total_videos_analyzed", 0)
    total_views = engagement_data.get("total_views", 0)
    avg_engagement = engagement_data.get("avg_engagement_rate", 0)
    like_ratio = engagement_data.get('like_to_view_ratio', 0)
    
    # Simple 4-column layout for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Videos Analyzed", total_videos)
    with col2:
        st.metric("Total Views", f"{total_views:,}")
    with col3:
        st.metric("Avg Engagement Rate", f"{avg_engagement}%")
    with col4:
        st.metric("Like-to-View Ratio", f"{like_ratio}%")
    
    # Performance tier
    tier = engagement_data.get("performance_tier", "Unknown")
    st.info(f"Performance Tier: {tier}")
    
    # Add some spacing
    st.markdown("---")
    
    # Averages & Consistency Section
    st.markdown("**📈 Averages & Consistency**")
    
    avg_views = engagement_data.get("avg_views_per_video", 0)
    avg_likes = engagement_data.get("avg_likes_per_video", 0)
    avg_comments = engagement_data.get("avg_comments_per_video", 0)
    consistency = engagement_data.get("consistency_score", 0)
    
    # Simple 4-column layout for averages
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg Views per Video", f"{avg_views:,.0f}")
    with col2:
        st.metric("Avg Likes per Video", f"{avg_likes:.1f}")
    with col3:
        st.metric("Avg Comments per Video", f"{avg_comments:.1f}")
    with col4:
        st.metric("Consistency Score", f"{consistency:.2f}")
    
    # Consistency assessment
    if consistency > 0.7:
        cons_text = "Very Consistent"
    elif consistency > 0.5:
        cons_text = "Moderately Consistent"
    else:
        cons_text = "Variable Performance"
        
    st.info(f"Consistency: {cons_text}")
    
    # Add some spacing
    st.markdown("---")
    
    # Best performing video
    best_video = engagement_data.get("best_performing_video", {})
    if best_video:
        st.markdown("**🏆 Best Performing Video**")
        st.markdown(f"**{best_video.get('title', 'Unknown')}**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Views", f"{best_video.get('views', 0):,}")
        with col2:
            st.metric("Likes", f"{best_video.get('likes', 0):,}")
        with col3:
            video_id = best_video.get('video_id')
            if video_id:
                st.markdown(f"[Watch Video](https://youtube.com/watch?v={video_id})")


def _display_oauth_audience_insights(analytics_data: dict, period: str):
    """Display OAuth-only audience demographics and geography."""
    
    oauth_data = analytics_data.get("oauth", {})
    if not oauth_data or oauth_data.get("error"):
        return
    
    st.header(f"👥 Demographics & Geography (OAuth) - {period}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Demographics
        demo = oauth_data.get("demographics", {})
        if demo.get("rows"):
            st.subheader("📊 Demographics")
            age_gender_map = {}
            for age, gender, percentage in demo["rows"]:
                if age not in age_gender_map:
                    age_gender_map[age] = {}
                age_gender_map[age][gender] = round(float(percentage), 1)
            
            demo_table = []
            for age_group in sorted(age_gender_map.keys()):
                row_data = {"Age Group": age_group}
                row_data.update({f"{gender.title()} %": f"{pct}%" for gender, pct in age_gender_map[age_group].items()})
                demo_table.append(row_data)
            
            st.table(demo_table)
        else:
            st.info("Demographics data not available")
    
    with col2:
        # Geography
        geo = oauth_data.get("geography", {})
        if geo.get("rows"):
            st.subheader("🌍 Top Countries")
            geo_rows = sorted(geo["rows"], key=lambda x: int(x[1]), reverse=True)[:10]
            geo_table = []
            for row in geo_rows:
                country, views, watch_time = row[:3]  # Only take the first three columns
                geo_table.append({
                    "Country": country,
                    "Views": f"{int(views):,}",
                    "Watch Time": f"{int(watch_time):,} min"
                })
            st.table(geo_table)
        else:
            st.info("Geographic data not available")
    



def _display_oauth_traffic_sources(analytics_data: dict):
    """Display traffic sources from OAuth data."""
    oauth_data = analytics_data.get("oauth", {})
    traffic = oauth_data.get("traffic_sources", {})
    
    if traffic and traffic.get("rows") and len(traffic["rows"]) > 0:
        st.subheader("🚀 Traffic Sources (OAuth)")
        total_views = sum(int(row[1]) for row in traffic["rows"])
        
        # Friendly source name mapping
        source_names = {
            "YT_SEARCH": "YouTube Search",
            "SUGGESTED_VIDEO": "Suggested Videos",
            "EXTERNAL_URL": "External Links",
            "BROWSE_FEATURES": "Browse Features",
            "NOTIFICATION": "Notifications",
            "DIRECT_OR_UNKNOWN": "Direct/Unknown",
            "PLAYLIST": "Playlists",
            "CHANNEL": "Channel Pages",
            "SUBSCRIBER": "Subscribers"
        }
        
        traffic_table = []
        for row in traffic["rows"]:
            source = row[0]
            views = row[1]
            friendly_name = source_names.get(source, source)
            view_count = int(views)
            percentage = (view_count / total_views * 100) if total_views > 0 else 0
            
            traffic_table.append({
                "Traffic Source": friendly_name,
                "Views": f"{view_count:,}",
                "Percentage": f"{percentage:.1f}%"
            })
        
        if traffic_table:
            st.table(traffic_table)
        else:
            st.info("Traffic sources data format not recognized")
    else:
        st.info("Traffic sources data not available for this channel or period. This is common for newer channels or channels with limited analytics data.")


def _display_oauth_monetization(analytics_data: dict, period: str):
    """Display monetization metrics from OAuth data."""
    oauth_data = analytics_data.get("oauth", {})
    monetization = oauth_data.get("monetization", {})
    
    if monetization.get("rows"):
        st.subheader(f"💰 Monetization (OAuth) - {period}")
        row = monetization["rows"][0]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            revenue = float(row[0]) if len(row) > 0 else 0
            st.metric("💵 Estimated Revenue", f"${revenue:.2f}")
        with col2:
            ad_revenue = float(row[1]) if len(row) > 1 else 0
            st.metric("📺 Ad Revenue", f"${ad_revenue:.2f}")
        with col3:
            cpm = float(row[2]) if len(row) > 2 else 0
            st.metric("📊 CPM", f"${cpm:.2f}")
        with col4:
            rpm = float(row[3]) if len(row) > 3 else 0
            st.metric("💰 RPM", f"${rpm:.2f}")
    else:
        st.info("Monetization data not available")


def _display_oauth_growth_trends(analytics_data: dict, period: str):
    """Display growth trends from OAuth data."""
    oauth_data = analytics_data.get("oauth", {})
    growth = oauth_data.get("growth_metrics", {})
    
    if growth.get("rows"):
        st.subheader(f"📈 Growth Trends (OAuth) - {period}")
        
        # Extract data for charts
        
        dates = []
        views = []
        subscribers = []
        watch_time = []
        
        for row in growth["rows"]:
            if len(row) >= 4:
                dates.append(row[0])
                views.append(int(row[1]))
                subscribers.append(int(row[2]))
                watch_time.append(int(row[3]))
        
        if dates:
            df = pd.DataFrame({
                'Date': dates,
                'Views': views,
                'Subscribers': subscribers,
                'Watch Time': watch_time
            })
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📈 Views Over Time**")
                st.line_chart(df.set_index('Date')['Views'])
            with col2:
                st.markdown("**👥 Subscribers Over Time**")
                st.line_chart(df.set_index('Date')['Subscribers'])
            
            st.markdown("**⏱️ Watch Time Over Time**")
            st.line_chart(df.set_index('Date')['Watch Time'])
    else:
        st.info("Growth trends data not available")


def _display_oauth_performance_summary(analytics_data: dict, period: str):
    """Display performance summary from OAuth data."""
    oauth_data = analytics_data.get("oauth", {})
    performance = oauth_data.get("performance_summary", {})
    
    if performance and performance.get("rows") and len(performance["rows"]) > 0:
        st.subheader(f"📊 Performance Summary (OAuth) - {period}")
        row = performance["rows"][0]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_views = int(row[0]) if len(row) > 0 else 0
            st.metric("👀 Total Views", f"{total_views:,}")
        with col2:
            total_watch_time = int(row[1]) if len(row) > 1 else 0
            st.metric("⏱️ Watch Time (min)", f"{total_watch_time:,}")
        with col3:
            avg_view_duration = float(row[2]) if len(row) > 2 else 0
            st.metric("📊 Avg View Duration", f"{avg_view_duration:.1f}s")
        with col4:
            subscriber_gain = int(row[3]) if len(row) > 3 else 0
            st.metric("👥 Subscribers Gained", f"{subscriber_gain:,}")
    else:
        st.info("Performance summary data not available for this channel or period. This is common for newer channels or channels with limited analytics data.")


def _display_public_upload_patterns(analytics_data: dict):
    """Display upload frequency and timing analysis - from old streamlit."""
    
    st.header("📅 Upload Patterns & Consistency")
    
    upload_data = analytics_data.get("upload_patterns", {})
    
    if not upload_data:
        st.info("No upload pattern data available")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("⏰ Upload Frequency")
        
        total_videos = upload_data.get("total_videos", 0)
        avg_days = upload_data.get("avg_days_between_uploads", 0)
        consistency = upload_data.get("upload_consistency", "Unknown")
        
        st.metric("📹 Videos Analyzed", total_videos)
        st.metric("📅 Avg Days Between Uploads", f"{avg_days:.1f}")
        
        consistency_colors = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
        st.markdown(f"**Consistency**: {consistency_colors.get(consistency, '⚪')} {consistency}")
    
    with col2:
        st.subheader("📊 Optimal Upload Times")
        
        best_day = upload_data.get("most_common_upload_day")
        best_hour = upload_data.get("most_common_upload_hour")
        
        if best_day:
            st.metric("📅 Most Common Day", best_day)
        if best_hour is not None:
            st.metric("🕐 Most Common Hour", f"{best_hour}:00")
    
    with col3:
        st.subheader("🎬 Content Length")
        
        avg_duration = upload_data.get("avg_video_duration_seconds", 0)
        
        if avg_duration > 0:
            minutes = int(avg_duration // 60)
            seconds = int(avg_duration % 60)
            st.metric("⏱️ Avg Video Length", f"{minutes}m {seconds}s")


def _display_public_top_content(analytics_data: dict):
    """Display top performing videos and playlists - from old streamlit."""
    
    st.header("🏆 Top Performing Content")
    
    popular_videos = analytics_data.get("popular_videos", [])
    playlists = analytics_data.get("playlists", [])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📹 Most Popular Videos")
        
        if popular_videos:
            top_videos_data = []
            for i, video in enumerate(popular_videos[:10]):
                title = video["snippet"]["title"]
                views = int(video["statistics"].get("viewCount", 0))
                likes = int(video["statistics"].get("likeCount", 0))
                published = video["snippet"]["publishedAt"][:10]  # Date only
                
                # Shorten title for display
                short_title = title[:40] + "..." if len(title) > 40 else title
                
                top_videos_data.append({
                    "#": i + 1,
                    "Title": short_title,
                    "Views": f"{views:,}",
                    "Likes": f"{likes:,}",
                    "Published": published
                })
            
            st.table(top_videos_data)
        else:
            st.info("No video data available")
    
    with col2:
        st.subheader("📝 Channel Playlists")
        
        if playlists:
            playlist_data = []
            for playlist in playlists[:10]:
                title = playlist["snippet"]["title"]
                video_count = playlist["contentDetails"]["itemCount"]
                
                # Shorten title for display
                short_title = title[:40] + "..." if len(title) > 40 else title
                
                playlist_data.append({
                    "Playlist": short_title,
                    "Videos": video_count
                })
            
            st.table(playlist_data)
        else:
            st.info("No public playlists found")


def _display_oauth_revenue_metrics(analytics_data: dict, period: str):
    """Display monetization and revenue data - from old streamlit."""
    
    oauth_data = analytics_data.get("oauth", {})
    if not oauth_data or oauth_data.get("error"):
        return
    
    monetization = oauth_data.get("monetization", {})
    if monetization.get("error") or not monetization.get("rows"):
        st.header("💰 Monetization")
        st.info("💡 Monetization data not available - channel may not be monetized or data access restricted")
        return
    
    st.header(f"💰 Revenue Analytics (OAuth) - {period}")
    
    row = monetization["rows"][0]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        revenue = float(row[0]) if row[0] else 0
        st.metric("💵 Est. Revenue", f"${revenue:.2f}")
    
    with col2:
        ad_revenue = float(row[1]) if len(row) > 1 and row[1] else 0
        st.metric("📺 Ad Revenue", f"${ad_revenue:.2f}")
    
    with col3:
        cpm = float(row[4]) if len(row) > 4 and row[4] else 0
        st.metric("📊 CPM", f"${cpm:.2f}")
    
    with col4:
        playback_cpm = float(row[5]) if len(row) > 5 and row[5] else 0
        st.metric("▶️ Playback CPM", f"${playback_cpm:.2f}")


def _display_oauth_subscriber_status(analytics_data: dict, period: str):
    """Display subscriber status breakdown - from old streamlit."""
    
    # Channel-level subscriber status breakdown (OAuth only) - exactly like old streamlit
    try:
        # Get the channel ID and days_back from analytics data
        channel_info = analytics_data.get("channel_info", {})
        if not channel_info:
            st.info("ℹ️ Channel information not available for subscriber status breakdown.")
            return
            
        channel_id = channel_info.get("id")
        if not channel_id:
            st.info("ℹ️ Channel ID not available for subscriber status breakdown.")
            return
        
        # Calculate days_back based on period
        if period == "All Time":
            # Try to get actual channel age for All Time
            try:
                from datetime import datetime
                published_at = channel_info.get("snippet", {}).get("publishedAt")
                if published_at:
                    created_date = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                    days_back = (datetime.now(created_date.tzinfo) - created_date).days
                else:
                    days_back = 3650  # 10 years fallback
            except Exception:
                days_back = 3650  # 10 years fallback
        else:
            # Extract number from period string like "Last 30 days"
            import re
            match = re.search(r'(\d+)', period)
            days_back = int(match.group(1)) if match else 30
        
        # Get OAuth service to fetch subscriber status data
        from src.analysis.channel_analysis import ChannelAnalysisService
        import os
        from src.config.settings import SETTINGS
        
        api_key = os.getenv("YT_API_KEY") or SETTINGS.youtube_api_key
        if not api_key:
            st.info("ℹ️ API key not available for subscriber status breakdown.")
            return
            
        service = ChannelAnalysisService(api_key)
        try:
            oauth_service, access_type = service.get_service_for_channel(channel_id)
            if access_type != "oauth":
                st.info("ℹ️ OAuth access required for subscriber status breakdown.")
                return
        except Exception:
            st.info("ℹ️ OAuth service not available for subscriber status breakdown.")
            return
        
        # Fetch subscriber status breakdown data
        from src.youtube.analytics import channel_subscriber_status_breakdown
        sub_status_data = channel_subscriber_status_breakdown(
            oauth_service, channel_id, days_back=days_back
        )
        
        rows = sub_status_data.get("rows", [])
        if rows:
            st.markdown("### 👥 Views by Subscriber Status")
            table = []
            for row in rows:
                status = row[0].capitalize() if row[0] else "Unknown"
                views = int(row[1]) if len(row) > 1 else 0
                watch_time = int(row[2]) if len(row) > 2 else 0
                avg_view_duration = int(row[3]) if len(row) > 3 else 0
                mins, secs = divmod(avg_view_duration, 60)
                table.append({
                    "Status": status,
                    "Views": f"{views:,}",
                    "Watch Time (min)": f"{watch_time:,}",
                    "Avg View Duration": f"{mins}m {secs}s"
                })
            st.table(table)
        else:
            st.info("ℹ️ Subscriber status breakdown (views, watch time, avg view duration) is not available for this channel or period. This requires sufficient data and OAuth access.")
    except Exception as e:
        st.info(f"ℹ️ Could not fetch subscriber status breakdown: {e}")