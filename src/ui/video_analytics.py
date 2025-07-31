"""Streamlit UI for single-video analytics."""
from __future__ import annotations

import streamlit as st
import json
from pathlib import Path

from src.helpers import video_analytics as va
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


def _get_videos(channel_id: str):
    try:
        return va.fetch_recent_videos(channel_id, max_videos=20)
    except Exception as e:
        st.error(f"Failed to fetch videos: {e}")
        return []


def render_video_analytics():
    st.header("🎬 Video Analytics")
    st.markdown(
        "Comprehensive analysis of individual videos – audio, visual frames, comments & statistics (same as channel analytics)."
    )

    channel_id = st.text_input("Channel ID", placeholder="UC...", key="va_channel")

    if not channel_id:
        st.info("Enter a channel ID to list recent videos.")
        return

    if st.button("📥 Fetch Videos", key="va_fetch") or "va_videos" not in st.session_state:
        with st.spinner("Fetching videos..."):
            st.session_state["va_videos"] = _get_videos(channel_id)

    videos = st.session_state.get("va_videos", [])
    if not videos:
        st.warning("No videos found or failed to fetch.")
        return

    video_titles = {f"{v['title']} ({v['video_id']})": v["video_id"] for v in videos}
    selected_label = st.selectbox("Select Video", list(video_titles.keys()), key="va_select")
    video_id = video_titles[selected_label]

    if st.button("🚀 Run Complete Analysis", key="va_run"):
        with st.spinner("Running comprehensive video analysis – this may take several minutes..."):
            try:
                result = va.analyze_video(video_id)
                st.session_state["va_result"] = result
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                return

    result = st.session_state.get("va_result")
    if not result:
        return

    if not result.get("success"):
        st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
        return

    st.success("✅ Video analysis complete!")

    video_title = result.get("title", "Unknown Title")
    output_dir = result.get("output_dir")

    st.markdown(f"## 📹 Analysis Results for: {video_title}")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Get stats for metrics
    stats = result.get("statistics", {})
    statistics = stats.get("statistics", {}) if stats else {}
    
    col1.metric("Views", statistics.get("viewCount", "N/A"))
    col2.metric("Likes", statistics.get("likeCount", "N/A"))
    col3.metric("Comments", statistics.get("commentCount", "N/A"))
    # Parse duration from ISO format to human readable
    duration_iso = stats.get("contentDetails", {}).get("duration", "PT0S") if stats else "PT0S"
    duration_minutes = parse_iso_duration_to_minutes(duration_iso)
    duration_formatted = f"{duration_minutes} min" if duration_minutes > 0 else "N/A"
    col4.metric("Duration", duration_formatted)

    # Complete Video Analysis Summary
    summary_file = output_dir / f"{video_id}_summary.md"
    summary_content = _safe_read_file(summary_file)
    if summary_content:
        st.markdown("---")
        st.markdown("## 📄 Complete Video Analysis Summary")
        st.markdown(summary_content)
        st.download_button(
            "⬇️ Download Complete Summary", 
            summary_content.encode('utf-8'), 
            file_name=f"{video_id}_summary.md",
            mime="text/markdown",
            key="summary_download"
        )

    st.markdown("---")

    # a. Audio Analysis (same as channel analytics)
    st.markdown("## 🎤 Audio Analysis")
    
    audio_analysis = result.get("audio_analysis")
    if audio_analysis and audio_analysis != "Enhanced audio analysis failed":
        st.markdown("### Enhanced Audio Analysis (LLM)")
        st.markdown(audio_analysis)
    
    # Audio transcript data
    audio_file = output_dir / f"{video_id}_audio.json"
    audio_data = _safe_read_json(audio_file)
    if audio_data:
        st.markdown("### Transcript Segments with Sentiment")
        
        segments = audio_data if isinstance(audio_data, list) else []
        if segments:
            # Show sample segments
            sample_segments = segments[:5]  # Show first 5 segments
            for i, seg in enumerate(sample_segments):
                sentiment = seg.get('sentiment', 'N/A')
                text = seg.get('text', '').strip()
                start_time = seg.get('start', 0)
                
                if text:
                    st.markdown(f"**[{start_time:.1f}s]** (Sentiment: {sentiment}): {text}")
            
            if len(segments) > 5:
                st.markdown(f"... and {len(segments) - 5} more segments")
        
        st.download_button(
            "⬇️ Download Audio Analysis JSON", 
            json.dumps(audio_data, indent=2).encode('utf-8'), 
            file_name=f"{video_id}_audio.json",
            mime="application/json",
            key="audio_download"
        )

    # b. Video Analysis (same as channel analytics)
    st.markdown("---")
    st.markdown("## 🎬 Video Frame Analysis")
    
    video_analysis = result.get("video_analysis")
    if video_analysis and video_analysis != "Vision analysis failed":
        st.markdown("### LLM Vision Analysis (Emotions, Products, Creator Nature)")
        st.markdown(video_analysis)
    
    # Frames data
    frames_file = output_dir / f"{video_id}_frames.json"
    frames_data = _safe_read_json(frames_file)
    if frames_data:
        frame_count = len(frames_data) if isinstance(frames_data, list) else 0
        st.markdown(f"**Extracted {frame_count} frames for analysis**")
        
        st.download_button(
            "⬇️ Download Frames JSON", 
            json.dumps(frames_data, indent=2).encode('utf-8'), 
            file_name=f"{video_id}_frames.json",
            mime="application/json",
            key="frames_download"
        )

    # c. Comments Analysis (same as channel analytics)
    st.markdown("---")
    st.markdown("## 💬 Comments Analysis")
    
    comments_analysis = result.get("comments_analysis")
    if comments_analysis:
        st.markdown("### Comments Summary")
        st.markdown(comments_analysis)
    
    # Detailed comments data
    comments_file = output_dir / f"{video_id}_comments.json"
    comments_data = _safe_read_json(comments_file)
    if comments_data:
        comments_list = comments_data if isinstance(comments_data, list) else []
        if comments_list:
            # Calculate and display sentiment stats
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
            st.markdown("### Sample Comments with Sentiment")
            for i, comment in enumerate(comments_list[:5]):
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
            key="comments_download"
        )

    # d. Statistics (Public + OAuth)
    st.markdown("---")
    st.markdown("## 📊 Video Statistics")
    
    # Public Statistics
    st.markdown("### Public Statistics")
    if stats:
        statistics = stats.get('statistics', {})
        snippet = stats.get('snippet', {})
        content_details = stats.get('contentDetails', {})
        
        # Display key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Views", statistics.get('viewCount', 0))
        col2.metric("Likes", statistics.get('likeCount', 0))
        col3.metric("Comments", statistics.get('commentCount', 0))
        
        # Parse and format duration
        duration_iso = content_details.get('duration', 'PT0S')
        duration_formatted = _format_duration(duration_iso)
        col4.metric("Duration", duration_formatted)
        
        # Additional info
        if snippet:
            st.markdown(f"**Published:** {snippet.get('publishedAt', 'N/A')}")
            st.markdown(f"**Channel:** {snippet.get('channelTitle', 'N/A')}")
            description = snippet.get('description', '')[:300]
            if description:
                st.markdown(f"**Description:** {description}...")
        
        st.download_button(
            "⬇️ Download Public Statistics JSON", 
            json.dumps(stats, indent=2).encode('utf-8'), 
            file_name=f"{video_id}_stats.json",
            mime="application/json",
            key="stats_download"
        )
    
    # OAuth Analytics (if available)
    oauth_analytics = result.get("oauth_analytics")
    if oauth_analytics and (not isinstance(oauth_analytics, dict) or not oauth_analytics.get("error")):
        st.markdown("### 🔐 OAuth Available")
        
        # Display comprehensive OAuth analytics using the same functions as simple streamlit app
        _display_enhanced_analytics(oauth_analytics, video_title, result.get("title", "Video"))
        
        oauth_file = output_dir / f"{video_id}_oauth_analytics.json"
        if oauth_file.exists():
            st.download_button(
                "⬇️ Download OAuth Analytics JSON", 
                json.dumps(oauth_analytics, indent=2).encode('utf-8'), 
                file_name=f"{video_id}_oauth_analytics.json",
                mime="application/json",
                key="oauth_download"
            )
    else:
        st.markdown("### 🔐 OAuth Analytics")
        
        # Check for specific OAuth issues
        if oauth_analytics and isinstance(oauth_analytics, dict) and oauth_analytics.get("error"):
            st.error(f"OAuth analytics failed: {oauth_analytics['error']}")
        else:
            st.info("OAuth analytics not available. Only public statistics shown.")
            
            # Provide helpful information about OAuth setup
            with st.expander("ℹ️ How to enable OAuth Analytics"):
                st.markdown("""
                To enable OAuth analytics, you need:
                
                1. **OAuth Client Configuration** (one of these):
                   - `client_secret.json` file in project root, OR
                   - Environment variables: `OAUTH_CLIENT_ID` and `OAUTH_CLIENT_SECRET`
                
                2. **OAuth Tokens**: Valid tokens in the `data/tokens/` directory
                
                3. **YouTube Analytics API Access**: Your OAuth app needs YouTube Analytics API permissions
                
                **Current Status:**
                - OAuth tokens found: ✅ (1 token available)
                - Client configuration: ❌ (missing client_secret.json and environment variables)
                """)

    # e. Combined Analysis JSON for LLM
    st.markdown("---")
    st.markdown("## 🔗 Combined Analysis Data")
    
    analysis_file = output_dir / f"{video_id}_analysis.json"
    analysis_data = _safe_read_json(analysis_file)
    if analysis_data:
        st.markdown("This file contains all analysis results combined for further LLM processing.")
        st.json({k: v for k, v in analysis_data.items() if k != "transcript"}, expanded=False)
        
        st.download_button(
            "⬇️ Download Combined Analysis JSON", 
            json.dumps(analysis_data, indent=2).encode('utf-8'), 
            file_name=f"{video_id}_analysis.json",
            mime="application/json",
            key="analysis_download"
        )

    # Output directory info
    st.markdown("---")
    st.markdown("## 📁 Analysis Files")
    if output_dir and output_dir.exists():
        st.markdown(f"**Analysis output directory:** `{output_dir}`")
        st.info("All analysis files have been saved to the output directory and can be downloaded using the buttons above.")


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
        import pandas as pd
        
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
        
        import pandas as pd
        
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
        
        import pandas as pd
        
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
            
            import pandas as pd
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
        
        import pandas as pd
        
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
        
        import pandas as pd
        
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


def _format_duration(iso_duration: str) -> str:
    """Convert ISO-8601 duration (PT#H#M#S) to human-readable format."""
    if not iso_duration:
        return "N/A"
    
    import re
    
    # Parse ISO duration using regex
    pattern = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")
    match = pattern.match(iso_duration)
    
    if not match:
        return iso_duration  # Return original if parsing fails
    
    hours, minutes, seconds = (int(x) if x else 0 for x in match.groups())
    
    # Format based on duration length
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"