"""Video summary prompts for comprehensive analysis compilation."""

def get_comprehensive_video_summary_prompt(
    video_title: str,
    audio_analysis: str,
    vision_analysis: str,
    comments_analysis: str,
    stats: dict,
    oauth_analytics: dict = None
) -> str:
    """Generate comprehensive video summary prompt.
    
    Args:
        video_title: Title of the video
        audio_analysis: Audio analysis results
        vision_analysis: Video frame analysis results
        comments_analysis: Comments analysis results
        stats: Video statistics dictionary
        oauth_analytics: OAuth analytics data (if available)
        
    Returns:
        Formatted prompt for comprehensive summary generation
    """
    stats_dict = stats or {}
    statistics = stats_dict.get('statistics', {})
    snippet = stats_dict.get('snippet', {})
    
    return f"""Create a comprehensive video analysis summary for: {video_title}

Based on the following analysis data:

AUDIO ANALYSIS:
{audio_analysis or 'No audio analysis available'}

VIDEO ANALYSIS:
{vision_analysis or 'No video analysis available'}

COMMENTS ANALYSIS:
{comments_analysis or 'No comments analysis available'}

VIDEO STATISTICS:
- Views: {statistics.get('viewCount', 'N/A')}
- Likes: {statistics.get('likeCount', 'N/A')}
- Comments: {statistics.get('commentCount', 'N/A')}
- Duration: {stats_dict.get('contentDetails', {}).get('duration', 'N/A')}
- Published: {snippet.get('publishedAt', 'N/A')}

{_format_oauth_analytics(oauth_analytics) if oauth_analytics else ""}

IMPORTANT: Use the provided statistics throughout your analysis to support ALL claims and observations. Reference specific numbers, percentages, and metrics wherever possible. Every major statement should be backed by data from the video statistics provided above.

Please create a structured markdown summary with the following sections:

## Executive Summary
Provide a high-level overview citing specific performance metrics. Include:
- Key performance indicators (views, engagement rate, retention)
- Overall content assessment with statistical backing
- Primary audience insights from demographic data
- Revenue/monetization performance (if OAuth data available)

## Content Analysis
Reference retention data and engagement patterns to assess:
- **Content Type & Category**: Identify specific content type (tutorial, review, vlog, unboxing, etc.) and YouTube category. Support with retention patterns and engagement metrics that indicate content type effectiveness.
- **Topic Resonance**: Correlate engagement spikes/dips with specific topics or moments in the video timeline
- **Educational vs Entertainment Value**: Use watch time vs. views ratio, retention curve analysis, and comment themes to determine primary content value proposition
- **Content Structure Optimization**: Identify retention drop-off points and suggest improvements based on audience behavior data

## Creator Style & Authenticity
Use engagement metrics and comment sentiment to evaluate:
- **Creator Style & Tone**: Analyze communication style (professional, casual, energetic, calm), presentation tone, and personality traits evident in audio/visual content. Reference retention patterns that correlate with style effectiveness.
- **Content Authenticity Level**: Rate authenticity on a scale of 1-10 based on natural delivery, genuine reactions, and audience response. Support with comment sentiment analysis and engagement authenticity indicators.
- **Production Quality Impact**: Assess technical production quality and its correlation with retention patterns and audience engagement
- **Audience Connection Strength**: Measure parasocial relationship indicators using subscriber conversion rates, comment intimacy, and repeat engagement patterns

## Products & Brand Analysis
Analyze commercial content and sponsorship indicators:
- **Products/Brands Featured**: List all products, brands, or services shown/mentioned in the video with timestamps if available from visual/audio analysis
- **Sponsorship Detection**: Determine if content is sponsored based on disclosure language, presentation style, and promotional tone. Look for FTC compliance indicators.
- **Commercial Intent**: Assess whether content is primarily commercial vs. informational, using engagement patterns around product mentions and audience reactions
- **Brand Integration Quality**: Evaluate how naturally products/brands are integrated vs. intrusive advertising, correlating with retention data during promotional segments

## Audience Engagement & Sentiment Analysis
Provide data-driven analysis using specific metrics:
- **Engagement Rate Calculation**: (likes + comments + shares) / views × 100
- **Sentiment Analysis**: Analyze audience sentiment distribution from comments with specific percentages (e.g., 65% positive, 20% neutral, 15% negative). Reference comment analysis data.
- **Comment Quality & Authenticity**: Assess comment authenticity vs. bot/spam comments, comment depth and thoughtfulness, and genuine audience interaction indicators
- **Audience Response Patterns**: Identify common themes in comments, questions asked, and community interaction quality
- **Subscriber Conversion Rate**: subscribers gained / total views × 100
- **Share Rate and Viral Indicators**: Analyze sharing behavior and viral potential based on engagement velocity
- **Geographic and Demographic Engagement**: Patterns based on available audience data

## Performance Benchmarking
Compare against typical YouTube performance standards:
- View-to-impression ratio (CTR analysis if available)
- Average view duration vs. video length percentage
- Engagement rate vs. industry benchmarks (2-5% typical range)
- Retention curve analysis vs. platform averages
- Revenue per thousand views (RPM) if monetized

## Traffic & Discovery Analysis
Analyze traffic sources and discovery patterns:
- Primary discovery methods with percentage breakdown
- Search vs. suggested vs. external traffic performance
- Geographic performance distribution
- Optimal posting and discovery timing patterns

## Key Insights & Data-Driven Recommendations
Base ALL recommendations on statistical evidence:
- Performance strengths (cite specific metrics showing success)
- Improvement opportunities (reference underperforming metrics)
- Content strategy optimization (use retention and engagement data)
- Audience development tactics (leverage demographic and geographic insights)
- Monetization optimization (if revenue data available)

## Statistical Evidence Summary
Conclude with a bullet-point list of the key statistics that support your analysis:
- Most compelling performance indicators
- Critical engagement metrics
- Audience behavior insights
- Revenue/growth indicators

CRITICAL: Every section must include specific numbers, percentages, and statistical references. Avoid generic statements - ground every observation in the actual data provided.

VERIFICATION REQUIREMENTS:
- Quote exact statistics when making performance claims
- Calculate and show your work for engagement rates and percentages  
- Compare metrics to industry benchmarks where applicable (e.g., 2-5% engagement rate is typical)
- Identify statistical outliers and explain their significance
- Use conditional language when data is limited (e.g., "Based on available data..." or "The X metric of Y% suggests...")
- Always state the data source for OAuth vs. public statistics when making comparisons"""


def _format_oauth_analytics(oauth_analytics: dict) -> str:
    """Format OAuth analytics data for inclusion in the video summary prompt."""
    if not oauth_analytics or isinstance(oauth_analytics, dict) and oauth_analytics.get("error"):
        return ""
    
    oauth_section = "ENHANCED ANALYTICS (OAuth Enabled):\n"
    
    # Summary metrics (views, watch time, likes, comments from Analytics API)
    summary_metrics = oauth_analytics.get("summary_metrics", {})
    total_views = 0
    total_watch_time = 0
    avg_duration = 0
    
    if summary_metrics.get("rows"):
        row = summary_metrics["rows"][0]
        total_views = int(row[1]) if len(row) > 1 else 0
        total_watch_time = int(row[2]) if len(row) > 2 else 0
        avg_duration = int(row[3]) if len(row) > 3 else 0
        
        oauth_section += f"- Analytics Views: {total_views:,}\n"
        oauth_section += f"- Total Watch Time: {total_watch_time:,} minutes ({total_watch_time/60:.1f} hours)\n"
        oauth_section += f"- Average View Duration: {avg_duration:,} seconds ({avg_duration/60:.1f} minutes)\n"
        oauth_section += f"- Analytics Likes: {int(row[4]) if len(row) > 4 else 'N/A':,}\n"
        oauth_section += f"- Analytics Comments: {int(row[5]) if len(row) > 5 else 'N/A':,}\n"
        
        # Calculate view duration percentage (if we can estimate video length)
        if avg_duration > 0:
            oauth_section += f"- Average View Duration Ratio: {(avg_duration/300)*100:.1f}% (assuming 5min video)\n"
    
    # Engagement metrics with calculated rates
    engagement_metrics = oauth_analytics.get("engagement_metrics", {})
    if engagement_metrics.get("rows"):
        eng_row = engagement_metrics["rows"][0]
        eng_views = int(eng_row[0]) if len(eng_row) > 0 else total_views
        eng_likes = int(eng_row[1]) if len(eng_row) > 1 else 0
        eng_comments = int(eng_row[3]) if len(eng_row) > 3 else 0
        eng_shares = int(eng_row[4]) if len(eng_row) > 4 else 0
        subs_gained = int(eng_row[5]) if len(eng_row) > 5 else 0
        playlist_adds = int(eng_row[7]) if len(eng_row) > 7 else 0
        saves = int(eng_row[8]) if len(eng_row) > 8 else 0
        
        oauth_section += f"- Shares: {eng_shares:,}\n"
        oauth_section += f"- Subscribers Gained: {subs_gained:,}\n"
        oauth_section += f"- Playlist Adds: {playlist_adds:,}\n"
        oauth_section += f"- Saves: {saves:,}\n"
        
        # Calculate engagement rates
        if eng_views > 0:
            engagement_rate = ((eng_likes + eng_comments + eng_shares) / eng_views) * 100
            like_rate = (eng_likes / eng_views) * 100
            comment_rate = (eng_comments / eng_views) * 100
            share_rate = (eng_shares / eng_views) * 100
            sub_conversion_rate = (subs_gained / eng_views) * 100
            
            oauth_section += f"- Overall Engagement Rate: {engagement_rate:.2f}%\n"
            oauth_section += f"- Like Rate: {like_rate:.2f}%\n"
            oauth_section += f"- Comment Rate: {comment_rate:.2f}%\n"
            oauth_section += f"- Share Rate: {share_rate:.2f}%\n"
            oauth_section += f"- Subscriber Conversion Rate: {sub_conversion_rate:.3f}%\n"
    
    # Impressions and CTR with calculated ratios
    impressions_data = oauth_analytics.get("impressions", {})
    if impressions_data.get("rows") and not impressions_data.get("error"):
        imp_row = impressions_data["rows"][0]
        impressions = int(imp_row[0]) if len(imp_row) > 0 else 0
        ctr = float(imp_row[1]) if len(imp_row) > 1 else 0
        unique_viewers = int(imp_row[2]) if len(imp_row) > 2 else 0
        
        oauth_section += f"- Impressions: {impressions:,}\n"
        oauth_section += f"- Click-through Rate: {ctr:.2f}%\n"
        oauth_section += f"- Unique Viewers: {unique_viewers:,}\n"
        
        # Calculate view-to-impression ratio and unique viewer rate
        if impressions > 0 and total_views > 0:
            views_per_impression = (total_views / impressions) * 100
            oauth_section += f"- Views per Impression: {views_per_impression:.1f}%\n"
        
        if total_views > 0 and unique_viewers > 0:
            repeat_view_rate = ((total_views - unique_viewers) / total_views) * 100
            oauth_section += f"- Repeat View Rate: {repeat_view_rate:.1f}%\n"
    
    # Traffic sources (top 3) with percentages
    traffic_sources = oauth_analytics.get("traffic_sources", [])
    if traffic_sources and not isinstance(traffic_sources, dict):
        oauth_section += "- Top Traffic Sources:\n"
        source_names = {
            'PLAYLIST': 'Playlists',
            'SEARCH': 'YouTube Search', 
            'SUGGESTED_VIDEO': 'Suggested Videos',
            'BROWSE': 'Browse Features',
            'CHANNEL': 'Channel Page',
            'EXTERNAL': 'External Sources',
            'DIRECT': 'Direct Links',
            'NOTIFICATION': 'Notifications'
        }
        
        # Calculate total traffic views for percentages
        total_traffic_views = sum(int(row[1]) for row in traffic_sources[:5] if len(row) >= 2)
        
        for i, row in enumerate(traffic_sources[:3]):
            if len(row) >= 2:
                source = source_names.get(row[0], row[0])
                views = int(row[1])
                percentage = (views / total_traffic_views * 100) if total_traffic_views > 0 else 0
                oauth_section += f"  {i+1}. {source}: {views:,} views ({percentage:.1f}%)\n"
    
    # Geographic data (top 3 countries) with percentages
    geography_data = oauth_analytics.get("geography", [])
    if geography_data and not isinstance(geography_data, dict):
        oauth_section += "- Top Geographic Regions:\n"
        country_names = {
            'US': 'United States', 'GB': 'United Kingdom', 'CA': 'Canada',
            'AU': 'Australia', 'DE': 'Germany', 'FR': 'France', 'IN': 'India',
            'JP': 'Japan', 'BR': 'Brazil', 'MX': 'Mexico', 'IT': 'Italy',
            'ES': 'Spain', 'RU': 'Russia', 'KR': 'South Korea', 'NL': 'Netherlands'
        }
        
        # Calculate total geographic views for percentages
        total_geo_views = sum(int(row[1]) for row in geography_data[:5] if len(row) >= 2)
        
        for i, row in enumerate(geography_data[:3]):
            if len(row) >= 2:
                country = country_names.get(row[0], row[0])
                views = int(row[1])
                percentage = (views / total_geo_views * 100) if total_geo_views > 0 else 0
                oauth_section += f"  {i+1}. {country}: {views:,} views ({percentage:.1f}%)\n"
    
    # Audience retention insights
    retention_data = oauth_analytics.get("audience_retention", [])
    if retention_data and not isinstance(retention_data, dict):
        retention_rates = [float(row[1]) * 100 for row in retention_data if len(row) >= 2]
        if retention_rates:
            avg_retention = sum(retention_rates) / len(retention_rates)
            max_retention = max(retention_rates)
            min_retention = min(retention_rates)
            oauth_section += f"- Average Audience Retention: {avg_retention:.1f}%\n"
            oauth_section += f"- Peak Retention: {max_retention:.1f}%\n"
            oauth_section += f"- Lowest Retention: {min_retention:.1f}%\n"
    
    # Demographics (top age/gender groups)
    demographics_data = oauth_analytics.get("demographics", [])
    if demographics_data and not isinstance(demographics_data, dict):
        oauth_section += "- Top Demographics:\n"
        for i, row in enumerate(demographics_data[:3]):
            if len(row) >= 3:
                age_group = row[0]
                gender = row[1]
                percentage = float(row[2])
                oauth_section += f"  {i+1}. {age_group} {gender}: {percentage:.1f}%\n"
    
    # Monetization (if available) with calculated rates
    monetization_data = oauth_analytics.get("monetization", {})
    if monetization_data.get("rows") and not monetization_data.get("error"):
        mon_row = monetization_data["rows"][0]
        estimated_revenue = float(mon_row[0]) if len(mon_row) > 0 else 0
        ad_revenue = float(mon_row[1]) if len(mon_row) > 1 else 0
        if estimated_revenue > 0:
            oauth_section += f"- Estimated Revenue: ${estimated_revenue:.2f}\n"
            oauth_section += f"- Ad Revenue: ${ad_revenue:.2f}\n"
            
            cpm = float(mon_row[4]) if len(mon_row) > 4 else 0
            oauth_section += f"- CPM: ${cpm:.2f}\n"
            
            # Calculate RPM (Revenue per 1000 views)
            if total_views > 0:
                rpm = (estimated_revenue / total_views) * 1000
                oauth_section += f"- RPM (Revenue per 1000 views): ${rpm:.2f}\n"
                
            # Calculate revenue per watch hour
            if total_watch_time > 0:
                revenue_per_hour = (estimated_revenue / (total_watch_time / 60))
                oauth_section += f"- Revenue per Watch Hour: ${revenue_per_hour:.2f}\n"
    
    return oauth_section


def get_channel_collective_analysis_prompt(
    channel_title: str,
    video_analyses: list,
    total_duration: int,
    avg_authenticity: float,
    products_count: int,
    content_type_distribution: dict
) -> str:
    """Generate collective channel analysis prompt.
    
    Args:
        channel_title: Name of the channel
        video_analyses: List of individual video analysis results
        total_duration: Total duration across all videos
        avg_authenticity: Average authenticity score
        products_count: Total number of products mentioned
        content_type_distribution: Distribution of content types
        
    Returns:
        Formatted prompt for collective channel analysis
    """
    return f"""You are an expert YouTube channel analyst. Based on the comprehensive analysis of {len(video_analyses)} videos from "{channel_title}", create a strategic channel assessment.

CHANNEL OVERVIEW:
- Total Videos Analyzed: {len(video_analyses)}
- Total Content Duration: {total_duration} minutes
- Average Authenticity Score: {avg_authenticity:.1f}/10
- Products/Brands Mentioned: {products_count}
- Primary Content Types: {dict(list(content_type_distribution.items())[:3]) if content_type_distribution else 'Mixed'}

INDIVIDUAL VIDEO INSIGHTS:
{chr(10).join([f"Video {i+1}: {analysis.get('title', 'Unknown')}" for i, analysis in enumerate(video_analyses[:5])])}
{'... and more' if len(video_analyses) > 5 else ''}

Create a comprehensive markdown report with these sections:

## 🎯 Executive Summary
High-level assessment of channel performance, content strategy, and growth potential.

## 📊 Content Strategy Analysis
- Content type distribution and effectiveness
- Topic consistency and variety
- Production quality trends
- Content authenticity assessment

## 🎤 Creator Brand Analysis
- Communication style consistency
- Personality and brand voice
- Authenticity and trustworthiness
- Audience connection strength

## 💬 Community Engagement Insights
- Overall audience sentiment
- Comment quality and engagement depth
- Community building effectiveness
- Feedback patterns and concerns

## 🛍️ Commercial Viability
- Brand partnership potential
- Product integration effectiveness
- Monetization opportunities
- Sponsor alignment assessment

## 📈 Growth Recommendations
- Content optimization strategies
- Audience development tactics
- Production improvements
- Strategic focus areas

## ⚠️ Risk Assessment
- Potential brand safety concerns
- Content consistency issues
- Audience retention risks
- Competitive positioning

Provide specific, actionable insights based on the analyzed data."""