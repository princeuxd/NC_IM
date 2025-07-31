"""Video summary prompts for comprehensive analysis compilation."""

def get_comprehensive_video_summary_prompt(
    video_title: str,
    audio_analysis: str,
    vision_analysis: str,
    comments_analysis: str,
    stats: dict
) -> str:
    """Generate comprehensive video summary prompt.
    
    Args:
        video_title: Title of the video
        audio_analysis: Audio analysis results
        vision_analysis: Video frame analysis results
        comments_analysis: Comments analysis results
        stats: Video statistics dictionary
        
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

Please create a structured markdown summary with the following sections:

## Executive Summary
Provide a high-level overview of the video's performance, content quality, and key insights.

## Content Analysis
- Content type and category
- Main topics and themes
- Educational vs entertainment value
- Content structure and organization

## Creator Style & Authenticity
- Communication style and personality
- Authenticity assessment
- Production quality
- Engagement techniques

## Audience Engagement
- Comment sentiment and feedback
- Engagement metrics analysis
- Community response patterns
- Areas of positive/negative feedback

## Technical Quality
- Audio quality and clarity
- Visual presentation
- Production values
- Technical recommendations

## Performance Metrics
- View performance relative to channel average
- Engagement rate analysis
- Growth potential assessment

## Key Insights & Recommendations
- What worked well in this video
- Areas for improvement
- Content strategy recommendations
- Audience development suggestions

Make the summary comprehensive, actionable, and data-driven."""


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