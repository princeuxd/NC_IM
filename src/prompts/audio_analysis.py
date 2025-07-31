"""Audio analysis prompts for LLM-based transcript analysis."""

def get_enhanced_audio_analysis_prompt(full_transcript: str) -> str:
    """Generate enhanced audio analysis prompt for transcript analysis.
    
    Args:
        full_transcript: The complete video transcript to analyze
        
    Returns:
        Formatted prompt string for LLM analysis
    """
    return f"""You are an expert content analyst. Analyze this video transcript thoroughly and provide detailed insights in the following structured format:

## CONTENT TYPE CLASSIFICATION
Classify the primary content type and any secondary types. Choose from: Tutorial, Review, Vlog, Gaming, Educational, Entertainment, News, Interview, Reaction, Unboxing, Comparison, How-to, Commentary, Music, Comedy, Travel, Food, Fashion, Technology, Finance, Health, Lifestyle.

## VOICE AND COMMUNICATION STYLE
Analyze the creator's communication style including:
- Speaking pace (fast/medium/slow)
- Tone (casual/professional/enthusiastic/serious)
- Language complexity (simple/moderate/advanced)
- Use of slang, technical terms, or jargon
- Personality traits evident in speech
- Engagement techniques used

## CONTENT QUALITY AND PRODUCTION STYLE
Assess:
- Script quality (improvised/semi-scripted/fully scripted)
- Information density (high/medium/low)
- Structure and organization
- Use of examples, stories, or analogies
- Educational value
- Entertainment value

## SENTIMENT ANALYSIS
Provide:
- Overall sentiment (Positive/Neutral/Negative/Mixed)
- Emotional tone throughout the video
- Energy level (high/medium/low)
- Mood indicators

## CONTENT AUTHENTICITY ASSESSMENT
Rate authenticity level (1-10) and provide reasoning:
- Genuine personal opinions vs. scripted content
- Spontaneous reactions vs. planned responses
- Personal experience sharing
- Transparency about sponsorships/partnerships
- Natural speech patterns vs. overly polished delivery

## PRODUCTS, BRANDS, AND SPONSORSHIPS
List ALL products, brands, services, or companies mentioned with:
- Product/brand name
- Context of mention (review, sponsorship, casual reference, comparison)
- Approximate timestamp or segment description
- Sponsored content indicators (explicit or implicit)
- Creator's apparent relationship with the brand

## KEY MOMENTS AND TIMESTAMPS
Identify important segments with approximate time references:
- Main topic introductions
- Product demonstrations or mentions
- Sponsored content segments
- Key information or takeaways

Transcript to analyze:
{full_transcript}"""


def get_quick_audio_summary_prompt(full_transcript: str) -> str:
    """Generate a quick audio summary prompt for basic transcript analysis.
    
    Args:
        full_transcript: The complete video transcript to analyze
        
    Returns:
        Shorter prompt for basic analysis
    """
    return f"""Analyze this video transcript and provide a concise summary covering:

1. **Content Type**: What type of content is this?
2. **Main Topics**: What are the key topics discussed?
3. **Creator Style**: How does the creator communicate?
4. **Key Products/Brands**: What products or brands are mentioned?
5. **Overall Sentiment**: What's the general tone and mood?

Keep the analysis concise but informative.

Transcript:
{full_transcript}"""