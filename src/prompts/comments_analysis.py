"""Comments analysis prompts for LLM-based sentiment and authenticity analysis."""

def get_comments_sentiment_analysis_prompt(comments_text: str) -> str:
    """Generate prompt for analyzing comment sentiment and authenticity.
    
    Args:
        comments_text: Combined text of all comments to analyze
        
    Returns:
        Formatted prompt string for LLM analysis
    """
    return f"""You are an expert in social media sentiment analysis and comment authenticity assessment. Analyze the following YouTube comments and provide detailed insights:

## OVERALL SENTIMENT ANALYSIS
Provide:
- Overall sentiment distribution (% Positive, Neutral, Negative)
- Dominant emotional themes
- Sentiment intensity (mild, moderate, strong)
- Emotional tone patterns

## COMMENT AUTHENTICITY ASSESSMENT
Evaluate comment authenticity:
- Genuine vs. potentially fake/bot comments (%)
- Spam or promotional content indicators
- Authentic engagement patterns
- Community interaction quality

## POSITIVE FEEDBACK ANALYSIS
Identify positive aspects mentioned:
- What viewers appreciate most
- Specific praise for content, creator, or production
- Engagement with creator's message
- Community building indicators

## CONCERNS AND CRITICISMS
Identify concerns or negative feedback:
- Common complaints or criticisms
- Technical issues mentioned
- Content-related concerns
- Constructive vs. destructive feedback

## AUDIENCE ENGAGEMENT PATTERNS
Analyze engagement quality:
- Depth of comments (superficial vs. thoughtful)
- Questions and discussions generated
- Community interaction between viewers
- Creator-audience interaction quality

## KEY THEMES AND TOPICS
Extract main discussion themes:
- Most discussed topics from the video
- Questions viewers are asking
- Suggestions or requests from audience
- Off-topic vs. on-topic discussions

## COMMERCIAL RECEPTION
If applicable, analyze reception of commercial content:
- Response to sponsored content or products
- Purchase intent indicators
- Brand/product sentiment
- Trust in creator's recommendations

Comments to analyze:
{comments_text}"""


def get_comments_summary_prompt(comments_data: list) -> str:
    """Generate prompt for creating a summary of comments analysis.
    
    Args:
        comments_data: List of comment dictionaries with sentiment scores
        
    Returns:
        Formatted prompt for comments summary
    """
    # Extract sample comments for analysis
    sample_comments = []
    for i, comment in enumerate(comments_data[:10]):  # First 10 comments
        author = comment.get('author', 'Unknown')
        text = comment.get('text', comment.get('textDisplay', ''))
        sentiment = comment.get('sentiment', 'N/A')
        likes = comment.get('likeCount', 0)
        
        sample_comments.append(f"Comment {i+1}: [{author}] (Sentiment: {sentiment}, Likes: {likes})\n{text[:200]}{'...' if len(text) > 200 else ''}")
    
    comments_text = "\n\n".join(sample_comments)
    
    return f"""Analyze these YouTube comments and provide a concise summary:

## Comment Statistics:
- Total Comments: {len(comments_data)}
- Sample Comments Shown: {min(10, len(comments_data))}

## Sample Comments:
{comments_text}

Please provide:

1. **Overall Sentiment**: General mood and reception
2. **Key Themes**: Main topics discussed by viewers
3. **Audience Engagement**: Quality of interaction and discussion
4. **Authenticity Assessment**: How genuine the comments appear
5. **Notable Feedback**: Important praise or concerns mentioned

Keep the analysis concise but informative, focusing on actionable insights."""


def get_comment_authenticity_prompt(comments_data: list) -> str:
    """Generate prompt specifically for comment authenticity analysis.
    
    Args:
        comments_data: List of comment dictionaries
        
    Returns:
        Formatted prompt for authenticity assessment
    """
    return f"""Analyze the authenticity of these {len(comments_data)} YouTube comments. Look for patterns that indicate genuine vs. potentially fake engagement:

## AUTHENTICITY INDICATORS TO ASSESS:

**Genuine Comment Patterns:**
- Natural language variation and personal expressions
- Specific references to video content
- Varied comment lengths and complexity
- Personal experiences or opinions shared
- Questions or discussions with other viewers

**Suspicious Patterns:**
- Generic or template-like comments
- Excessive promotional content
- Repetitive phrasing across multiple comments
- Unrelated or off-topic content
- Unusual engagement patterns

**Community Engagement Quality:**
- Meaningful discussions between viewers
- Responses to creator's content points
- Constructive feedback and suggestions
- Evidence of genuine viewership

Provide an authenticity score (1-10, where 10 is highly authentic) and explain your reasoning based on the patterns you observe in the comment set."""