"""Vision analysis prompts for video frame analysis."""

def get_frame_analysis_prompt() -> str:
    """Generate prompt for video frame analysis using vision models.
    
    Returns:
        Standard prompt for analyzing video frames
    """
    return (
        "You are a video-analysis assistant. Based ONLY on the supplied frames "
        "(and provided timestamps), output a concise markdown report with the "
        "following sections:\n\n"
        "1. **Summary** – 3-5 bullet points describing what happens in the video.\n"
        "2. **Overall Sentiment** – one of Positive / Neutral / Negative with a short justification.\n"
        "3. **Category** – choose the single best fit from this list: "
        "{Lifestyle, Education, Technology, Gaming, Health, Finance, Entertainment, Travel, Food, Sports, News}.\n"
        "4. **Products / Brands Shown** – bullet list of distinct products, logos or brand names visible in the frames. If none, write 'None visible'."
    )


def get_detailed_frame_analysis_prompt() -> str:
    """Generate detailed prompt for comprehensive video frame analysis.
    
    Returns:
        Detailed prompt for in-depth frame analysis
    """
    return (
        "You are an expert video content analyst. Analyze the provided video frames and timestamps to create a comprehensive report:\n\n"
        
        "## Visual Content Summary\n"
        "Describe what happens in the video based on the frames, including:\n"
        "- Main activities or events shown\n"
        "- Setting and environment\n"
        "- People present and their actions\n"
        "- Key visual elements\n\n"
        
        "## Creator Emotions and Behavior\n"
        "Analyze the creator's emotional state and behavior:\n"
        "- Facial expressions and body language\n"
        "- Energy level and engagement\n"
        "- Interaction with camera/audience\n"
        "- Overall demeanor and personality traits\n\n"
        
        "## Products and Brands Analysis\n"
        "Identify all visible products, brands, and commercial elements:\n"
        "- Product names and brands clearly visible\n"
        "- Logos, packaging, or branded items\n"
        "- Context of product appearance (review, demo, casual use)\n"
        "- Sponsored content indicators\n\n"
        
        "## Production Quality Assessment\n"
        "Evaluate the technical and production aspects:\n"
        "- Video quality and lighting\n"
        "- Camera work and framing\n"
        "- Set design or background\n"
        "- Overall production value\n\n"
        
        "## Content Category and Style\n"
        "Classify the content type and style:\n"
        "- Primary content category\n"
        "- Content style (professional, casual, educational, entertainment)\n"
        "- Target audience indicators\n"
        "- Content format (tutorial, vlog, review, etc.)\n\n"
        
        "Base your analysis strictly on what you can observe in the provided frames."
    )