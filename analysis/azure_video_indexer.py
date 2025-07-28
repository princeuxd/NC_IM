"""Azure Video Indexer API integration.

This module provides a simplified interface to Azure Video Indexer's REST API
for uploading videos, checking processing status, and retrieving insights.

Based on the Azure Video Indexer Python samples and API documentation from:
- https://github.com/bklim5/python_video_indexer_lib
- https://github.com/tomconte/videoindexer-samples-python
- https://learn.microsoft.com/en-us/azure/azure-video-indexer/

Main functions:
- upload_video() - Upload video for indexing
- get_video_status() - Check processing status
- get_video_insights() - Retrieve analysis results
- list_videos() - List all videos in account
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
from urllib.parse import quote

from config.settings import SETTINGS

logger = logging.getLogger(__name__)


class AzureVideoIndexerError(Exception):
    """Custom exception for Azure Video Indexer API errors."""
    pass


class AzureVideoIndexer:
    """Azure Video Indexer API client."""
    
    def __init__(
        self,
        subscription_key: str | None = None,
        location: str | None = None,
        account_id: str | None = None,
    ):
        """Initialize Azure Video Indexer client.
        
        Args:
            subscription_key: Azure Video Indexer subscription key
            location: Azure region (e.g., 'eastus', 'westus2')
            account_id: Video Indexer account ID
        """
        self.subscription_key = subscription_key or SETTINGS.azure_vi_subscription_key
        self.location = location or SETTINGS.azure_vi_location
        self.account_id = account_id or SETTINGS.azure_vi_account_id
        
        if not self.subscription_key:
            raise AzureVideoIndexerError("Azure Video Indexer subscription key is required")
        if not self.account_id:
            raise AzureVideoIndexerError("Azure Video Indexer account ID is required")
            
        self.base_url = f"https://api.videoindexer.ai/{self.location}/Accounts/{self.account_id}"
        self.access_token = None
        self.token_expires_at = 0
        
    def _get_access_token(self) -> str:
        """Get or refresh access token for API calls using ARM-based authentication."""
        current_time = time.time()
        
        # Check if we have a valid token
        if self.access_token and current_time < self.token_expires_at:
            return self.access_token
            
        # For ARM-based accounts, use the new authentication endpoint
        # The subscription key should be from the API Management Developer Portal
        url = f"https://api.videoindexer.ai/Auth/{self.location}/Accounts/{self.account_id}/AccessToken"
        
        headers = {
            "Ocp-Apim-Subscription-Key": self.subscription_key,
            "Content-Type": "application/json"
        }
        
        # For ARM accounts, we can optionally specify permissions scope
        params = {
            "allowEdit": "true"  # Request edit permissions
        }
        
        logger.info("Getting new ARM access token...")
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code != 200:
            error_text = response.text
            logger.error(f"Access token request failed: {response.status_code} - {error_text}")
            raise AzureVideoIndexerError(f"Failed to get access token: {response.status_code} - {error_text}")
            
        # The response should be a JSON string containing the token
        try:
            self.access_token = response.json().strip('"')  # Remove quotes if present
        except:
            # If response is plain text token
            self.access_token = response.text.strip().strip('"')
            
        # Tokens typically expire in 1 hour, refresh 5 minutes early
        self.token_expires_at = current_time + 3300  # 55 minutes
        
        return self.access_token
    
    def upload_video(
        self,
        video_path: Path | str,
        video_name: str | None = None,
        language: str = "English",
        privacy: str = "Private",
        **kwargs
    ) -> str:
        """Upload a video file for indexing.
        
        Args:
            video_path: Path to video file
            video_name: Name for the video (defaults to filename)
            language: Video language for processing
            privacy: Privacy setting ('Private' or 'Public')
            **kwargs: Additional parameters for the API
            
        Returns:
            Video ID for tracking the upload
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise AzureVideoIndexerError(f"Video file not found: {video_path}")
            
        if video_name is None:
            video_name = video_path.stem
            
        access_token = self._get_access_token()
        
        url = f"{self.base_url}/Videos"
        params = {
            "accessToken": access_token,
            "name": video_name,
            "privacy": privacy,
            "language": language,
            **kwargs
        }
        
        logger.info(f"Uploading video: {video_name}")
        
        with open(video_path, 'rb') as video_file:
            files = {'file': (video_path.name, video_file, 'video/mp4')}
            response = requests.post(url, params=params, files=files)
            
        if response.status_code not in [200, 201]:
            raise AzureVideoIndexerError(f"Upload failed: {response.status_code} - {response.text}")
            
        result = response.json()
        video_id = result.get('id')
        
        if not video_id:
            raise AzureVideoIndexerError(f"No video ID in response: {result}")
            
        logger.info(f"Video uploaded successfully. ID: {video_id}")
        return video_id
    
    def get_video_status(self, video_id: str) -> Dict[str, Any]:
        """Get the processing status of a video.
        
        Args:
            video_id: Video ID from upload
            
        Returns:
            Status information including processing state
        """
        access_token = self._get_access_token()
        
        url = f"{self.base_url}/Videos/{video_id}/Index"
        params = {"accessToken": access_token}
        
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            raise AzureVideoIndexerError(f"Failed to get status: {response.status_code} - {response.text}")
            
        return response.json()
    
    def wait_for_processing(self, video_id: str, timeout: int = 1800, poll_interval: int = 30) -> Dict[str, Any]:
        """Wait for video processing to complete.
        
        Args:
            video_id: Video ID to monitor
            timeout: Maximum time to wait in seconds (default: 30 minutes)
            poll_interval: Time between status checks in seconds
            
        Returns:
            Final status information
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_video_status(video_id)
            state = status.get('state', 'Unknown')
            
            logger.info(f"Video {video_id} status: {state}")
            
            if state in ['Processed', 'Failed']:
                return status
            elif state == 'Processing':
                time.sleep(poll_interval)
            else:
                # States like 'Uploaded', 'Queued' - continue waiting
                time.sleep(poll_interval)
                
        raise AzureVideoIndexerError(f"Processing timeout after {timeout} seconds")
    
    def get_video_insights(self, video_id: str, include_video_insights: bool = True) -> Dict[str, Any]:
        """Get detailed insights for a processed video.
        
        Args:
            video_id: Video ID
            include_video_insights: Whether to include video-specific insights
            
        Returns:
            Complete insights data
        """
        access_token = self._get_access_token()
        
        url = f"{self.base_url}/Videos/{video_id}/Index"
        params = {
            "accessToken": access_token,
            "includeVideoInsights": str(include_video_insights).lower()
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            raise AzureVideoIndexerError(f"Failed to get insights: {response.status_code} - {response.text}")
            
        return response.json()
    
    def list_videos(self, page_size: int = 25) -> List[Dict[str, Any]]:
        """List all videos in the account.
        
        Args:
            page_size: Number of videos per page
            
        Returns:
            List of video information
        """
        access_token = self._get_access_token()
        
        url = f"{self.base_url}/Videos"
        params = {
            "accessToken": access_token,
            "pageSize": page_size
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            raise AzureVideoIndexerError(f"Failed to list videos: {response.status_code} - {response.text}")
            
        result = response.json()
        return result.get('results', [])
    
    def delete_video(self, video_id: str) -> bool:
        """Delete a video from the account.
        
        Args:
            video_id: Video ID to delete
            
        Returns:
            True if deletion was successful
        """
        access_token = self._get_access_token()
        
        url = f"{self.base_url}/Videos/{video_id}"
        params = {"accessToken": access_token}
        
        response = requests.delete(url, params=params)
        
        if response.status_code != 200:
            logger.warning(f"Delete failed: {response.status_code} - {response.text}")
            return False
            
        logger.info(f"Video {video_id} deleted successfully")
        return True


def format_insights_summary(insights: Dict[str, Any]) -> str:
    """Format Azure Video Indexer insights into a readable summary.
    
    Args:
        insights: Raw insights from get_video_insights()
        
    Returns:
        Formatted markdown summary
    """
    if not insights:
        return "No insights available."
        
    summary_parts = []
    
    # Basic video info
    video_info = insights.get('videos', [{}])[0] if insights.get('videos') else {}
    if video_info:
        duration = video_info.get('insights', {}).get('duration', {})
        duration_str = duration.get('time', 'Unknown') if duration else 'Unknown'
        
        summary_parts.append(f"## 📹 Video Information")
        summary_parts.append(f"- **Name**: {video_info.get('name', 'Unknown')}")
        summary_parts.append(f"- **Duration**: {duration_str}")
        summary_parts.append(f"- **State**: {video_info.get('state', 'Unknown')}")
        summary_parts.append("")
    
    # Get insights from the video
    video_insights = video_info.get('insights', {}) if video_info else {}
    
    # Transcript
    transcript = video_insights.get('transcript', [])
    if transcript:
        summary_parts.append("## 🎤 Transcript")
        full_text = " ".join([t.get('text', '') for t in transcript[:10]])  # First 10 segments
        if len(transcript) > 10:
            full_text += f"... ({len(transcript) - 10} more segments)"
        summary_parts.append(full_text[:500] + ("..." if len(full_text) > 500 else ""))
        summary_parts.append("")
    
    # Topics
    topics = video_insights.get('topics', [])
    if topics:
        summary_parts.append("## 🏷️ Topics")
        for topic in topics[:5]:  # Top 5 topics
            name = topic.get('name', 'Unknown')
            confidence = topic.get('confidence', 0)
            summary_parts.append(f"- **{name}** (confidence: {confidence:.2f})")
        summary_parts.append("")
    
    # Keywords
    keywords = video_insights.get('keywords', [])
    if keywords:
        summary_parts.append("## 🔑 Keywords")
        keyword_names = [kw.get('name', '') for kw in keywords[:10]]
        summary_parts.append(", ".join(keyword_names))
        summary_parts.append("")
    
    # Sentiment
    sentiments = video_insights.get('sentiments', [])
    if sentiments:
        # Calculate average sentiment
        positive = sum(1 for s in sentiments if s.get('sentimentType') == 'Positive')
        negative = sum(1 for s in sentiments if s.get('sentimentType') == 'Negative')
        neutral = len(sentiments) - positive - negative
        
        summary_parts.append("## 😊 Sentiment Analysis")
        summary_parts.append(f"- **Positive**: {positive} segments")
        summary_parts.append(f"- **Neutral**: {neutral} segments") 
        summary_parts.append(f"- **Negative**: {negative} segments")
        summary_parts.append("")
    
    # Faces (if available)
    faces = video_insights.get('faces', [])
    if faces:
        summary_parts.append("## 👤 Detected Faces")
        for face in faces[:5]:  # Top 5 faces
            name = face.get('name', 'Unknown Person')
            confidence = face.get('confidence', 0)
            summary_parts.append(f"- **{name}** (confidence: {confidence:.2f})")
        summary_parts.append("")
    
    # Labels/objects
    labels = video_insights.get('labels', [])
    if labels:
        summary_parts.append("## 🏷️ Visual Labels")
        label_names = [label.get('name', '') for label in labels[:10]]
        summary_parts.append(", ".join(label_names))
        summary_parts.append("")
    
    return "\n".join(summary_parts) if summary_parts else "No detailed insights available."


# Convenience functions for common operations
def upload_and_analyze(
    video_path: Path | str,
    video_name: str | None = None,
    language: str = "English",
    wait_for_completion: bool = True,
    timeout: int = 1800
) -> Dict[str, Any]:
    """Upload a video and optionally wait for analysis to complete.
    
    Args:
        video_path: Path to video file
        video_name: Name for the video
        language: Video language
        wait_for_completion: Whether to wait for processing
        timeout: Maximum wait time in seconds
        
    Returns:
        Dictionary with video_id and insights (if completed)
    """
    vi = AzureVideoIndexer()
    
    # Upload video
    video_id = vi.upload_video(video_path, video_name, language)
    result = {"video_id": video_id, "status": "uploaded"}
    
    if wait_for_completion:
        # Wait for processing
        final_status = vi.wait_for_processing(video_id, timeout)
        result["status"] = final_status.get("state", "unknown")
        
        if final_status.get("state") == "Processed":
            # Get insights
            insights = vi.get_video_insights(video_id)
            result["insights"] = insights
            result["summary"] = format_insights_summary(insights)
    
    return result


__all__ = [
    "AzureVideoIndexer",
    "AzureVideoIndexerError", 
    "format_insights_summary",
    "upload_and_analyze",
] 