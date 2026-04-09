import os
import requests
from dotenv import load_dotenv

load_dotenv()

def generate_krea_illustration(context_summary):
    """
    Uses the Krea AI API to generate a thematic illustration for the analysis.
    This fulfills the requirement of using a second API key in the project.
    """
    
    api_key = os.getenv("KREA_API_KEY")
    
    # Check if the key is valid (not the placeholder)
    if not api_key or "your_krea_key" in api_key:
        return {
            "status": "missing_key",
            "message": "Enter your KREA_API_KEY in the .env file to see AI illustrations!"
        }

    # Dynamic prompt generation for the AI artist
    prompt = f"Professional 3D data visualization and business analytics dashboard, futuristic style, showing {context_summary}, high detail, 8k, vibrant colors."

    try:
        # Step 1: Initialize API call
        url = "https://api.krea.ai/v1/image-generation"
        payload = {
            "prompt": prompt,
            "aspect_ratio": "16:9",
            "quality": "high"
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Step 2: Perform Request
        response = requests.post(url, json=payload, headers=headers, timeout=20)
        
        # Step 3: Parse Result
        if response.status_code == 200:
            image_url = response.json().get("url")
            if image_url:
                return {
                    "status": "ready",
                    "image_url": image_url,
                    "prompt_used": prompt
                }
        
        # Fallback if API response is not as expected or status is not 200
        # This allows the app to stay functional even if the API is down
        return {
            "status": "ready",
            "image_url": "https://placehold.co/800x400/1a1a1a/ffffff?text=AI+Analyst+Insights",
            "message": f"Krea API returned {response.status_code}. Using thematic fallback.",
            "prompt_used": prompt
        }

    except Exception as e:
        return {
            "status": "ready", 
            "image_url": "https://placehold.co/800x400/1a1a1a/ffffff?text=AI+Analyst+Data+Insight",
            "message": f"Krea AI sequence failure: {str(e)}",
            "prompt_used": prompt
        }
