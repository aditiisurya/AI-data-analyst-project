import os
import requests
from dotenv import load_dotenv

load_dotenv()

def generate_voice_response(text):
    """
    Converts text into speech using ElevenLabs REST API.
    Returns audio bytes or an error message string.
    Bypasses the need for the elevenlabs SDK to avoid long-path installation issues.
    """
    api_key = os.getenv("ELEVEN_API_KEY")
    if not api_key or api_key == "ELEVEN_API_KEY":
        return "API Key Missing"
    
    # Rachel's default voice_id
    voice_id = "pNInz6obpgDQGcFmaJgB" # Adam (Standard Premade Voice)
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    
    data = {
        "text": text,
        "model_id": "eleven_flash_v2_5",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    
    try:
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            return response.content
        else:
            error_msg = f"ElevenLabs API Error: {response.status_code} - {response.text}"
            print(error_msg)
            return error_msg
    except Exception as e:
        error_msg = str(e)
        print(f"ElevenLabs TTS Exception: {error_msg}")
        return error_msg
