import base64
import requests
import base64
from dotenv import load_dotenv
import os
import sounddevice as sd
import numpy as np
import io
import wave
import json

load_dotenv()

def record_audio_to_base64(duration=6, samplerate=16000, channels=1):
    """
    Records audio in real-time, converts it to base64-encoded string.

    Parameters:
    - duration (int): Duration to record in seconds.
    - samplerate (int): Sampling rate of audio.
    - channels (int): Number of audio channels.

    Returns:
    - str: Base64-encoded audio string.
    """
    print("Recording... Speak now!")
    audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=channels, dtype=np.int16)
    sd.wait()  # Wait for recording to finish
    print("Recording finished.")

    # Save audio data to a buffer in WAV format
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 2 bytes per sample for np.int16
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())

    # Encode buffer content to base64
    base64_audio = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return base64_audio

def speech_to_text_translate(base64_audio):
    """
    Sends a base64-encoded audio file to the speech-to-text API and returns the transcribed text in english.
    
    Link: https://sarvam.ai/docs/speech-to-text-translate

    Parameters:
    - base64_audio (str): Base64-encoded audio content.

    Returns:
    - str: Transcribed text or error message from the API.
    """
    
    url = "https://api.sarvam.ai/speech-to-text-translate"

    data = {
        "model": "saaras:v2.5",
    }

    files = {
        "file": ("input.wav", base64.b64decode(base64_audio), "audio/wav")
    }

    headers = {
    'api-subscription-key': os.getenv('SARVAM_AI_API_KEY')
    }

    try:
        response = requests.post(url, data=data, files=files,headers=headers)
        print(f"API Response Status: {response.status_code}")
        print(f"API Response: {response.json()}")
        
        if 'transcript' in response.json():
            return [response.json()['transcript'],response.json()['language_code']]
        else:
            print("Error: 'transcript' not found in API response")
            return {"error": "No transcript in response", "details": response.json()}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def speech_to_text_original(base64_audio,language_code):
    
    """
    Sends a base64-encoded audio file to the speech-to-text API and returns the transcribed text in original language.
    
    Link: https://sarvam.ai/docs/speech-to-text-translate

    Parameters:
    - base64_audio (str): Base64-encoded audio content.
    - language_code (str): Language code of the audio content.

    Returns:
    - str: Transcribed text or error message from the API.
    """
    
    url = " https://api.sarvam.ai/speech-to-text"

    data = {
        "model": "saarika:v2.5",
        "language_code": language_code
    }

    files = {
        "file": ("input.wav", base64.b64decode(base64_audio), "audio/wav"),
    }

    headers = {
    'api-subscription-key': os.getenv('SARVAM_AI_API_KEY')
    }

    try:
        response = requests.post(url, data=data, files=files,headers=headers)
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def wavToBase64(wav_file_path):
    # Open the WAV file in binary mode
    with open(wav_file_path, 'rb') as wav_file:
        # Read the file contents
        wav_data = wav_file.read()
        
        # Convert the WAV file data to Base64
        base64_wav = base64.b64encode(wav_data).decode('utf-8')
        
    return base64_wav

 
if __name__ == "__main__":
        
    base64_audio = record_audio_to_base64()
    # base64_audio = wavToBase64("sample_input.wav")

    # Call the speech-to-text function
    response = speech_to_text_translate(base64_audio)
    print("Transcription:", response[0],response[1])

    if response[1] == None:
        print('No language detected')
        response[1] = 'en-IN'

    if response[1] != "en-IN":
        response_original = speech_to_text_original(base64_audio,response[1])
        print("Transcription in original number:", response_original)