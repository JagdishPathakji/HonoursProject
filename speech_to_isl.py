import asyncio
import os
import string
from dotenv import load_dotenv

# Import speech functionalities
from sarvam_ai_speech import record_audio_to_base64, speech_to_text_translate

# Import ISL video processing functionalities
from text_isl import process_sentence
from helper.local_video_dictionary import LOCAL_VIDEO_ID
from helper.general_dictionary import VIDEO_ID

def main():
    load_dotenv()
    
    # 1. Choose video dictionary (Prefer local videos since you have them downloaded)
    video_dict = LOCAL_VIDEO_ID if LOCAL_VIDEO_ID else VIDEO_ID
    if not video_dict:
        print("Error: No videos found in dictionaries!")
        print("Please check your helper dictionaries.")
        return

    # 2. Record audio and get base64 encoded data
    print("\n=== Step 1: Recording Speech ===")
    base64_audio = record_audio_to_base64(duration=6) # 6 seconds recording
    
    # 3. Transcribe audio to English text (ISL relies on English words mapping)
    print("\n=== Step 2: Transcribing Speech ===")
    response = speech_to_text_translate(base64_audio)
    
    if isinstance(response, dict) and "error" in response:
        print(f"Error during transcription: {response['error']}")
        return
        
    transcript = response[0]
    language_code = response[1]
    
    print(f"Detected English Transcription: {transcript}")
    
    if not transcript:
        print("No transcript was returned. Please try speaking again.")
        return

    # 4. Clean up text and convert to ISL videos
    # Clean the text by removing punctuation and converting to lowercase
    cleaned_transcript = transcript.translate(str.maketrans('', '', string.punctuation)).lower()
    words = cleaned_transcript.split()
    
    print(f"\n=== Step 3: Playing ISL Videos for words: {words} ===")
    asyncio.run(process_sentence(words, video_dict))
    print("\nDone!")

if __name__ == "__main__":
    main()
