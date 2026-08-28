"""
Download ISL gesture videos from Google Drive and save locally.
This script downloads videos for a given set of words.
"""

import os
import gdown
from pathlib import Path
from helper.general_dictionary import VIDEO_ID
from dotenv import load_dotenv

load_dotenv()

# Create videos directory if it doesn't exist
VIDEOS_DIR = Path("videos")
VIDEOS_DIR.mkdir(exist_ok=True)

def extract_drive_id(drive_url):
    """Extract file ID from Google Drive URL."""
    if "id=" in drive_url:
        return drive_url.split("id=")[1]
    return drive_url

def download_video(word, drive_url, output_dir=VIDEOS_DIR):
    """Download a single video from Google Drive."""
    try:
        # Extract the file ID from the drive URL
        file_id = extract_drive_id(drive_url)
        
        # Create output path
        output_path = output_dir / f"{word}.mp4"
        
        # Skip if already downloaded
        if output_path.exists():
            print(f"✓ Already downloaded: {word}")
            return True
        
        # Download using gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"⬇ Downloading: {word}...")
        gdown.download(url, str(output_path), quiet=False)
        
        if output_path.exists():
            print(f"✓ Successfully downloaded: {word}")
            return True
        else:
            print(f"✗ Failed to download: {word}")
            return False
            
    except Exception as e:
        print(f"✗ Error downloading {word}: {e}")
        return False

def download_all_videos(words=None):
    """Download videos for specific words or all words."""
    if words is None:
        words = list(VIDEO_ID.keys())
    
    total = len(words)
    success = 0
    
    print(f"\n{'='*60}")
    print(f"Starting download of {total} videos...")
    print(f"Saving to: {VIDEOS_DIR.absolute()}")
    print(f"{'='*60}\n")
    
    for i, word in enumerate(words, 1):
        if word in VIDEO_ID:
            drive_url = VIDEO_ID[word]
            if download_video(word, drive_url):
                success += 1
        print(f"Progress: {i}/{total}\n")
    
    print(f"\n{'='*60}")
    print(f"Download Complete!")
    print(f"Successfully downloaded: {success}/{total}")
    print(f"Location: {VIDEOS_DIR.absolute()}")
    print(f"{'='*60}\n")

def create_local_video_dict():
    """Create a dictionary mapping words to local video paths."""
    local_dict = {}
    video_files = list(VIDEOS_DIR.glob("*.mp4"))
    
    for video_file in video_files:
        word = video_file.stem
        local_dict[word] = str(video_file)
    
    return local_dict

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Download specific words
        words_to_download = sys.argv[1:]
        print(f"Downloading specific words: {words_to_download}")
        download_all_videos(words_to_download)
    else:
        # Ask user what to do
        print("\nOptions:")
        print("1. Download ALL videos (this may take a while)")
        print("2. Download specific words (provide as arguments)")
        print("\nUsage:")
        print("  python download_videos.py              # Downloads ALL videos")
        print("  python download_videos.py word1 word2  # Downloads specific words")
        
        response = input("\nDo you want to download all videos? (y/n): ").strip().lower()
        if response == 'y':
            download_all_videos()
        else:
            print("Please run with specific words as arguments.")
