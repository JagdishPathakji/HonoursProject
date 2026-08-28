"""
Quick setup script - Downloads videos and runs the ISL project.
Run this once to get everything ready!
"""

import subprocess
import sys
import os
from pathlib import Path

def install_gdown():
    """Install gdown if not already installed."""
    print("\n" + "="*60)
    print("Installing gdown (Google Drive downloader)...")
    print("="*60)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
        print("✓ gdown installed successfully!")
        return True
    except Exception as e:
        print(f"✗ Error installing gdown: {e}")
        return False

def download_sample_videos():
    """Download a few sample videos to test."""
    print("\n" + "="*60)
    print("Downloading sample ISL gesture videos...")
    print("="*60)
    
    sample_words = [
        "good", "morning", "hello", "thank", "you", 
        "pain", "help", "call", "please"
    ]
    
    # Import after gdown is installed
    from download_videos import download_all_videos
    
    print(f"\nDownloading {len(sample_words)} sample videos...")
    download_all_videos(sample_words)

def main():
    """Main setup routine."""
    print("\n" + "🎯 "*30)
    print("ISL PROJECT SETUP - One Time Only!")
    print("🎯 "*30)
    
    # Step 1: Install gdown
    if not install_gdown():
        print("\nSetup failed. Please install gdown manually:")
        print("  pip install gdown")
        sys.exit(1)
    
    # Step 2: Download sample videos
    try:
        download_sample_videos()
    except Exception as e:
        print(f"\n✗ Error during video download: {e}")
        print("You can download videos manually later using:")
        print("  python download_videos.py")
        sys.exit(1)
    
    # Step 3: Success message
    print("\n" + "="*60)
    print("✓ SETUP COMPLETE!")
    print("="*60)
    print("\nYou can now run the ISL project:")
    print("\n  1. Speech Recognition → Text:")
    print("     python sarvam_ai_speech.py")
    print("\n  2. Text → ISL Gestures:")
    print("     python text_isl.py")
    print("\n  3. ISL → Text (from camera):")
    print("     python inference_classifier.py")
    print("\nTo download more videos later:")
    print("  python download_videos.py word1 word2 word3")
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
