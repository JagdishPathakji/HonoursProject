"""
Local video dictionary - maps words to local video file paths.
This file is auto-generated when videos are downloaded.
"""

from pathlib import Path

# Videos directory
VIDEOS_DIR = Path("videos")

# Create local video dictionary from downloaded videos
def get_local_video_dict():
    """Get dictionary mapping words to local video paths."""
    local_dict = {}
    
    if VIDEOS_DIR.exists():
        for video_file in VIDEOS_DIR.glob("*.mp4"):
            word = video_file.stem
            local_dict[word] = str(video_file)
    
    return local_dict

# Auto-load local videos
LOCAL_VIDEO_ID = get_local_video_dict()

if __name__ == "__main__":
    print(f"Found {len(LOCAL_VIDEO_ID)} local videos")
    print("\nSample videos:")
    for word, path in list(LOCAL_VIDEO_ID.items())[:5]:
        print(f"  {word}: {path}")
