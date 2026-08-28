# SETUP GUIDE - Running Speech → ISL Locally

## Step 1: Install gdown (for downloading from Google Drive)

```bash
pip install gdown
```

Or add to requirements:
```bash
pip install -r requirements.txt gdown
```

---

## Step 2: Download Videos

### Option A: Download ALL Videos
```bash
python download_videos.py
```

⏱️ **Warning**: This downloads 200+ videos (5-10 GB). May take 1-2 hours depending on internet speed.

### Option B: Download Specific Words Only
```bash
python download_videos.py pain good morning hello
```

Replace `pain good morning hello` with the words you want to test.

### Option C: Manual Download
1. Copy any Google Drive link:
   ```
   https://drive.google.com/uc?export=download&id=1U-R3V89qbKDIQL_2vjKxyoQQBqs9C6nF
   ```
2. Videos will be saved to `videos/` folder with names like `word.mp4`

---

## Step 3: Run the Project (No Internet Required!)

### Activate Virtual Environment
```bash
.\venv312\Scripts\Activate.ps1
```

### Run Speech → ISL
```bash
python sarvam_ai_speech.py
```
- Speak when prompted
- Text will be recognized

### Run Text → ISL (with local videos)
```bash
python text_isl.py
```
- Program will look for videos in `videos/` folder
- Videos will be played with ISL landmarks

---

## File Structure After Setup

```
ml_sih_isl/
├── videos/                          # Downloaded videos
│   ├── pain.mp4
│   ├── good.mp4
│   ├── morning.mp4
│   ├── hello.mp4
│   └── ... (200+ more videos)
├── download_videos.py               # Video downloader script
├── text_isl.py                      # Modified to use local videos
├── helper/
│   ├── local_video_dictionary.py   # Auto-loads local videos
│   └── ...
└── ...
```

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| "No videos found" | Run: `python download_videos.py word_name` |
| "Video file not found" | Check if video exists in `videos/` folder |
| "gdown not installed" | Run: `pip install gdown` |
| "Download too slow" | Run multiple download sessions with specific words |

---

## System Requirements

- **Disk Space**: ~50MB per video (for 10 videos = 500MB minimum)
- **Download Speed**: ~5-10 minutes per video
- **Internet**: Only needed for initial download

---

## Next Steps

1. ✅ Install gdown: `pip install gdown`
2. ✅ Download videos: `python download_videos.py`
3. ✅ Run the project: `python sarvam_ai_speech.py` then `python text_isl.py`

---

**Now you can run the entire project offline! 🚀**
