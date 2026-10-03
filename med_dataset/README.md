# Medical ISL dataset (patient -> doctor)

Training data for recognising a patient's medical signs. Built from the ISL dictionary videos the
project already links to in `med_chatbot/helper/general_dictionary.py` (`MED_VIDEO_IDS`).

| Step | Command | Output |
|---|---|---|
| 1. Download reference videos | `python med_dataset/download_videos.py` | `videos/<word>.mp4` (gitignored, ~134 MB) |
| 2. Extract landmarks | `python med_dataset/extract_landmarks.py` | `landmarks/<word>.npz` |
| 3. Train the model | `python med_dataset/train_model.py` (~5 min on CPU) | `model/med_sign_model.pt`, `model/metrics.json` |
| 4. Test without a webcam | `python med_dataset/test_live_pipeline.py` | prediction for every reference video |
| 5. Use it live | `python patient_to_doctor.py` | webcam app |

To test on your own recording: `python med_dataset/test_live_pipeline.py --video my_clip.mp4`.

The word list is in `vocab.py` (45 words: symptoms, body parts, conditions, needs, time).
`preview.jpg` shows the extracted skeleton on the middle frame of every sign.

Each `.npz` holds per-frame MediaPipe Holistic landmarks (25 upper-body pose points + both hands, x/y/z)
and `segment`, the frames of the sign itself. Dictionary videos often continue with an explanation
after the sign; `extract_landmarks.py` finds the first signing segment and cuts it at a pause, capped at 4 s.

## Results

- Trained on 18,000 generated variations (400 per word). Validation and stress test (variations 1.5x stronger than training): 100%.
- Full reference videos played through the live pipeline (sign detection + recognition): 45/45 correct, ~92% confidence; also correct when simulating a 12 fps webcam.
- **Not yet measured: accuracy on new people.** Every test above comes from the same 45 videos, so real-world accuracy will be lower.

## Limitations and how to improve

There is only one reference video per word, so the training step multiplies each sign with
augmentation (speed, scale, shift, rotation, mirroring, jitter). Recording 5-10 real clips per word
with different people will improve accuracy a lot.

The live app recognises one sign at a time: the patient lowers their hands between signs, like the
dictionary videos. Continuous signing without pauses is not supported yet.
