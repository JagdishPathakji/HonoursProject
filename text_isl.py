import asyncio
import cv2
import numpy as np
import mediapipe as mp
import torch
import time
from collections import deque
import concurrent.futures
from dotenv import load_dotenv

from helper.drive_link_placeholder import DRIVE_LINK_PLACEHOLDER
from helper.connections import CONNECTIONS_NOT_NEEDED
from helper.general_dictionary import VIDEO_ID
from helper.local_video_dictionary import LOCAL_VIDEO_ID
from text_isl_preprocessing import RailwaysAnnouncementPreprocessor

class GPULandmarkDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3
        )
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3
        )

        # Define a filtered list of connections for face outlines
        self.FACEMESH_OUTLINE_CONNECTIONS = (
            list(self.mp_face_mesh.FACEMESH_LIPS) +  # Mouth outline
            list(self.mp_face_mesh.FACEMESH_LEFT_EYE) +  # Left eye outline
            list(self.mp_face_mesh.FACEMESH_RIGHT_EYE) +  # Right eye outline
            list(self.mp_face_mesh.FACEMESH_FACE_OVAL)+# Face outline
            list(self.mp_face_mesh.FACEMESH_NOSE)  # Nose outline
        )

    def preprocess_for_gpu(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_rgb.astype(np.uint8)

    def draw_landmarks(self, canvas, pose_landmarks, hand_landmarks, face_mesh_landmarks):
        pose_color = (0, 255, 0)  # Green
        hand_color = (255, 0, 0)  # Blue 
        face_color = (0, 0, 255)  # Red

        if pose_landmarks:
            for connection in self.mp_pose.POSE_CONNECTIONS:
                if connection[0] in CONNECTIONS_NOT_NEEDED or connection[1] in CONNECTIONS_NOT_NEEDED:
                    continue

                start_x, start_y = int(pose_landmarks.landmark[connection[0]].x * canvas.shape[1]), int(pose_landmarks.landmark[connection[0]].y * canvas.shape[0])
                end_x, end_y = int(pose_landmarks.landmark[connection[1]].x * canvas.shape[1]), int(pose_landmarks.landmark[connection[1]].y * canvas.shape[0])
                cv2.line(canvas, (start_x, start_y), (end_x, end_y), pose_color, 2)

        if hand_landmarks:
            for hand_landmark in hand_landmarks:
                for id, lm in enumerate(hand_landmark.landmark):
                    h, w, c = canvas.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(canvas, (cx, cy), 2, hand_color, cv2.FILLED)

                for connection in self.mp_hands.HAND_CONNECTIONS:
                    start_x, start_y = int(hand_landmark.landmark[connection[0]].x * w), int(hand_landmark.landmark[connection[0]].y * h)
                    end_x, end_y = int(hand_landmark.landmark[connection[1]].x * w), int(hand_landmark.landmark[connection[1]].y * h)
                    cv2.line(canvas, (start_x, start_y), (end_x, end_y), hand_color, 1)

        if face_mesh_landmarks:
            for face_landmarks in face_mesh_landmarks:
                for connection in self.FACEMESH_OUTLINE_CONNECTIONS:
                    start_x, start_y = int(face_landmarks.landmark[connection[0]].x * canvas.shape[1]), int(face_landmarks.landmark[connection[0]].y * canvas.shape[0])
                    end_x, end_y = int(face_landmarks.landmark[connection[1]].x * canvas.shape[1]), int(face_landmarks.landmark[connection[1]].y * canvas.shape[0])
                    cv2.line(canvas, (start_x, start_y), (end_x, end_y), face_color, 1)

        return canvas

    def detect_landmarks(self, image):
        image_rgb = self.preprocess_for_gpu(image)
        pose_results = self.pose.process(image_rgb)
        hands_results = self.hands.process(image_rgb)
        face_mesh_results = self.face_mesh.process(image_rgb)

        # White canvas
        canvas = 255 * np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8)

        canvas = self.draw_landmarks(canvas, pose_results.pose_landmarks, hands_results.multi_hand_landmarks, face_mesh_results.multi_face_landmarks)

        return canvas


async def load_video_frames(video_path):
    """Asynchronously loads video frames into memory and returns them as a list."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return []

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (500, 500))  # Resize for consistency
        frames.append(frame)

    cap.release()
    return frames


def process_video(frames, detector, word):
    """Processes a video by performing landmark detection on each frame."""
    prev_time = 0
    print(f"Streaming video for word: {word}")

    for frame in frames:
        # Detect landmarks
        landmark_canvas = detector.detect_landmarks(frame)

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time

        # Display FPS on the video frame
        cv2.putText(
            landmark_canvas,
            f"FPS: {int(fps)}",
            (10, 470),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        
        cv2.putText(
            landmark_canvas,
            f"Word: {word}",
            (10, 60),  # Position below the FPS
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),  # Blue text
            1,
            cv2.LINE_AA,
        )

        # Show the frame
        cv2.imshow("Landmark Canvas", landmark_canvas)

        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


async def buffer_videos(queue, words, word_to_video_map):
    """Continuously buffers video frames into the queue while maintaining sequence."""
    buffer_index = 0  # Tracks which word is being buffered

    while buffer_index < len(words):
        if queue.full():
            await asyncio.sleep(0.1)  # Wait until there's space in the buffer
            continue

        # Buffer the next video
        word = words[buffer_index]
        video_path = word_to_video_map.get(word)
        if video_path:
            print(f"Buffering video for: {word}")
            frames = await load_video_frames(video_path)
            await queue.put((frames, word))  # Add frames and word to the queue
        else:
            print(f"No video found for word: {word}")
            await queue.put(([], word))  # Placeholder for missing videos

        buffer_index += 1  # Move to the next word

    # Signal that buffering is complete
    await queue.put(None)  # None signals end of buffering


async def stream_videos(queue, detector):
    """Streams videos from the buffer asynchronously."""
    with concurrent.futures.ThreadPoolExecutor() as pool:
        while True:
            item = await queue.get()  # Wait for frames and word from the buffer
            if item is None:
                print("Buffering complete. No more videos to stream.")
                break  # Exit when buffering is done

            frames, word = item
            if not frames:
                print(f"Skipping empty frames for word: {word}")
                continue

            print(f"Processing video for word: {word}")
            # Process the video in a separate thread
            await asyncio.get_event_loop().run_in_executor(pool, process_video, frames, detector, word)


async def process_sentence(words, word_to_video_map):
    """Processes a sentence and handles buffering and streaming concurrently."""
    detector = GPULandmarkDetector()

    # Shared queue for buffering and streaming
    queue = asyncio.Queue(maxsize=3)  # Buffer size of 3 videos

    # Create tasks for buffering and streaming
    buffer_task = asyncio.create_task(buffer_videos(queue, words, word_to_video_map))
    stream_task = asyncio.create_task(stream_videos(queue, detector))

    # Run both tasks concurrently
    await asyncio.gather(buffer_task, stream_task)

# Example usage
if __name__ == "__main__":
    load_dotenv()
    sentence = "Good morning"
    
    # Use LOCAL_VIDEO_ID for local videos, fallback to VIDEO_ID for Drive videos
    video_dict = LOCAL_VIDEO_ID if LOCAL_VIDEO_ID else VIDEO_ID
    
    if not video_dict:
        print("Error: No videos found!")
        print("Please download videos using: python download_videos.py")
    else:
        asyncio.run(process_sentence(sentence.lower().split(), video_dict))
