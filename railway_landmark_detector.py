import cv2
import numpy as np
import pandas as pd
import ast
import json
import mediapipe as mp
import time

from text_isl_preprocessing import RailwaysAnnouncementPreprocessor

class LandmarkRenderer:
    def __init__(self, pose_connections_to_skip=None, hand_color=(0, 0, 255), pose_color=(255, 0, 0)):
        """
        Initialize the LandmarkRenderer with optional connection skipping and custom colors
        
        :param pose_connections_to_skip: List of pose landmark indices to skip when drawing
        :param hand_color: Color for hand landmarks (BGR format)
        :param pose_color: Color for pose landmarks (BGR format)
        """
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        
        # Connections to skip in pose landmarks
        self.CONNECTIONS_NOT_NEEDED = [20,18,22,21,19,17,26,25,28,27,32,29,30,31]
        
        # Colors
        self.hand_color = hand_color
        self.pose_color = pose_color

    def render_landmarks(self, canvas, pose_landmarks=None, hand_landmarks=None):
        """
        Render landmarks on the given canvas
        
        :param canvas: OpenCV image to draw on
        :param pose_landmarks: List of pose landmarks (dictionary format)
        :param hand_landmarks: List of hand landmarks
        :return: Canvas with landmarks drawn
        """
        # Create landmark objects for pose if exists
        if pose_landmarks:
            # Convert list of dictionaries to landmark objects compatible with drawing
            class PoseLandmarkObject:
                def __init__(self, landmarks):
                    self.landmark = [type('Landmark', (), {'x': lm['x'], 'y': lm['y'], 'z': lm.get('z', 0)}) for lm in landmarks]

            pose_landmark_obj = PoseLandmarkObject(pose_landmarks)

            # Draw connections
            for connection in self.mp_pose.POSE_CONNECTIONS:
                # Skip specified connections
                if connection[0] in self.CONNECTIONS_NOT_NEEDED or connection[1] in self.CONNECTIONS_NOT_NEEDED:
                    continue

                # Scale coordinates to canvas size
                start_x = int(pose_landmark_obj.landmark[connection[0]].x * canvas.shape[1])
                start_y = int(pose_landmark_obj.landmark[connection[0]].y * canvas.shape[0])
                end_x = int(pose_landmark_obj.landmark[connection[1]].x * canvas.shape[1])
                end_y = int(pose_landmark_obj.landmark[connection[1]].y * canvas.shape[0])
                
                # Draw connection line
                cv2.line(canvas, (start_x, start_y), (end_x, end_y), self.pose_color, 2)

        # Draw hand landmarks and connections
        if hand_landmarks:
            # Create landmark objects for hands
            class HandLandmarkObject:
                def __init__(self, landmarks):
                    self.landmark = [type('Landmark', (), {'x': lm['x'], 'y': lm['y'], 'z': lm.get('z', 0)}) for lm in landmarks]

            for hand_landmark_dict in hand_landmarks:
                hand_landmark_obj = HandLandmarkObject(hand_landmark_dict)
                
                # Get canvas dimensions
                h, w, _ = canvas.shape

                # Draw individual hand landmarks as circles
                for lm in hand_landmark_obj.landmark:
                    cx = int(lm.x * w)
                    cy = int(lm.y * h)
                    cv2.circle(canvas, (cx, cy), 2, self.hand_color, cv2.FILLED)

                # Draw hand connections
                for connection in self.mp_hands.HAND_CONNECTIONS:
                    start_x = int(hand_landmark_obj.landmark[connection[0]].x * w)
                    start_y = int(hand_landmark_obj.landmark[connection[0]].y * h)
                    end_x = int(hand_landmark_obj.landmark[connection[1]].x * w)
                    end_y = int(hand_landmark_obj.landmark[connection[1]].y * h)
                    
                    # Draw connection line
                    cv2.line(canvas, (start_x, start_y), (end_x, end_y), self.hand_color, 1)

        return canvas

def render_sentence(words, renderer=None, coordinates_file="coordinates.csv"):
    """
    Render landmarks for a given sentence by fetching from CSV file.
    
    :param sentence: Sentence to render landmarks for
    :param renderer: Optional LandmarkRenderer instance
    :param coordinates_file: Path to the CSV file containing coordinates
    """
    # Load coordinates from CSV file
    df = pd.read_csv(coordinates_file)

    # Create renderer if not provided
    if renderer is None:
        renderer = LandmarkRenderer()

    for word in words:
        word_data = df[df['word'] == word]
        if word_data.empty:
            print(f"No landmarks found for word: {word}")
            continue

        print(f"Rendering landmarks for word: {word}")
        for _, row in word_data.iterrows():
            # Create blank canvas
            canvas = 255 * np.ones((500, 500, 3), dtype=np.uint8)

            # Prepare landmarks
            pose_landmarks = ast.literal_eval(row['pose']) if row['pose'] != '[]' else None
            hand_landmarks = ast.literal_eval(row['hands']) if row['hands'] != '[]' else None
            
            # Display the word in the bottom-left corner
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            text_color = (0, 0, 0)  # Black text
            position = (10, canvas.shape[0] - 10)  # Bottom-left corner

            # Render landmarks
            canvas = renderer.render_landmarks(canvas, pose_landmarks, hand_landmarks)

            cv2.putText(canvas, word, position, font, font_scale, text_color, font_thickness)
            
            # Display canvas
            cv2.imshow("Rendered Landmarks", canvas)

            # Press 'q' to quit early
            if cv2.waitKey(15) & 0xFF == ord("q"):
                return

    cv2.destroyAllWindows()

from dotenv import load_dotenv

def main():    
    load_dotenv()
    preprocessor = RailwaysAnnouncementPreprocessor()
    print("=== Text to ISL Translator ===")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        sentence = input("Enter the text to translate to ISL: ")
        if sentence.lower() in ['exit', 'quit']:
            break
        if not sentence.strip():
            continue
            
        print("Processing text via NLP...")
        words = preprocessor.preprocess(sentence)
        print(f"ISL Tokens extracted: {words}")
        print("Rendering ISL gestures on screen...")
        
        render_sentence(words)


if __name__ == '__main__':
    main()
