"""Pose detection and angle computation using MediaPipe"""
import cv2
import mediapipe as mp
import numpy as np
import urllib.request
from pathlib import Path
from typing import Optional, Dict, List, Tuple


def download_pose_model() -> str:
    """Download MediaPipe pose landmarker model if not present"""
    model_dir = Path.home() / ".gaitdiff" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "pose_landmarker_lite.task"
    
    if not model_path.exists():
        print("Downloading pose detection model...")
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
        try:
            # Add user agent to avoid 403 errors
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(model_path, 'wb') as out_file:
                out_file.write(response.read())
            print(f"Model downloaded to {model_path}")
        except Exception as e:
            print(f"Error downloading model: {e}")
            print("Please download manually from:")
            print(url)
            print(f"and save to: {model_path}")
            raise
    
    return str(model_path)


class PoseDetector:
    """Detect pose landmarks using MediaPipe"""
    
    def __init__(self):
        # MediaPipe 0.10+ uses tasks API
        model_path = download_pose_model()
        
        base_options = mp.tasks.BaseOptions(
            model_asset_path=model_path
        )
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
    
    def detect(self, frame: np.ndarray) -> Optional[object]:
        """Detect pose in frame"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect pose landmarks
        results = self.landmarker.detect(mp_image)
        return results
    
    def draw_landmarks(self, frame: np.ndarray, results: object) -> np.ndarray:
        """Draw pose landmarks on frame"""
        if results and results.pose_landmarks:
            annotated_frame = frame.copy()
            h, w, _ = frame.shape
            
            # Draw landmarks
            for landmarks in results.pose_landmarks:
                for landmark in landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(annotated_frame, (x, y), 3, (0, 255, 0), -1)
                
                # Draw connections
                connections = mp.tasks.vision.PoseLandmarksConnections.POSE_CONNECTIONS
                for connection in connections:
                    start_idx = connection.start
                    end_idx = connection.end
                    if start_idx < len(landmarks) and end_idx < len(landmarks):
                        start = landmarks[start_idx]
                        end = landmarks[end_idx]
                        start_point = (int(start.x * w), int(start.y * h))
                        end_point = (int(end.x * w), int(end.y * h))
                        cv2.line(annotated_frame, start_point, end_point, (0, 0, 255), 2)
            
            return annotated_frame
        return frame
    
    def release(self):
        """Release resources"""
        self.landmarker.close()


def calculate_angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """Calculate angle between three points (in degrees)"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    
    return angle


def extract_joint_angles(results: object) -> Optional[Dict[str, float]]:
    """Extract knee and hip angles from pose landmarks"""
    if not results or not results.pose_landmarks or len(results.pose_landmarks) == 0:
        return None
    
    # Get first set of landmarks
    landmarks = results.pose_landmarks[0]
    
    # Get relevant landmarks (using MediaPipe pose landmark indices)
    # Left side: 11=shoulder, 23=hip, 25=knee, 27=ankle
    # Right side: 12=shoulder, 24=hip, 26=knee, 28=ankle
    left_hip = (landmarks[23].x, landmarks[23].y)
    left_knee = (landmarks[25].x, landmarks[25].y)
    left_ankle = (landmarks[27].x, landmarks[27].y)
    left_shoulder = (landmarks[11].x, landmarks[11].y)
    
    right_hip = (landmarks[24].x, landmarks[24].y)
    right_knee = (landmarks[26].x, landmarks[26].y)
    right_ankle = (landmarks[28].x, landmarks[28].y)
    right_shoulder = (landmarks[12].x, landmarks[12].y)
    
    angles = {
        'left_knee': calculate_angle(left_hip, left_knee, left_ankle),
        'right_knee': calculate_angle(right_hip, right_knee, right_ankle),
        'left_hip': calculate_angle(left_shoulder, left_hip, left_knee),
        'right_hip': calculate_angle(right_shoulder, right_hip, right_knee),
    }
    
    return angles


def compute_rom(angle_history: List[float]) -> Dict[str, float]:
    """Compute Range of Motion statistics"""
    if not angle_history:
        return {'min': 0, 'max': 0, 'range': 0, 'mean': 0}
    
    angles = np.array(angle_history)
    return {
        'min': float(np.min(angles)),
        'max': float(np.max(angles)),
        'range': float(np.max(angles) - np.min(angles)),
        'mean': float(np.mean(angles))
    }
