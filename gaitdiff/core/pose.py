"""Pose detection and angle computation using MediaPipe"""
import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Dict, List, Tuple


class PoseDetector:
    """Detect pose landmarks using MediaPipe"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def detect(self, frame: np.ndarray) -> Optional[object]:
        """Detect pose in frame"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        return results
    
    def draw_landmarks(self, frame: np.ndarray, results: object) -> np.ndarray:
        """Draw pose landmarks on frame"""
        if results.pose_landmarks:
            annotated_frame = frame.copy()
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            return annotated_frame
        return frame
    
    def release(self):
        """Release resources"""
        self.pose.close()


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
    if not results.pose_landmarks:
        return None
    
    landmarks = results.pose_landmarks.landmark
    
    # Get relevant landmarks
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
