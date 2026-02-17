"""
MediaPipe Utilities - Converted to PySide6 and Tasks API
Handles MediaPipe pose detection and video processing.
"""

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import urllib.request
from pathlib import Path
from .pose_format_utils import process_mediapipe_to_rr21, SUPPORTED_FORMATS


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


# Initialize MediaPipe Pose Landmarker
_model_path = download_pose_model()
_base_options = mp.tasks.BaseOptions(model_asset_path=_model_path)
_options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=_base_options,
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(_options)


def get_pose_landmarks_from_frame(frame):
    """
    Detect pose landmarks for a single frame using MediaPipe
    
    Args:
        frame: OpenCV frame (BGR)
    
    Returns:
        tuple: (landmarks_list as flat [x1,y1,x2,y2...], annotated_frame)
    """
    if frame is None:
        return [], None
    
    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    
    # Detect pose landmarks
    results = _landmarker.detect(mp_image)
    
    if not results or not results.pose_landmarks:
        return [], frame
    
    # Create a copy for annotations
    annotated_frame = frame.copy()
    
    # Draw the pose annotation on the image
    h, w, _ = frame.shape
    try:
        for landmarks in results.pose_landmarks:
            # Draw landmarks
            for landmark in landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(annotated_frame, (x, y), 5, (0, 255, 0), -1)
            
            # Draw connections
            connections = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS
            for connection in connections:
                start_idx = connection.start
                end_idx = connection.end
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start = landmarks[start_idx]
                    end = landmarks[end_idx]
                    start_point = (int(start.x * w), int(start.y * h))
                    end_point = (int(end.x * w), int(end.y * h))
                    cv2.line(annotated_frame, start_point, end_point, (0, 0, 255), 2)
    except Exception as e:
        print(f"Error drawing landmarks: {e}")
        # Continue without annotations
    
    # Extract landmarks as flat list [x1,y1,x2,y2...]
    landmarks_list = []
    try:
        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            for landmark in results.pose_landmarks[0]:  # Take first pose
                # Normalize coordinates to image dimensions
                landmarks_list.append(landmark.x * w)
                landmarks_list.append(landmark.y * h)
    except Exception as e:
        print(f"Error extracting landmarks: {e}")
        return [], frame
    
    return landmarks_list, annotated_frame


def process_video_with_mediapipe(video_path, progress_callback=None):
    """
    Process an entire video with MediaPipe pose detection
    
    Args:
        video_path: Path to the video file
        progress_callback: Optional callback function for progress updates (receives int 0-100)
    
    Returns:
        tuple: (DataFrame with pose data in RR21 format, success flag)
    """
    try:
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, False
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Initialize pose detector
        video_options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=_base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        video_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(video_options)
        
        # Create empty DataFrame for RR21 format
        column_names = []
        for name in SUPPORTED_FORMATS["rr21"]:
            column_names.extend([f'{name}_X', f'{name}_Y'])
        
        # Initialize with zeros
        pose_data = pd.DataFrame(np.zeros((frame_count, len(column_names))), columns=column_names)
        
        # Process frames
        frame_idx = 0
        cancelled = False
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update progress
            if progress_callback is not None:
                progress_percent = min(100, int((frame_idx / frame_count) * 100))
                # progress_callback returns True if cancelled
                if progress_callback(progress_percent):
                    cancelled = True
                    break
            
            # Process frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            # Detect pose landmarks
            results = video_landmarker.detect_for_video(mp_image, frame_idx)
            
            # Extract landmarks if detected
            if results and results.pose_landmarks and len(results.pose_landmarks) > 0:
                landmarks_list = []
                h, w, _ = frame.shape
                try:
                    for landmark in results.pose_landmarks[0]:  # Take first pose
                        landmarks_list.append(landmark.x * w)
                        landmarks_list.append(landmark.y * h)
                except Exception as e:
                    print(f"Error extracting landmarks for frame {frame_idx}: {e}")
                    continue
                
                # Convert to RR21 format
                rr21_landmarks = process_mediapipe_to_rr21(landmarks_list)
                
                # Update data
                for i in range(0, len(rr21_landmarks), 2):
                    if i+1 < len(rr21_landmarks) and i//2 < len(column_names)//2:
                        pose_data.iloc[frame_idx, i] = rr21_landmarks[i]
                        pose_data.iloc[frame_idx, i+1] = rr21_landmarks[i+1]
            
            frame_idx += 1
        
        # Clean up
        cap.release()
        video_landmarker.close()
        
        if cancelled:
            return None, False
        
        return pose_data, True
            
    except Exception as e:
        print(f"Error processing video: {e}")
        if 'cap' in locals() and cap is not None:
            cap.release()
        return None, False


def get_frame_with_pose_overlay(frame, pose_row, keypoint_names=None, connections=None,
                                 selected_point=None, point_radius=5, line_thickness=2):
    """
    Draw pose overlay on a frame using RR21 pose data
    
    Args:
        frame: OpenCV frame (BGR)
        pose_row: Series or array with x,y coordinates for all keypoints
        keypoint_names: List of keypoint names (default: RR21)
        connections: List of (idx1, idx2) tuples for skeleton connections
        selected_point: Index of selected keypoint to highlight
        point_radius: Radius of keypoint circles
        line_thickness: Thickness of skeleton lines
    
    Returns:
        numpy array: Annotated frame
    """
    if keypoint_names is None:
        keypoint_names = SUPPORTED_FORMATS["rr21"]
    
    if connections is None:
        from .pose_format_utils import get_keypoint_connections
        connections = get_keypoint_connections("rr21")
    
    annotated = frame.copy()
    
    # Convert pose_row to numpy array if it's a pandas Series
    if hasattr(pose_row, 'values'):
        coords = pose_row.values.reshape(-1, 2)
    else:
        coords = np.array(pose_row).reshape(-1, 2)
    
    # Draw skeleton connections
    for (idx1, idx2) in connections:
        if idx1 < len(coords) and idx2 < len(coords):
            pt1 = coords[idx1]
            pt2 = coords[idx2]
            
            # Skip if either point is at origin (not detected)
            if (pt1[0] == 0 and pt1[1] == 0) or (pt2[0] == 0 and pt2[1] == 0):
                continue
                
            cv2.line(annotated, 
                    (int(pt1[0]), int(pt1[1])),
                    (int(pt2[0]), int(pt2[1])),
                    (0, 255, 0), line_thickness)
    
    # Draw keypoints
    for i, (x, y) in enumerate(coords):
        if x == 0 and y == 0:
            continue
            
        # Highlight selected point
        if i == selected_point:
            color = (0, 0, 255)  # Red for selected
            radius = point_radius + 3
        else:
            color = (255, 0, 0)  # Blue for others
            radius = point_radius
            
        cv2.circle(annotated, (int(x), int(y)), radius, color, -1)
    
    return annotated
