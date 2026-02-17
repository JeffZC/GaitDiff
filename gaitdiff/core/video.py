"""Video processing utilities"""
import cv2
import numpy as np
from typing import Optional, Tuple


class VideoReader:
    """Read and process video frames"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Validate that we got reasonable values
        if self.frame_count <= 0 or self.width <= 0 or self.height <= 0:
            self.cap.release()
            raise RuntimeError(f"Invalid video properties: frames={self.frame_count}, size={self.width}x{self.height}")
        
    def read_frame(self, frame_number: Optional[int] = None) -> Optional[np.ndarray]:
        """Read a specific frame or the next frame"""
        if frame_number is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def sample_frames(self, num_samples: int = 30) -> list:
        """Sample frames uniformly across the video"""
        if num_samples >= self.frame_count:
            frame_indices = range(self.frame_count)
        else:
            frame_indices = np.linspace(0, self.frame_count - 1, num_samples, dtype=int)
        
        frames = []
        for idx in frame_indices:
            frame = self.read_frame(idx)
            if frame is not None:
                frames.append((idx, frame))
        
        return frames
    
    def get_current_position(self) -> int:
        """Get current frame position"""
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    def release(self):
        """Release video capture"""
        if self.cap:
            self.cap.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
