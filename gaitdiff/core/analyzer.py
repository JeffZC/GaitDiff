"""Analysis engine for gait comparison"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import signal

from .video import VideoReader
from .pose import PoseDetector, extract_joint_angles, compute_rom


def detect_gait_cycles(ankle_positions: List[float], fps: float) -> Tuple[int, float]:
    """
    Detect gait cycles from ankle vertical position oscillations.
    Returns (num_steps, avg_step_time_seconds)
    """
    if len(ankle_positions) < 10:
        return 0, 0.0
    
    # Smooth the signal
    positions = np.array(ankle_positions)
    if len(positions) > 5:
        # Simple moving average smoothing
        kernel = np.ones(5) / 5
        positions = np.convolve(positions, kernel, mode='valid')
    
    # Find peaks (heel strikes) using scipy
    try:
        peaks, _ = signal.find_peaks(positions, distance=max(3, len(positions) // 10))
        num_steps = len(peaks)
        
        if num_steps > 1 and fps > 0:
            # Calculate average step time
            avg_frames_per_step = len(ankle_positions) / num_steps
            avg_step_time = avg_frames_per_step / fps
            return num_steps, avg_step_time
    except:
        pass
    
    return 0, 0.0


def estimate_step_length(hip_positions: List[Tuple[float, float]], 
                         ankle_positions: List[Tuple[float, float]],
                         frame_height: int) -> float:
    """
    Estimate relative step length from hip-ankle distance variation.
    Returns normalized step length (0-100 scale based on body proportion)
    """
    if len(hip_positions) < 5 or len(ankle_positions) < 5:
        return 0.0
    
    # Calculate hip-to-ankle distances over time
    distances = []
    for hip, ankle in zip(hip_positions, ankle_positions):
        dist = np.sqrt((hip[0] - ankle[0])**2 + (hip[1] - ankle[1])**2)
        distances.append(dist)
    
    if not distances:
        return 0.0
    
    # Step length correlates with max extension
    max_dist = np.max(distances)
    min_dist = np.min(distances)
    
    # Normalize by frame height to get body-relative measurement
    step_length = (max_dist - min_dist) / frame_height * 100
    
    return float(step_length)


def estimate_walking_speed(hip_positions: List[Tuple[float, float]], 
                          fps: float, frame_width: int) -> float:
    """
    Estimate relative walking speed from hip horizontal movement.
    Returns speed as percentage of frame width per second.
    """
    if len(hip_positions) < 2 or fps <= 0:
        return 0.0
    
    # Calculate total horizontal displacement
    x_positions = [p[0] for p in hip_positions]
    total_displacement = abs(x_positions[-1] - x_positions[0])
    
    # Calculate duration
    duration = len(hip_positions) / fps
    
    if duration > 0:
        # Normalize by frame width for relative speed
        speed = (total_displacement / frame_width) * 100 / duration
        return float(speed)
    
    return 0.0


class GaitAnalyzer:
    """Analyze gait from video"""
    
    def __init__(self, runs_dir: str = "runs"):
        self.runs_dir = Path(runs_dir)
        self.pose_detector = PoseDetector()
    
    def analyze_video(self, video_path: str, num_samples: int = 30) -> Dict:
        """Analyze a single video"""
        results = {
            'video_path': video_path,
            'timestamp': datetime.now().isoformat(),
            'num_samples': num_samples,
            'angle_data': {
                'left_knee': [],
                'right_knee': [],
                'left_hip': [],
                'right_hip': []
            },
            'position_data': {
                'left_ankle_y': [],
                'right_ankle_y': [],
                'left_hip_pos': [],
                'right_hip_pos': [],
                'left_ankle_pos': [],
                'right_ankle_pos': []
            }
        }
        
        fps = 30.0
        frame_width = 1920
        frame_height = 1080
        
        with VideoReader(video_path) as reader:
            fps = reader.fps if reader.fps > 0 else 30.0
            frame_width = reader.width
            frame_height = reader.height
            frames = reader.sample_frames(num_samples)
            
            for frame_idx, frame in frames:
                pose_results = self.pose_detector.detect(frame)
                angles = extract_joint_angles(pose_results)
                
                if angles:
                    for joint, angle in angles.items():
                        results['angle_data'][joint].append(angle)
                
                # Extract position data for gait metrics
                if pose_results and pose_results.pose_landmarks and len(pose_results.pose_landmarks) > 0:
                    landmarks = pose_results.pose_landmarks[0]
                    h, w = frame.shape[:2]
                    
                    # Ankle Y positions (for step detection)
                    results['position_data']['left_ankle_y'].append(landmarks[27].y * h)
                    results['position_data']['right_ankle_y'].append(landmarks[28].y * h)
                    
                    # Full positions for step length/speed
                    results['position_data']['left_hip_pos'].append(
                        (landmarks[23].x * w, landmarks[23].y * h))
                    results['position_data']['right_hip_pos'].append(
                        (landmarks[24].x * w, landmarks[24].y * h))
                    results['position_data']['left_ankle_pos'].append(
                        (landmarks[27].x * w, landmarks[27].y * h))
                    results['position_data']['right_ankle_pos'].append(
                        (landmarks[28].x * w, landmarks[28].y * h))
        
        # Compute ROM for each joint
        results['rom'] = {}
        for joint, angles in results['angle_data'].items():
            results['rom'][joint] = compute_rom(angles)
        
        # Compute gait metrics
        results['gait_metrics'] = self._compute_gait_metrics(
            results['position_data'], fps, frame_width, frame_height
        )
        
        return results
    
    def _compute_gait_metrics(self, position_data: Dict, fps: float, 
                               frame_width: int, frame_height: int) -> Dict:
        """Compute gait-specific metrics"""
        # Detect steps from ankle oscillations
        left_steps, left_step_time = detect_gait_cycles(
            position_data['left_ankle_y'], fps)
        right_steps, right_step_time = detect_gait_cycles(
            position_data['right_ankle_y'], fps)
        
        # Average step time
        if left_step_time > 0 and right_step_time > 0:
            avg_step_time = (left_step_time + right_step_time) / 2
        elif left_step_time > 0:
            avg_step_time = left_step_time
        elif right_step_time > 0:
            avg_step_time = right_step_time
        else:
            avg_step_time = 0.0
        
        # Estimate step length (average of both legs)
        left_step_length = estimate_step_length(
            position_data['left_hip_pos'],
            position_data['left_ankle_pos'],
            frame_height
        )
        right_step_length = estimate_step_length(
            position_data['right_hip_pos'],
            position_data['right_ankle_pos'],
            frame_height
        )
        avg_step_length = (left_step_length + right_step_length) / 2 if (left_step_length + right_step_length) > 0 else 0
        
        # Estimate walking speed (from hip movement)
        left_speed = estimate_walking_speed(
            position_data['left_hip_pos'], fps, frame_width)
        right_speed = estimate_walking_speed(
            position_data['right_hip_pos'], fps, frame_width)
        avg_speed = (left_speed + right_speed) / 2 if (left_speed + right_speed) > 0 else 0
        
        # Cadence (steps per minute)
        total_steps = left_steps + right_steps
        cadence = (total_steps / (len(position_data['left_ankle_y']) / fps)) * 60 if len(position_data['left_ankle_y']) > 0 and fps > 0 else 0
        
        return {
            'step_time': round(avg_step_time, 3),          # seconds
            'step_length': round(avg_step_length, 2),       # relative units
            'walking_speed': round(avg_speed, 2),           # relative units/sec
            'cadence': round(cadence, 1),                   # steps/min
            'total_steps': total_steps
        }
    
    def analyze_comparison(self, video_a_path: str, video_b_path: str, 
                          num_samples: int = 30) -> Dict:
        """Analyze and compare two videos"""
        print("Analyzing Video A...")
        video_a_results = self.analyze_video(video_a_path, num_samples)
        
        print("Analyzing Video B...")
        video_b_results = self.analyze_video(video_b_path, num_samples)
        
        comparison = {
            'video_a': video_a_results,
            'video_b': video_b_results,
            'comparison': self._compute_comparison(video_a_results, video_b_results),
            'gait_comparison': self._compute_gait_comparison(
                video_a_results.get('gait_metrics', {}),
                video_b_results.get('gait_metrics', {})
            )
        }
        
        return comparison
    
    def _compute_gait_comparison(self, metrics_a: Dict, metrics_b: Dict) -> Dict:
        """Compute gait metrics comparison between two videos"""
        comparison = {}
        
        for metric in ['step_time', 'step_length', 'walking_speed', 'cadence']:
            val_a = metrics_a.get(metric, 0)
            val_b = metrics_b.get(metric, 0)
            comparison[metric] = {
                'video_a': val_a,
                'video_b': val_b,
                'difference': round(val_b - val_a, 3),
                'percent_change': round(((val_b - val_a) / val_a * 100) if val_a != 0 else 0, 1)
            }
        
        return comparison
    
    def _compute_comparison(self, results_a: Dict, results_b: Dict) -> Dict:
        """Compute comparison metrics between two videos"""
        comparison = {}
        
        for joint in ['left_knee', 'right_knee', 'left_hip', 'right_hip']:
            rom_a = results_a['rom'][joint]
            rom_b = results_b['rom'][joint]
            
            comparison[joint] = {
                'range_diff': rom_b['range'] - rom_a['range'],
                'mean_diff': rom_b['mean'] - rom_a['mean'],
                'video_a_range': rom_a['range'],
                'video_b_range': rom_b['range'],
                'video_a_mean': rom_a['mean'],
                'video_b_mean': rom_b['mean']
            }
        
        return comparison
    
    def save_results(self, results: Dict, run_id: Optional[str] = None) -> Path:
        """Save analysis results to JSON file"""
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = run_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {results_path}")
        return results_path
    
    def release(self):
        """Release resources"""
        self.pose_detector.release()
