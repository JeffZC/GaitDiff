"""Analysis engine for gait comparison"""
import json
import os
import cv2
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import signal

from .video import VideoReader
from .pose import PoseDetector, extract_joint_angles, compute_rom, calculate_angle


def _estimate_body_height_median(pose_data) -> Optional[float]:
    """Estimate body height robustly using median from all pose frames (Jan's method).
    Returns pixels (with Jan's 1.1 buffer) or None if no valid samples.
    """
    if pose_data is None or pose_data.empty:
        return None
        
    estimates = []
    
    # Check more frames to find valid pose data
    num_frames_to_check = min(100, len(pose_data))  # Check up to 100 frames
    
    for frame_idx in range(num_frames_to_check):
        frame_pose = pose_data.iloc[frame_idx].values
        
        if len(frame_pose) >= 42:  # Ensure we have enough data (21 keypoints * 2 coords)
            keypoints = frame_pose.reshape(-1, 2)  # (21, 2)
            
            if len(keypoints) >= 17:  # Ensure we have nose and ankles
                nose = keypoints[0]      # nose
                left_ankle = keypoints[15]   # left ankle
                right_ankle = keypoints[16]  # right ankle
                
                # Validate coordinates are reasonable and non-zero
                if (nose[0] > 0 and nose[1] > 0 and 
                    left_ankle[0] > 0 and left_ankle[1] > 0 and 
                    right_ankle[0] > 0 and right_ankle[1] > 0 and
                    nose[1] < left_ankle[1] and nose[1] < right_ankle[1]):  # nose should be above ankles
                    
                    ankle_y = (left_ankle[1] + right_ankle[1]) / 2.0
                    height = abs(nose[1] - ankle_y)
                    
                    if height >= 50:  # reasonable minimum height in pixels
                        estimates.append(height)
                        
                        # If we have enough estimates, we can stop early
                        if len(estimates) >= 10:
                            break
    
    if not estimates:
        return None
    
    median_height = float(np.median(estimates))
    return median_height * 1.1  # Jan's 10% buffer


def _get_frame_timestamps(video_path: str, num_frames: int) -> List[float]:
    """Get accurate timestamps for each frame using video properties"""
    timestamps = []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # Fallback to simple frame-based timing
        fps = 30.0
        return [i / fps for i in range(num_frames)]
    
    # Get FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    
    # Try to get accurate timestamps
    frame_idx = 0
    last_pos_msec = 0
    
    while frame_idx < num_frames:
        # Get current position in milliseconds
        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        
        if pos_msec > 0:
            timestamps.append(pos_msec / 1000.0)  # convert to seconds
            last_pos_msec = pos_msec
        else:
            # Fallback: use frame index and fps
            timestamps.append(frame_idx / fps)
        
        # Read next frame
        ret = cap.grab()
        if not ret:
            break
        frame_idx += 1
    
    cap.release()
    
    # Ensure we have the right number of timestamps
    if len(timestamps) < num_frames:
        # Pad with estimated timestamps
        while len(timestamps) < num_frames:
            timestamps.append((len(timestamps)) / fps)
    
    return timestamps[:num_frames]


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


class GaitAnalyzer:
    """Analyze gait from video"""
    
    def __init__(self, runs_dir: str = "runs"):
        self.runs_dir = Path(runs_dir)
        self.pose_detector = PoseDetector()
    
    def analyze_video(self, video_path: str, video_id: str) -> Dict:
        """Analyze a single video using pre-processed pose data"""
        from ..pose_editor import get_shared_state
        
        shared_state = get_shared_state()
        pose_data = shared_state.get_pose_data(video_id)
        
        if pose_data is None or pose_data.empty:
            raise ValueError(f"No pose data available for video {video_id}")
        
        results = {
            'video_path': video_path,
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(pose_data),
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
                'right_ankle_pos': [],
                'nose_pos': []  # Add nose position for body height estimation
            }
        }
        
        # Estimate body height robustly from all pose data (Jan's method)
        body_height = _estimate_body_height_median(pose_data)
        if body_height is None or body_height < 10:
            print(f"Warning: Could not estimate body height reliably for video {video_id}, using fallback")
            body_height = 1.0  # fallback to avoid division by zero
        else:
            print(f"Estimated body height for video {video_id}: {body_height:.1f} pixels")
        
        results['body_height'] = body_height
        
        # Get accurate timestamps for each frame
        timestamps = _get_frame_timestamps(video_path, len(pose_data))
        results['timestamps'] = timestamps
        
        # Get video properties for pixel-based calculations
        with VideoReader(video_path) as reader:
            fps = reader.fps if reader.fps > 0 else 30.0
            frame_width = reader.width
            frame_height = reader.height
        
        results['fps'] = fps
        results['frame_width'] = frame_width
        results['frame_height'] = frame_height
        
        # Process each frame's pose data
        for frame_idx in range(len(pose_data)):
            frame_pose = pose_data.iloc[frame_idx].values
            
            # Convert flat array to RR21 format (21 keypoints, x,y each = 42 values)
            if len(frame_pose) >= 42:  # Ensure we have enough data
                keypoints = frame_pose.reshape(-1, 2)  # (21, 2)
                
                # Extract joint angles using pose keypoints
                angles = self._extract_angles_from_keypoints(keypoints, frame_width, frame_height)
                
                if angles:
                    for joint, angle in angles.items():
                        results['angle_data'][joint].append(float(angle))
                else:
                    for joint in results['angle_data'].keys():
                        results['angle_data'][joint].append(float("nan"))
                
                # Extract position data for gait metrics (pixel coordinates)
                # RR21 format: https://github.com/anibali/rr21-pose-format
                # Keypoints: 0-20 for different body parts
                # We need: hips (11,12), ankles (15,16)
                if len(keypoints) >= 17:  # Ensure we have ankle keypoints
                    # Left hip (11), right hip (12), left ankle (15), right ankle (16)
                    left_hip = keypoints[11]   # [x, y]
                    right_hip = keypoints[12]  # [x, y] 
                    left_ankle = keypoints[15] # [x, y]
                    right_ankle = keypoints[16] # [x, y]
                    nose = keypoints[0]        # [x, y] - nose for body height
                    
                    # Store Y positions for step detection (pixel coordinates)
                    results['position_data']['left_ankle_y'].append(left_ankle[1])
                    results['position_data']['right_ankle_y'].append(right_ankle[1])
                    
                    # Store full positions for step length/speed calculations
                    results['position_data']['left_hip_pos'].append((left_hip[0], left_hip[1]))
                    results['position_data']['right_hip_pos'].append((right_hip[0], right_hip[1]))
                    results['position_data']['left_ankle_pos'].append((left_ankle[0], left_ankle[1]))
                    results['position_data']['right_ankle_pos'].append((right_ankle[0], right_ankle[1]))
                    results['position_data']['nose_pos'].append((nose[0], nose[1]))
            else:
                for joint in results['angle_data'].keys():
                    results['angle_data'][joint].append(float("nan"))
        
        # Detect gait events for cycle-based ROM
        gait_events = self._detect_gait_events_jangait(results['position_data'], fps)
        results['gait_events'] = gait_events

        # Compute ROM for each joint using gait cycles
        results['rom'] = {}
        for joint, angles in results['angle_data'].items():
            cycle_roms = self._compute_cycle_based_rom(angles, gait_events)
            if cycle_roms:
                results['rom'][joint] = {
                    'min': float(np.min(cycle_roms)),
                    'max': float(np.max(cycle_roms)),
                    'range': float(np.mean(cycle_roms)),  # Mean ROM across cycles
                    'mean': float(np.mean(angles)) if angles else 0.0
                }
            else:
                # Fallback to overall ROM
                results['rom'][joint] = compute_rom(angles)
        
        # Compute gait metrics
        results['gait_metrics'] = self._compute_gait_metrics(
            results['position_data'], fps, frame_width, frame_height, body_height, timestamps
        )
        
        return results

    def _compute_cycle_rom_median(self, angle_series: List[float], cycle_frames: List[int]) -> Optional[Dict[str, float]]:
        """Compute per-cycle ROM stats and return median of cycles."""
        if not cycle_frames or len(cycle_frames) < 2:
            return None
        
        series = np.array(angle_series, dtype=float)
        per_cycle = []
        
        for i in range(len(cycle_frames) - 1):
            start = int(cycle_frames[i])
            end = int(cycle_frames[i + 1])
            if end <= start:
                continue
            if start < 0 or end >= len(series):
                continue
            segment = series[start:end + 1]
            segment = segment[np.isfinite(segment)]
            if len(segment) < 2:
                continue
            seg_min = float(np.min(segment))
            seg_max = float(np.max(segment))
            seg_mean = float(np.mean(segment))
            per_cycle.append({
                'min': seg_min,
                'max': seg_max,
                'range': float(seg_max - seg_min),
                'mean': seg_mean
            })
        
        if not per_cycle:
            return None
        
        mins = np.array([c['min'] for c in per_cycle], dtype=float)
        maxs = np.array([c['max'] for c in per_cycle], dtype=float)
        ranges = np.array([c['range'] for c in per_cycle], dtype=float)
        means = np.array([c['mean'] for c in per_cycle], dtype=float)
        
        return {
            'min': float(np.median(mins)),
            'max': float(np.median(maxs)),
            'range': float(np.median(ranges)),
            'mean': float(np.median(means))
        }
    
    def _extract_angles_from_keypoints(self, keypoints: np.ndarray, frame_width: int, frame_height: int) -> Optional[Dict[str, float]]:
        """Extract joint angles from RR21 keypoints"""
        if len(keypoints) < 17:  # Need at least up to ankles
            return None
        
        # RR21 keypoints:
        # 5: LEFT_SHOULDER, 6: RIGHT_SHOULDER
        # 11: LEFT_HIP, 12: RIGHT_HIP  
        # 13: LEFT_KNEE, 14: RIGHT_KNEE
        # 15: LEFT_ANKLE, 16: RIGHT_ANKLE
        
        try:
            left_shoulder = keypoints[5]   # [x, y]
            right_shoulder = keypoints[6]  # [x, y]
            left_hip = keypoints[11]       # [x, y]
            right_hip = keypoints[12]      # [x, y]
            left_knee = keypoints[13]      # [x, y]
            right_knee = keypoints[14]     # [x, y]
            left_ankle = keypoints[15]     # [x, y]
            right_ankle = keypoints[16]    # [x, y]
            
            # Calculate angles using the same method as extract_joint_angles
            angles = {
                'left_knee': calculate_angle(left_hip, left_knee, left_ankle),
                'right_knee': calculate_angle(right_hip, right_knee, right_ankle),
                'left_hip': calculate_angle(left_shoulder, left_hip, left_knee),
                'right_hip': calculate_angle(right_shoulder, right_hip, right_knee),
            }
            
            return angles
        except (IndexError, TypeError):
            return None
    
    def _compute_gait_metrics(self, position_data: Dict, fps: float, 
                               frame_width: int, frame_height: int, body_height: float, timestamps: List[float]) -> Dict:
        """Compute gait-specific metrics using JanGait method"""
        # Use JanGait's robust event detection method
        # Calculate horizontal distance between ankle and mid-hip (pelvis)
        gait_events = self._detect_gait_events_jangait(position_data, fps)
        
        # Calculate gait parameters from detected events
        gait_params = self._calculate_gait_parameters_jangait(gait_events, position_data, fps, frame_width, frame_height, body_height, timestamps)
        
        return gait_params
    
    def _detect_gait_events_jangait(self, position_data: Dict, fps: float) -> Dict:
        """Detect gait events using JanGait's method (horizontal ankle-pelvis distance)"""
        import scipy.signal
        
        # Calculate mid-hip positions (average of left and right hips)
        mid_hip_x = []
        for i in range(len(position_data['left_hip_pos'])):
            left_hip_x = position_data['left_hip_pos'][i][0]
            right_hip_x = position_data['right_hip_pos'][i][0]
            mid_hip_x.append((left_hip_x + right_hip_x) / 2)
        
        # Calculate horizontal distance between ankle and mid-hip (JanGait method)
        # Left: ankle_x - mid_hip_x
        # Right: ankle_x - mid_hip_x
        left_signal = []
        right_signal = []
        
        for i in range(len(position_data['left_ankle_pos'])):
            left_ankle_x = position_data['left_ankle_pos'][i][0]
            right_ankle_x = position_data['right_ankle_pos'][i][0]
            
            left_signal.append(left_ankle_x - mid_hip_x[i])
            right_signal.append(right_ankle_x - mid_hip_x[i])
        
        left_signal = np.array(left_signal)
        right_signal = np.array(right_signal)
        
        # Detect events using find_peaks (JanGait approach)
        # LHS: peaks in left signal
        # LTO: peaks in negative left signal (valleys)
        # RHS: peaks in right signal  
        # RTO: peaks in negative right signal (valleys)
        
        # Find peaks with minimum distance to avoid false positives
        min_distance = max(10, len(left_signal) // 20)  # Adaptive minimum distance
        
        try:
            # Left heel strikes: peaks in left signal
            lhs_peaks, _ = scipy.signal.find_peaks(left_signal, distance=min_distance, prominence=np.std(left_signal)*0.5)
            
            # Left toe offs: peaks in negative left signal (valleys in original)
            lto_peaks, _ = scipy.signal.find_peaks(-left_signal, distance=min_distance, prominence=np.std(left_signal)*0.5)
            
            # Right heel strikes: peaks in right signal
            rhs_peaks, _ = scipy.signal.find_peaks(right_signal, distance=min_distance, prominence=np.std(right_signal)*0.5)
            
            # Right toe offs: peaks in negative right signal (valleys in original)
            rto_peaks, _ = scipy.signal.find_peaks(-right_signal, distance=min_distance, prominence=np.std(right_signal)*0.5)
            
        except:
            # Fallback if scipy fails
            lhs_peaks = np.array([])
            lto_peaks = np.array([])
            rhs_peaks = np.array([])
            rto_peaks = np.array([])
        
        return {
            'lhs_frames': lhs_peaks.tolist(),  # Left heel strike frames
            'lto_frames': lto_peaks.tolist(),  # Left toe off frames
            'rhs_frames': rhs_peaks.tolist(),  # Right heel strike frames
            'rto_frames': rto_peaks.tolist(),  # Right toe off frames
            'fps': fps
        }

    def _calculate_gait_parameters_jangait(self, gait_events: Dict, position_data: Dict, 
                                         fps: float, frame_width: int, frame_height: int, body_height: float, timestamps: List[float]) -> Dict:
        """Calculate gait parameters using JanGait's exact method"""
        lhs_frames = np.array(gait_events['lhs_frames'])
        rhs_frames = np.array(gait_events['rhs_frames'])
        
        # Calculate step times (JanGait method) using accurate timestamps
        step_times_left = []
        step_times_right = []
        
        # Left step time: time between consecutive RHS after LHS
        for i in range(len(lhs_frames)):
            lhs_frame = int(lhs_frames[i])
            if lhs_frame < len(timestamps):
                lhs_time = timestamps[lhs_frame]
                # Find next RHS
                next_rhs = rhs_frames[rhs_frames > lhs_frames[i]]
                if len(next_rhs) > 0:
                    rhs_frame = int(next_rhs[0])
                    if rhs_frame < len(timestamps):
                        rhs_time = timestamps[rhs_frame]
                        step_times_left.append(rhs_time - lhs_time)
        
        # Right step time: time between consecutive LHS after RHS
        for i in range(len(rhs_frames)):
            rhs_frame = int(rhs_frames[i])
            if rhs_frame < len(timestamps):
                rhs_time = timestamps[rhs_frame]
                # Find next LHS
                next_lhs = lhs_frames[lhs_frames > rhs_frames[i]]
                if len(next_lhs) > 0:
                    lhs_frame = int(next_lhs[0])
                    if lhs_frame < len(timestamps):
                        lhs_time = timestamps[lhs_frame]
                        step_times_right.append(lhs_time - rhs_time)
        
        avg_step_time = np.mean(step_times_left + step_times_right) if (step_times_left + step_times_right) else 0.0
        
        # Calculate step lengths (JanGait method - normalized by body height)
        step_lengths_left = []
        step_lengths_right = []
        
        # Left step length: distance between left and right ankle at RHS, normalized by body height
        for rhs_frame in rhs_frames:
            if rhs_frame < len(position_data['left_ankle_pos']):
                left_ankle = position_data['left_ankle_pos'][rhs_frame]
                right_ankle = position_data['right_ankle_pos'][rhs_frame]
                # Use horizontal distance in pixels, then normalize by body height
                step_length_pixels = abs(left_ankle[0] - right_ankle[0])
                if body_height > 0:
                    step_length_normalized = (step_length_pixels / body_height) * 100  # percentage of body height
                else:
                    step_length_normalized = step_length_pixels  # fallback to pixels
                step_lengths_left.append(step_length_normalized)
        
        # Right step length: distance between left and right ankle at LHS, normalized by body height
        for lhs_frame in lhs_frames:
            if lhs_frame < len(position_data['left_ankle_pos']):
                left_ankle = position_data['left_ankle_pos'][lhs_frame]
                right_ankle = position_data['right_ankle_pos'][lhs_frame]
                # Use horizontal distance in pixels, then normalize by body height
                step_length_pixels = abs(left_ankle[0] - right_ankle[0])
                if body_height > 0:
                    step_length_normalized = (step_length_pixels / body_height) * 100  # percentage of body height
                else:
                    step_length_normalized = step_length_pixels  # fallback to pixels
                step_lengths_right.append(step_length_normalized)
        
        avg_step_length = np.mean(step_lengths_left + step_lengths_right) if (step_lengths_left + step_lengths_right) else 0.0
        
        # Calculate walking speed (JanGait method - normalized units per second)
        # Speed = average normalized step length / average step time
        walking_speed = avg_step_length / avg_step_time if avg_step_time > 0 else 0.0
        
        # Calculate cadence (steps per minute) using actual video duration
        total_steps = len(lhs_frames) + len(rhs_frames)
        if len(timestamps) > 1:
            duration = timestamps[-1] - timestamps[0]  # actual video duration in seconds
        else:
            duration = len(position_data['left_ankle_pos']) / fps  # fallback
        cadence = (total_steps / duration) * 60 if duration > 0 else 0.0
        
        return {
            'step_time': round(avg_step_time, 3),          # seconds
            'step_length': round(avg_step_length, 2),      # percentage of body height
            'walking_speed': round(walking_speed, 2),      # percentage/second
            'cadence': round(cadence, 1),                  # steps/min
            'total_steps': total_steps,
            'events_detected': {
                'lhs': len(lhs_frames),
                'rhs': len(rhs_frames)
            }
        }
    
    def _compute_cycle_based_rom(self, angles: List[float], gait_events: Dict) -> List[float]:
        """Compute ROM for each gait cycle and return list of ROMs"""
        if not angles or not gait_events:
            return []
        
        # Get all heel strike frames and sort
        lhs_frames = gait_events.get('lhs_frames', [])
        rhs_frames = gait_events.get('rhs_frames', [])
        all_strikes = sorted(set(lhs_frames + rhs_frames))
        
        if len(all_strikes) < 2:
            return []
        
        cycle_roms = []
        for i in range(len(all_strikes) - 1):
            start_frame = int(all_strikes[i])
            end_frame = int(all_strikes[i + 1])
            
            if start_frame >= len(angles) or end_frame >= len(angles) or start_frame >= end_frame:
                continue
            
            # Get angles in this cycle
            cycle_angles = angles[start_frame:end_frame]
            if len(cycle_angles) > 5:  # Minimum frames for valid cycle
                cycle_min = np.min(cycle_angles)
                cycle_max = np.max(cycle_angles)
                rom = cycle_max - cycle_min
                if rom > 0:  # Valid ROM
                    cycle_roms.append(rom)
        
        return cycle_roms
    
    def analyze_comparison(self, video_a_path: str, video_b_path: str) -> Dict:
        """Analyze and compare two videos using available pose data"""
        print("Analyzing Video A...")
        video_a_results = self.analyze_video(video_a_path, 'A')
        
        print("Analyzing Video B...")
        video_b_results = self.analyze_video(video_b_path, 'B')
        
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
        
        # Compare main gait metrics
        for metric in ['step_time', 'stance_time', 'swing_time', 'step_length', 'walking_speed', 'cadence']:
            val_a = metrics_a.get(metric, 0)
            val_b = metrics_b.get(metric, 0)
            comparison[metric] = {
                'video_a': val_a,
                'video_b': val_b,
                'difference': round(val_b - val_a, 3),
                'percent_change': round(((val_b - val_a) / val_a * 100) if val_a != 0 else 0, 1)
            }
        
        # Compare total steps
        total_steps_a = metrics_a.get('total_steps', 0)
        total_steps_b = metrics_b.get('total_steps', 0)
        comparison['total_steps'] = {
            'video_a': total_steps_a,
            'video_b': total_steps_b,
            'difference': total_steps_b - total_steps_a,
            'percent_change': round(((total_steps_b - total_steps_a) / total_steps_a * 100) if total_steps_a != 0 else 0, 1)
        }
        
        # Compare events detected
        events_a = metrics_a.get('events_detected', {})
        events_b = metrics_b.get('events_detected', {})
        comparison['events_detected'] = {
            'video_a': events_a,
            'video_b': events_b
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
