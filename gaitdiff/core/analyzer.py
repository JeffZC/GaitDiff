"""Analysis engine for gait comparison"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from .video import VideoReader
from .pose import PoseDetector, extract_joint_angles, compute_rom


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
            }
        }
        
        with VideoReader(video_path) as reader:
            frames = reader.sample_frames(num_samples)
            
            for frame_idx, frame in frames:
                pose_results = self.pose_detector.detect(frame)
                angles = extract_joint_angles(pose_results)
                
                if angles:
                    for joint, angle in angles.items():
                        results['angle_data'][joint].append(angle)
        
        # Compute ROM for each joint
        results['rom'] = {}
        for joint, angles in results['angle_data'].items():
            results['rom'][joint] = compute_rom(angles)
        
        return results
    
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
            'comparison': self._compute_comparison(video_a_results, video_b_results)
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
