#!/usr/bin/env python3
"""
Example: Using GaitDiff Core API Programmatically

This example demonstrates how to use the GaitDiff core functionality
without the GUI for batch processing or integration into other applications.
"""

from gaitdiff.core.analyzer import GaitAnalyzer
from pathlib import Path


def example_single_video_analysis():
    """Analyze a single video"""
    print("Example 1: Single Video Analysis")
    print("-" * 50)
    
    analyzer = GaitAnalyzer(runs_dir="runs")
    
    # Analyze video
    video_path = "path/to/gait_video.mp4"
    results = analyzer.analyze_video(video_path, num_samples=30)
    
    # Print ROM results
    print(f"\nAnalyzed: {video_path}")
    print("\nRange of Motion (ROM):")
    for joint, rom in results['rom'].items():
        print(f"  {joint:15} - Range: {rom['range']:6.2f}°  Mean: {rom['mean']:6.2f}°")
    
    # Save results
    results_path = analyzer.save_results(results)
    print(f"\nResults saved to: {results_path}")
    
    analyzer.release()


def example_video_comparison():
    """Compare two videos"""
    print("\nExample 2: Video Comparison")
    print("-" * 50)
    
    analyzer = GaitAnalyzer(runs_dir="runs")
    
    # Compare two videos
    video_a = "path/to/baseline_gait.mp4"
    video_b = "path/to/treatment_gait.mp4"
    
    results = analyzer.analyze_comparison(video_a, video_b, num_samples=30)
    
    # Print comparison
    print(f"\nComparing:")
    print(f"  Video A: {video_a}")
    print(f"  Video B: {video_b}")
    print("\nComparison Results:")
    
    for joint, comparison in results['comparison'].items():
        print(f"\n{joint.replace('_', ' ').title()}:")
        print(f"  Video A ROM: {comparison['video_a_range']:6.2f}°")
        print(f"  Video B ROM: {comparison['video_b_range']:6.2f}°")
        print(f"  Difference:  {comparison['range_diff']:+6.2f}°")
    
    # Save results
    results_path = analyzer.save_results(results)
    print(f"\nResults saved to: {results_path}")
    
    analyzer.release()


def example_batch_processing():
    """Process multiple videos in batch"""
    print("\nExample 3: Batch Processing")
    print("-" * 50)
    
    analyzer = GaitAnalyzer(runs_dir="runs/batch")
    
    # List of videos to process
    video_files = [
        "path/to/patient1_gait.mp4",
        "path/to/patient2_gait.mp4",
        "path/to/patient3_gait.mp4",
    ]
    
    all_results = []
    
    for video_path in video_files:
        print(f"\nProcessing: {video_path}")
        
        try:
            results = analyzer.analyze_video(video_path, num_samples=20)
            all_results.append(results)
            
            # Print summary
            print(f"  ✓ Completed")
            for joint, rom in results['rom'].items():
                print(f"    {joint}: {rom['range']:.1f}°")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Save combined results
    if all_results:
        combined_results = {
            'batch_analysis': True,
            'videos': all_results,
            'summary': {
                'total_videos': len(all_results),
                'successful': len([r for r in all_results if r])
            }
        }
        results_path = analyzer.save_results(combined_results, run_id="batch_analysis")
        print(f"\n\nBatch results saved to: {results_path}")
    
    analyzer.release()


def example_custom_analysis():
    """Custom analysis with specific parameters"""
    print("\nExample 4: Custom Analysis")
    print("-" * 50)
    
    from gaitdiff.core.video import VideoReader
    from gaitdiff.core.pose import PoseDetector, extract_joint_angles, compute_rom
    
    video_path = "path/to/video.mp4"
    
    # Custom frame sampling
    with VideoReader(video_path) as reader:
        print(f"\nVideo info:")
        print(f"  Frames: {reader.frame_count}")
        print(f"  FPS: {reader.fps}")
        print(f"  Size: {reader.width}x{reader.height}")
        
        # Sample specific frame range
        start_frame = 100
        end_frame = 200
        step = 5
        
        detector = PoseDetector()
        angle_history = {'left_knee': [], 'right_knee': [], 'left_hip': [], 'right_hip': []}
        
        print(f"\nAnalyzing frames {start_frame} to {end_frame} (step={step})...")
        
        for frame_num in range(start_frame, end_frame, step):
            frame = reader.read_frame(frame_num)
            if frame is not None:
                pose_results = detector.detect(frame)
                angles = extract_joint_angles(pose_results)
                
                if angles:
                    for joint, angle in angles.items():
                        angle_history[joint].append(angle)
        
        # Compute ROM
        print("\nCustom ROM Analysis:")
        for joint, angles in angle_history.items():
            if angles:
                rom = compute_rom(angles)
                print(f"  {joint:15} - Range: {rom['range']:6.2f}°")
        
        detector.release()


if __name__ == "__main__":
    print("=" * 50)
    print("GaitDiff Core API Examples")
    print("=" * 50)
    print("\nNote: Update video paths before running")
    print("\nExamples:")
    
    # Uncomment the examples you want to run:
    
    # example_single_video_analysis()
    # example_video_comparison()
    # example_batch_processing()
    # example_custom_analysis()
    
    print("\n✓ See code for usage examples")
