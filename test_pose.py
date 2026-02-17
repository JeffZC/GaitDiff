"""Quick test script for pose detection"""
from gaitdiff.core.pose import PoseDetector
from gaitdiff.core.video import VideoReader
import cv2
import numpy as np

print('Creating detector...')
pd = PoseDetector()

# Test with actual video
video_path = "test_videos/IMG_7290.MOV"
print(f'\nTesting with video: {video_path}')

vr = VideoReader(video_path)
print(f'Video: {vr.frame_count} frames, {vr.fps} fps, {vr.width}x{vr.height}')

# Read middle frame
frame_idx = vr.frame_count // 2
frame = vr.read_frame(frame_idx)
print(f'Frame {frame_idx} shape: {frame.shape}')

print('\nRunning pose detection...')
result = pd.detect(frame)

if result and result.pose_landmarks and len(result.pose_landmarks) > 0:
    print(f'✅ Pose detected! Found {len(result.pose_landmarks)} person(s)')
    print(f'   First person has {len(result.pose_landmarks[0])} landmarks')
    
    # Draw and save
    annotated = pd.draw_landmarks(frame, result)
    cv2.imwrite('test_pose_output.jpg', annotated)
    print(f'   Saved annotated frame to test_pose_output.jpg')
else:
    print('❌ No pose detected in this frame')
    print('   Trying different frames...')
    
    # Try a few more frames
    for test_idx in [0, vr.frame_count // 4, vr.frame_count * 3 // 4]:
        frame = vr.read_frame(test_idx)
        result = pd.detect(frame)
        if result and result.pose_landmarks and len(result.pose_landmarks) > 0:
            print(f'   ✅ Found pose at frame {test_idx}')
            annotated = pd.draw_landmarks(frame, result)
            cv2.imwrite('test_pose_output.jpg', annotated)
            print(f'   Saved annotated frame to test_pose_output.jpg')
            break
    else:
        print('   No poses found in any test frames')

vr.release()
pd.release()
print('\nTest complete!')
