"""
Pose Editor Module - PySide6 converted version
Integrated with GaitDiff for shared state management
"""

from .pose_format_utils import (
    SUPPORTED_FORMATS,
    MEDIAPIPE33_TO_RR21,
    BODY25_TO_RR21,
    detect_pose_format,
    convert_to_rr21,
    load_pose_data,
    save_pose_data,
    create_empty_pose_data,
    get_keypoint_connections,
    process_mediapipe_to_rr21,
)

from .mediapipe_utils import (
    get_pose_landmarks_from_frame,
    process_video_with_mediapipe,
    get_frame_with_pose_overlay,
)

from .plot_utils import (
    KeypointPlot,
    create_plot_widget,
    calculate_ankle_angle,
)

from .shared_state import (
    SharedPoseState,
    get_shared_state,
)

from .editor_window import (
    PoseEditorWindow,
)

__all__ = [
    # Format utilities
    'SUPPORTED_FORMATS',
    'MEDIAPIPE33_TO_RR21', 
    'BODY25_TO_RR21',
    'detect_pose_format',
    'convert_to_rr21',
    'load_pose_data',
    'save_pose_data',
    'create_empty_pose_data',
    'get_keypoint_connections',
    'process_mediapipe_to_rr21',
    # MediaPipe utilities
    'get_pose_landmarks_from_frame',
    'process_video_with_mediapipe',
    'get_frame_with_pose_overlay',
    # Plot utilities
    'KeypointPlot',
    'create_plot_widget',
    'calculate_ankle_angle',
    # Shared state
    'SharedPoseState',
    'get_shared_state',
    # Editor window
    'PoseEditorWindow',
]
