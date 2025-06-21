"""Constants and configurations for biomechanical analysis."""

# MediaPipe keypoint mapping (MATLAB indices)
KEYPOINT_MAPPING = {
    'nose': 1,
    'left_eye': 2,
    'right_eye': 5,
    'left_shoulder': 12,
    'right_shoulder': 13,
    'left_elbow': 14,
    'right_elbow': 15,
    'left_wrist': 16,
    'right_wrist': 17,
    'left_hip': 24,
    'right_hip': 25,
    'left_knee': 26,
    'right_knee': 27,
    'left_ankle': 28,
    'right_ankle': 29,
    'left_heel': 30,
    'right_heel': 31
}

# Keypoint pairs for analysis
KEYPOINT_PAIRS = {
    'Hip': ('left_hip', 'right_hip'),
    'Wrist': ('left_wrist', 'right_wrist'),
    'Ankle': ('left_ankle', 'right_ankle'),
    'Elbow': ('left_elbow', 'right_elbow'),
    'Shoulder': ('left_shoulder', 'right_shoulder'),
    'Knee': ('left_knee', 'right_knee'),
    'Eye': ('left_eye', 'right_eye'),
    'Heel': ('left_heel', 'right_heel')
}

# Joint angle definitions (3-point angles)
JOINT_ANGLES = {
    'Left Knee': ('left_hip', 'left_knee', 'left_ankle'),
    'Right Knee': ('right_hip', 'right_knee', 'right_ankle'),
    'Left Elbow': ('left_shoulder', 'left_elbow', 'left_wrist'),
    'Right Elbow': ('right_shoulder', 'right_elbow', 'right_wrist'),
    'Left Hip': ('left_shoulder', 'left_hip', 'left_knee'),
    'Right Hip': ('right_shoulder', 'right_hip', 'right_knee'),
    'Left Shoulder': ('left_hip', 'left_shoulder', 'left_elbow'),
    'Right Shoulder': ('right_hip', 'right_shoulder', 'right_elbow')
}

# Colors for plotting
COLORS = {
    'left': '#2E86AB',   # Blue
    'right': '#A23B72',  # Red/Pink
    'joint': '#F18F01',  # Orange
    'background': '#C73E1D'  # Dark red
}

# Default FPS
DEFAULT_FPS = 40


# Full keypoint mapping for skeleton visualization
FULL_KEYPOINT_MAPPING = {
    'nose': 1, 'left_eye_inner': 2, 'left_eye': 3, 'left_eye_outer': 4,
    'right_eye_inner': 5, 'right_eye': 6, 'right_eye_outer': 7, 'left_ear': 8,
    'right_ear': 9, 'mouth_left': 10, 'mouth_right': 11,
    'left_shoulder': 12, 'right_shoulder': 13, 'left_elbow': 14, 'right_elbow': 15,
    'left_wrist': 16, 'right_wrist': 17, 'left_pinky': 18, 'right_pinky': 19,
    'left_index': 20, 'right_index': 21, 'left_thumb': 22, 'right_thumb': 23,
    'left_hip': 24, 'right_hip': 25, 'left_knee': 26, 'right_knee': 27,
    'left_ankle': 28, 'right_ankle': 29, 'left_heel': 30, 'right_heel': 31,
    'left_foot_index': 32, 'right_foot_index': 33
}
