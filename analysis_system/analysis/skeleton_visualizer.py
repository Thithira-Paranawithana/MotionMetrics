"""Full body skeleton visualization from 3D keypoints."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib
from typing import Dict, List, Tuple, Callable
from utils.constants import KEYPOINT_MAPPING

# Force matplotlib to use HTML writer
matplotlib.rcParams['animation.html'] = 'html5'


class SkeletonVisualizer:
    """Create full body skeleton visualization from 3D keypoints."""

    def __init__(self, data_loader):
        """Initialize with data loader."""
        self.data_loader = data_loader

        # MediaPipe skeleton connections (MATLAB indices)
        self.skeleton_connections = [
            # Face outline
            (1, 2), (2, 3), (3, 7), (7, 6), (6, 5), (5, 4), (4, 1),

            # Torso
            (12, 13),  # left_shoulder to right_shoulder
            (12, 24),  # left_shoulder to left_hip
            (13, 25),  # right_shoulder to right_hip
            (24, 25),  # left_hip to right_hip

            # Left arm
            (12, 14),  # left_shoulder to left_elbow
            (14, 16),  # left_elbow to left_wrist
            (16, 18),  # left_wrist to left_pinky
            (16, 20),  # left_wrist to left_index
            (16, 22),  # left_wrist to left_thumb

            # Right arm
            (13, 15),  # right_shoulder to right_elbow
            (15, 17),  # right_elbow to right_wrist
            (17, 19),  # right_wrist to right_pinky
            (17, 21),  # right_wrist to right_index
            (17, 23),  # right_wrist to right_thumb

            # Left leg
            (24, 26),  # left_hip to left_knee
            (26, 28),  # left_knee to left_ankle
            (28, 30),  # left_ankle to left_heel
            (28, 32),  # left_ankle to left_foot_index

            # Right leg
            (25, 27),  # right_hip to right_knee
            (27, 29),  # right_knee to right_ankle
            (29, 31),  # right_ankle to right_heel
            (29, 33),  # right_ankle to right_foot_index
        ]

        # Store animation data as instance variables
        self.skeleton_data = None
        self.skeleton_lines = []
        self.keypoint_scatter = None
        self.frame_text = None
        self.time_text = None
        self.fps = 25

    def load_skeleton_data(self, progress_callback: Callable = None) -> np.ndarray:
        """Load and organize 3D keypoint data for skeleton visualization."""
        if progress_callback:
            progress_callback(10, "Loading skeleton data...")

        # Get all available keypoints
        available_keypoints = self.data_loader.get_available_keypoints()

        # Get unique frames
        frames = sorted(self.data_loader.data['Frame'].unique())
        num_frames = len(frames)

        if progress_callback:
            progress_callback(30, f"Processing {num_frames} frames...")

        # Initialize skeleton data array [keypoint_id, xyz, frame_index]
        max_keypoint = 33  # MediaPipe has 33 keypoints
        skeleton_data = np.full((max_keypoint + 1, 3, num_frames), np.nan)

        # Create frame index mapping
        frame_to_idx = {frame: idx for idx, frame in enumerate(frames)}

        # Fill skeleton data
        for _, row in self.data_loader.data.iterrows():
            kp_id = int(row['Keypoint'])
            frame_idx = frame_to_idx[row['Frame']]
            skeleton_data[kp_id, 0, frame_idx] = row['X_mm']
            skeleton_data[kp_id, 1, frame_idx] = row['Y_mm']
            skeleton_data[kp_id, 2, frame_idx] = row['Z_mm']

        if progress_callback:
            progress_callback(50, "Skeleton data loaded successfully")

        return skeleton_data, frames

    def update_skeleton_frame(self, frame_idx):
        """Update skeleton for animation frame."""
        if frame_idx >= self.skeleton_data.shape[2]:
            return self.skeleton_lines + [self.keypoint_scatter, self.frame_text, self.time_text]

        current_time = (frame_idx + 1) / self.fps

        # Update text displays
        self.frame_text.set_text(f'Frame: {frame_idx + 1}')
        self.time_text.set_text(f'Time: {current_time:.2f}s')

        # Clear previous skeleton lines
        for line in self.skeleton_lines:
            line.set_data([], [])
            line.set_3d_properties([])

        # Get current frame keypoints
        current_keypoints = self.skeleton_data[:, :, frame_idx]

        # Update skeleton connections
        line_idx = 0
        for connection in self.skeleton_connections:
            if line_idx >= len(self.skeleton_lines):
                break

            kp1_id, kp2_id = connection

            # Check if both keypoints exist and are valid
            if (kp1_id < current_keypoints.shape[0] and kp2_id < current_keypoints.shape[0]):
                p1 = current_keypoints[kp1_id, :]
                p2 = current_keypoints[kp2_id, :]

                if not (np.any(np.isnan(p1)) or np.any(np.isnan(p2))):
                    # Update line
                    self.skeleton_lines[line_idx].set_data([p1[0], p2[0]], [p1[1], p2[1]])
                    self.skeleton_lines[line_idx].set_3d_properties([p1[2], p2[2]])

            line_idx += 1

        # Update keypoint scatter
        valid_points = ~np.isnan(current_keypoints[:, 0])
        if np.any(valid_points):
            valid_keypoints = current_keypoints[valid_points, :]
            self.keypoint_scatter._offsets3d = (valid_keypoints[:, 0],
                                                valid_keypoints[:, 1],
                                                valid_keypoints[:, 2])
        else:
            self.keypoint_scatter._offsets3d = ([], [], [])

        return self.skeleton_lines + [self.keypoint_scatter, self.frame_text, self.time_text]

    def create_full_body_skeleton(self, fps: int = 25, progress_callback: Callable = None) -> Dict:
        """Create full body skeleton visualization."""
        self.fps = fps

        if progress_callback:
            progress_callback(5, "Starting skeleton visualization...")

        # Load skeleton data
        self.skeleton_data, frames = self.load_skeleton_data(progress_callback)

        if progress_callback:
            progress_callback(60, "Creating 3D visualization...")

            # Create 3D figure
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')

            # EXTREME ZOOM: Calculate skeleton bounds
            all_points = self.skeleton_data.reshape(-1, self.skeleton_data.shape[2])
            valid_points = all_points[~np.isnan(all_points[:, 0]), :]

            if len(valid_points) > 0:
                # Get skeleton center
                skeleton_center_x = np.nanmean(valid_points[:, 0])
                skeleton_center_y = np.nanmean(valid_points[:, 1])
                skeleton_center_z = np.nanmean(valid_points[:, 2])

                # EXTREME ZOOM: Use very tiny range (skeleton fills entire view)
                extreme_zoom = 1000  # Try 150, 100, or even 50mm for massive zoom!

                x_range = [skeleton_center_x - extreme_zoom, skeleton_center_x + extreme_zoom]
                y_range = [skeleton_center_y - extreme_zoom, skeleton_center_y + extreme_zoom]
                z_range = [-500, 1500]

                print(f"EXTREME ZOOM: Using {extreme_zoom * 2}mm total range")
                print(f"Skeleton center: ({skeleton_center_x:.1f}, {skeleton_center_y:.1f}, {skeleton_center_z:.1f})")

            else:
                # Tiny fallback range
                extreme_zoom = 1000
                x_range = [-extreme_zoom, extreme_zoom]
                y_range = [-extreme_zoom, extreme_zoom]
                z_range = [-extreme_zoom, extreme_zoom]

            # EXTREME ZOOM: Set ultra-tiny axis limits
            ax.set_xlim(x_range)
            ax.set_ylim(y_range)
            ax.set_zlim(z_range)

            # Keep equal aspect ratio
            ax.set_box_aspect((1, 1, 1))

            ax.view_init(elev=7, azim=-50)

            # Initialize skeleton lines with NORMAL thickness (not thick)
            self.skeleton_lines = []
            colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

            for i, connection in enumerate(self.skeleton_connections):
                color = colors[i % len(colors)]
                line, = ax.plot([], [], [], color=color, linewidth=2, alpha=0.8)  # Normal thickness
                self.skeleton_lines.append(line)

            # Normal-sized keypoints
            self.keypoint_scatter = ax.scatter([], [], [], c='red', s=80, alpha=0.8)

        # Text displays
        self.frame_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        self.time_text = ax.text2D(0.02, 0.88, '', transform=ax.transAxes, fontsize=12,
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))

        # Set labels and title
        ax.set_xlabel('X (mm)', fontsize=12)
        ax.set_ylabel('Y (mm)', fontsize=12)
        ax.set_zlabel('Z (mm)', fontsize=12)
        ax.set_title('Full Body Skeleton Visualization', fontsize=16, fontweight='bold')

        if progress_callback:
            progress_callback(80, "Creating animation...")

        # Create animation
        num_frames = self.skeleton_data.shape[2]
        ani = animation.FuncAnimation(fig, self.update_skeleton_frame, frames=num_frames,
                                      interval=1000 / fps, blit=False, repeat=True)

        results = {
            'animation': ani,
            'figure': fig,
            'num_frames': num_frames
        }

        if progress_callback:
            progress_callback(90, "Saving HTML file...")

        # Save HTML file
        html_file = 'full_body_skeleton_visualization.html'
        try:
            ani.save(html_file, writer='html', fps=fps)
            results['html_file'] = html_file
            print(f"✅ HTML skeleton animation saved: {html_file}")
        except Exception as e:
            print(f"⚠ Could not save HTML: {e}")

        if progress_callback:
            progress_callback(100, "Skeleton visualization completed!")

        # Show the interactive matplotlib window
        plt.show()

        return results
