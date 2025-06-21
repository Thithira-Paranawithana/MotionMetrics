# """Joint angle calculation utilities."""
# """Joint angle calculation utilities."""
#
# import numpy as np
# import pandas as pd
# from typing import Dict, List, Tuple
#
# # FIXED: Change relative imports to absolute imports
# from utils.constants import JOINT_ANGLES, KEYPOINT_MAPPING
#
#
#
# class JointAngleCalculator:
#     """Calculate joint angles from 3D keypoint data."""
#
#     def __init__(self, data_loader):
#         """Initialize with data loader."""
#         self.data_loader = data_loader
#
#     def calculate_angle_3points(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
#         """Calculate angle at p2 formed by p1-p2-p3."""
#         try:
#             # Vectors from p2 to p1 and p2 to p3
#             v1 = p1 - p2
#             v2 = p3 - p2
#
#             # Calculate angle using dot product
#             cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
#
#             # Clamp to valid range to avoid numerical errors
#             cos_angle = np.clip(cos_angle, -1.0, 1.0)
#
#             # Convert to degrees
#             angle = np.degrees(np.arccos(cos_angle))
#
#             return angle
#
#         except (ValueError, ZeroDivisionError):
#             return np.nan
#
#     def calculate_joint_angle_timeseries(self, joint_name: str) -> pd.DataFrame:
#         """Calculate joint angle time series for a specific joint."""
#         if joint_name not in JOINT_ANGLES:
#             raise ValueError(f"Unknown joint: {joint_name}")
#
#         p1_name, p2_name, p3_name = JOINT_ANGLES[joint_name]
#
#         # Get keypoint data
#         p1_data = self.data_loader.get_keypoint_data(p1_name)
#         p2_data = self.data_loader.get_keypoint_data(p2_name)
#         p3_data = self.data_loader.get_keypoint_data(p3_name)
#
#         if any(data is None or data.empty for data in [p1_data, p2_data, p3_data]):
#             raise ValueError(f"Missing data for joint {joint_name}")
#
#         # Merge data on frame
#         merged = pd.merge(p1_data[['Frame', 'Time', 'X_mm', 'Y_mm', 'Z_mm']],
#                           p2_data[['Frame', 'X_mm', 'Y_mm', 'Z_mm']],
#                           on='Frame', suffixes=('_p1', '_p2'))
#         merged = pd.merge(merged, p3_data[['Frame', 'X_mm', 'Y_mm', 'Z_mm']],
#                           on='Frame')
#         merged.rename(columns={'X_mm': 'X_mm_p3', 'Y_mm': 'Y_mm_p3', 'Z_mm': 'Z_mm_p3'}, inplace=True)
#
#         # Calculate angles
#         angles = []
#         for _, row in merged.iterrows():
#             p1 = np.array([row['X_mm_p1'], row['Y_mm_p1'], row['Z_mm_p1']])
#             p2 = np.array([row['X_mm_p2'], row['Y_mm_p2'], row['Z_mm_p2']])
#             p3 = np.array([row['X_mm_p3'], row['Y_mm_p3'], row['Z_mm_p3']])
#
#             angle = self.calculate_angle_3points(p1, p2, p3)
#             angles.append(angle)
#
#         merged['Angle'] = angles
#         merged['Joint'] = joint_name
#
#         return merged[['Frame', 'Time', 'Angle', 'Joint']]
#
#     def calculate_all_joint_angles(self) -> Dict[str, pd.DataFrame]:
#         """Calculate all available joint angles."""
#         joint_data = {}
#
#         for joint_name in JOINT_ANGLES.keys():
#             try:
#                 joint_data[joint_name] = self.calculate_joint_angle_timeseries(joint_name)
#                 print(f"✓ Calculated {joint_name} angles")
#             except Exception as e:
#                 print(f"✗ Failed to calculate {joint_name}: {str(e)}")
#
#         return joint_data
#
#     def calculate_velocity_acceleration(self, position_data: pd.DataFrame) -> pd.DataFrame:
#         """Calculate velocity and acceleration from position data."""
#         result = position_data.copy()
#
#         # Calculate velocity (first derivative)
#         dt = np.diff(result['Time'])
#         dx = np.diff(result['X_mm'])
#         dy = np.diff(result['Y_mm'])
#         dz = np.diff(result['Z_mm'])
#
#         velocity_x = np.concatenate([[np.nan], dx / dt])
#         velocity_y = np.concatenate([[np.nan], dy / dt])
#         velocity_z = np.concatenate([[np.nan], dz / dt])
#
#         result['Velocity_X'] = velocity_x
#         result['Velocity_Y'] = velocity_y
#         result['Velocity_Z'] = velocity_z
#         result['Velocity_Magnitude'] = np.sqrt(velocity_x ** 2 + velocity_y ** 2 + velocity_z ** 2)
#
#         # Calculate acceleration (second derivative)
#         dvx = np.diff(velocity_x[1:])  # Skip first NaN
#         dvy = np.diff(velocity_y[1:])
#         dvz = np.diff(velocity_z[1:])
#         dt_acc = dt[1:]
#
#         accel_x = np.concatenate([[np.nan, np.nan], dvx / dt_acc])
#         accel_y = np.concatenate([[np.nan, np.nan], dvy / dt_acc])
#         accel_z = np.concatenate([[np.nan, np.nan], dvz / dt_acc])
#
#         result['Acceleration_X'] = accel_x
#         result['Acceleration_Y'] = accel_y
#         result['Acceleration_Z'] = accel_z
#         result['Acceleration_Magnitude'] = np.sqrt(accel_x ** 2 + accel_y ** 2 + accel_z ** 2)
#
#         return result



##########################

"""Joint angle calculation utilities with visualization support."""
"""Joint angle calculation utilities with visualization support - FIXED VERSION."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib
from typing import Dict, List, Tuple
from utils.constants import JOINT_ANGLES, KEYPOINT_MAPPING

# Force matplotlib to use HTML writer for interactive controls
matplotlib.rcParams['animation.html'] = 'html5'


class JointAngleCalculator:
    """Calculate joint angles from 3D keypoint data with visualization."""

    def __init__(self, data_loader):
        """Initialize with data loader."""
        self.data_loader = data_loader
        # Store animation data as instance variables to avoid pickle issues
        self.viz_data = None
        self.line1 = None
        self.line2 = None
        self.joint_point = None
        self.angle_text = None
        self.frame_text = None
        self.time_text = None
        self.fps = 40

    def calculate_angle_3points(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle at p2 formed by p1-p2-p3."""
        try:
            v1 = p1 - p2
            v2 = p3 - p2
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            return angle
        except (ValueError, ZeroDivisionError):
            return np.nan

    def calculate_joint_angle_timeseries(self, joint_name: str, progress_callback=None) -> pd.DataFrame:
        """Calculate joint angle time series for a specific joint with progress tracking."""
        if joint_name not in JOINT_ANGLES:
            raise ValueError(f"Unknown joint: {joint_name}")

        if progress_callback:
            progress_callback(10, f"Loading keypoint data for {joint_name}...")

        p1_name, p2_name, p3_name = JOINT_ANGLES[joint_name]

        # Get keypoint data
        p1_data = self.data_loader.get_keypoint_data(p1_name)
        p2_data = self.data_loader.get_keypoint_data(p2_name)
        p3_data = self.data_loader.get_keypoint_data(p3_name)

        if any(data is None or data.empty for data in [p1_data, p2_data, p3_data]):
            raise ValueError(f"Missing data for joint {joint_name}")

        if progress_callback:
            progress_callback(30, f"Merging keypoint data...")

        # Merge data on frame
        merged = pd.merge(p1_data[['Frame', 'Time', 'X_mm', 'Y_mm', 'Z_mm']],
                          p2_data[['Frame', 'X_mm', 'Y_mm', 'Z_mm']],
                          on='Frame', suffixes=('_p1', '_p2'))
        merged = pd.merge(merged, p3_data[['Frame', 'X_mm', 'Y_mm', 'Z_mm']],
                          on='Frame')
        merged.rename(columns={'X_mm': 'X_mm_p3', 'Y_mm': 'Y_mm_p3', 'Z_mm': 'Z_mm_p3'}, inplace=True)

        if progress_callback:
            progress_callback(50, f"Calculating joint angles...")

        # Calculate angles with progress
        angles = []
        total_rows = len(merged)

        for idx, (_, row) in enumerate(merged.iterrows()):
            p1 = np.array([row['X_mm_p1'], row['Y_mm_p1'], row['Z_mm_p1']])
            p2 = np.array([row['X_mm_p2'], row['Y_mm_p2'], row['Z_mm_p2']])
            p3 = np.array([row['X_mm_p3'], row['Y_mm_p3'], row['Z_mm_p3']])

            angle = self.calculate_angle_3points(p1, p2, p3)
            angles.append(angle)

            # Update progress every 10% of rows
            if progress_callback and idx % max(1, total_rows // 10) == 0:
                progress = 50 + int(40 * idx / total_rows)
                progress_callback(progress, f"Calculating angles... {idx}/{total_rows}")

        merged['Angle'] = angles
        merged['Joint'] = joint_name

        if progress_callback:
            progress_callback(90, f"Finalizing angle calculations...")

        return merged[['Frame', 'Time', 'Angle', 'Joint']]

    def update_animation_frame(self, frame_idx):
        """FIXED: Instance method to avoid pickle issues."""
        if frame_idx >= len(self.viz_data):
            return self.line1, self.line2, self.joint_point, self.angle_text, self.frame_text, self.time_text

        row = self.viz_data.iloc[frame_idx]

        # Get current points
        p1 = np.array([row['X_mm_p1'], row['Y_mm_p1'], row['Z_mm_p1']])
        p2 = np.array([row['X_mm_p2'], row['Y_mm_p2'], row['Z_mm_p2']])
        p3 = np.array([row['X_mm_p3'], row['Y_mm_p3'], row['Z_mm_p3']])

        # Update text displays
        if np.isnan(row['Angle']):
            self.angle_text.set_text('Angle: N/A (Missing Data)')
        else:
            self.angle_text.set_text(f'Angle: {row["Angle"]:.1f}°')

        self.frame_text.set_text(f'Frame: {row["Frame"]}')
        self.time_text.set_text(f'Time: {row["Time"]:.2f}s')

        # Update lines if data is valid
        if not (np.any(np.isnan(p1)) or np.any(np.isnan(p2)) or np.any(np.isnan(p3))):
            # Line from p1 to p2
            self.line1.set_data([p1[0], p2[0]], [p1[1], p2[1]])
            self.line1.set_3d_properties([p1[2], p2[2]])

            # Line from p2 to p3
            self.line2.set_data([p2[0], p3[0]], [p2[1], p3[1]])
            self.line2.set_3d_properties([p2[2], p3[2]])

            # Joint center point
            self.joint_point.set_data([p2[0]], [p2[1]])
            self.joint_point.set_3d_properties([p2[2]])
        else:
            # Clear lines if data is invalid
            self.line1.set_data([], [])
            self.line1.set_3d_properties([])
            self.line2.set_data([], [])
            self.line2.set_3d_properties([])
            self.joint_point.set_data([], [])
            self.joint_point.set_3d_properties([])

        return self.line1, self.line2, self.joint_point, self.angle_text, self.frame_text, self.time_text

    def create_joint_angle_visualization(self, joint_name: str, fps: int = 40, save_files: bool = True,
                                         progress_callback=None) -> Dict:
        """Create interactive joint angle visualization - FIXED VERSION."""

        self.fps = fps

        if progress_callback:
            progress_callback(5, f"Starting {joint_name} visualization...")

        # Get joint angle data
        joint_data = self.calculate_joint_angle_timeseries(joint_name, progress_callback)

        if joint_data.empty:
            raise ValueError(f"No data available for joint: {joint_name}")

        if progress_callback:
            progress_callback(95, "Creating 3D visualization...")

        # Get the three keypoints for this joint
        p1_name, p2_name, p3_name = JOINT_ANGLES[joint_name]

        # Get 3D position data for visualization
        p1_data = self.data_loader.get_keypoint_data(p1_name)
        p2_data = self.data_loader.get_keypoint_data(p2_name)
        p3_data = self.data_loader.get_keypoint_data(p3_name)

        # Merge position data
        pos_data = pd.merge(p1_data[['Frame', 'X_mm', 'Y_mm', 'Z_mm']],
                            p2_data[['Frame', 'X_mm', 'Y_mm', 'Z_mm']],
                            on='Frame', suffixes=('_p1', '_p2'))
        pos_data = pd.merge(pos_data, p3_data[['Frame', 'X_mm', 'Y_mm', 'Z_mm']],
                            on='Frame')
        pos_data.rename(columns={'X_mm': 'X_mm_p3', 'Y_mm': 'Y_mm_p3', 'Z_mm': 'Z_mm_p3'}, inplace=True)

        # Merge with angle data
        self.viz_data = pd.merge(pos_data, joint_data[['Frame', 'Time', 'Angle']], on='Frame')

        # Create 3D visualization
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # Calculate plot bounds
        all_x = np.concatenate([self.viz_data['X_mm_p1'], self.viz_data['X_mm_p2'], self.viz_data['X_mm_p3']])
        all_y = np.concatenate([self.viz_data['Y_mm_p1'], self.viz_data['Y_mm_p2'], self.viz_data['Y_mm_p3']])
        all_z = np.concatenate([self.viz_data['Z_mm_p1'], self.viz_data['Z_mm_p2'], self.viz_data['Z_mm_p3']])

        margin = 200  # mm
        x_range = [np.nanmin(all_x) - margin, np.nanmax(all_x) + margin]
        y_range = [np.nanmin(all_y) - margin, np.nanmax(all_y) + margin]
        z_range = [np.nanmin(all_z) - margin, np.nanmax(all_z) + margin]

        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_zlim(z_range)

        # Initialize plot elements as instance variables
        self.line1, = ax.plot([], [], [], 'ro-', linewidth=3, markersize=8, label=f'{p1_name} → {p2_name}')
        self.line2, = ax.plot([], [], [], 'bo-', linewidth=3, markersize=8, label=f'{p2_name} → {p3_name}')
        self.joint_point, = ax.plot([], [], [], 'go', markersize=12, label=f'{p2_name} (Joint Center)')

        # Text displays as instance variables
        self.angle_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=14,
                                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
        self.frame_text = ax.text2D(0.02, 0.88, '', transform=ax.transAxes, fontsize=12,
                                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        self.time_text = ax.text2D(0.02, 0.81, '', transform=ax.transAxes, fontsize=12,
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))

        # Set labels and title
        ax.set_xlabel('X (mm)', fontsize=12)
        ax.set_ylabel('Y (mm)', fontsize=12)
        ax.set_zlabel('Z (mm)', fontsize=12)
        ax.set_title(f'{joint_name} Joint Angle Visualization', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')

        # Create animation using instance method
        ani = animation.FuncAnimation(fig, self.update_animation_frame, frames=len(self.viz_data),
                                      interval=1000 / fps, blit=False, repeat=True)

        results = {
            'joint_name': joint_name,
            'animation': ani,
            'figure': fig,
            'data': joint_data,
            'visualization_data': self.viz_data
        }

        # Save files if requested
        if save_files:
            if progress_callback:
                progress_callback(98, "Saving visualization files...")

            # 1. Save interactive HTML (web-based controls)
            html_file = f'{joint_name}_joint_angle_visualization.html'
            ani.save(html_file, writer='html', fps=fps)
            results['html_file'] = html_file

            # 2. FIXED: Save animation object (not pickle - avoid local function issue)
            # Instead, save the data needed to recreate the visualization
            # import json
            # viz_config = {
            #     'joint_name': joint_name,
            #     'fps': fps,
            #     'p1_name': p1_name,
            #     'p2_name': p2_name,
            #     'p3_name': p3_name,
            #     'x_range': x_range,
            #     'y_range': y_range,
            #     'z_range': z_range
            # }
            #
            # config_file = f'{joint_name}_visualization_config.json'
            # with open(config_file, 'w') as f:
            #     json.dump(viz_config, f, indent=2)
            # results['config_file'] = config_file

            # 3. Save angle data CSV
            csv_file = f'{joint_name}_angles.csv'
            joint_data.to_csv(csv_file, index=False)
            results['csv_file'] = csv_file

            print(f"✅ Joint angle visualization saved:")
            print(f"  - Interactive HTML: {html_file}")
            # print(f"  - Visualization config: {config_file}")
            print(f"  - Angle data CSV: {csv_file}")

        if progress_callback:
            progress_callback(100, f"{joint_name} visualization completed!")

        return results

    def calculate_all_joint_angles(self) -> Dict[str, pd.DataFrame]:
        """Calculate all available joint angles."""
        joint_data = {}

        for joint_name in JOINT_ANGLES.keys():
            try:
                joint_data[joint_name] = self.calculate_joint_angle_timeseries(joint_name)
                print(f"✓ Calculated {joint_name} angles")
            except Exception as e:
                print(f"✗ Failed to calculate {joint_name}: {str(e)}")

        return joint_data

    def calculate_velocity_acceleration(self, position_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate velocity and acceleration from position data."""
        result = position_data.copy()

        # Calculate velocity (first derivative)
        dt = np.diff(result['Time'])
        dx = np.diff(result['X_mm'])
        dy = np.diff(result['Y_mm'])
        dz = np.diff(result['Z_mm'])

        velocity_x = np.concatenate([[np.nan], dx / dt])
        velocity_y = np.concatenate([[np.nan], dy / dt])
        velocity_z = np.concatenate([[np.nan], dz / dt])

        result['Velocity_X'] = velocity_x
        result['Velocity_Y'] = velocity_y
        result['Velocity_Z'] = velocity_z
        result['Velocity_Magnitude'] = np.sqrt(velocity_x ** 2 + velocity_y ** 2 + velocity_z ** 2)

        # Calculate acceleration (second derivative)
        dvx = np.diff(velocity_x[1:])  # Skip first NaN
        dvy = np.diff(velocity_y[1:])
        dvz = np.diff(velocity_z[1:])
        dt_acc = dt[1:]

        accel_x = np.concatenate([[np.nan, np.nan], dvx / dt_acc])
        accel_y = np.concatenate([[np.nan, np.nan], dvy / dt_acc])
        accel_z = np.concatenate([[np.nan, np.nan], dvz / dt_acc])

        result['Acceleration_X'] = accel_x
        result['Acceleration_Y'] = accel_y
        result['Acceleration_Z'] = accel_z
        result['Acceleration_Magnitude'] = np.sqrt(accel_x ** 2 + accel_y ** 2 + accel_z ** 2)

        return result
