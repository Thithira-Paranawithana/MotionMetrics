"""Movement analysis functions for biomechanical data."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# FIXED: Change relative imports to absolute imports
from utils.constants import COLORS, KEYPOINT_PAIRS



class MovementAnalyzer:
    """Analyze movement patterns from 3D keypoint data."""

    def __init__(self, data_loader):
        """Initialize with data loader."""
        self.data_loader = data_loader

        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def create_position_charts(self, pair_name: str, save_path: str = None) -> str:
        """Create 3-direction position charts for a keypoint pair."""
        if pair_name not in KEYPOINT_PAIRS:
            raise ValueError(f"Unknown keypoint pair: {pair_name}")

        left_keypoint, right_keypoint = KEYPOINT_PAIRS[pair_name]

        try:
            left_data, right_data = self.data_loader.get_keypoint_pair_data(left_keypoint, right_keypoint)
        except ValueError as e:
            raise ValueError(f"Cannot create charts for {pair_name}: {str(e)}")

        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(f'{pair_name} Movement Analysis', fontsize=16, fontweight='bold')

        directions = ['X', 'Y', 'Z']
        direction_labels = ['X Direction (Left-Right)', 'Y Direction (Height)', 'Z Direction (Depth)']

        for i, (direction, label) in enumerate(zip(directions, direction_labels)):
            ax = axes[i]

            # Plot data
            ax.plot(left_data['Time'], left_data[f'{direction}_mm'],
                    color=COLORS['left'], linewidth=2, label=f'Left {pair_name}')
            ax.plot(right_data['Time'], right_data[f'{direction}_mm'],
                    color=COLORS['right'], linewidth=2, label=f'Right {pair_name}')

            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel(f'{direction} Position (mm)')
            ax.set_title(f'{pair_name} Movement in {label}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure with output folder support
        if save_path is None:
            import os
            os.makedirs('output_movements', exist_ok=True)
            save_path = os.path.join('output_movements', f'{pair_name}_Movement_Analysis.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        # plt.close()  # Close figure to free memory

        return save_path

    def create_velocity_charts(self, keypoint_name: str, save_path: str = None) -> str:
        """Create velocity analysis charts for a keypoint."""
        from analysis.joint_angles import JointAngleCalculator

        keypoint_data = self.data_loader.get_keypoint_data(keypoint_name)
        if keypoint_data is None:
            raise ValueError(f"No data available for {keypoint_name}")

        # Calculate velocity and acceleration
        calc = JointAngleCalculator(self.data_loader)
        enhanced_data = calc.calculate_velocity_acceleration(keypoint_data)

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{keypoint_name.title()} Velocity & Acceleration Analysis', fontsize=16, fontweight='bold')

        # Velocity magnitude
        axes[0, 0].plot(enhanced_data['Time'], enhanced_data['Velocity_Magnitude'],
                        color=COLORS['joint'], linewidth=2)
        axes[0, 0].set_title('Velocity Magnitude')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Velocity (mm/s)')
        axes[0, 0].grid(True, alpha=0.3)

        # Acceleration magnitude
        axes[0, 1].plot(enhanced_data['Time'], enhanced_data['Acceleration_Magnitude'],
                        color=COLORS['background'], linewidth=2)
        axes[0, 1].set_title('Acceleration Magnitude')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Acceleration (mm/s²)')
        axes[0, 1].grid(True, alpha=0.3)

        # 3D velocity components
        axes[1, 0].plot(enhanced_data['Time'], enhanced_data['Velocity_X'], label='X', linewidth=2)
        axes[1, 0].plot(enhanced_data['Time'], enhanced_data['Velocity_Y'], label='Y', linewidth=2)
        axes[1, 0].plot(enhanced_data['Time'], enhanced_data['Velocity_Z'], label='Z', linewidth=2)
        axes[1, 0].set_title('Velocity Components')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Velocity (mm/s)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 3D acceleration components
        axes[1, 1].plot(enhanced_data['Time'], enhanced_data['Acceleration_X'], label='X', linewidth=2)
        axes[1, 1].plot(enhanced_data['Time'], enhanced_data['Acceleration_Y'], label='Y', linewidth=2)
        axes[1, 1].plot(enhanced_data['Time'], enhanced_data['Acceleration_Z'], label='Z', linewidth=2)
        axes[1, 1].set_title('Acceleration Components')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Acceleration (mm/s²)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # UPDATED: Save figure with output folder support
        if save_path is None:
            import os
            os.makedirs('output_velocities', exist_ok=True)
            save_path = os.path.join('output_velocities', f'{keypoint_name}_Velocity_Analysis.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.close()  # Close figure to free memory

        return save_path

    def create_joint_angle_charts(self, joint_angles_data: Dict[str, pd.DataFrame], save_path: str = None) -> str:
        """Create joint angle analysis charts."""
        if not joint_angles_data:
            raise ValueError("No joint angle data provided")

        # Create figure with subplots
        n_joints = len(joint_angles_data)
        cols = 2
        rows = (n_joints + 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        fig.suptitle('Joint Angle Analysis', fontsize=16, fontweight='bold')

        if n_joints == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)

        for i, (joint_name, joint_data) in enumerate(joint_angles_data.items()):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]

            ax.plot(joint_data['Time'], joint_data['Angle'],
                    linewidth=2, label=joint_name)
            ax.set_title(f'{joint_name} Angle')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Angle (degrees)')
            ax.grid(True, alpha=0.3)

            # Add statistics
            mean_angle = joint_data['Angle'].mean()
            std_angle = joint_data['Angle'].std()
            ax.axhline(mean_angle, color='red', linestyle='--', alpha=0.7,
                       label=f'Mean: {mean_angle:.1f}°')
            ax.legend()

        # Hide empty subplots
        for i in range(n_joints, rows * cols):
            row = i // cols
            col = i % cols
            if rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)

        plt.tight_layout()

        if save_path is None:
            import os
            os.makedirs('output_joint_angles', exist_ok=True)
            save_path = os.path.join('output_joint_angles', 'Joint_Angle_Analysis.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        # # Save figure
        # if save_path is None:
        #     save_path = 'Joint_Angle_Analysis.png'
        #
        # plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return save_path

    def calculate_movement_statistics(self, pair_name: str) -> Dict:
        """Calculate movement statistics for a keypoint pair."""
        left_keypoint, right_keypoint = KEYPOINT_PAIRS[pair_name]
        left_data, right_data = self.data_loader.get_keypoint_pair_data(left_keypoint, right_keypoint)

        stats = {
            'pair_name': pair_name,
            'duration': left_data['Time'].max() - left_data['Time'].min(),
            'data_points': len(left_data),
            'left_stats': {},
            'right_stats': {}
        }

        for side, data in [('left', left_data), ('right', right_data)]:
            stats[f'{side}_stats'] = {
                'x_range': data['X_mm'].max() - data['X_mm'].min(),
                'y_range': data['Y_mm'].max() - data['Y_mm'].min(),
                'z_range': data['Z_mm'].max() - data['Z_mm'].min(),
                'x_mean': data['X_mm'].mean(),
                'y_mean': data['Y_mm'].mean(),
                'z_mean': data['Z_mm'].mean(),
                'x_std': data['X_mm'].std(),
                'y_std': data['Y_mm'].std(),
                'z_std': data['Z_mm'].std()
            }

        return stats

    def calculate_velocity_statistics(self, keypoint_name: str) -> Dict:
        """Calculate velocity statistics for a keypoint for report inclusion."""
        keypoint_data = self.data_loader.get_keypoint_data(keypoint_name)
        if keypoint_data is None:
            raise ValueError(f"No data available for {keypoint_name}")

        # Calculate velocity and acceleration
        from analysis.joint_angles import JointAngleCalculator
        calc = JointAngleCalculator(self.data_loader)
        enhanced_data = calc.calculate_velocity_acceleration(keypoint_data)

        # Calculate statistics
        stats = {
            'keypoint': keypoint_name,
            'vel_mean': enhanced_data['Velocity_Magnitude'].mean(),
            'vel_max': enhanced_data['Velocity_Magnitude'].max(),
            'vel_std': enhanced_data['Velocity_Magnitude'].std(),
            'acc_mean': enhanced_data['Acceleration_Magnitude'].mean(),
            'acc_max': enhanced_data['Acceleration_Magnitude'].max(),
            'acc_std': enhanced_data['Acceleration_Magnitude'].std(),
            'vel_x_max': enhanced_data['Velocity_X'].mean(),
            'vel_x_peak': enhanced_data['Velocity_X'].max(),
            'vel_x_std': enhanced_data['Velocity_X'].std(),
            'vel_y_max': enhanced_data['Velocity_Y'].mean(),
            'vel_y_peak': enhanced_data['Velocity_Y'].max(),
            'vel_y_std': enhanced_data['Velocity_Y'].std(),
            'vel_z_max': enhanced_data['Velocity_Z'].mean(),
            'vel_z_peak': enhanced_data['Velocity_Z'].max(),
            'vel_z_std': enhanced_data['Velocity_Z'].std()
        }

        return stats

    def calculate_knee_ankle_velocity_statistics(self, keypoint_name: str) -> Dict:
        """Calculate specialized velocity statistics for knee and ankle joints."""
        keypoint_data = self.data_loader.get_keypoint_data(keypoint_name)
        if keypoint_data is None:
            raise ValueError(f"No data available for {keypoint_name}")

        # Calculate velocity and acceleration
        from analysis.joint_angles import JointAngleCalculator
        calc = JointAngleCalculator(self.data_loader)
        enhanced_data = calc.calculate_velocity_acceleration(keypoint_data)

        # Enhanced statistics for knee/ankle based on search results
        stats = {
            'keypoint': keypoint_name,
            'joint_type': 'knee' if 'knee' in keypoint_name else 'ankle',

            # Linear velocity statistics (mm/s)
            'vel_mean': enhanced_data['Velocity_Magnitude'].mean(),
            'vel_max': enhanced_data['Velocity_Magnitude'].max(),
            'vel_std': enhanced_data['Velocity_Magnitude'].std(),
            'vel_peak_percentile_95': enhanced_data['Velocity_Magnitude'].quantile(0.95),

            # Acceleration statistics (mm/s²)
            'acc_mean': enhanced_data['Acceleration_Magnitude'].mean(),
            'acc_max': enhanced_data['Acceleration_Magnitude'].max(),
            'acc_std': enhanced_data['Acceleration_Magnitude'].std(),
            'acc_peak_percentile_95': enhanced_data['Acceleration_Magnitude'].quantile(0.95),

            # Directional velocity components
            'vel_x_mean': enhanced_data['Velocity_X'].mean(),
            'vel_x_max': enhanced_data['Velocity_X'].max(),
            'vel_x_std': enhanced_data['Velocity_X'].std(),
            'vel_y_mean': enhanced_data['Velocity_Y'].mean(),
            'vel_y_max': enhanced_data['Velocity_Y'].max(),
            'vel_y_std': enhanced_data['Velocity_Y'].std(),
            'vel_z_mean': enhanced_data['Velocity_Z'].mean(),
            'vel_z_max': enhanced_data['Velocity_Z'].max(),
            'vel_z_std': enhanced_data['Velocity_Z'].std(),

            # Clinical metrics based on search results
            'peak_velocity_timing': enhanced_data.loc[enhanced_data['Velocity_Magnitude'].idxmax(), 'Time'],
            'movement_efficiency': enhanced_data['Velocity_Magnitude'].mean() / enhanced_data[
                'Velocity_Magnitude'].max(),
            'velocity_variability': enhanced_data['Velocity_Magnitude'].std() / enhanced_data[
                'Velocity_Magnitude'].mean()
        }

        return stats

