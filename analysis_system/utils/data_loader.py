"""Data loading utilities for biomechanical analysis."""
"""Data loading utilities for biomechanical analysis."""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional

# FIXED: Change relative imports to absolute imports
from utils.constants import KEYPOINT_MAPPING, DEFAULT_FPS



class DataLoader:
    """Handles loading and preprocessing of 3D keypoint data."""

    def __init__(self, csv_path: str, fps: float = DEFAULT_FPS):
        """Initialize data loader with CSV file path."""
        self.csv_path = csv_path
        self.fps = fps
        self.data = None
        self.time_data = None

    def load_data(self) -> pd.DataFrame:
        """Load CSV data and add time column."""
        try:
            self.data = pd.read_csv(self.csv_path)

            # Add time column
            self.data['Time'] = (self.data['Frame'] - 1) / self.fps

            print(f"✓ Loaded {len(self.data)} data points")
            print(f"✓ Frame range: {self.data['Frame'].min()} to {self.data['Frame'].max()}")
            print(f"✓ Time range: {self.data['Time'].min():.2f} to {self.data['Time'].max():.2f} seconds")

            return self.data

        except Exception as e:
            raise Exception(f"Error loading CSV file: {str(e)}")

    def get_keypoint_data(self, keypoint_name: str) -> Optional[pd.DataFrame]:
        """Get data for a specific keypoint."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        keypoint_id = KEYPOINT_MAPPING.get(keypoint_name)
        if keypoint_id is None:
            raise ValueError(f"Unknown keypoint: {keypoint_name}")

        keypoint_data = self.data[self.data['Keypoint'] == keypoint_id].copy()
        return keypoint_data.sort_values('Frame') if not keypoint_data.empty else None

    def get_keypoint_pair_data(self, left_keypoint: str, right_keypoint: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get data for a pair of keypoints (left and right)."""
        left_data = self.get_keypoint_data(left_keypoint)
        right_data = self.get_keypoint_data(right_keypoint)

        if left_data is None or right_data is None:
            raise ValueError(f"Missing data for keypoints: {left_keypoint}, {right_keypoint}")

        return left_data, right_data

    def get_available_keypoints(self) -> list:
        """Get list of available keypoints in the data."""
        if self.data is None:
            return []

        available_ids = set(self.data['Keypoint'].unique())
        available_keypoints = []

        for name, keypoint_id in KEYPOINT_MAPPING.items():
            if keypoint_id in available_ids:
                available_keypoints.append(name)

        return sorted(available_keypoints)
