
"""Main GUI window for biomechanical analysis."""
from matplotlib import pyplot as plt

"""Main GUI window for biomechanical analysis."""

import sys
import os
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QLabel, QPushButton, QComboBox, QTextEdit, QFileDialog,
                            QProgressBar, QGroupBox, QTabWidget, QMessageBox,
                            QGridLayout, QSpinBox, QDoubleSpinBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont, QPixmap

# FIXED: Change relative imports to absolute imports
from utils.data_loader import DataLoader
from utils.constants import KEYPOINT_PAIRS, JOINT_ANGLES
from analysis.movement_analyzer import MovementAnalyzer
from analysis.joint_angles import JointAngleCalculator
from analysis.report_generator import BiomechanicalReportGenerator
from gui.styles import MAIN_STYLE



class AnalysisWorker(QThread):
    """Worker thread for analysis tasks."""

    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, data_loader, selected_pairs, selected_joints, fps):
        super().__init__()
        self.data_loader = data_loader
        self.selected_pairs = selected_pairs
        self.selected_joints = selected_joints
        self.fps = fps

    def run(self):
        """Run analysis in background thread."""
        try:
            results = {
                'chart_paths': [],
                'movement_stats': {},
                'joint_stats': {},
                'joint_chart_path': None,
                'velocity_stats': {},
                'velocity_chart_paths': []
            }

            total_tasks = len(self.selected_pairs) + (1 if self.selected_joints else 0) + 1
            current_task = 0

            # Movement analysis
            analyzer = MovementAnalyzer(self.data_loader)

            for pair_name in self.selected_pairs:
                self.status.emit(f"Analyzing {pair_name} movement...")

                try:
                    chart_path = analyzer.create_position_charts(pair_name)
                    results['chart_paths'].append(chart_path)

                    stats = analyzer.calculate_movement_statistics(pair_name)
                    results['movement_stats'][pair_name] = stats

                except Exception as e:
                    self.error.emit(f"Error analyzing {pair_name}: {str(e)}")

                current_task += 1
                self.progress.emit(int(100 * current_task / total_tasks))

            # Joint angle analysis
            if self.selected_joints:
                self.status.emit("Calculating joint angles...")

                try:
                    joint_calc = JointAngleCalculator(self.data_loader)
                    joint_data = {}

                    for joint_name in self.selected_joints:
                        joint_timeseries = joint_calc.calculate_joint_angle_timeseries(joint_name)
                        joint_data[joint_name] = joint_timeseries

                        # Calculate statistics
                        results['joint_stats'][joint_name] = {
                            'mean': joint_timeseries['Angle'].mean(),
                            'std': joint_timeseries['Angle'].std(),
                            'min': joint_timeseries['Angle'].min(),
                            'max': joint_timeseries['Angle'].max()
                        }

                    # Create joint angle charts
                    if joint_data:
                        chart_path = analyzer.create_joint_angle_charts(joint_data)
                        results['joint_chart_path'] = chart_path

                except Exception as e:
                    self.error.emit(f"Error calculating joint angles: {str(e)}")

                current_task += 1
                self.progress.emit(int(100 * current_task / total_tasks))

            # Velocity analysis for key keypoints
            key_keypoints = [
                'left_hip', 'right_hip',
                'left_wrist', 'right_wrist',
                'left_knee', 'right_knee',
                'left_ankle', 'right_ankle',
                'left_shoulder', 'right_shoulder',
                'left_elbow', 'right_elbow'
            ]

            total_velocity_tasks = len(key_keypoints)

            for idx, keypoint in enumerate(key_keypoints):
                try:
                    self.status.emit(f"Calculating {keypoint} velocity... ({idx + 1}/{total_velocity_tasks})")

                    # Create velocity chart
                    velocity_chart = analyzer.create_velocity_charts(keypoint)
                    results['velocity_chart_paths'].append(velocity_chart)

                    # Calculate velocity statistics
                    velocity_stats = analyzer.calculate_velocity_statistics(keypoint)
                    results['velocity_stats'][keypoint] = velocity_stats

                    # Update progress
                    velocity_progress = 70 + int(20 * (idx + 1) / total_velocity_tasks)
                    self.progress.emit(velocity_progress)

                except Exception as e:
                    self.error.emit(f"Error analyzing {keypoint} velocity: {str(e)}")




            # Generate report
            self.status.emit("Generating report...")

            try:
                report_gen = BiomechanicalReportGenerator()

                data_stats = {
                    'total_points': len(self.data_loader.data),
                    'duration': self.data_loader.data['Time'].max() - self.data_loader.data['Time'].min(),
                    'fps': self.fps,
                    'available_keypoints': self.data_loader.get_available_keypoints()
                }

                report_path = report_gen.generate_report(
                    data_stats=data_stats,
                    movement_stats=results['movement_stats'],
                    joint_stats=results['joint_stats'],
                    chart_paths=results['chart_paths'],
                    joint_chart_path=results['joint_chart_path'],
                    velocity_stats=results['velocity_stats'],
                    velocity_chart_paths=results['velocity_chart_paths'],
                    csv_file_path=self.data_loader.csv_path
                )

                results['report_path'] = report_path

            except Exception as e:
                self.error.emit(f"Error generating report: {str(e)}")

            current_task += 1
            self.progress.emit(100)

            self.finished.emit(results)

        except Exception as e:
            self.error.emit(f"Analysis failed: {str(e)}")


class BiomechanicalAnalyzerGUI(QMainWindow):
    """Main GUI window for biomechanical analysis."""

    def __init__(self):
        super().__init__()
        self.data_loader = None
        self.worker = None

        self.init_ui()
        self.setStyleSheet(MAIN_STYLE)

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("MotionMetrics - Biomechanical Motion Analyzer v1.0")
        self.setGeometry(100, 100, 1000, 700)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)

        # Title
        title_label = QLabel("MotionMetrics - Biomechanical Motion Analyzer")
        title_label.setObjectName("title")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # File selection section
        file_group = QGroupBox("Data Input")
        file_layout = QVBoxLayout(file_group)

        file_row = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        self.file_button = QPushButton("Select CSV File")
        self.file_button.clicked.connect(self.select_file)

        file_row.addWidget(QLabel("CSV File:"))
        file_row.addWidget(self.file_label, 1)
        file_row.addWidget(self.file_button)
        file_layout.addLayout(file_row)

        # FPS setting
        fps_row = QHBoxLayout()
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(1, 120)
        self.fps_spinbox.setValue(40)
        fps_row.addWidget(QLabel("Video FPS:"))
        fps_row.addWidget(self.fps_spinbox)
        fps_row.addStretch()
        file_layout.addLayout(fps_row)

        main_layout.addWidget(file_group)

        # Joint Visualization Section
        viz_group = QGroupBox("Joint Angle Visualization")
        viz_layout = QVBoxLayout(viz_group)

        viz_label = QLabel("Create Interactive Joint Angle Visualization:")
        viz_label.setObjectName("subtitle")
        viz_layout.addWidget(viz_label)

        # Joint selection for visualization
        viz_row = QHBoxLayout()
        viz_row.addWidget(QLabel("Select Joint:"))

        self.viz_joint_combo = QComboBox()
        self.viz_joint_combo.addItems(list(JOINT_ANGLES.keys()))
        viz_row.addWidget(self.viz_joint_combo)

        self.viz_button = QPushButton("Create Visualization")
        self.viz_button.clicked.connect(self.create_joint_visualization)
        self.viz_button.setEnabled(False)
        viz_row.addWidget(self.viz_button)

        viz_row.addStretch()
        viz_layout.addLayout(viz_row)

        main_layout.addWidget(viz_group)

        # Full Body Skeleton Visualization Section
        skeleton_group = QGroupBox("Full Body Skeleton Visualization")
        skeleton_layout = QVBoxLayout(skeleton_group)

        skeleton_label = QLabel("Create Full Body Skeleton Visualization:")
        skeleton_label.setObjectName("subtitle")
        skeleton_layout.addWidget(skeleton_label)

        # Skeleton visualization controls
        skeleton_row = QHBoxLayout()

        skeleton_row.addWidget(QLabel("FPS:"))
        self.skeleton_fps_spinbox = QSpinBox()
        self.skeleton_fps_spinbox.setRange(10, 60)
        self.skeleton_fps_spinbox.setValue(25)
        skeleton_row.addWidget(self.skeleton_fps_spinbox)

        self.skeleton_button = QPushButton("Create Full Body Skeleton")
        self.skeleton_button.clicked.connect(self.create_skeleton_visualization)
        self.skeleton_button.setEnabled(False)
        skeleton_row.addWidget(self.skeleton_button)

        skeleton_row.addStretch()
        skeleton_layout.addLayout(skeleton_row)

        main_layout.addWidget(skeleton_group)


        # Analysis configuration
        config_group = QGroupBox("Analysis Configuration")
        config_layout = QVBoxLayout(config_group)

        # Movement pairs selection
        pairs_label = QLabel("Select Keypoint Pairs for Movement Analysis:")
        pairs_label.setObjectName("subtitle")
        config_layout.addWidget(pairs_label)

        self.pairs_layout = QGridLayout()
        self.pair_checkboxes = {}

        row, col = 0, 0
        for pair_name in KEYPOINT_PAIRS.keys():
            from PyQt5.QtWidgets import QCheckBox
            checkbox = QCheckBox(pair_name)
            checkbox.setChecked(True)  # Default all selected
            self.pair_checkboxes[pair_name] = checkbox
            self.pairs_layout.addWidget(checkbox, row, col)

            col += 1
            if col >= 3:  # 3 columns
                col = 0
                row += 1

        config_layout.addLayout(self.pairs_layout)

        # Joint angles selection
        joints_label = QLabel("Select Joints for Angle Analysis:")
        joints_label.setObjectName("subtitle")
        config_layout.addWidget(joints_label)

        self.joints_layout = QGridLayout()
        self.joint_checkboxes = {}

        row, col = 0, 0
        for joint_name in JOINT_ANGLES.keys():
            from PyQt5.QtWidgets import QCheckBox
            checkbox = QCheckBox(joint_name)
            checkbox.setChecked(True)  # Default all selected
            self.joint_checkboxes[joint_name] = checkbox
            self.joints_layout.addWidget(checkbox, row, col)

            col += 1
            if col >= 3:  # 3 columns
                col = 0
                row += 1

        config_layout.addLayout(self.joints_layout)
        main_layout.addWidget(config_group)

        # Control buttons
        button_layout = QHBoxLayout()

        self.analyze_button = QPushButton("Start Analysis")
        self.analyze_button.clicked.connect(self.start_analysis)
        self.analyze_button.setEnabled(False)

        self.clear_button = QPushButton("Clear Results")
        self.clear_button.clicked.connect(self.clear_results)

        button_layout.addWidget(self.analyze_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()

        main_layout.addLayout(button_layout)

        # Progress section
        progress_group = QGroupBox("Analysis Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Ready")

        progress_layout.addWidget(self.status_label)
        progress_layout.addWidget(self.progress_bar)

        main_layout.addWidget(progress_group)

        # Results section
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)

        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(150)
        results_layout.addWidget(self.results_text)

        main_layout.addWidget(results_group)

    def select_file(self):
        """Select CSV file for analysis."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "", "CSV Files (*.csv);;All Files (*)"
        )

        if file_path:
            try:
                self.data_loader = DataLoader(file_path, self.fps_spinbox.value())
                self.data_loader.load_data()

                self.file_label.setText(os.path.basename(file_path))
                self.analyze_button.setEnabled(True)
                # self.viz_button.setEnabled(True)
                # self.skeleton_button.setEnabled(True)

                # Update results
                available_keypoints = self.data_loader.get_available_keypoints()
                self.results_text.append(f"✓ File loaded: {os.path.basename(file_path)}")
                self.results_text.append(f"✓ Data points: {len(self.data_loader.data)}")
                self.results_text.append(f"✓ Available keypoints: {', '.join(available_keypoints)}")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file:\n{str(e)}")

    def start_analysis(self):
        """Start the biomechanical analysis."""
        if not self.data_loader:
            QMessageBox.warning(self, "Warning", "Please select a CSV file first.")
            return

        # Get selected pairs and joints
        selected_pairs = [name for name, checkbox in self.pair_checkboxes.items()
                          if checkbox.isChecked()]
        selected_joints = [name for name, checkbox in self.joint_checkboxes.items()
                           if checkbox.isChecked()]

        if not selected_pairs and not selected_joints:
            QMessageBox.warning(self, "Warning", "Please select at least one analysis option.")
            return

        # Disable controls during analysis
        self.analyze_button.setEnabled(False)
        self.file_button.setEnabled(False)

        # Clear previous results
        self.results_text.clear()
        self.progress_bar.setValue(0)

        # Start worker thread
        self.worker = AnalysisWorker(
            self.data_loader, selected_pairs, selected_joints, self.fps_spinbox.value()
        )

        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(self.status_label.setText)
        self.worker.finished.connect(self.analysis_finished)
        self.worker.error.connect(self.analysis_error)

        self.worker.start()

    def analysis_finished(self, results):
        """Handle analysis completion."""
        self.status_label.setText("Analysis completed!")

        # Display results
        self.results_text.append("=== Analysis Complete ===")

        if results['chart_paths']:
            self.results_text.append(f"✓ Generated {len(results['chart_paths'])} movement charts")
            for path in results['chart_paths']:
                self.results_text.append(f"  - {os.path.basename(path)}")

        if results['joint_chart_path']:
            self.results_text.append(f"✓ Generated joint angle chart: {os.path.basename(results['joint_chart_path'])}")

        if 'report_path' in results:
            self.results_text.append(f"✓ Generated report: {os.path.basename(results['report_path'])}")

        self.results_text.append("\n=== Movement Statistics ===")
        for pair_name, stats in results['movement_stats'].items():
            self.results_text.append(f"{pair_name}:")
            self.results_text.append(f"  Duration: {stats['duration']:.2f}s")
            self.results_text.append(f"  Data points: {stats['data_points']}")

        if results['joint_stats']:
            self.results_text.append("\n=== Joint Angle Statistics ===")
            for joint_name, stats in results['joint_stats'].items():
                self.results_text.append(f"{joint_name}: {stats['mean']:.1f}° ± {stats['std']:.1f}°")

        if results['velocity_stats']:
            self.results_text.append("\n=== Velocity Analysis Results ===")

            # Group by joint type
            hip_velocities = {k: v for k, v in results['velocity_stats'].items() if 'hip' in k}
            knee_velocities = {k: v for k, v in results['velocity_stats'].items() if 'knee' in k}
            ankle_velocities = {k: v for k, v in results['velocity_stats'].items() if 'ankle' in k}
            wrist_velocities = {k: v for k, v in results['velocity_stats'].items() if 'wrist' in k}

            # Display knee velocities
            if knee_velocities:
                self.results_text.append("📊 Knee Velocities:")
                for joint_name, stats in knee_velocities.items():
                    side = 'Left' if 'left' in joint_name else 'Right'
                    self.results_text.append(
                        f"  {side} Knee: Peak {stats['vel_max']:.1f} mm/s, Mean {stats['vel_mean']:.1f} mm/s")

            # Display ankle velocities
            if ankle_velocities:
                self.results_text.append("📊 Ankle Velocities:")
                for joint_name, stats in ankle_velocities.items():
                    side = 'Left' if 'left' in joint_name else 'Right'
                    self.results_text.append(
                        f"  {side} Ankle: Peak {stats['vel_max']:.1f} mm/s, Mean {stats['vel_mean']:.1f} mm/s")

            # Display hip and wrist velocities
            if hip_velocities:
                self.results_text.append("📊 Hip Velocities:")
                for joint_name, stats in hip_velocities.items():
                    side = 'Left' if 'left' in joint_name else 'Right'
                    self.results_text.append(
                        f"  {side} Hip: Peak {stats['vel_max']:.1f} mm/s, Mean {stats['vel_mean']:.1f} mm/s")

            if wrist_velocities:
                self.results_text.append("📊 Wrist Velocities:")
                for joint_name, stats in wrist_velocities.items():
                    side = 'Left' if 'left' in joint_name else 'Right'
                    self.results_text.append(
                        f"  {side} Wrist: Peak {stats['vel_max']:.1f} mm/s, Mean {stats['vel_mean']:.1f} mm/s")

        # Re-enable controls
        self.analyze_button.setEnabled(True)
        self.file_button.setEnabled(True)

        # Show completion message
        QMessageBox.information(self, "Success",
                                "Analysis completed successfully!\nCheck the generated files in the current directory.")

    def analysis_error(self, error_msg):
        """Handle analysis errors."""
        self.status_label.setText("Analysis failed")
        self.results_text.append(f"✗ Error: {error_msg}")

        # Re-enable controls
        self.analyze_button.setEnabled(True)
        self.file_button.setEnabled(True)

        QMessageBox.critical(self, "Analysis Error", f"Analysis failed:\n{error_msg}")

    def clear_results(self):
        """Clear all results and reset the interface."""
        self.results_text.clear()
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready")
        self.file_label.setText("No file selected")
        self.data_loader = None
        self.analyze_button.setEnabled(False)

    def create_joint_visualization(self):
        """Create joint angle visualization with progress bar."""
        if not self.data_loader:
            QMessageBox.warning(self, "Warning", "Please select a CSV file first.")
            return

        selected_joint = self.viz_joint_combo.currentText()

        try:
            from analysis.joint_angles import JointAngleCalculator
            joint_calc = JointAngleCalculator(self.data_loader)

            # Reset progress bar
            self.progress_bar.setValue(0)
            self.status_label.setText("Starting visualization...")

            # Progress callback function
            def update_progress(value, message):
                self.progress_bar.setValue(value)
                self.status_label.setText(message)
                self.results_text.append(f"[{value}%] {message}")
                # Process events to update GUI
                from PyQt5.QtWidgets import QApplication
                QApplication.processEvents()

            results = joint_calc.create_joint_angle_visualization(
                selected_joint,
                fps=self.fps_spinbox.value(),
                save_files=True,
                progress_callback=update_progress
            )

            # Update results display
            self.results_text.append(f"✅ {selected_joint} visualization completed!")
            self.results_text.append(f"  📱 HTML: {results.get('html_file', 'N/A')}")
            self.results_text.append(f"  ⚙️ Config: {results.get('config_file', 'N/A')}")
            self.results_text.append(f"  📊 CSV: {results.get('csv_file', 'N/A')}")

            # Show the matplotlib window
            plt.show()

            QMessageBox.information(self, "Success",
                                    f"{selected_joint} visualization created!\n\n"
                                    f"Files saved:\n"
                                    f"• HTML: Interactive web controls\n"
                                    # f"• JSON: Visualization configuration\n"
                                    f"• CSV: Angle time series data")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create visualization:\n{str(e)}")
            self.results_text.append(f"✗ Error: {str(e)}")

    def create_skeleton_visualization(self):
        """Create full body skeleton visualization."""
        if not self.data_loader:
            QMessageBox.warning(self, "Warning", "Please select a CSV file first.")
            return

        try:
            from analysis.skeleton_visualizer import SkeletonVisualizer

            # Reset progress bar
            self.progress_bar.setValue(0)
            self.status_label.setText("Creating skeleton visualization...")

            # Progress callback function
            def update_progress(value, message):
                self.progress_bar.setValue(value)
                self.status_label.setText(message)
                from PyQt5.QtWidgets import QApplication
                QApplication.processEvents()

            # Create skeleton visualizer
            skeleton_viz = SkeletonVisualizer(self.data_loader)

            results = skeleton_viz.create_full_body_skeleton(
                fps=self.skeleton_fps_spinbox.value(),
                progress_callback=update_progress
            )

            # Update results display
            self.results_text.append(f"✅ Full body skeleton visualization created!")
            self.results_text.append(f"  📱 HTML: {results.get('html_file', 'N/A')}")
            self.results_text.append(f"  🎯 3D Window: Interactive matplotlib window opened")

            # Show success message
            QMessageBox.information(self, "Success",
                                    f"Full body skeleton visualization created!\n\n"
                                    f"Files created:\n"
                                    f"• HTML: Interactive web animation\n"
                                    f"• 3D Window: Interactive matplotlib window\n\n"
                                    f"Use mouse to rotate, zoom, and pan in 3D window!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create skeleton:\n{str(e)}")
            self.results_text.append(f"✗ Error: {str(e)}")


    # Update the select_file method to enable visualization button
    def select_file(self):
        """Select CSV file for analysis."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "", "CSV Files (*.csv);;All Files (*)"
        )

        if file_path:
            try:
                self.data_loader = DataLoader(file_path, self.fps_spinbox.value())
                self.data_loader.load_data()

                self.file_label.setText(os.path.basename(file_path))
                self.analyze_button.setEnabled(True)
                self.viz_button.setEnabled(True)  # Enable visualization button
                self.skeleton_button.setEnabled(True)

                # Update results
                available_keypoints = self.data_loader.get_available_keypoints()
                self.results_text.append(f"✓ File loaded: {os.path.basename(file_path)}")
                self.results_text.append(f"✓ Data points: {len(self.data_loader.data)}")
                self.results_text.append(f"✓ Available keypoints: {', '.join(available_keypoints)}")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file:\n{str(e)}")
