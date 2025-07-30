
# MotionMetrics - 3D Sports Biomechanics Analysis System

![Demo Video Placeholder](./assets/3d_motion.gif)

<div align="center">

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)](https://opencv.org/)
[![MATLAB](https://img.shields.io/badge/MATLAB-0076A8?style=for-the-badge&logo=mathworks&logoColor=white)](https://mathworks.com/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-4285F4.svg?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev) 
[![SpinView](https://img.shields.io/badge/SpinView-FF6B35.svg?style=for-the-badge)](https://www.flir.com/support/products/spinview/)

*Democratizing precision motion capture for sports biomechanics analysis*

</div>

---

## 🚀 Project Overview

**MotionMetrics** is a comprehensive 3D sports biomechanics analysis system that bridges the gap between expensive professional motion capture systems (>$100,000) and low-accuracy alternatives. Our innovative solution combines synchronized stereo cameras with advanced computer vision techniques to deliver accurate, accessible, and cost-effective motion analysis.

### 🎯 Why MotionMetrics?

- **Accessibility**: Affordable alternative to high-end systems like Vicon and Qualisys
- **Precision**: Sub-centimeter accuracy with 0.07 pixel stereo calibration error
- **Versatility**: Hybrid marker-based and markerless detection approaches
- **User-Friendly**: Automated analysis and professional report generation
- **Clinical Ready**: Validated through comparative analysis of normal vs. abnormal gait patterns

---

## ✨ Key Features

### 🎥 **Dual-Mode Motion Capture**
- **Markerless Detection**: MediaPipe integration for 33 anatomical landmarks
- **Marker-Based Tracking**: Green spherical markers on 14 major body joints
- **Smart Fallback**: MediaPipe prediction during marker occlusion

### 🔬 **Precision 3D Reconstruction**
- High-accuracy stereo camera calibration (0.07 pixels reprojection error)
- Advanced lens distortion correction
- Direct Linear Transformation (DLT) triangulation algorithms
- 70-85% triangulation success rates with sub-centimeter precision

### 📊 **Comprehensive Biomechanical Analysis**
- Automated joint angle calculations using vector dot product analysis
- Velocity and acceleration profiling
- Bilateral symmetry assessment
- Real-time movement visualization

### 📋 **Professional Reporting**
- Automated PDF report generation
- 3D skeleton visualizations
- Joint angle progression charts
- Movement statistics and clinical insights
- Bilateral comparison tables

---

## 🛠️ Technologies Used

<div align="center">

| **Language/Framework** | **Purpose** |
|:----------------------:|:------------|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) | Core detection algorithms, GUI application |
| ![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=flat&logo=OpenCV&logoColor=white) | Computer vision processing |
| ![MediaPipe](https://img.shields.io/badge/MediaPipe-4285F4?style=flat&logo=google&logoColor=white) | Markerless pose estimation |
| ![MATLAB](https://img.shields.io/badge/MATLAB-0076A8?style=flat&logo=mathworks&logoColor=white) | Stereo calibration, 3D reconstruction |
| ![PyQt5](https://img.shields.io/badge/PyQt5-41CD52?style=flat&logo=qt&logoColor=white) | Interactive analysis application |
| ![SpinView](https://img.shields.io/badge/SpinView-FF6B35?style=flat) | Synchronized video capture |

</div>

---

## 👥 Contributors

<div align="center">

|:---------------:|
| **Dilranga Dissanayake** | 
| **Thithira Paranawithana** | 
| **Nipini Tennakoon** |

</div>

---

## 🏆 Achievements

- ✅ Achieved 0.07 pixel mean reprojection error in stereo calibration
- ✅ Successfully validated through normal vs. abnormal gait analysis
- ✅ Developed comprehensive clinical reporting system
- ✅ Created user-friendly analysis interface for non-technical users

---

## 📈 Validation Results

Our system successfully detected movement asymmetries in comparative analysis:
- **Normal gait**: Bilateral knee standard deviation (13.1° vs 13.6°)
- **Injured gait**: Clear asymmetry detected (12.4° vs 8.5°)
- **Clinical significance**: Objective evidence of compensatory movement patterns

---

## 🙏 Acknowledgments

Special thanks to:
- Our project supervisors and advisors
- The open-source community
- MediaPipe and OpenCV development teams
- MATLAB Computer Vision Toolbox developers

---

<div align="center">

*Made with ❤️ by MotionMetrics Team for the sports biomechanics community*

</div>
