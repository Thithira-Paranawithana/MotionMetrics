import cv2
import mediapipe as mp
import numpy as np
import sys
import json
import os

def detect_keypoints_stereo(video1_path, video2_path, output1_path, output2_path):
    """
    Detect keypoints in stereo video pair using MediaPipe
    """
    print(f"Processing Video 1: {video1_path}")
    print(f"Processing Video 2: {video2_path}")
    
    # Verify video files exist
    if not os.path.exists(video1_path):
        print(f"Error: Video 1 not found: {video1_path}")
        return False
    
    if not os.path.exists(video2_path):
        print(f"Error: Video 2 not found: {video2_path}")
        return False
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,  # Highest accuracy for stereo
        enable_segmentation=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    # Open both video files
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    if not cap1.isOpened():
        print(f"Error: Could not open video 1: {video1_path}")
        return False
        
    if not cap2.isOpened():
        print(f"Error: Could not open video 2: {video2_path}")
        cap1.release()
        return False
    
    # Get video properties
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video 1: {frame_count1} frames at {fps1:.2f} fps")
    print(f"Video 2: {frame_count2} frames at {fps2:.2f} fps")
    
    # Use minimum frame count for synchronization
    total_frames = min(frame_count1, frame_count2)
    print(f"Processing {total_frames} synchronized frames")
    
    keypoints_data1 = []
    keypoints_data2 = []
    
    frame_idx = 0
    
    while frame_idx < total_frames:
        # Read frames from both cameras
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            print(f"End of video reached at frame {frame_idx}")
            break
        
        # Process both frames
        frame1_keypoints = process_frame(frame1, pose)
        frame2_keypoints = process_frame(frame2, pose)
        
        keypoints_data1.append(frame1_keypoints)
        keypoints_data2.append(frame2_keypoints)
        
        frame_idx += 1
        
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames ({100*frame_idx/total_frames:.1f}%)")
    
    # Release video captures
    cap1.release()
    cap2.release()
    
    # Save keypoints to JSON files
    print(f"Saving keypoints to {output1_path}")
    try:
        with open(output1_path, 'w', encoding='utf-8') as f:
            json.dump(keypoints_data1, f)
        print(f"SUCCESS: Camera 1 keypoints saved: {output1_path}")
    except Exception as e:
        print(f"Error saving camera 1 keypoints: {e}")
        return False
    
    print(f"Saving keypoints to {output2_path}")
    try:
        with open(output2_path, 'w', encoding='utf-8') as f:
            json.dump(keypoints_data2, f)
        print(f"SUCCESS: Camera 2 keypoints saved: {output2_path}")
    except Exception as e:
        print(f"Error saving camera 2 keypoints: {e}")
        return False
    
    print(f"SUCCESS: Successfully processed {frame_idx} frames")
    
    return True

def process_frame(frame, pose):
    """
    Process a single frame and extract keypoints
    """
    # Convert BGR to RGB (MediaPipe expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame
    results = pose.process(rgb_frame)
    
    frame_keypoints = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            # Convert normalized coordinates to pixel coordinates
            h, w, _ = frame.shape
            x = landmark.x * w
            y = landmark.y * h
            visibility = landmark.visibility
            frame_keypoints.append([x, y, visibility])
    else:
        # No pose detected, fill with NaN
        frame_keypoints = [[float('nan'), float('nan'), 0.0] for _ in range(33)]
    
    return frame_keypoints

def main():
    if len(sys.argv) != 5:
        print("Usage: python detect_stereo_keypoints.py <video1_path> <video2_path> <output1_path> <output2_path>")
        print("Example: python detect_stereo_keypoints.py vid1.avi vid2.avi keypoints_vid1.json keypoints_vid2.json")
        sys.exit(1)
    
    video1_path = sys.argv[1]
    video2_path = sys.argv[2]
    output1_path = sys.argv[3]
    output2_path = sys.argv[4]
    
    print("=== Starting Stereo Keypoint Detection ===")
    print(f"Video 1: {video1_path}")
    print(f"Video 2: {video2_path}")
    print(f"Output 1: {output1_path}")
    print(f"Output 2: {output2_path}")
    
    success = detect_keypoints_stereo(video1_path, video2_path, output1_path, output2_path)
    
    if success:
        print("=== Stereo keypoint detection completed successfully! ===")
    else:
        print("=== Stereo keypoint detection failed! ===")
        sys.exit(1)

if __name__ == "__main__":
    main()
