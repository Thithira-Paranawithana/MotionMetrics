import cv2
import mediapipe as mp
import numpy as np
import json

def extract_keypoints_from_video(video_path, output_json):
    """
    Extract MediaPipe keypoints from video and save to JSON
    """
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing {video_path}: {frame_count} frames at {fps:.1f} fps")
    
    keypoints_data = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = pose.process(rgb_frame)
        
        # Extract keypoints
        frame_keypoints = []
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                h, w, _ = frame.shape
                x = landmark.x * w
                y = landmark.y * h
                visibility = landmark.visibility
                frame_keypoints.append([x, y, visibility])
        else:
            # No pose detected - fill with NaN
            frame_keypoints = [[float('nan'), float('nan'), 0.0] for _ in range(33)]
        
        keypoints_data.append(frame_keypoints)
        frame_idx += 1
        
        if frame_idx % 50 == 0:
            print(f"  Processed {frame_idx}/{frame_count} frames")
    
    cap.release()
    
    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump(keypoints_data, f)
    
    print(f"SUCCESS: Saved {len(keypoints_data)} frames to {output_json}")
    return True

def main():
    # Process both videos
    videos = [
        ("vid/vid1.avi", "keypoints_vid1.json"),
        ("vid/vid2.avi", "keypoints_vid2.json")
    ]
    
    print("=== Extracting Stereo Keypoints ===")
    
    for video_path, json_path in videos:
        print(f"\nExtracting from {video_path}...")
        success = extract_keypoints_from_video(video_path, json_path)
        
        if not success:
            print(f"Failed to process {video_path}")
            return
    
    print("\n=== Keypoint Extraction Complete ===")
    print("JSON files created: keypoints_vid1.json, keypoints_vid2.json")

if __name__ == "__main__":
    main()
