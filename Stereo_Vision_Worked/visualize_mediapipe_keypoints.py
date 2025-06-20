import cv2
import mediapipe as mp
import numpy as np

def visualize_keypoints_on_video(video_path, output_path):
    """
    Visualize MediaPipe keypoints on video and save result
    """
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,  # Highest accuracy
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
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing {video_path}:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {frame_count}")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    keypoints_detected = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = pose.process(rgb_frame)
        
        # Draw keypoints if detected
        if results.pose_landmarks:
            keypoints_detected += 1
            
            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Add frame info
            cv2.putText(frame, f'Frame: {frame_idx}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, 'POSE DETECTED', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # No pose detected
            cv2.putText(frame, f'Frame: {frame_idx}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, 'NO POSE', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Write frame
        out.write(frame)
        frame_idx += 1
        
        if frame_idx % 50 == 0:
            print(f"  Processed {frame_idx}/{frame_count} frames")
    
    cap.release()
    out.release()
    
    detection_rate = (keypoints_detected / frame_count) * 100
    print(f"SUCCESS: Completed {output_path}")
    print(f"  Pose detection rate: {detection_rate:.1f}% ({keypoints_detected}/{frame_count} frames)")
    
    return True

def main():
    # Process both videos
    video_dir = "vid/"  # Adjust path as needed
    videos = [
        (video_dir + "vid1.avi", "vid1_with_keypoints.mp4"),
        (video_dir + "vid2.avi", "vid2_with_keypoints.mp4")
    ]
    
    print("=== MediaPipe Keypoint Visualization ===")
    
    for input_video, output_video in videos:
        print(f"\nProcessing {input_video}...")
        success = visualize_keypoints_on_video(input_video, output_video)
        
        if not success:
            print(f"Failed to process {input_video}")
        else:
            print(f"Success! Check {output_video}")
    
    print("\n=== Visualization Complete ===")
    print("Review the output videos to check MediaPipe detection quality")

if __name__ == "__main__":
    main()
