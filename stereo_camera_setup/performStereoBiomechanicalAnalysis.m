function performStereoBiomechanicalAnalysis(rotatedXyzPoints)
    % Biomechanical analysis for stereo setup
    
    fprintf('=== Stereo Biomechanical Analysis ===\n');
    
    % MediaPipe keypoint mapping
    keypoints_map = struct();
    keypoints_map.nose = 1;
    keypoints_map.left_shoulder = 11;
    keypoints_map.right_shoulder = 12;
    keypoints_map.left_elbow = 13;
    keypoints_map.right_elbow = 14;
    keypoints_map.left_wrist = 15;
    keypoints_map.right_wrist = 16;
    keypoints_map.left_hip = 23;
    keypoints_map.right_hip = 24;
    keypoints_map.left_knee = 25;
    keypoints_map.right_knee = 26;
    keypoints_map.left_ankle = 27;
    keypoints_map.right_ankle = 28;
    
    numFrames = size(rotatedXyzPoints, 3);
    frameRate = 25; % fps
    
    % Calculate joint angles
    function angle = calculateJointAngle(xyzPoints, frameIdx, pt1Idx, pt2Idx, pt3Idx)
        p1 = squeeze(xyzPoints(pt1Idx, :, frameIdx));
        p2 = squeeze(xyzPoints(pt2Idx, :, frameIdx));
        p3 = squeeze(xyzPoints(pt3Idx, :, frameIdx));
        
        if any(isnan([p1, p2, p3]))
            angle = NaN;
            return;
        end
        
        v1 = p1 - p2;
        v2 = p3 - p2;
        cosAngle = dot(v1, v2) / (norm(v1) * norm(v2));
        cosAngle = max(-1, min(1, cosAngle));
        angle = acosd(cosAngle);
    end
    
    % Initialize angle arrays
    angles = struct();
    angles.rightKnee = nan(numFrames, 1);
    angles.leftKnee = nan(numFrames, 1);
    angles.rightElbow = nan(numFrames, 1);
    angles.leftElbow = nan(numFrames, 1);
    
    for frameIdx = 1:numFrames
        angles.rightKnee(frameIdx) = calculateJointAngle(rotatedXyzPoints, frameIdx, ...
            keypoints_map.right_hip, keypoints_map.right_knee, keypoints_map.right_ankle);
        
        angles.leftKnee(frameIdx) = calculateJointAngle(rotatedXyzPoints, frameIdx, ...
            keypoints_map.left_hip, keypoints_map.left_knee, keypoints_map.left_ankle);
        
        angles.rightElbow(frameIdx) = calculateJointAngle(rotatedXyzPoints, frameIdx, ...
            keypoints_map.right_shoulder, keypoints_map.right_elbow, keypoints_map.right_wrist);
        
        angles.leftElbow(frameIdx) = calculateJointAngle(rotatedXyzPoints, frameIdx, ...
            keypoints_map.left_shoulder, keypoints_map.left_elbow, keypoints_map.left_wrist);
    end
    
    % Create biomechanical plots
    figure('Position', [100, 100, 1400, 800]);
    
    timeVector = (1:numFrames) / frameRate;
    
    subplot(2, 2, 1);
    hold on;
    plot(timeVector, angles.rightKnee, 'r-', 'LineWidth', 2, 'DisplayName', 'Right Knee');
    plot(timeVector, angles.leftKnee, 'b-', 'LineWidth', 2, 'DisplayName', 'Left Knee');
    xlabel('Time (s)'); ylabel('Angle (degrees)');
    title('Knee Joint Angles (Stereo)'); legend; grid on;
    
    subplot(2, 2, 2);
    hold on;
    plot(timeVector, angles.rightElbow, 'r-', 'LineWidth', 2, 'DisplayName', 'Right Elbow');
    plot(timeVector, angles.leftElbow, 'b-', 'LineWidth', 2, 'DisplayName', 'Left Elbow');
    xlabel('Time (s)'); ylabel('Angle (degrees)');
    title('Elbow Joint Angles (Stereo)'); legend; grid on;
    
    sgtitle('Stereo Biomechanical Analysis Results');
    
    % Save biomechanical data
    biomechanicalData = struct();
    biomechanicalData.angles = angles;
    biomechanicalData.xyzPoints = rotatedXyzPoints;
    biomechanicalData.keypoints_map = keypoints_map;
    biomechanicalData.frameRate = frameRate;
    biomechanicalData.setupType = 'stereo';
    
    save('stereo_biomechanical_analysis.mat', 'biomechanicalData');
    
    fprintf('✓ Stereo biomechanical analysis completed\n');
end
