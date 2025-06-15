%% Stereo Human Motion Analysis Pipeline - Main Script
% Complete pipeline for 2-camera sports biomechanics analysis

clear; clc; close all;

%% Configuration
fprintf('=== Stereo Human Motion Analysis Pipeline ===\n');
fprintf('Starting 2-camera stereo analysis...\n\n');

% UPDATED: Use vid subfolder in current directory
videoDir = fullfile(pwd, 'vid');
patternDims = [7, 8]; % Adjust based on your checkerboard
squareSize = 60; % Your actual square size in mm

% Verify the video directory exists
if ~isfolder(videoDir)
    error('Video directory not found: %s\nPlease copy the vid folder to: %s', videoDir, pwd);
else
    fprintf('Video directory: %s\n', videoDir);
end

%% Step 1: Load stereo calibration
fprintf('\n=== Step 1: Loading Stereo Calibration ===\n');

try
    % Load the calibration session
    if exist('stereoParams.mat', 'file')
        data = load('stereoParams.mat');
        
        % Extract stereo parameters from calibration session
        if isfield(data, 'calibrationSession')
            calibrationSession = data.calibrationSession;
            fprintf('✓ Loaded stereo calibration session\n');
            
            % Extract stereo parameters from the session
            stereoParams = calibrationSession.CameraParameters;
            fprintf('✓ Extracted stereo parameters from session\n');
            
        else
            error('No calibrationSession found in stereoParams.mat');
        end
    else
        error('stereoParams.mat file not found. Please run stereo calibration first.');
    end
    
    fprintf('Stereo calibration loaded successfully:\n');
    fprintf('  Type: %s\n', class(stereoParams));
    fprintf('  Mean reprojection error: %.3f pixels\n', stereoParams.MeanReprojectionError);
    fprintf('  Number of calibration patterns: %d\n', stereoParams.NumPatterns);
    fprintf('  World units: %s\n', stereoParams.WorldUnits);
    
    % Calculate and display baseline distance
    baseline = norm(stereoParams.PoseCamera2.Translation);
    fprintf('  Baseline distance: %.1f mm\n', baseline);
    
    % Display individual camera errors
    fprintf('  Camera 1 reprojection error: %.3f pixels\n', stereoParams.CameraParameters1.MeanReprojectionError);
    fprintf('  Camera 2 reprojection error: %.3f pixels\n', stereoParams.CameraParameters2.MeanReprojectionError);
    
catch ME
    error('Failed to load stereo calibration: %s', ME.message);
end

%% Step 1.5: MediaPipe Keypoint Detection
fprintf('\n=== Step 1.5: MediaPipe Keypoint Detection ===\n');

% FIXED: Set video paths using proper string concatenation
videoFiles = {fullfile('vid', 'vid1.avi'), ...
              fullfile('vid', 'vid2.avi')};

% Check if keypoint files already exist
keypointFiles = {'keypoints_vid1.json', 'keypoints_vid2.json'};
needsProcessing = false;

for i = 1:2
    if ~exist(keypointFiles{i}, 'file')
        needsProcessing = true;
        break;
    end
end

if needsProcessing
    fprintf('Running MediaPipe keypoint detection...\n');
    
    % Verify Python script exists in current directory
    if ~exist('detect_stereo_keypoints.py', 'file')
        error('Python script not found. Please copy detect_stereo_keypoints.py to: %s', pwd);
    end
    
    % Verify video files exist
    for i = 1:2
        fprintf('Checking video file: %s\n', videoFiles{i});
        if ~isfile(videoFiles{i})
            error('Video file not found: %s', videoFiles{i});
        else
            info = dir(videoFiles{i});
            fprintf('Video %d: %s (%d bytes)\n', i, videoFiles{i}, info.bytes);
        end
    end
    
    % Build command with relative paths
    command = sprintf('python detect_stereo_keypoints.py "%s" "%s" "%s" "%s"', ...
                     videoFiles{1}, videoFiles{2}, keypointFiles{1}, keypointFiles{2});
    
    fprintf('Executing: %s\n', command);
    fprintf('Working directory: %s\n', pwd);
    
    % Execute command
    [status, result] = system(command);
    
    fprintf('Exit status: %d\n', status);
    fprintf('Python output:\n%s\n', result);
    
    if status == 0
        % Verify output files were created
        success = true;
        for i = 1:2
            outputFile = keypointFiles{i};
            if exist(outputFile, 'file')
                info = dir(outputFile);
                fprintf('✓ Created %s (%d bytes)\n', outputFile, info.bytes);
            else
                fprintf('✗ Failed to create %s\n', outputFile);
                success = false;
            end
        end
        
        if success
            fprintf('✓ MediaPipe processing completed successfully\n');
        else
            error('Python script ran but failed to create output files');
        end
    else
        error('MediaPipe keypoint detection failed. Exit code: %d\nError: %s', status, result);
    end
else
    fprintf('✓ Keypoint files already exist, skipping MediaPipe processing\n');
end


%% Step 2: Load keypoints
fprintf('\n=== Step 2: Loading MediaPipe Keypoints ===\n');

% Load keypoints with enhanced parsing
humanKeypoints = loadStereoKeypoints(keypointFiles);

if isempty(humanKeypoints)
    error('No keypoints loaded. Check MediaPipe processing and JSON files.');
end

%% Step 3: Undistort keypoints
fprintf('\n=== Step 3: Stereo Keypoint Undistortion ===\n');

undistortedKeypoints = undistortStereoKeypoints(humanKeypoints, stereoParams);

%% Step 4: Fixed Stereo triangulation
fprintf('\n=== Step 4: Fixed Stereo Triangulation ===\n');

[xyzPoints, successStats] = stereoTriangulation(undistortedKeypoints, stereoParams);

% Verify we have good results
if successStats.successfulTriangulations > 0
    successRate = (successStats.successfulTriangulations / successStats.totalAttempts) * 100;
    fprintf('✓ Triangulation successful: %.1f%% success rate\n', successRate);
else
    error('Triangulation failed completely. Check stereo calibration and keypoints.');
end

%% Step 5: Post-processing
fprintf('\n=== Step 5: Post-Processing Pipeline ===\n');

processedXyzPoints = stereoPostProcessing(xyzPoints);

% Apply coordinate rotation for proper orientation
fprintf('Applying coordinate rotation for proper orientation...\n');
rotationMatrix = [1, 0, 0; 0, 0, 1; 0, -1, 0];
rotatedXyzPoints = applyStereoRotation(processedXyzPoints, rotationMatrix);

%% Step 6: Create visualization
fprintf('\n=== Step 6: Creating Stereo 3D Animation ===\n');

createStereo3DVideo(rotatedXyzPoints, videoFiles, stereoParams);

%% Step 7: Biomechanical analysis
fprintf('\n=== Step 7: Stereo Biomechanical Analysis ===\n');

performStereoBiomechanicalAnalysis(rotatedXyzPoints);

%% Step 8: Generate final report
fprintf('\n=== Step 8: Final Report ===\n');

generateStereoFinalReport(stereoParams, successStats, rotatedXyzPoints);

%% Save all results
fprintf('\n=== Saving Results ===\n');

save('stereo_analysis_complete.mat', 'stereoParams', 'rotatedXyzPoints', 'humanKeypoints', ...
     'successStats', 'videoFiles', '-v7.3');

fprintf('✓ All results saved to stereo_analysis_complete.mat\n');
fprintf('✓ Stereo Human Motion Analysis Pipeline Complete!\n');
