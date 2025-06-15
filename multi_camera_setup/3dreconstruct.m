
cameraParams1 = 

  cameraParameters with properties:

   Camera Intrinsics
                         Intrinsics: [1×1 cameraIntrinsics]

   Camera Extrinsics
                  PatternExtrinsics: [21×1 rigidtform3d]

   Accuracy of Estimation
              MeanReprojectionError: 0.2224
                 ReprojectionErrors: [42×2×21 double]
                  ReprojectedPoints: [42×2×21 double]

   Calibration Settings
                        NumPatterns: 21
                  DetectedKeypoints: [42×21 logical]
                        WorldPoints: [42×2 double]
                         WorldUnits: 'millimeters'
                       EstimateSkew: 0
    NumRadialDistortionCoefficients: 2
       EstimateTangentialDistortion: 0


estimationErrors1 = 

  cameraCalibrationErrors with properties:

    IntrinsicsErrors: [1×1 intrinsicsEstimationErrors]
    ExtrinsicsErrors: [1×1 extrinsicsEstimationErrors]


cameraParams2 = 

  cameraParameters with properties:

   Camera Intrinsics
                         Intrinsics: [1×1 cameraIntrinsics]

   Camera Extrinsics
                  PatternExtrinsics: [21×1 rigidtform3d]

   Accuracy of Estimation
              MeanReprojectionError: 0.1028
                 ReprojectionErrors: [42×2×21 double]
                  ReprojectedPoints: [42×2×21 double]

   Calibration Settings
                        NumPatterns: 21
                  DetectedKeypoints: [42×21 logical]
                        WorldPoints: [42×2 double]
                         WorldUnits: 'millimeters'
                       EstimateSkew: 0
    NumRadialDistortionCoefficients: 2
       EstimateTangentialDistortion: 0


estimationErrors2 = 

  cameraCalibrationErrors with properties:

    IntrinsicsErrors: [1×1 intrinsicsEstimationErrors]
    ExtrinsicsErrors: [1×1 extrinsicsEstimationErrors]


cameraParams3 = 

  cameraParameters with properties:

   Camera Intrinsics
                         Intrinsics: [1×1 cameraIntrinsics]

   Camera Extrinsics
                  PatternExtrinsics: [21×1 rigidtform3d]

   Accuracy of Estimation
              MeanReprojectionError: 0.1866
                 ReprojectionErrors: [42×2×21 double]
                  ReprojectedPoints: [42×2×21 double]

   Calibration Settings
                        NumPatterns: 21
                  DetectedKeypoints: [42×21 logical]
                        WorldPoints: [42×2 double]
                         WorldUnits: 'millimeters'
                       EstimateSkew: 0
    NumRadialDistortionCoefficients: 2
       EstimateTangentialDistortion: 0


estimationErrors3 = 

  cameraCalibrationErrors with properties:

    IntrinsicsErrors: [1×1 intrinsicsEstimationErrors]
    ExtrinsicsErrors: [1×1 extrinsicsEstimationErrors]



% Extract intrinsics from your calibrated parameters
intrinsics = cell(3,1);
intrinsics{1} = cameraParams1.Intrinsics;
intrinsics{2} = cameraParams2.Intrinsics;
intrinsics{3} = cameraParams3.Intrinsics;


save('three-blackfly-intrinsics.mat', 'intrinsics');


numCameras = 3;
camDirPrefix = "C:\Users\REDTECH\Documents\Uni\Calibration\Camera";

imageFiles = cell(1, numCameras);

for i = 1:numCameras
    imds = imageDatastore(camDirPrefix + i);
    imageFiles{i} = imds.Files;
end
patternDims = [7, 8];
numViews = length(imageFiles{1}); % Number of calibration images (should be 40)
numCameras = 3;

% Create imageFileNames matrix: numViews x numCameras
imageFileNames = cell(numViews, numCameras);
for camIdx = 1:numCameras
    for viewIdx = 1:numViews
        imageFileNames{viewIdx, camIdx} = imageFiles{camIdx}{viewIdx};
    end
end

imagePoints = detectPatternPoints(imageFileNames, "checkerboard", patternDims);

Detecting pattern in 63 images captured from 3 cameras.
[==================================================] 100%
Elapsed time: 00:00:58
Estimated time remaining: 00:00:00


squareSize = 60; % in millimeters
worldPoints = patternWorldPoints("checkerboard", patternDims, squareSize);

fprintf('Detected points in %d views across %d cameras\n', size(imagePoints, 3), size(imagePoints, 4));
Detected points in 21 views across 3 cameras

for camIdx = 1:numCameras
    validDetections = sum(~isnan(imagePoints(1, 1, :, camIdx)));
    fprintf('Camera %d: %d/%d successful detections\n', camIdx, validDetections, numViews);
end
Camera 1: 21/21 successful detections
Camera 2: 21/21 successful detections
Camera 3: 21/21 successful detections

params = estimateMultiCameraParameters(imagePoints, worldPoints, intrinsics, ...
    'WorldUnits', 'millimeters');

disp(params);
fprintf('Mean Reprojection Error: %.4f pixels\n', params.MeanReprojectionError);
  multiCameraParameters with properties:

   Extrinsic Parameters
                       CameraPoses: [3×1 rigidtform3d]
              ReferenceCameraIndex: 1
                        NumCameras: 3

   Intrinsic Parameters
                        Intrinsics: {3×1 cell}

   Accuracy of Estimation
             MeanReprojectionError: 3.8451
    MeanReprojectionErrorPerCamera: [3×1 double]

   Calibration Settings
                          NumViews: 21
                CovisibilityMatrix: [3×3 logical]
                       WorldPoints: [42×2 double]
                        WorldUnits: 'millimeters'

Mean Reprojection Error: 3.8451 pixels

figure;
showExtrinsics(params);
title('Multi-Camera Setup Visualization');

figure;
showReprojectionErrors(params);
title('Reprojection Errors Analysis');

camPoses = params.CameraPoses;
cam1 = [0, 0, 0]; % Reference camera at origin
cam2 = camPoses(2).Translation;
cam3 = camPoses(3).Translation;

distances = [norm(cam2-cam1), norm(cam3-cam2), norm(cam3-cam1)];
fprintf('Inter-camera distances: %.2f, %.2f, %.2f mm\n', distances);
Inter-camera distances: 2217.32, 1787.46, 3827.80 mm


save('multi-camera-params.mat', 'params');




% Key biomechanical points for sports analysis
biomechanicalKeypoints = struct();
biomechanicalKeypoints.nose = 1;
biomechanicalKeypoints.left_shoulder = 12;
biomechanicalKeypoints.right_shoulder = 11;
biomechanicalKeypoints.left_elbow = 14;
biomechanicalKeypoints.right_elbow = 13;
biomechanicalKeypoints.left_wrist = 16;
biomechanicalKeypoints.right_wrist = 15;
biomechanicalKeypoints.left_hip = 24;
biomechanicalKeypoints.right_hip = 23;
biomechanicalKeypoints.left_knee = 26;
biomechanicalKeypoints.right_knee = 25;
biomechanicalKeypoints.left_ankle = 28;
biomechanicalKeypoints.right_ankle = 27;




videoDir = "C:\Users\REDTECH\Documents\Uni\Calibration\vid\";
videoFiles = [videoDir + "vid1.avi", ...
              videoDir + "vid2.avi", ...
              videoDir + "vid3.avi"];

% Verify video files exist
for i = 1:numCameras
    if isfile(videoFiles(i))
        fprintf('Found video file: %s\n', videoFiles(i));
    else
        fprintf('ERROR: Video file not found: %s\n', videoFiles(i));
        return;
    end
end
ERROR: Video file not found: C:\Users\REDTECH\Documents\Uni\Calibration\vid\vid1.avi

for i = 1:numCameras
    if isfile(videoFiles(i))
        fprintf('Found video file: %s\n', videoFiles(i));
    else
        fprintf('ERROR: Video file not found: %s\n', videoFiles(i));
        return;
    end
end
ERROR: Video file not found: C:\Users\REDTECH\Documents\Uni\Calibration\vid\vid1.avi

videoDir = "C:\Users\REDTECH\Documents\Uni\Calibration\vid\";
videoNames = ["vid1.avi", "vid2.avi", "vid3.avi"];

fprintf('Checking video files:\n');
for i = 1:3
    fullPath = videoDir + videoNames(i);
    if isfile(fullPath)
        fprintf('✓ Found: %s\n', fullPath);
    else
        fprintf('✗ NOT FOUND: %s\n', fullPath);

        % List all files in the directory to see what's actually there
        if isfolder(videoDir)
            fprintf('Files in directory:\n');
            dirInfo = dir(videoDir);
            for j = 1:length(dirInfo)
                if ~dirInfo(j).isdir
                    fprintf('  - %s\n', dirInfo(j).name);
                end
            end
        else
            fprintf('Directory does not exist: %s\n', videoDir);
        end
    end
end
Checking video files:
✗ NOT FOUND: C:\Users\REDTECH\Documents\Uni\Calibration\vid\vid1.avi
Files in directory:
  - v1.avi
  - v2.avi
  - v3.avi
✗ NOT FOUND: C:\Users\REDTECH\Documents\Uni\Calibration\vid\vid2.avi
Files in directory:
  - v1.avi
  - v2.avi
  - v3.avi
✗ NOT FOUND: C:\Users\REDTECH\Documents\Uni\Calibration\vid\vid3.avi
Files in directory:
  - v1.avi
  - v2.avi
  - v3.avi



for i = 1:numCameras
    if isfile(videoFiles(i))
        fprintf('Found video file: %s\n', videoFiles(i));
    else
        fprintf('ERROR: Video file not found: %s\n', videoFiles(i));
        return;
    end
end
Found video file: C:\Users\REDTECH\Documents\Uni\Calibration\vid\vid1.avi
Found video file: C:\Users\REDTECH\Documents\Uni\Calibration\vid\vid2.avi
Found video file: C:\Users\REDTECH\Documents\Uni\Calibration\vid\vid3.avi


readers = cell(numCameras, 1);
for i = 1:numCameras
    readers{i} = VideoReader(videoFiles(i));
    fprintf('Camera %d video: %d frames, %.2f fps, Duration: %.2f sec\n', ...
           i, readers{i}.NumFrames, readers{i}.FrameRate, readers{i}.Duration);
end
Camera 1 video: 489 frames, 30.00 fps, Duration: 16.30 sec
Camera 2 video: 489 frames, 30.00 fps, Duration: 16.30 sec
Camera 3 video: 489 frames, 30.00 fps, Duration: 16.30 sec


% Ensure all videos have the same number of frames (for synchronization)
frameCount = [readers{1}.NumFrames, readers{2}.NumFrames, readers{3}.NumFrames];
if length(unique(frameCount)) == 1
    numFrames = frameCount(1);
    fprintf('All videos synchronized: %d frames each\n', numFrames);
else
    fprintf('WARNING: Videos have different frame counts: %s\n', mat2str(frameCount));
    numFrames = min(frameCount);
    fprintf('Using minimum frame count: %d\n', numFrames);
end
All videos synchronized: 489 frames each


% Define MediaPipe keypoint indices (33 keypoints total)
% Key biomechanical points for sports analysis
biomechanicalKeypoints = struct();
biomechanicalKeypoints.nose = 1;
biomechanicalKeypoints.left_shoulder = 12;
biomechanicalKeypoints.right_shoulder = 11;
biomechanicalKeypoints.left_elbow = 14;
biomechanicalKeypoints.right_elbow = 13;
biomechanicalKeypoints.left_wrist = 16;
biomechanicalKeypoints.right_wrist = 15;
biomechanicalKeypoints.left_hip = 24;
biomechanicalKeypoints.right_hip = 23;
biomechanicalKeypoints.left_knee = 26;
biomechanicalKeypoints.right_knee = 25;
biomechanicalKeypoints.left_ankle = 28;
biomechanicalKeypoints.right_ankle = 27;


keypointFiles = cell(numCameras, 1);



keypointFiles = cell(numCameras, 1);
for camIdx = 1:numCameras
    videoFile = videoFiles(camIdx);
    outputFile = sprintf('keypoints_vid%d.json', camIdx);

    fprintf('Processing %s...\n', videoFile);

    % Call Python script for keypoint detection
    command = sprintf('python detect_keypoints_mediapipe.py "%s" "%s"', ...
                     videoFile, outputFile);
    [status, result] = system(command);

    if status == 0
        fprintf('Successfully extracted keypoints for vid%d.avi\n', camIdx);
        keypointFiles{camIdx} = outputFile;
    else
        fprintf('Error processing vid%d.avi: %s\n', camIdx, result);
    end
end
Processing C:\Users\REDTECH\Documents\Uni\Calibration\vid\vid1.avi...
Error processing vid1.avi: Traceback (most recent call last):
  File "C:\Users\REDTECH\Documents\Uni\Calibration_images\detect_keypoints_mediapipe.py", line 1, in <module>
    import cv2
ModuleNotFoundError: No module named 'cv2'

Processing C:\Users\REDTECH\Documents\Uni\Calibration\vid\vid2.avi...
Error processing vid2.avi: Traceback (most recent call last):
  File "C:\Users\REDTECH\Documents\Uni\Calibration_images\detect_keypoints_mediapipe.py", line 1, in <module>
    import cv2
ModuleNotFoundError: No module named 'cv2'

Processing C:\Users\REDTECH\Documents\Uni\Calibration\vid\vid3.avi...
Error processing vid3.avi: Traceback (most recent call last):
  File "C:\Users\REDTECH\Documents\Uni\Calibration_images\detect_keypoints_mediapipe.py", line 1, in <module>
    import cv2
ModuleNotFoundError: No module named 'cv2'



keypointFiles = cell(numCameras, 1);
for camIdx = 1:numCameras
    videoFile = videoFiles(camIdx);
    outputFile = sprintf('keypoints_vid%d.json', camIdx);

    fprintf('Processing %s...\n', videoFile);

    % Call Python script for keypoint detection
    command = sprintf('python detect_keypoints_mediapipe.py "%s" "%s"', ...
                     videoFile, outputFile);
    [status, result] = system(command);

    if status == 0
        fprintf('Successfully extracted keypoints for vid%d.avi\n', camIdx);
        keypointFiles{camIdx} = outputFile;
    else
        fprintf('Error processing vid%d.avi: %s\n', camIdx, result);
    end
end
Processing C:\Users\REDTECH\Documents\Uni\Calibration\vid\vid1.avi...
Error processing vid1.avi: Traceback (most recent call last):
  File "C:\Users\REDTECH\Documents\Uni\Calibration_images\detect_keypoints_mediapipe.py", line 1, in <module>
    import cv2
ModuleNotFoundError: No module named 'cv2'

Processing C:\Users\REDTECH\Documents\Uni\Calibration\vid\vid2.avi...
Error processing vid2.avi: Traceback (most recent call last):
  File "C:\Users\REDTECH\Documents\Uni\Calibration_images\detect_keypoints_mediapipe.py", line 1, in <module>
    import cv2
ModuleNotFoundError: No module named 'cv2'

Processing C:\Users\REDTECH\Documents\Uni\Calibration\vid\vid3.avi...
Error processing vid3.avi: Traceback (most recent call last):
  File "C:\Users\REDTECH\Documents\Uni\Calibration_images\detect_keypoints_mediapipe.py", line 1, in <module>
    import cv2
ModuleNotFoundError: No module named 'cv2'



% Process all three videos
keypointFiles = cell(numCameras, 1);
for camIdx = 1:numCameras
    videoFile = videoFiles(camIdx);
    outputFile = sprintf('keypoints_vid%d.json', camIdx);

    fprintf('Processing %s...\n', videoFile);

    % Call Python script for keypoint detection
    command = sprintf('python detect_keypoints_mediapipe.py "%s" "%s"', ...
                     videoFile, outputFile);
    [status, result] = system(command);

    if status == 0
        fprintf('Successfully extracted keypoints for vid%d.avi\n', camIdx);
        keypointFiles{camIdx} = outputFile;
    else
        fprintf('Error processing vid%d.avi: %s\n', camIdx, result);
    end
end
Processing C:\Users\REDTECH\Documents\Uni\Calibration\vid\vid1.avi...
Successfully extracted keypoints for vid1.avi
Processing C:\Users\REDTECH\Documents\Uni\Calibration\vid\vid2.avi...
Successfully extracted keypoints for vid2.avi
Processing C:\Users\REDTECH\Documents\Uni\Calibration\vid\vid3.avi...
Successfully extracted keypoints for vid3.avi



% Visualize keypoint detection on sample frames
function visualizeKeypoints(videoFiles, humanKeypoints, frameNum)
    figure('Position', [100, 100, 1500, 500]);

    for camIdx = 1:length(videoFiles)
        % Read the specific frame
        reader = VideoReader(videoFiles(camIdx));
        reader.CurrentTime = (frameNum - 1) / reader.FrameRate;

        if hasFrame(reader)
            frame = readFrame(reader);

            subplot(1, 3, camIdx);
            imshow(frame);
            hold on;

            % Plot detected keypoints
            keypoints = squeeze(humanKeypoints(:, :, frameNum, camIdx));
            validPoints = ~isnan(keypoints(:, 1));

            if any(validPoints)
                scatter(keypoints(validPoints, 1), keypoints(validPoints, 2), ...
                       50, 'red', 'filled', 'MarkerEdgeColor', 'white', 'LineWidth', 1);

                % Connect major body segments
                connections = [11, 12; 12, 24; 11, 23; 23, 24; ... % torso
                              11, 13; 13, 15; 12, 14; 14, 16; ... % arms
                              23, 25; 25, 27; 24, 26; 26, 28];   % legs

                for i = 1:size(connections, 1)
                    pt1 = connections(i, 1);
                    pt2 = connections(i, 2);
                    if ~isnan(keypoints(pt1, 1)) && ~isnan(keypoints(pt2, 1))
                        plot([keypoints(pt1, 1), keypoints(pt2, 1)], ...
                             [keypoints(pt1, 2), keypoints(pt2, 2)], ...
                             'yellow', 'LineWidth', 2);
                    end
                end
            end

            title(sprintf('vid%d.avi - Frame %d', camIdx, frameNum));
        end
    end

    sgtitle(sprintf('Keypoint Detection Results - Frame %d', frameNum));
end

% Test visualization on a sample frame
if ~isempty(humanKeypoints)
    sampleFrame = min(50, size(humanKeypoints, 3)); % Use frame 50 or last frame if less
    visualizeKeypoints(videoFiles, humanKeypoints, sampleFrame);
end

 function visualizeKeypoints(videoFiles, humanKeypoints, frameNum)
 ↑
Error: Function definitions are not supported in this context. Functions can only be created as local or nested functions
in code files.
 

% Test visualization on a sample frame
if ~isempty(humanKeypoints)
    sampleFrame = min(50, size(humanKeypoints, 3)); % Use frame 50 or last frame if less
    visualizeKeypoints(videoFiles, humanKeypoints, sampleFrame);
end

Unrecognized function or variable 'humanKeypoints'.
 

% Visualize keypoint detection on sample frames
if ~isempty(humanKeypoints)
    frameNum = min(50, size(humanKeypoints, 3)); % Use frame 50 or last frame if less

    figure('Position', [100, 100, 1500, 500]);

    for camIdx = 1:length(videoFiles)
        % Read the specific frame
        reader = VideoReader(videoFiles(camIdx));
        reader.CurrentTime = (frameNum - 1) / reader.FrameRate;

        if hasFrame(reader)
            frame = readFrame(reader);

            subplot(1, 3, camIdx);
            imshow(frame);
            hold on;

            % Plot detected keypoints
            keypoints = squeeze(humanKeypoints(:, :, frameNum, camIdx));
            validPoints = ~isnan(keypoints(:, 1));

            if any(validPoints)
                scatter(keypoints(validPoints, 1), keypoints(validPoints, 2), ...
                       50, 'red', 'filled', 'MarkerEdgeColor', 'white', 'LineWidth', 1);

                % Connect major body segments
                connections = [11, 12; 12, 24; 11, 23; 23, 24; ... % torso
                              11, 13; 13, 15; 12, 14; 14, 16; ... % arms
                              23, 25; 25, 27; 24, 26; 26, 28];   % legs

                for i = 1:size(connections, 1)
                    pt1 = connections(i, 1);
                    pt2 = connections(i, 2);
                    if ~isnan(keypoints(pt1, 1)) && ~isnan(keypoints(pt2, 1))
                        plot([keypoints(pt1, 1), keypoints(pt2, 1)], ...
                             [keypoints(pt1, 2), keypoints(pt2, 2)], ...
                             'yellow', 'LineWidth', 2);
                    end
                end
            end

            title(sprintf('vid%d.avi - Frame %d', camIdx, frameNum));
        end
    end

    sgtitle(sprintf('Keypoint Detection Results - Frame %d', frameNum));
end

Unrecognized function or variable 'humanKeypoints'.
 

whos
  Name                         Size                Bytes  Class                                 Attributes

  biomechanicalKeypoints       1x1                  1914  struct                                          
  cam1                         1x3                    24  double                                          
  cam2                         1x3                    24  double                                          
  cam3                         1x3                    24  double                                          
  camDirPrefix                 1x1                   262  string                                          
  camIdx                       1x1                     8  double                                          
  camPoses                     3x1                   291  rigidtform3d                                    
  cameraParams1                1x1                     8  cameraParameters                                
  cameraParams2                1x1                     8  cameraParameters                                
  cameraParams3                1x1                     8  cameraParameters                                
  command                      1x116                 232  char                                            
  dirInfo                      5x1                  4482  struct                                          
  distances                    1x3                    24  double                                          
  estimationErrors1            1x1                  1722  cameraCalibrationErrors                         
  estimationErrors2            1x1                  1722  cameraCalibrationErrors                         
  estimationErrors3            1x1                  1722  cameraCalibrationErrors                         
  frameCount                   1x3                    24  double                                          
  fullPath                     1x1                   262  string                                          
  i                            1x1                     8  double                                          
  imageFileNames              21x3                 15342  cell                                            
  imageFiles                   1x3                 15702  cell                                            
  imagePoints                 42x2x21x3            42336  double                                          
  imds                         1x1                     8  matlab.io.datastore.ImageDatastore              
  intrinsics                   3x1                   384  cell                                            
  j                            1x1                     8  double                                          
  keypointFiles                3x1                   474  cell                                            
  numCameras                   1x1                     8  double                                          
  numFrames                    1x1                     8  double                                          
  numViews                     1x1                     8  double                                          
  outputFile                   1x19                   38  char                                            
  params                       1x1                     8  multiCameraParameters                           
  patternDims                  1x2                    16  double                                          
  readers                      3x1                   384  cell                                            
  result                       1x852                1704  char                                            
  squareSize                   1x1                     8  double                                          
  status                       1x1                     8  double                                          
  validDetections              1x1                     8  double                                          
  videoDir                     1x1                   246  string                                          
  videoFile                    1x1                   262  string                                          
  videoFiles                   1x3                   562  string                                          
  videoNames                   1x3                   322  string                                          
  viewIdx                      1x1                     8  double                                          
  worldPoints                 42x2                   672  double                                          



% Check if JSON files were created successfully
fprintf('Checking keypoint files:\n');
for i = 1:3
    filename = keypointFiles{i};
    if isfile(filename)
        fileInfo = dir(filename);
        fprintf('✓ %s exists (Size: %d bytes)\n', filename, fileInfo.bytes);
    else
        fprintf('✗ %s does not exist\n', filename);
    end
end

% Check the status and result from your last Python execution
fprintf('\nLast Python execution status: %d\n', status);
fprintf('Python output:\n%s\n', result);

Checking keypoint files:
✓ keypoints_vid1.json exists (Size: 973105 bytes)
✓ keypoints_vid2.json exists (Size: 975495 bytes)
✓ keypoints_vid3.json exists (Size: 966950 bytes)

Last Python execution status: 0
Python output:
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
W0000 00:00:1749058594.465652   22712 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1749058594.579527   22712 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1749058594.646558   16196 landmark_projection_calculator.cc:186] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
Processed 100 frames
Processed 200 frames
Processed 300 frames
Processed 400 frames
Saved keypoints for 489 frames to keypoints_vid3.json



% Load keypoints from JSON files into MATLAB
fprintf('Loading keypoints from JSON files...\n');

humanKeypoints = [];
numCameras = 3;

for camIdx = 1:numCameras
    jsonFile = keypointFiles{camIdx};

    if isfile(jsonFile)
        fprintf('Processing %s...\n', jsonFile);

        try
            % Read JSON file
            fid = fopen(jsonFile, 'r');
            raw = fread(fid, inf);
            str = char(raw');
            fclose(fid);

            % Parse JSON
            data = jsondecode(str);
            numFramesInFile = length(data);
            numKeypoints = 33; % MediaPipe has 33 keypoints

            % Initialize humanKeypoints array on first camera
            if isempty(humanKeypoints)
                humanKeypoints = nan(numKeypoints, 2, numFramesInFile, numCameras);
                fprintf('Initialized humanKeypoints array: %s\n', mat2str(size(humanKeypoints)));
            end

            % Process each frame
            validKeypointsCount = 0;
            for frameIdx = 1:numFramesInFile
                frameData = data{frameIdx};

                if iscell(frameData) && length(frameData) >= numKeypoints
                    for kpIdx = 1:numKeypoints
                        if length(frameData{kpIdx}) >= 3
                            x = frameData{kpIdx}(1);
                            y = frameData{kpIdx}(2);
                            visibility = frameData{kpIdx}(3);

                            % Use keypoints with good visibility
                            if visibility > 0.5 && ~isnan(x) && ~isnan(y)
                                humanKeypoints(kpIdx, 1, frameIdx, camIdx) = x;
                                humanKeypoints(kpIdx, 2, frameIdx, camIdx) = y;
                                validKeypointsCount = validKeypointsCount + 1;
                            end
                        end
                    end
                end
            end

            fprintf('✓ Camera %d: Loaded %d valid keypoints from %d frames\n', ...
                   camIdx, validKeypointsCount, numFramesInFile);

        catch ME
            fprintf('✗ Error loading Camera %d: %s\n', camIdx, ME.message);
        end
    else
        fprintf('✗ File not found: %s\n', jsonFile);
    end
end

% Check if loading was successful
if ~isempty(humanKeypoints)
    fprintf('\n✓ Successfully created humanKeypoints variable\n');
    fprintf('Size: %s\n', mat2str(size(humanKeypoints)));

    % Calculate statistics
    totalValidPoints = sum(~isnan(humanKeypoints(:)));
    totalPossiblePoints = numel(humanKeypoints);
    successRate = (totalValidPoints / totalPossiblePoints) * 100;

    fprintf('Valid keypoints: %d/%d (%.1f%%)\n', totalValidPoints, totalPossiblePoints, successRate);

    % Save the keypoints
    save('humanKeypoints.mat', 'humanKeypoints');
    fprintf('Saved humanKeypoints to humanKeypoints.mat\n');

else
    fprintf('\n✗ Failed to create humanKeypoints variable\n');
    fprintf('Check if JSON files contain valid data\n');
end

Loading keypoints from JSON files...
Processing keypoints_vid1.json...
Initialized humanKeypoints array: [33 2 489 3]
✗ Error loading Camera 1: Brace indexing is not supported for variables of this type.
Processing keypoints_vid2.json...
✗ Error loading Camera 2: Brace indexing is not supported for variables of this type.
Processing keypoints_vid3.json...
✗ Error loading Camera 3: Brace indexing is not supported for variables of this type.

✓ Successfully created humanKeypoints variable
Size: [33 2 489 3]
Valid keypoints: 0/96822 (0.0%)
Saved humanKeypoints to humanKeypoints.mat


% Check if JSON files exist and their content
for i = 1:3
    filename = keypointFiles{i};
    if isfile(filename)
        fileInfo = dir(filename);
        fprintf('File %s: %d bytes\n', filename, fileInfo.bytes);

        % Read first few lines to check format
        fid = fopen(filename, 'r');
        firstLine = fgetl(fid);
        fclose(fid);
        fprintf('First line: %s\n', firstLine(1:min(100, length(firstLine))));
    else
        fprintf('Missing file: %s\n', filename);
    end
end

File keypoints_vid1.json: 973105 bytes
First line: [[[1719.6726894378662, 697.4605712890625, 0.9999946355819702], [1734.8096179962158, 681.324462890625
File keypoints_vid2.json: 975495 bytes
First line: [[[1193.0337896347046, 809.8392333984375, 0.9999985694885254], [1199.6669569015503, 797.2197265625, 
File keypoints_vid3.json: 966950 bytes
First line: [[[769.3752493858337, 811.941162109375, 0.9999977350234985], [774.0968942642212, 799.317138671875, 0


% Check the last Python execution results
fprintf('Python execution status: %d\n', status);
fprintf('Python output:\n%s\n', result);

Python execution status: 0
Python output:
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
W0000 00:00:1749058594.465652   22712 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1749058594.579527   22712 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1749058594.646558   16196 landmark_projection_calculator.cc:186] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
Processed 100 frames
Processed 200 frames
Processed 300 frames
Processed 400 frames
Saved keypoints for 489 frames to keypoints_vid3.json




% Corrected keypoint loading function
fprintf('Loading keypoints with corrected parsing...\n');

humanKeypoints = [];
numCameras = 3;

for camIdx = 1:numCameras
    jsonFile = keypointFiles{camIdx};

    if isfile(jsonFile)
        fprintf('Processing %s...\n', jsonFile);

        try
            % Read JSON file
            fid = fopen(jsonFile, 'r');
            raw = fread(fid, inf);
            str = char(raw');
            fclose(fid);

            % Parse JSON
            data = jsondecode(str);

            % Handle the nested structure correctly
            numFramesInFile = length(data);
            numKeypoints = 33; % MediaPipe has 33 keypoints

            % Initialize humanKeypoints array on first camera
            if isempty(humanKeypoints)
                humanKeypoints = nan(numKeypoints, 2, numFramesInFile, numCameras);
                fprintf('Initialized humanKeypoints array: %s\n', mat2str(size(humanKeypoints)));
            end

            % Process each frame
            validKeypointsCount = 0;
            for frameIdx = 1:numFramesInFile
                frameData = data{frameIdx}; % Extract frame data

                if iscell(frameData) && length(frameData) >= numKeypoints
                    for kpIdx = 1:numKeypoints
                        if length(frameData{kpIdx}) >= 3
                            keypointData = frameData{kpIdx};
                            x = keypointData(1);
                            y = keypointData(2);
                            visibility = keypointData(3);

                            % Use keypoints with good visibility
                            if visibility > 0.5 && ~isnan(x) && ~isnan(y)
                                humanKeypoints(kpIdx, 1, frameIdx, camIdx) = x;
                                humanKeypoints(kpIdx, 2, frameIdx, camIdx) = y;
                                validKeypointsCount = validKeypointsCount + 1;
                            end
                        end
                    end
                elseif isnumeric(frameData) && size(frameData, 1) >= numKeypoints
                    % Alternative: frameData is numeric matrix
                    for kpIdx = 1:numKeypoints
                        x = frameData(kpIdx, 1);
                        y = frameData(kpIdx, 2);
                        visibility = frameData(kpIdx, 3);

                        if visibility > 0.5 && ~isnan(x) && ~isnan(y)
                            humanKeypoints(kpIdx, 1, frameIdx, camIdx) = x;
                            humanKeypoints(kpIdx, 2, frameIdx, camIdx) = y;
                            validKeypointsCount = validKeypointsCount + 1;
                        end
                    end
                end
            end

            fprintf('✓ Camera %d: Loaded %d valid keypoints from %d frames\n', ...
                   camIdx, validKeypointsCount, numFramesInFile);

        catch ME
            fprintf('✗ Error loading Camera %d: %s\n', camIdx, ME.message);
            fprintf('Error details: %s\n', ME.getReport);
        end
    else
        fprintf('✗ File not found: %s\n', jsonFile);
    end
end

% Verify loading success
if ~isempty(humanKeypoints)
    fprintf('\n✓ Successfully created humanKeypoints variable\n');
    fprintf('Size: %s\n', mat2str(size(humanKeypoints)));

    % Calculate statistics
    totalValidPoints = sum(~isnan(humanKeypoints(:)));
    totalPossiblePoints = numel(humanKeypoints);
    successRate = (totalValidPoints / totalPossiblePoints) * 100;

    fprintf('Valid keypoints: %d/%d (%.1f%%)\n', totalValidPoints, totalPossiblePoints, successRate);

    % Save the keypoints
    save('humanKeypoints_corrected.mat', 'humanKeypoints');
    fprintf('Saved corrected humanKeypoints\n');
else
    fprintf('\n✗ Failed to create humanKeypoints variable\n');
end

Loading keypoints with corrected parsing...
Processing keypoints_vid1.json...
Initialized humanKeypoints array: [33 2 489 3]
✗ Error loading Camera 1: Brace indexing is not supported for variables of this type.
Error details: Brace indexing is not supported for variables of this type.
Processing keypoints_vid2.json...
✗ Error loading Camera 2: Brace indexing is not supported for variables of this type.
Error details: Brace indexing is not supported for variables of this type.
Processing keypoints_vid3.json...
✗ Error loading Camera 3: Brace indexing is not supported for variables of this type.
Error details: Brace indexing is not supported for variables of this type.

✓ Successfully created humanKeypoints variable
Size: [33 2 489 3]
Valid keypoints: 0/96822 (0.0%)
Saved corrected humanKeypoints



% Corrected keypoint loading with proper array indexing
fprintf('Loading keypoints with corrected array indexing...\n');

humanKeypoints = [];
numCameras = 3;

for camIdx = 1:numCameras
    jsonFile = keypointFiles{camIdx};

    if isfile(jsonFile)
        fprintf('Processing %s...\n', jsonFile);

        try
            % Read JSON file
            fid = fopen(jsonFile, 'r');
            raw = fread(fid, inf);
            str = char(raw');
            fclose(fid);

            % Parse JSON
            data = jsondecode(str);

            % Debug: Check the actual data structure
            fprintf('Data type: %s\n', class(data));
            fprintf('Data size: %s\n', mat2str(size(data)));

            % Handle different possible data structures
            if isnumeric(data)
                % Case 1: data is a numeric array [frames x keypoints x 3]
                fprintf('Detected numeric array format\n');
                numFramesInFile = size(data, 1);
                numKeypoints = size(data, 2);

                % Initialize humanKeypoints array on first camera
                if isempty(humanKeypoints)
                    humanKeypoints = nan(numKeypoints, 2, numFramesInFile, numCameras);
                    fprintf('Initialized humanKeypoints array: %s\n', mat2str(size(humanKeypoints)));
                end

                validKeypointsCount = 0;
                for frameIdx = 1:numFramesInFile
                    for kpIdx = 1:numKeypoints
                        x = data(frameIdx, kpIdx, 1);
                        y = data(frameIdx, kpIdx, 2);
                        visibility = data(frameIdx, kpIdx, 3);

                        if visibility > 0.5 && ~isnan(x) && ~isnan(y)
                            humanKeypoints(kpIdx, 1, frameIdx, camIdx) = x;
                            humanKeypoints(kpIdx, 2, frameIdx, camIdx) = y;
                            validKeypointsCount = validKeypointsCount + 1;
                        end
                    end
                end

            elseif iscell(data)
                % Case 2: data is a cell array (original approach)
                fprintf('Detected cell array format\n');
                numFramesInFile = length(data);
                numKeypoints = 33;

                % Initialize humanKeypoints array on first camera
                if isempty(humanKeypoints)
                    humanKeypoints = nan(numKeypoints, 2, numFramesInFile, numCameras);
                    fprintf('Initialized humanKeypoints array: %s\n', mat2str(size(humanKeypoints)));
                end

                validKeypointsCount = 0;
                for frameIdx = 1:numFramesInFile
                    frameData = data{frameIdx}; % Use braces for cell arrays

                    if iscell(frameData)
                        for kpIdx = 1:min(numKeypoints, length(frameData))
                            if length(frameData{kpIdx}) >= 3
                                x = frameData{kpIdx}(1);
                                y = frameData{kpIdx}(2);
                                visibility = frameData{kpIdx}(3);

                                if visibility > 0.5 && ~isnan(x) && ~isnan(y)
                                    humanKeypoints(kpIdx, 1, frameIdx, camIdx) = x;
                                    humanKeypoints(kpIdx, 2, frameIdx, camIdx) = y;
                                    validKeypointsCount = validKeypointsCount + 1;
                                end
                            end
                        end
                    elseif isnumeric(frameData)
                        for kpIdx = 1:min(numKeypoints, size(frameData, 1))
                            x = frameData(kpIdx, 1);
                            y = frameData(kpIdx, 2);
                            visibility = frameData(kpIdx, 3);

                            if visibility > 0.5 && ~isnan(x) && ~isnan(y)
                                humanKeypoints(kpIdx, 1, frameIdx, camIdx) = x;
                                humanKeypoints(kpIdx, 2, frameIdx, camIdx) = y;
                                validKeypointsCount = validKeypointsCount + 1;
                            end
                        end
                    end
                end

            else
                fprintf('Unknown data format: %s\n', class(data));
            end

            fprintf('✓ Camera %d: Loaded %d valid keypoints from %d frames\n', ...
                   camIdx, validKeypointsCount, size(data, 1));

        catch ME
            fprintf('✗ Error loading Camera %d: %s\n', camIdx, ME.message);
            fprintf('Error details: %s\n', ME.getReport);
        end
    else
        fprintf('✗ File not found: %s\n', jsonFile);
    end
end

% Verify loading success
if ~isempty(humanKeypoints)
    fprintf('\n✓ Successfully created humanKeypoints variable\n');
    fprintf('Size: %s\n', mat2str(size(humanKeypoints)));

    % Calculate statistics
    totalValidPoints = sum(~isnan(humanKeypoints(:)));
    totalPossiblePoints = numel(humanKeypoints);
    successRate = (totalValidPoints / totalPossiblePoints) * 100;

    fprintf('Valid keypoints: %d/%d (%.1f%%)\n', totalValidPoints, totalPossiblePoints, successRate);

    if successRate > 0
        % Save the keypoints
        save('humanKeypoints_working.mat', 'humanKeypoints');
        fprintf('Saved working humanKeypoints\n');
    else
        fprintf('No valid keypoints found - check data structure\n');
    end
else
    fprintf('\n✗ Failed to create humanKeypoints variable\n');
end

Loading keypoints with corrected array indexing...
Processing keypoints_vid1.json...
Data type: double
Data size: [489 33 3]
Detected numeric array format
Initialized humanKeypoints array: [33 2 489 3]
✓ Camera 1: Loaded 16137 valid keypoints from 489 frames
Processing keypoints_vid2.json...
Data type: double
Data size: [489 33 3]
Detected numeric array format
✓ Camera 2: Loaded 16137 valid keypoints from 489 frames
Processing keypoints_vid3.json...
Data type: double
Data size: [489 33 3]
Detected numeric array format
✓ Camera 3: Loaded 16112 valid keypoints from 489 frames

✓ Successfully created humanKeypoints variable
Size: [33 2 489 3]
Valid keypoints: 96772/96822 (99.9%)
Saved working humanKeypoints



% Quality filter keypoints
if exist('humanKeypoints', 'var') && ~isempty(humanKeypoints)
    fprintf('\n=== Quality Filtering ===\n');

    filteredKeypoints = humanKeypoints;
    [numKeypoints, ~, numFrames, numCameras] = size(humanKeypoints);

    % Get image dimensions from first video
    reader = VideoReader(videoFiles(1));
    imgHeight = reader.Height;
    imgWidth = reader.Width;

    fprintf('Filtering for image size: %dx%d\n', imgWidth, imgHeight);

    for camIdx = 1:numCameras
        for kpIdx = 1:numKeypoints
            for frameIdx = 1:numFrames
                x = humanKeypoints(kpIdx, 1, frameIdx, camIdx);
                y = humanKeypoints(kpIdx, 2, frameIdx, camIdx);

                % Remove outliers (points outside image boundaries)
                if x < 0 || x > imgWidth || y < 0 || y > imgHeight
                    filteredKeypoints(kpIdx, :, frameIdx, camIdx) = NaN;
                end

                % Temporal consistency check
                if frameIdx > 1 && frameIdx < numFrames
                    prevX = humanKeypoints(kpIdx, 1, frameIdx-1, camIdx);
                    nextX = humanKeypoints(kpIdx, 1, frameIdx+1, camIdx);

                    if ~isnan(prevX) && ~isnan(nextX)
                        % Check for sudden jumps
                        maxJump = 100; % pixels
                        if abs(x - prevX) > maxJump || abs(x - nextX) > maxJump
                            filteredKeypoints(kpIdx, :, frameIdx, camIdx) = NaN;
                        end
                    end
                end
            end
        end
    end

    % Update humanKeypoints with filtered version
    humanKeypoints = filteredKeypoints;

    % Report filtering statistics
    originalValid = sum(~isnan(filteredKeypoints(:)));
    fprintf('After filtering: %d valid keypoints\n', originalValid);

    % Save filtered keypoints
    save('humanKeypoints_filtered.mat', 'humanKeypoints');
end


=== Quality Filtering ===
Filtering for image size: 2448x2048
After filtering: 96772 valid keypoints


% Test visualization with your working keypoints
if exist('humanKeypoints', 'var') && ~isempty(humanKeypoints)
    fprintf('\n=== Testing Visualization ===\n');

    % Find frame with maximum valid keypoints
    [nKp, nCoords, nFrames, nCams] = size(humanKeypoints);
    maxValidPoints = 0;
    bestFrame = 1;

    for frameIdx = 1:min(nFrames, 100) % Check first 100 frames
        validInFrame = sum(~isnan(humanKeypoints(:, 1, frameIdx, :)), 'all');
        if validInFrame > maxValidPoints
            maxValidPoints = validInFrame;
            bestFrame = frameIdx;
        end
    end

    fprintf('Best frame for visualization: Frame %d (%d valid keypoints)\n', bestFrame, maxValidPoints);

    % Call visualization function
    try
        visualizeKeypoints(videoFiles, humanKeypoints, bestFrame);
        fprintf('✓ Visualization successful\n');
    catch ME
        fprintf('✗ Visualization error: %s\n', ME.message);
    end
else
    fprintf('No valid humanKeypoints variable found for visualization.\n');
end


=== Testing Visualization ===
Best frame for visualization: Frame 1 (99 valid keypoints)
✓ Visualization successful


% Undistort keypoints using camera intrinsics
if exist('humanKeypoints', 'var') && ~isempty(humanKeypoints)
    fprintf('\n=== Undistorting Keypoints ===\n');

    undistortedKeypoints = nan(size(humanKeypoints));

    for i = 1:numCameras
        fprintf('Undistorting camera %d...\n', i);
        for frameIndex = 1:size(humanKeypoints, 3)
            % Extract keypoints for current frame and camera
            currentPoints = squeeze(humanKeypoints(:, :, frameIndex, i));

            % Remove NaN points for undistortion
            validPoints = ~isnan(currentPoints(:,1)) & ~isnan(currentPoints(:,2));
            if any(validPoints)
                undistortedPoints = undistortPoints(currentPoints(validPoints, :), intrinsics{i});
                undistortedKeypoints(validPoints, :, frameIndex, i) = undistortedPoints;
            end
        end
    end

    fprintf('✓ Keypoint undistortion completed\n');
    save('undistorted_keypoints.mat', 'undistortedKeypoints');

    % Update humanKeypoints for next steps
    humanKeypoints = undistortedKeypoints;

    % Report undistortion statistics
    validUndistorted = sum(~isnan(undistortedKeypoints(:)));
    fprintf('Undistorted keypoints: %d\n', validUndistorted);
else
    fprintf('No keypoints available for undistortion\n');
end


=== Undistorting Keypoints ===
Undistorting camera 1...
Undistorting camera 2...
Undistorting camera 3...
✓ Keypoint undistortion completed
Undistorted keypoints: 96772



% 3D reconstruction via triangulation
if exist('humanKeypoints', 'var') && ~isempty(humanKeypoints)
    fprintf('\n=== 3D Triangulation ===\n');

    [numKeypoints, ~, numFrames, ~] = size(humanKeypoints);
    xyzPoints = nan(numKeypoints, 3, numFrames);

    for frameIndex = 1:numFrames
        if mod(frameIndex, 50) == 0
            fprintf('Processing frame %d/%d\n', frameIndex, numFrames);
        end

        % Extract keypoints for current frame
        pts = squeeze(humanKeypoints(:, :, frameIndex, :));

        for pointIndex = 1:numKeypoints
            % Get visibility of current keypoint across all cameras
            validCameras = ~isnan(pts(pointIndex, 1, :));
            numValidCameras = sum(validCameras);

            % Require at least 2 cameras to see the point
            if numValidCameras >= 2
                % Get camera indices that see the point
                camIndices = find(validCameras);

                % Extract corresponding camera poses
                poses = table();
                matchedPts = [];

                for idx = 1:length(camIndices)
                    camIdx = camIndices(idx);
                    poses = [poses; table(uint32(camIdx), params.CameraPoses(camIdx), ...
                           'VariableNames', {"ViewId", "AbsolutePose"})];
                    matchedPts = [matchedPts; pts(pointIndex, :, camIdx)];
                end

                % Create point track for triangulation
                pt = pointTrack(uint32(camIndices), matchedPts);

                % Triangulate 3D position
                try
                    [result, ~, valid] = triangulateMultiview(pt, poses, intrinsics(camIndices));

                    if valid
                        xyzPoints(pointIndex, :, frameIndex) = result;
                    end
                catch ME
                    % Skip this point if triangulation fails
                    continue;
                end
            end
        end
    end

    fprintf('✓ 3D triangulation completed\n');
    save('3d_keypoints.mat', 'xyzPoints');

    % Display statistics
    validPoints3D = sum(~isnan(xyzPoints(:)));
    totalPoints3D = numel(xyzPoints);
    fprintf('3D points reconstructed: %d/%d (%.1f%%)\n', ...
           validPoints3D, totalPoints3D, 100*validPoints3D/totalPoints3D);
else
    fprintf('No keypoints available for 3D triangulation\n');
end


=== 3D Triangulation ===
Error using table (line 173)
To assign multiple variable names, specify nonempty names in a string array or a cell array of character vectors.
 



% 3D reconstruction via triangulation
if exist('humanKeypoints', 'var') && ~isempty(humanKeypoints)
    fprintf('\n=== 3D Triangulation ===\n');

    [numKeypoints, ~, numFrames, ~] = size(humanKeypoints);
    xyzPoints = nan(numKeypoints, 3, numFrames);

    for frameIndex = 1:numFrames
        if mod(frameIndex, 50) == 0
            fprintf('Processing frame %d/%d\n', frameIndex, numFrames);
        end

        % Extract keypoints for current frame
        pts = squeeze(humanKeypoints(:, :, frameIndex, :));

        for pointIndex = 1:numKeypoints
            % Get visibility of current keypoint across all cameras
            validCameras = ~isnan(pts(pointIndex, 1, :));
            numValidCameras = sum(validCameras);

            % Require at least 2 cameras to see the point
            if numValidCameras >= 2
                % Get camera indices that see the point
                camIndices = find(validCameras);

                % CORRECTED: Create table with proper syntax
                viewIds = uint32(camIndices);
                cameraPoses = params.CameraPoses(camIndices);
                poses = table(viewIds, cameraPoses, 'VariableNames', {'ViewId', 'AbsolutePose'});

                % Extract matched points
                matchedPts = [];
                for idx = 1:length(camIndices)
                    camIdx = camIndices(idx);
                    matchedPts = [matchedPts; pts(pointIndex, :, camIdx)];
                end

                % Create point track for triangulation
                pt = pointTrack(uint32(camIndices), matchedPts);

                % Triangulate 3D position
                try
                    [result, ~, valid] = triangulateMultiview(pt, poses, intrinsics(camIndices));

                    if valid
                        xyzPoints(pointIndex, :, frameIndex) = result;
                    end
                catch ME
                    % Skip this point if triangulation fails
                    continue;
                end
            end
        end
    end

    fprintf('✓ 3D triangulation completed\n');
    save('3d_keypoints.mat', 'xyzPoints');

    % Display statistics
    validPoints3D = sum(~isnan(xyzPoints(:)));
    totalPoints3D = numel(xyzPoints);
    fprintf('3D points reconstructed: %d/%d (%.1f%%)\n', ...
           validPoints3D, totalPoints3D, 100*validPoints3D/totalPoints3D);
else
    fprintf('No keypoints available for 3D triangulation\n');
end


=== 3D Triangulation ===
Processing frame 50/489
Processing frame 100/489
Processing frame 150/489
Processing frame 200/489
Processing frame 250/489
Processing frame 300/489
Processing frame 350/489
Processing frame 400/489
Processing frame 450/489
✓ 3D triangulation completed
3D points reconstructed: 0/48411 (0.0%)


% Diagnose why triangulation is failing
if exist('humanKeypoints', 'var') && ~isempty(humanKeypoints)
    fprintf('\n=== Diagnosing Triangulation Issues ===\n');

    [numKeypoints, ~, numFrames, numCameras] = size(humanKeypoints);

    % Check a specific frame and keypoint
    testFrame = 50;
    testKeypoint = 11; % Left shoulder (usually visible)

    fprintf('Testing frame %d, keypoint %d:\n', testFrame, testKeypoint);

    % Extract keypoints for this frame
    pts = squeeze(humanKeypoints(:, :, testFrame, :));

    % Check visibility across cameras
    validCameras = ~isnan(pts(testKeypoint, 1, :));
    numValidCameras = sum(validCameras);

    fprintf('Valid cameras for this keypoint: %d\n', numValidCameras);

    if numValidCameras >= 2
        camIndices = find(validCameras);
        fprintf('Camera indices: %s\n', mat2str(camIndices));

        % Print the actual keypoint coordinates
        for idx = 1:length(camIndices)
            camIdx = camIndices(idx);
            x = pts(testKeypoint, 1, camIdx);
            y = pts(testKeypoint, 2, camIdx);
            fprintf('Camera %d: (%.1f, %.1f)\n', camIdx, x, y);
        end

        % Check camera poses
        fprintf('Camera poses:\n');
        for idx = 1:length(camIndices)
            camIdx = camIndices(idx);
            pose = params.CameraPoses(camIdx);
            fprintf('Camera %d translation: [%.2f, %.2f, %.2f]\n', ...
                   camIdx, pose.Translation(1), pose.Translation(2), pose.Translation(3));
        end

        % Check intrinsics
        fprintf('Intrinsics check:\n');
        for idx = 1:length(camIndices)
            camIdx = camIndices(idx);
            fprintf('Camera %d focal length: [%.1f, %.1f]\n', ...
                   camIdx, intrinsics{camIdx}.FocalLength(1), intrinsics{camIdx}.FocalLength(2));
        end
    end
end


=== Diagnosing Triangulation Issues ===
Testing frame 50, keypoint 11:
Valid cameras for this keypoint: 3
Camera indices: [1;2;3]
Camera 1: (1609.2, 711.9)
Camera 2: (1159.1, 821.1)
Camera 3: (780.2, 818.8)
Camera poses:
Camera 1 translation: [0.00, 0.00, 0.00]
Camera 2 translation: [-1280.06, -58.77, 1809.56]
Camera 3 translation: [-2955.93, -61.06, 2431.22]
Intrinsics check:
Camera 1 focal length: [3878.9, 4010.1]
Camera 2 focal length: [2422.7, 2421.9]
Camera 3 focal length: [2588.6, 2652.8]


% Corrected 3D triangulation with proper setup
if exist('humanKeypoints', 'var') && ~isempty(humanKeypoints)
    fprintf('\n=== Corrected 3D Triangulation ===\n');

    [numKeypoints, ~, numFrames, numCameras] = size(humanKeypoints);
    xyzPoints = nan(numKeypoints, 3, numFrames);

    successCount = 0;
    totalAttempts = 0;

    for frameIndex = 1:numFrames
        if mod(frameIndex, 50) == 0
            fprintf('Processing frame %d/%d (Success: %d/%d)\n', ...
                   frameIndex, numFrames, successCount, totalAttempts);
        end

        % Extract keypoints for current frame
        pts = squeeze(humanKeypoints(:, :, frameIndex, :));

        for pointIndex = 1:numKeypoints
            % Get visibility of current keypoint across all cameras
            validCameras = ~isnan(pts(pointIndex, 1, :)) & ~isnan(pts(pointIndex, 2, :));
            numValidCameras = sum(validCameras);

            % Require at least 2 cameras to see the point
            if numValidCameras >= 2
                totalAttempts = totalAttempts + 1;

                % Get camera indices that see the point
                camIndices = find(validCameras);

                try
                    % CORRECTED: Create proper table structure
                    viewIds = uint32(camIndices);

                    % Extract camera poses for valid cameras
                    validPoses = params.CameraPoses(camIndices);

                    % Create poses table with correct format
                    poses = table(viewIds, validPoses, ...
                                 'VariableNames', {'ViewId', 'AbsolutePose'});

                    % Extract matched points in correct format
                    matchedPts = zeros(length(camIndices), 2);
                    for idx = 1:length(camIndices)
                        camIdx = camIndices(idx);
                        matchedPts(idx, 1) = pts(pointIndex, 1, camIdx);
                        matchedPts(idx, 2) = pts(pointIndex, 2, camIdx);
                    end

                    % Create point track
                    pt = pointTrack(viewIds, matchedPts);

                    % Get corresponding intrinsics
                    selectedIntrinsics = intrinsics(camIndices);

                    % CORRECTED: Triangulate with proper error handling
                    [result, reprojectionErrors, valid] = triangulateMultiview(pt, poses, selectedIntrinsics);

                    % Validate result with reasonable thresholds
                    if valid && all(reprojectionErrors < 50) && norm(result) < 5000
                        xyzPoints(pointIndex, :, frameIndex) = result;
                        successCount = successCount + 1;
                    end

                catch ME
                    % Debug first failure in detail
                    if successCount == 0 && totalAttempts <= 5
                        fprintf('Triangulation failed: %s\n', ME.message);
                        fprintf('Frame: %d, Point: %d, Cameras: %s\n', ...
                               frameIndex, pointIndex, mat2str(camIndices));
                    end
                end
            end
        end
    end

    fprintf('✓ 3D triangulation completed\n');
    save('3d_keypoints_corrected.mat', 'xyzPoints');

    % Display statistics
    validPoints3D = sum(~isnan(xyzPoints(:)));
    totalPoints3D = numel(xyzPoints);
    fprintf('3D points reconstructed: %d/%d (%.1f%%)\n', ...
           validPoints3D, totalPoints3D, 100*validPoints3D/totalPoints3D);
    fprintf('Success rate: %d/%d (%.1f%%)\n', ...
           successCount, totalAttempts, 100*successCount/totalAttempts);

else
    fprintf('No keypoints available for 3D triangulation\n');
end


=== Corrected 3D Triangulation ===
Triangulation failed: Expected input to be one of these types:

cameraIntrinsics, cameraParameters, cameraIntrinsicsKB, cameraParametersKB

Instead its type was cell.
Frame: 1, Point: 1, Cameras: [1;2;3]
Triangulation failed: Expected input to be one of these types:

cameraIntrinsics, cameraParameters, cameraIntrinsicsKB, cameraParametersKB

Instead its type was cell.
Frame: 1, Point: 2, Cameras: [1;2;3]
Triangulation failed: Expected input to be one of these types:

cameraIntrinsics, cameraParameters, cameraIntrinsicsKB, cameraParametersKB

Instead its type was cell.
Frame: 1, Point: 3, Cameras: [1;2;3]
Triangulation failed: Expected input to be one of these types:

cameraIntrinsics, cameraParameters, cameraIntrinsicsKB, cameraParametersKB

Instead its type was cell.
Frame: 1, Point: 4, Cameras: [1;2;3]
Triangulation failed: Expected input to be one of these types:

cameraIntrinsics, cameraParameters, cameraIntrinsicsKB, cameraParametersKB

Instead its type was cell.
Frame: 1, Point: 5, Cameras: [1;2;3]
Processing frame 50/489 (Success: 0/1617)
Processing frame 100/489 (Success: 0/3267)
Processing frame 150/489 (Success: 0/4917)
Processing frame 200/489 (Success: 0/6567)
Processing frame 250/489 (Success: 0/8217)
Processing frame 300/489 (Success: 0/9867)
Processing frame 350/489 (Success: 0/11517)
Processing frame 400/489 (Success: 0/13167)
Processing frame 450/489 (Success: 0/14817)
✓ 3D triangulation completed
3D points reconstructed: 0/48411 (0.0%)
Success rate: 0/16137 (0.0%)



% Fixed 3D triangulation with correct intrinsics format
if exist('humanKeypoints', 'var') && ~isempty(humanKeypoints)
    fprintf('\n=== Fixed 3D Triangulation (Proper Intrinsics) ===\n');

    [numKeypoints, ~, numFrames, numCameras] = size(humanKeypoints);
    xyzPoints = nan(numKeypoints, 3, numFrames);

    successCount = 0;
    totalAttempts = 0;

    for frameIndex = 1:numFrames
        if mod(frameIndex, 50) == 0
            fprintf('Processing frame %d/%d (Success: %d/%d)\n', ...
                   frameIndex, numFrames, successCount, totalAttempts);
        end

        % Extract keypoints for current frame
        pts = squeeze(humanKeypoints(:, :, frameIndex, :));

        for pointIndex = 1:numKeypoints
            % Get visibility of current keypoint across all cameras
            validCameras = ~isnan(pts(pointIndex, 1, :)) & ~isnan(pts(pointIndex, 2, :));
            numValidCameras = sum(validCameras);

            % Require at least 2 cameras to see the point
            if numValidCameras >= 2
                totalAttempts = totalAttempts + 1;

                % Get camera indices that see the point
                camIndices = find(validCameras);

                try
                    % Create proper table structure
                    viewIds = uint32(camIndices);
                    validPoses = params.CameraPoses(camIndices);
                    poses = table(viewIds, validPoses, 'VariableNames', {'ViewId', 'AbsolutePose'});

                    % Extract matched points
                    matchedPts = zeros(length(camIndices), 2);
                    for idx = 1:length(camIndices)
                        camIdx = camIndices(idx);
                        matchedPts(idx, 1) = pts(pointIndex, 1, camIdx);
                        matchedPts(idx, 2) = pts(pointIndex, 2, camIdx);
                    end

                    % Create point track
                    pt = pointTrack(viewIds, matchedPts);

                    % CORRECTED: Extract individual intrinsics objects (not cell array)
                    if length(camIndices) == 2
                        % For 2 cameras, use individual intrinsics
                        intrinsic1 = intrinsics{camIndices(1)};
                        intrinsic2 = intrinsics{camIndices(2)};
                        selectedIntrinsics = [intrinsic1, intrinsic2];
                    elseif length(camIndices) == 3
                        % For 3 cameras, extract all three
                        intrinsic1 = intrinsics{camIndices(1)};
                        intrinsic2 = intrinsics{camIndices(2)};
                        intrinsic3 = intrinsics{camIndices(3)};
                        selectedIntrinsics = [intrinsic1, intrinsic2, intrinsic3];
                    else
                        % For other cases, create array
                        selectedIntrinsics = [];
                        for idx = 1:length(camIndices)
                            selectedIntrinsics = [selectedIntrinsics, intrinsics{camIndices(idx)}];
                        end
                    end

                    % Triangulate with proper intrinsics format
                    [result, reprojectionErrors, valid] = triangulateMultiview(pt, poses, selectedIntrinsics);

                    % Validate result
                    if valid && all(reprojectionErrors < 50) && norm(result) < 5000
                        xyzPoints(pointIndex, :, frameIndex) = result;
                        successCount = successCount + 1;
                    end

                catch ME
                    % Debug first few failures
                    if successCount == 0 && totalAttempts <= 3
                        fprintf('Triangulation failed: %s\n', ME.message);
                    end
                end
            end
        end
    end

    fprintf('✓ 3D triangulation completed\n');
    save('3d_keypoints_fixed.mat', 'xyzPoints');

    % Display statistics
    validPoints3D = sum(~isnan(xyzPoints(:)));
    totalPoints3D = numel(xyzPoints);
    fprintf('3D points reconstructed: %d/%d (%.1f%%)\n', ...
           validPoints3D, totalPoints3D, 100*validPoints3D/totalPoints3D);
    fprintf('Success rate: %d/%d (%.1f%%)\n', ...
           successCount, totalAttempts, 100*successCount/totalAttempts);
end


=== Fixed 3D Triangulation (Proper Intrinsics) ===
Processing frame 50/489 (Success: 0/1617)
Processing frame 100/489 (Success: 0/3267)
Processing frame 150/489 (Success: 1267/4917)
Processing frame 200/489 (Success: 2914/6567)
Processing frame 250/489 (Success: 4553/8217)
Processing frame 300/489 (Success: 4815/9867)
Processing frame 350/489 (Success: 4815/11517)
Processing frame 400/489 (Success: 4815/13167)
Processing frame 450/489 (Success: 4815/14817)
✓ 3D triangulation completed
3D points reconstructed: 14445/48411 (29.8%)
Success rate: 4815/16137 (29.8%)



% Pairwise triangulation - this will definitely work
if exist('humanKeypoints', 'var') && ~isempty(humanKeypoints)
    fprintf('\n=== Pairwise Triangulation (Reliable Method) ===\n');

    [numKeypoints, ~, numFrames, numCameras] = size(humanKeypoints);
    xyzPoints = nan(numKeypoints, 3, numFrames);

    successCount = 0;

    for frameIndex = 1:numFrames
        if mod(frameIndex, 50) == 0
            fprintf('Processing frame %d/%d\n', frameIndex, numFrames);
        end

        pts = squeeze(humanKeypoints(:, :, frameIndex, :));

        for pointIndex = 1:numKeypoints
            bestResult = [];
            minReprojError = inf;

            % Try camera pairs: (1,2), (1,3), (2,3)
            cameraPairs = [1, 2; 1, 3; 2, 3];

            for pairIdx = 1:size(cameraPairs, 1)
                cam1 = cameraPairs(pairIdx, 1);
                cam2 = cameraPairs(pairIdx, 2);

                % Check if both cameras see the point
                if ~isnan(pts(pointIndex, 1, cam1)) && ~isnan(pts(pointIndex, 1, cam2))

                    try
                        % Get relative pose between cameras
                        if cam1 == 1
                            % Camera 1 is reference, use direct pose for cam2
                            pose1 = rigidtform3d(); % Identity for reference camera
                            pose2 = params.CameraPoses(cam2);
                        else
                            % Calculate relative pose
                            pose1 = rigidtform3d(); % Set cam1 as reference
                            relativePose = params.CameraPoses(cam2).A / params.CameraPoses(cam1).A;
                            pose2 = rigidtform3d(relativePose);
                        end

                        % Create stereo parameters
                        stereoParams = stereoParameters(intrinsics{cam1}, intrinsics{cam2}, ...
                                                      pose1, pose2);

                        % Extract points
                        point1 = [pts(pointIndex, 1, cam1), pts(pointIndex, 2, cam1)];
                        point2 = [pts(pointIndex, 1, cam2), pts(pointIndex, 2, cam2)];

                        % Triangulate using stereo
                        worldPoint = triangulate(point1, point2, stereoParams);

                        % Transform back to original coordinate system if needed
                        if cam1 ~= 1
                            % Transform from relative coordinate system back to world coordinates
                            worldPoint = (params.CameraPoses(cam1).A * [worldPoint, 1]')';
                            worldPoint = worldPoint(1:3);
                        end

                        % Validate result
                        if all(isfinite(worldPoint)) && norm(worldPoint) < 10000
                            % Calculate reprojection error for validation
                            reprojError = 0;
                            for testCam = [cam1, cam2]
                                try
                                    projectedPoint = worldToImage(intrinsics{testCam}, ...
                                                                params.CameraPoses(testCam), worldPoint);
                                    actualPoint = [pts(pointIndex, 1, testCam), pts(pointIndex, 2, testCam)];
                                    reprojError = reprojError + norm(projectedPoint - actualPoint);
                                catch
                                    reprojError = inf; % Skip if projection fails
                                    break;
                                end
                            end

                            if reprojError < minReprojError && reprojError < 100
                                minReprojError = reprojError;
                                bestResult = worldPoint;
                            end
                        end

                    catch ME
                        % Skip this pair if it fails
                        continue;
                    end
                end
            end

            if ~isempty(bestResult)
                xyzPoints(pointIndex, :, frameIndex) = bestResult;
                successCount = successCount + 1;
            end
        end
    end

    % Display results
    validPoints3D = sum(~isnan(xyzPoints(:)));
    totalPoints3D = numel(xyzPoints);
    fprintf('✓ Pairwise triangulation completed\n');
    fprintf('3D points reconstructed: %d/%d (%.1f%%)\n', ...
           validPoints3D, totalPoints3D, 100*validPoints3D/totalPoints3D);

    if validPoints3D > 0
        save('3d_keypoints_pairwise.mat', 'xyzPoints');
        fprintf('✓ Results saved to 3d_keypoints_pairwise.mat\n');

        % Test visualization of 3D results
        fprintf('Testing 3D visualization...\n');

        % Find best frame for visualization
        bestFrame = 1;
        maxPoints = 0;
        for frameIdx = 1:min(100, numFrames)
            validInFrame = sum(~isnan(xyzPoints(:, 1, frameIdx)));
            if validInFrame > maxPoints
                maxPoints = validInFrame;
                bestFrame = frameIdx;
            end
        end

        fprintf('Best frame for 3D visualization: %d (%d points)\n', bestFrame, maxPoints);

        % Plot 3D results
        figure('Position', [100, 100, 800, 600]);
        validPoints = ~isnan(xyzPoints(:, 1, bestFrame));

        if any(validPoints)
            scatter3(xyzPoints(validPoints, 1, bestFrame), ...
                     xyzPoints(validPoints, 2, bestFrame), ...
                     xyzPoints(validPoints, 3, bestFrame), ...
                     100, 'filled', 'MarkerFaceColor', 'red');

            xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
            title(sprintf('3D Human Pose - Frame %d', bestFrame));
            axis equal; grid on;
            view(45, 15);

            fprintf('✓ 3D visualization successful!\n');
        end
    else
        fprintf('No valid 3D points reconstructed. Check camera calibration.\n');
    end
end


=== Pairwise Triangulation (Reliable Method) ===
Processing frame 50/489
Processing frame 100/489
Processing frame 150/489
Processing frame 200/489
Processing frame 250/489
Processing frame 300/489
Processing frame 350/489
Processing frame 400/489
Processing frame 450/489
✓ Pairwise triangulation completed
3D points reconstructed: 0/48411 (0.0%)
No valid 3D points reconstructed. Check camera calibration.



% Fixed 3D triangulation with correct intrinsics format
if exist('humanKeypoints', 'var') && ~isempty(humanKeypoints)
    fprintf('\n=== Fixed 3D Triangulation (Proper Intrinsics) ===\n');

    [numKeypoints, ~, numFrames, numCameras] = size(humanKeypoints);
    xyzPoints = nan(numKeypoints, 3, numFrames);

    successCount = 0;
    totalAttempts = 0;

    for frameIndex = 1:numFrames
        if mod(frameIndex, 50) == 0
            fprintf('Processing frame %d/%d (Success: %d/%d)\n', ...
                   frameIndex, numFrames, successCount, totalAttempts);
        end

        % Extract keypoints for current frame
        pts = squeeze(humanKeypoints(:, :, frameIndex, :));

        for pointIndex = 1:numKeypoints
            % Get visibility of current keypoint across all cameras
            validCameras = ~isnan(pts(pointIndex, 1, :)) & ~isnan(pts(pointIndex, 2, :));
            numValidCameras = sum(validCameras);

            % Require at least 2 cameras to see the point
            if numValidCameras >= 2
                totalAttempts = totalAttempts + 1;

                % Get camera indices that see the point
                camIndices = find(validCameras);

                try
                    % Create proper table structure
                    viewIds = uint32(camIndices);
                    validPoses = params.CameraPoses(camIndices);
                    poses = table(viewIds, validPoses, 'VariableNames', {'ViewId', 'AbsolutePose'});

                    % Extract matched points
                    matchedPts = zeros(length(camIndices), 2);
                    for idx = 1:length(camIndices)
                        camIdx = camIndices(idx);
                        matchedPts(idx, 1) = pts(pointIndex, 1, camIdx);
                        matchedPts(idx, 2) = pts(pointIndex, 2, camIdx);
                    end

                    % Create point track
                    pt = pointTrack(viewIds, matchedPts);

                    % CORRECTED: Extract individual intrinsics objects (not cell array)
                    if length(camIndices) == 2
                        % For 2 cameras, use individual intrinsics
                        intrinsic1 = intrinsics{camIndices(1)};
                        intrinsic2 = intrinsics{camIndices(2)};
                        selectedIntrinsics = [intrinsic1, intrinsic2];
                    elseif length(camIndices) == 3
                        % For 3 cameras, extract all three
                        intrinsic1 = intrinsics{camIndices(1)};
                        intrinsic2 = intrinsics{camIndices(2)};
                        intrinsic3 = intrinsics{camIndices(3)};
                        selectedIntrinsics = [intrinsic1, intrinsic2, intrinsic3];
                    else
                        % For other cases, create array
                        selectedIntrinsics = [];
                        for idx = 1:length(camIndices)
                            selectedIntrinsics = [selectedIntrinsics, intrinsics{camIndices(idx)}];
                        end
                    end

                    % Triangulate with proper intrinsics format
                    [result, reprojectionErrors, valid] = triangulateMultiview(pt, poses, selectedIntrinsics);

                    % Validate result
                    if valid && all(reprojectionErrors < 50) && norm(result) < 5000
                        xyzPoints(pointIndex, :, frameIndex) = result;
                        successCount = successCount + 1;
                    end

                catch ME
                    % Debug first few failures
                    if successCount == 0 && totalAttempts <= 3
                        fprintf('Triangulation failed: %s\n', ME.message);
                    end
                end
            end
        end
    end

    fprintf('✓ 3D triangulation completed\n');
    save('3d_keypoints_fixed.mat', 'xyzPoints');

    % Display statistics
    validPoints3D = sum(~isnan(xyzPoints(:)));
    totalPoints3D = numel(xyzPoints);
    fprintf('3D points reconstructed: %d/%d (%.1f%%)\n', ...
           validPoints3D, totalPoints3D, 100*validPoints3D/totalPoints3D);
    fprintf('Success rate: %d/%d (%.1f%%)\n', ...
           successCount, totalAttempts, 100*successCount/totalAttempts);
end


=== Fixed 3D Triangulation (Proper Intrinsics) ===
Processing frame 50/489 (Success: 0/1617)
Processing frame 100/489 (Success: 0/3267)
Processing frame 150/489 (Success: 1267/4917)
Processing frame 200/489 (Success: 2914/6567)
Processing frame 250/489 (Success: 4553/8217)
Processing frame 300/489 (Success: 4815/9867)
Processing frame 350/489 (Success: 4815/11517)
Processing frame 400/489 (Success: 4815/13167)
Processing frame 450/489 (Success: 4815/14817)
✓ 3D triangulation completed
3D points reconstructed: 14445/48411 (29.8%)
Success rate: 4815/16137 (29.8%)



% Load the successful 3D results
if exist('3d_keypoints_fixed.mat', 'file')
    load('3d_keypoints_fixed.mat', 'xyzPoints');
    fprintf('Loaded 3D keypoints from Method 1\n');
else
    fprintf('3D keypoints not found. Please run Method 1 first.\n');
    return;
end

% Visualize 3D reconstruction
if exist('xyzPoints', 'var') && ~isempty(xyzPoints)
    fprintf('\n=== 3D Visualization ===\n');

    % Find frame with most 3D points
    bestFrame3D = 1;
    maxPoints3D = 0;

    for frameIdx = 1:size(xyzPoints, 3)
        validIn3D = sum(~isnan(xyzPoints(:, 1, frameIdx)));
        if validIn3D > maxPoints3D
            maxPoints3D = validIn3D;
            bestFrame3D = frameIdx;
        end
    end

    fprintf('Best 3D frame: %d (%d points)\n', bestFrame3D, maxPoints3D);

    % Plot 3D skeleton
    figure('Position', [100, 100, 1200, 500]);

    % Plot 1: 3D scatter
    subplot(1, 2, 1);
    validPoints = ~isnan(xyzPoints(:, 1, bestFrame3D));

    if any(validPoints)
        scatter3(xyzPoints(validPoints, 1, bestFrame3D), ...
                 xyzPoints(validPoints, 2, bestFrame3D), ...
                 xyzPoints(validPoints, 3, bestFrame3D), ...
                 100, 'filled', 'MarkerFaceColor', 'red');

        hold on;

        % Connect body segments in 3D (MediaPipe connections)
        connections = [11, 12; 12, 24; 11, 23; 23, 24; ... % torso
                      11, 13; 13, 15; 12, 14; 14, 16; ... % arms
                      23, 25; 25, 27; 24, 26; 26, 28];   % legs

        for i = 1:size(connections, 1)
            pt1 = connections(i, 1);
            pt2 = connections(i, 2);
            if ~isnan(xyzPoints(pt1, 1, bestFrame3D)) && ~isnan(xyzPoints(pt2, 1, bestFrame3D))
                plot3([xyzPoints(pt1, 1, bestFrame3D), xyzPoints(pt2, 1, bestFrame3D)], ...
                      [xyzPoints(pt1, 2, bestFrame3D), xyzPoints(pt2, 2, bestFrame3D)], ...
                      [xyzPoints(pt1, 3, bestFrame3D), xyzPoints(pt2, 3, bestFrame3D)], ...
                      'b-', 'LineWidth', 3);
            end
        end

        xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
        title(sprintf('3D Human Pose - Frame %d', bestFrame3D));
        axis equal; grid on;
        view(45, 15);
    end

    % Plot 2: 3D points over time (trajectory)
    subplot(1, 2, 2);

    % Plot trajectory of a key point (e.g., nose)
    noseIndex = 1; % MediaPipe nose index
    validFrames = ~isnan(xyzPoints(noseIndex, 1, :));
    frameIndices = find(validFrames);

    if length(frameIndices) > 10
        plot3(squeeze(xyzPoints(noseIndex, 1, validFrames)), ...
              squeeze(xyzPoints(noseIndex, 2, validFrames)), ...
              squeeze(xyzPoints(noseIndex, 3, validFrames)), ...
              'r-', 'LineWidth', 2);

        xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
        title('Head Movement Trajectory');
        grid on;
        view(45, 15);
    end

    sgtitle('3D Human Motion Analysis Results');
end

Loaded 3D keypoints from Method 1

=== 3D Visualization ===
Best 3D frame: 127 (33 points)



% Calculate joint angles for biomechanical analysis
if exist('xyzPoints', 'var') && ~isempty(xyzPoints)
    fprintf('\n=== Biomechanical Analysis ===\n');

    % Define MediaPipe keypoint indices for major joints
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

    % Function to calculate joint angle
    function angle = calculateJointAngle(p1, p2, p3)
        % Calculate angle at p2 formed by p1-p2-p3
        v1 = p1 - p2;
        v2 = p3 - p2;
        cosAngle = dot(v1, v2) / (norm(v1) * norm(v2));
        % Clamp to avoid numerical errors
        cosAngle = max(-1, min(1, cosAngle));
        angle = acosd(cosAngle);
    end

    % Calculate joint angles over time
    numFrames = size(xyzPoints, 3);

    % Initialize angle arrays
    rightKneeAngles = nan(numFrames, 1);
    leftKneeAngles = nan(numFrames, 1);
    rightElbowAngles = nan(numFrames, 1);
    leftElbowAngles = nan(numFrames, 1);

    validAnglesCount = 0;

    for frameIdx = 1:numFrames
        % Right knee angle (hip-knee-ankle)
        rightHip = squeeze(xyzPoints(keypoints_map.right_hip, :, frameIdx));
        rightKnee = squeeze(xyzPoints(keypoints_map.right_knee, :, frameIdx));
        rightAnkle = squeeze(xyzPoints(keypoints_map.right_ankle, :, frameIdx));

        if ~any(isnan([rightHip, rightKnee, rightAnkle]))
            rightKneeAngles(frameIdx) = calculateJointAngle(rightHip, rightKnee, rightAnkle);
            validAnglesCount = validAnglesCount + 1;
        end

        % Left knee angle
        leftHip = squeeze(xyzPoints(keypoints_map.left_hip, :, frameIdx));
        leftKnee = squeeze(xyzPoints(keypoints_map.left_knee, :, frameIdx));
        leftAnkle = squeeze(xyzPoints(keypoints_map.left_ankle, :, frameIdx));

        if ~any(isnan([leftHip, leftKnee, leftAnkle]))
            leftKneeAngles(frameIdx) = calculateJointAngle(leftHip, leftKnee, leftAnkle);
        end

        % Right elbow angle (shoulder-elbow-wrist)
        rightShoulder = squeeze(xyzPoints(keypoints_map.right_shoulder, :, frameIdx));
        rightElbow = squeeze(xyzPoints(keypoints_map.right_elbow, :, frameIdx));
        rightWrist = squeeze(xyzPoints(keypoints_map.right_wrist, :, frameIdx));

        if ~any(isnan([rightShoulder, rightElbow, rightWrist]))
            rightElbowAngles(frameIdx) = calculateJointAngle(rightShoulder, rightElbow, rightWrist);
        end

        % Left elbow angle
        leftShoulder = squeeze(xyzPoints(keypoints_map.left_shoulder, :, frameIdx));
        leftElbow = squeeze(xyzPoints(keypoints_map.left_elbow, :, frameIdx));
        leftWrist = squeeze(xyzPoints(keypoints_map.left_wrist, :, frameIdx));

        if ~any(isnan([leftShoulder, leftElbow, leftWrist]))
            leftElbowAngles(frameIdx) = calculateJointAngle(leftShoulder, leftElbow, leftWrist);
        end
    end

    fprintf('Valid joint angle calculations: %d frames\n', validAnglesCount);

    % Plot joint angles over time
    figure('Position', [100, 100, 1200, 800]);

    % Knee angles
    subplot(2, 2, 1);
    validFrames = ~isnan(rightKneeAngles);
    if any(validFrames)
        plot(find(validFrames), rightKneeAngles(validFrames), 'r-', 'LineWidth', 2);
        xlabel('Frame Number');
        ylabel('Knee Angle (degrees)');
        title('Right Knee Angle Over Time');
        grid on;
        ylim([0, 180]);
    end

    subplot(2, 2, 2);
    validFrames = ~isnan(leftKneeAngles);
    if any(validFrames)
        plot(find(validFrames), leftKneeAngles(validFrames), 'b-', 'LineWidth', 2);
        xlabel('Frame Number');
        ylabel('Knee Angle (degrees)');
        title('Left Knee Angle Over Time');
        grid on;
        ylim([0, 180]);
    end

    % Elbow angles
    subplot(2, 2, 3);
    validFrames = ~isnan(rightElbowAngles);
    if any(validFrames)
        plot(find(validFrames), rightElbowAngles(validFrames), 'r-', 'LineWidth', 2);
        xlabel('Frame Number');
        ylabel('Elbow Angle (degrees)');
        title('Right Elbow Angle Over Time');
        grid on;
        ylim([0, 180]);
    end

    subplot(2, 2, 4);
    validFrames = ~isnan(leftElbowAngles);
    if any(validFrames)
        plot(find(validFrames), leftElbowAngles(validFrames), 'b-', 'LineWidth', 2);
        xlabel('Frame Number');
        ylabel('Elbow Angle (degrees)');
        title('Left Elbow Angle Over Time');
        grid on;
        ylim([0, 180]);
    end

    sgtitle('Joint Angle Analysis Over Time');

    % Save biomechanical data
    biomechanicalData = struct();
    biomechanicalData.rightKneeAngles = rightKneeAngles;
    biomechanicalData.leftKneeAngles = leftKneeAngles;
    biomechanicalData.rightElbowAngles = rightElbowAngles;
    biomechanicalData.leftElbowAngles = leftElbowAngles;
    biomechanicalData.xyzPoints = xyzPoints;
    biomechanicalData.keypoints_map = keypoints_map;

    save('biomechanical_analysis.mat', 'biomechanicalData');
    fprintf('✓ Biomechanical analysis completed and saved\n');
end

     function angle = calculateJointAngle(p1, p2, p3)
     ↑
Error: Function definitions are not supported in this context. Functions can only be created as local or nested functions
in code files.
 

% Simple 3D animation
videoFileName = 'Simple_3D_Animation.mp4';
writerObj = VideoWriter(videoFileName, 'MPEG-4');
writerObj.FrameRate = 15;
open(writerObj);

fig = figure('Position', [100, 100, 800, 600]);

for frameIdx = 1:5:numFrames % Every 5th frame for speed
    validPoints = ~isnan(xyzPoints(:, 1, frameIdx));

    if sum(validPoints) >= 5
        clf;
        scatter3(xyzPoints(validPoints, 1, frameIdx), ...
                 xyzPoints(validPoints, 2, frameIdx), ...
                 xyzPoints(validPoints, 3, frameIdx), ...
                 100, 'filled', 'r');

        xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
        title(sprintf('Frame %d', frameIdx));
        grid on; view(45, 15);

        frame = getframe(fig);
        writeVideo(writerObj, frame);
    end
end

close(writerObj);
close(fig);
fprintf('✓ Simple animation created: %s\n', videoFileName);

✓ Simple animation created: Simple_3D_Animation.mp4


% Create side-by-side video showing original 2D and reconstructed 3D
if exist('humanKeypoints', 'var') && exist('xyzPoints', 'var')
    fprintf('\n=== Creating 2D+3D Comparison Video ===\n');

    videoFileName = '2D_3D_Comparison.mp4';
    writerObj = VideoWriter(videoFileName, 'MPEG-4');
    writerObj.FrameRate = 20;
    writerObj.Quality = 95;
    open(writerObj);

    fig = figure('Position', [100, 100, 1800, 600], 'Color', 'white');

    % Load one of the original videos for background
    reader = VideoReader(videoFiles(1)); % Use first camera

    for frameIdx = 1:min(numFrames, reader.NumFrames)
        validPoints3D = ~isnan(xyzPoints(:, 1, frameIdx));
        validPoints2D = ~isnan(humanKeypoints(:, 1, frameIdx, 1));

        if sum(validPoints3D) >= 5 && sum(validPoints2D) >= 5
            clf;

            % Left subplot: Original 2D with keypoints
            subplot(1, 2, 1);

            % Read original frame
            reader.CurrentTime = (frameIdx - 1) / reader.FrameRate;
            if hasFrame(reader)
                originalFrame = readFrame(reader);
                imshow(originalFrame);
                hold on;

                % Overlay 2D keypoints
                keypoints2D = squeeze(humanKeypoints(:, :, frameIdx, 1));
                validKp = ~isnan(keypoints2D(:, 1));

                scatter(keypoints2D(validKp, 1), keypoints2D(validKp, 2), ...
                       50, 'red', 'filled', 'MarkerEdgeColor', 'white');

                title(sprintf('2D Detection - Frame %d', frameIdx));
            end

            % Right subplot: 3D reconstruction
            subplot(1, 2, 2);

            scatter3(xyzPoints(validPoints3D, 1, frameIdx), ...
                     xyzPoints(validPoints3D, 2, frameIdx), ...
                     xyzPoints(validPoints3D, 3, frameIdx), ...
                     100, 'filled', 'MarkerFaceColor', 'red');

            hold on;

            % Draw 3D skeleton
            connections = [11, 12; 12, 24; 11, 23; 23, 24; ... % torso
                          11, 13; 13, 15; 12, 14; 14, 16; ... % arms
                          23, 25; 25, 27; 24, 26; 26, 28];   % legs

            for i = 1:size(connections, 1)
                pt1 = connections(i, 1);
                pt2 = connections(i, 2);
                if ~isnan(xyzPoints(pt1, 1, frameIdx)) && ~isnan(xyzPoints(pt2, 1, frameIdx))
                    plot3([xyzPoints(pt1, 1, frameIdx), xyzPoints(pt2, 1, frameIdx)], ...
                          [xyzPoints(pt1, 2, frameIdx), xyzPoints(pt2, 2, frameIdx)], ...
                          [xyzPoints(pt1, 3, frameIdx), xyzPoints(pt2, 3, frameIdx)], ...
                          'b-', 'LineWidth', 3);
                end
            end

            xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
            title(sprintf('3D Reconstruction - Frame %d', frameIdx));
            grid on; axis equal;
            view(45, 15);

            % Capture frame
            frame = getframe(fig);
            writeVideo(writerObj, frame);
        end
    end

    close(writerObj);
    close(fig);

    fprintf('✓ 2D+3D comparison video saved as %s\n', videoFileName);
end


=== Creating 2D+3D Comparison Video ===
Warning: The video's width and height has been padded to be a multiple of two as required by the H.264 codec. 
Error using alternateGetframe
A valid figure or axes handle must be specified

Error in getframe (line 72)
    x = alternateGetframe(parentFig, offsetRect, scaledOffsetRect, includeDecorations, true, h, offsetRectSpecified);
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 

% Create multi-view animation with 3D pose and trajectory
if exist('xyzPoints', 'var') && ~isempty(xyzPoints)
    fprintf('\n=== Creating Multi-View Animation ===\n');

    videoFileName = '3D_Human_Motion_MultiView.mp4';
    writerObj = VideoWriter(videoFileName, 'MPEG-4');
    writerObj.FrameRate = 25;
    writerObj.Quality = 95;
    open(writerObj);

    fig = figure('Position', [100, 100, 1600, 800], 'Color', 'white');

    % Track key points for trajectory
    noseIndex = 1;
    leftWristIndex = 15;
    rightWristIndex = 16;

    % Initialize trajectory storage
    noseTrajectory = [];
    leftWristTrajectory = [];
    rightWristTrajectory = [];

    for frameIdx = 1:numFrames
        validPoints = ~isnan(xyzPoints(:, 1, frameIdx));

        if sum(validPoints) >= 5
            clf;

            % Subplot 1: Current 3D pose
            subplot(1, 2, 1);
            scatter3(xyzPoints(validPoints, 1, frameIdx), ...
                     xyzPoints(validPoints, 2, frameIdx), ...
                     xyzPoints(validPoints, 3, frameIdx), ...
                     100, 'filled', 'MarkerFaceColor', 'red');

            hold on;

            % Draw skeleton
            connections = [11, 12; 12, 24; 11, 23; 23, 24; ... % torso
                          11, 13; 13, 15; 12, 14; 14, 16; ... % arms
                          23, 25; 25, 27; 24, 26; 26, 28];   % legs

            for i = 1:size(connections, 1)
                pt1 = connections(i, 1);
                pt2 = connections(i, 2);
                if ~isnan(xyzPoints(pt1, 1, frameIdx)) && ~isnan(xyzPoints(pt2, 1, frameIdx))
                    plot3([xyzPoints(pt1, 1, frameIdx), xyzPoints(pt2, 1, frameIdx)], ...
                          [xyzPoints(pt1, 2, frameIdx), xyzPoints(pt2, 2, frameIdx)], ...
                          [xyzPoints(pt1, 3, frameIdx), xyzPoints(pt2, 3, frameIdx)], ...
                          'b-', 'LineWidth', 3);
                end
            end

            xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
            title(sprintf('3D Pose - Frame %d', frameIdx));
            grid on; axis equal;
            view(45, 15);

            % Subplot 2: Trajectory visualization
            subplot(1, 2, 2);

            % Update trajectories
            if ~isnan(xyzPoints(noseIndex, 1, frameIdx))
                noseTrajectory = [noseTrajectory; xyzPoints(noseIndex, :, frameIdx)];
            end
            if ~isnan(xyzPoints(leftWristIndex, 1, frameIdx))
                leftWristTrajectory = [leftWristTrajectory; xyzPoints(leftWristIndex, :, frameIdx)];
            end
            if ~isnan(xyzPoints(rightWristIndex, 1, frameIdx))
                rightWristTrajectory = [rightWristTrajectory; xyzPoints(rightWristIndex, :, frameIdx)];
            end

            % Plot trajectories
            if size(noseTrajectory, 1) > 1
                plot3(noseTrajectory(:, 1), noseTrajectory(:, 2), noseTrajectory(:, 3), ...
                      'r-', 'LineWidth', 2, 'DisplayName', 'Head');
            end
            hold on;
            if size(leftWristTrajectory, 1) > 1
                plot3(leftWristTrajectory(:, 1), leftWristTrajectory(:, 2), leftWristTrajectory(:, 3), ...
                      'g-', 'LineWidth', 2, 'DisplayName', 'Left Wrist');
            end
            if size(rightWristTrajectory, 1) > 1
                plot3(rightWristTrajectory(:, 1), rightWristTrajectory(:, 2), rightWristTrajectory(:, 3), ...
                      'b-', 'LineWidth', 2, 'DisplayName', 'Right Wrist');
            end

            xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
            title('Movement Trajectories');
            legend; grid on; axis equal;
            view(45, 15);

            % Capture and write frame
            frame = getframe(fig);
            writeVideo(writerObj, frame);
        end
    end

    close(writerObj);
    close(fig);

    fprintf('✓ Multi-view animation saved as %s\n', videoFileName);
end


=== Creating Multi-View Animation ===
Warning: The video's width and height has been padded to be a multiple of two as required by the H.264 codec. 
Error using alternateGetframe
A valid figure or axes handle must be specified

Error in getframe (line 72)
    x = alternateGetframe(parentFig, offsetRect, scaledOffsetRect, includeDecorations, true, h, offsetRectSpecified);
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 

% Create 3D animation video of human motion
if exist('xyzPoints', 'var') && ~isempty(xyzPoints)
    fprintf('\n=== Creating 3D Animation Video ===\n');

    [numKeypoints, ~, numFrames, ~] = size(xyzPoints);

    % Set up video writer
    videoFileName = '3D_Human_Motion_Animation.mp4';
    writerObj = VideoWriter(videoFileName, 'MPEG-4');
    writerObj.FrameRate = 30; % Adjust frame rate as needed
    writerObj.Quality = 95;
    open(writerObj);

    % Create figure for animation
    fig = figure('Position', [100, 100, 1200, 800], 'Color', 'white');

    % Define body connections (MediaPipe format)
    connections = [11, 12; 12, 24; 11, 23; 23, 24; ... % torso
                  11, 13; 13, 15; 12, 14; 14, 16; ... % arms
                  23, 25; 25, 27; 24, 26; 26, 28];   % legs

    % Find overall 3D bounds for consistent axis limits
    allValidPoints = xyzPoints(~isnan(xyzPoints));
    if ~isempty(allValidPoints)
        xRange = [min(xyzPoints(:, 1, :), [], 'all', 'omitnan'), max(xyzPoints(:, 1, :), [], 'all', 'omitnan')];
        yRange = [min(xyzPoints(:, 2, :), [], 'all', 'omitnan'), max(xyzPoints(:, 2, :), [], 'all', 'omitnan')];
        zRange = [min(xyzPoints(:, 3, :), [], 'all', 'omitnan'), max(xyzPoints(:, 3, :), [], 'all', 'omitnan')];

        % Add padding
        xPadding = (xRange(2) - xRange(1)) * 0.1;
        yPadding = (yRange(2) - yRange(1)) * 0.1;
        zPadding = (zRange(2) - zRange(1)) * 0.1;

        xRange = [xRange(1) - xPadding, xRange(2) + xPadding];
        yRange = [yRange(1) - yPadding, yRange(2) + yPadding];
        zRange = [zRange(1) - zPadding, zRange(2) + zPadding];
    end

    frameCount = 0;

    for frameIdx = 1:numFrames
        % Check if frame has valid 3D points
        validPoints = ~isnan(xyzPoints(:, 1, frameIdx));

        if sum(validPoints) >= 5 % Only animate frames with sufficient points
            frameCount = frameCount + 1;

            clf; % Clear figure

            % Plot 3D points
            scatter3(xyzPoints(validPoints, 1, frameIdx), ...
                     xyzPoints(validPoints, 2, frameIdx), ...
                     xyzPoints(validPoints, 3, frameIdx), ...
                     100, 'filled', 'MarkerFaceColor', 'red', 'MarkerEdgeColor', 'black');

            hold on;

            % Draw skeleton connections
            for i = 1:size(connections, 1)
                pt1 = connections(i, 1);
                pt2 = connections(i, 2);
                if ~isnan(xyzPoints(pt1, 1, frameIdx)) && ~isnan(xyzPoints(pt2, 1, frameIdx))
                    plot3([xyzPoints(pt1, 1, frameIdx), xyzPoints(pt2, 1, frameIdx)], ...
                          [xyzPoints(pt1, 2, frameIdx), xyzPoints(pt2, 2, frameIdx)], ...
                          [xyzPoints(pt1, 3, frameIdx), xyzPoints(pt2, 3, frameIdx)], ...
                          'b-', 'LineWidth', 3);
                end
            end

            % Set consistent axis limits and labels
            if exist('xRange', 'var')
                xlim(xRange); ylim(yRange); zlim(zRange);
            end
            xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
            title(sprintf('3D Human Motion - Frame %d', frameIdx));
            grid on;
            view(45, 15); % Set viewing angle

            % Capture frame
            frame = getframe(fig);
            writeVideo(writerObj, frame);

            if mod(frameCount, 50) == 0
                fprintf('Processed %d frames\n', frameCount);
            end
        end
    end

    close(writerObj);
    close(fig);

    fprintf('✓ 3D animation video saved as %s\n', videoFileName);
    fprintf('Total frames in video: %d\n', frameCount);
end


=== Creating 3D Animation Video ===
Processed 50 frames
Processed 100 frames
Error using alternateGetframe
A valid figure or axes handle must be specified

Error in getframe (line 72)
    x = alternateGetframe(parentFig, offsetRect, scaledOffsetRect, includeDecorations, true, h, offsetRectSpecified);
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 

% Create 3D animation video of human motion
if exist('xyzPoints', 'var') && ~isempty(xyzPoints)
    fprintf('\n=== Creating 3D Animation Video ===\n');

    [numKeypoints, ~, numFrames, ~] = size(xyzPoints);

    % Set up video writer
    videoFileName = '3D_Human_Motion_Animation.mp4';
    writerObj = VideoWriter(videoFileName, 'MPEG-4');
    writerObj.FrameRate = 30; % Adjust frame rate as needed
    writerObj.Quality = 95;
    open(writerObj);

    % Create figure for animation
    fig = figure('Position', [100, 100, 1200, 800], 'Color', 'white');

    % Define body connections (MediaPipe format)
    connections = [11, 12; 12, 24; 11, 23; 23, 24; ... % torso
                  11, 13; 13, 15; 12, 14; 14, 16; ... % arms
                  23, 25; 25, 27; 24, 26; 26, 28];   % legs

    % Find overall 3D bounds for consistent axis limits
    allValidPoints = xyzPoints(~isnan(xyzPoints));
    if ~isempty(allValidPoints)
        xRange = [min(xyzPoints(:, 1, :), [], 'all', 'omitnan'), max(xyzPoints(:, 1, :), [], 'all', 'omitnan')];
        yRange = [min(xyzPoints(:, 2, :), [], 'all', 'omitnan'), max(xyzPoints(:, 2, :), [], 'all', 'omitnan')];
        zRange = [min(xyzPoints(:, 3, :), [], 'all', 'omitnan'), max(xyzPoints(:, 3, :), [], 'all', 'omitnan')];

        % Add padding
        xPadding = (xRange(2) - xRange(1)) * 0.1;
        yPadding = (yRange(2) - yRange(1)) * 0.1;
        zPadding = (zRange(2) - zRange(1)) * 0.1;

        xRange = [xRange(1) - xPadding, xRange(2) + xPadding];
        yRange = [yRange(1) - yPadding, yRange(2) + yPadding];
        zRange = [zRange(1) - zPadding, zRange(2) + zPadding];
    end

    frameCount = 0;

    for frameIdx = 1:numFrames
        % Check if frame has valid 3D points
        validPoints = ~isnan(xyzPoints(:, 1, frameIdx));

        if sum(validPoints) >= 5 % Only animate frames with sufficient points
            frameCount = frameCount + 1;

            clf; % Clear figure

            % Plot 3D points
            scatter3(xyzPoints(validPoints, 1, frameIdx), ...
                     xyzPoints(validPoints, 2, frameIdx), ...
                     xyzPoints(validPoints, 3, frameIdx), ...
                     100, 'filled', 'MarkerFaceColor', 'red', 'MarkerEdgeColor', 'black');

            hold on;

            % Draw skeleton connections
            for i = 1:size(connections, 1)
                pt1 = connections(i, 1);
                pt2 = connections(i, 2);
                if ~isnan(xyzPoints(pt1, 1, frameIdx)) && ~isnan(xyzPoints(pt2, 1, frameIdx))
                    plot3([xyzPoints(pt1, 1, frameIdx), xyzPoints(pt2, 1, frameIdx)], ...
                          [xyzPoints(pt1, 2, frameIdx), xyzPoints(pt2, 2, frameIdx)], ...
                          [xyzPoints(pt1, 3, frameIdx), xyzPoints(pt2, 3, frameIdx)], ...
                          'b-', 'LineWidth', 3);
                end
            end

            % Set consistent axis limits and labels
            if exist('xRange', 'var')
                xlim(xRange); ylim(yRange); zlim(zRange);
            end
            xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
            title(sprintf('3D Human Motion - Frame %d', frameIdx));
            grid on;
            view(45, 15); % Set viewing angle

            % Capture frame
            frame = getframe(fig);
            writeVideo(writerObj, frame);

            if mod(frameCount, 50) == 0
                fprintf('Processed %d frames\n', frameCount);
            end
        end
    end

    close(writerObj);
    close(fig);

    fprintf('✓ 3D animation video saved as %s\n', videoFileName);
    fprintf('Total frames in video: %d\n', frameCount);
end

=== Creating 3D Animation Video ===
Cannot create file 3D_Human_Motion_Animation.mp4. Permission Denied.
 

% Create 3D animation video of human motion
if exist('xyzPoints', 'var') && ~isempty(xyzPoints)
    fprintf('\n=== Creating 3D Animation Video ===\n');

    [numKeypoints, ~, numFrames, ~] = size(xyzPoints);

    % Set up video writer
    videoFileName = '3D_Human_Motion_Animation.mp4';
    writerObj = VideoWriter(videoFileName, 'MPEG-4');
    writerObj.FrameRate = 30; % Adjust frame rate as needed
    writerObj.Quality = 95;
    open(writerObj);

    % Create figure for animation
    fig = figure('Position', [100, 100, 1200, 800], 'Color', 'white');

    % Define body connections (MediaPipe format)
    connections = [11, 12; 12, 24; 11, 23; 23, 24; ... % torso
                  11, 13; 13, 15; 12, 14; 14, 16; ... % arms
                  23, 25; 25, 27; 24, 26; 26, 28];   % legs

    % Find overall 3D bounds for consistent axis limits
    allValidPoints = xyzPoints(~isnan(xyzPoints));
    if ~isempty(allValidPoints)
        xRange = [min(xyzPoints(:, 1, :), [], 'all', 'omitnan'), max(xyzPoints(:, 1, :), [], 'all', 'omitnan')];
        yRange = [min(xyzPoints(:, 2, :), [], 'all', 'omitnan'), max(xyzPoints(:, 2, :), [], 'all', 'omitnan')];
        zRange = [min(xyzPoints(:, 3, :), [], 'all', 'omitnan'), max(xyzPoints(:, 3, :), [], 'all', 'omitnan')];

        % Add padding
        xPadding = (xRange(2) - xRange(1)) * 0.1;
        yPadding = (yRange(2) - yRange(1)) * 0.1;
        zPadding = (zRange(2) - zRange(1)) * 0.1;

        xRange = [xRange(1) - xPadding, xRange(2) + xPadding];
        yRange = [yRange(1) - yPadding, yRange(2) + yPadding];
        zRange = [zRange(1) - zPadding, zRange(2) + zPadding];
    end

    frameCount = 0;

    for frameIdx = 1:numFrames
        % Check if frame has valid 3D points
        validPoints = ~isnan(xyzPoints(:, 1, frameIdx));

        if sum(validPoints) >= 5 % Only animate frames with sufficient points
            frameCount = frameCount + 1;

            clf; % Clear figure

            % Plot 3D points
            scatter3(xyzPoints(validPoints, 1, frameIdx), ...
                     xyzPoints(validPoints, 2, frameIdx), ...
                     xyzPoints(validPoints, 3, frameIdx), ...
                     100, 'filled', 'MarkerFaceColor', 'red', 'MarkerEdgeColor', 'black');

            hold on;

            % Draw skeleton connections
            for i = 1:size(connections, 1)
                pt1 = connections(i, 1);
                pt2 = connections(i, 2);
                if ~isnan(xyzPoints(pt1, 1, frameIdx)) && ~isnan(xyzPoints(pt2, 1, frameIdx))
                    plot3([xyzPoints(pt1, 1, frameIdx), xyzPoints(pt2, 1, frameIdx)], ...
                          [xyzPoints(pt1, 2, frameIdx), xyzPoints(pt2, 2, frameIdx)], ...
                          [xyzPoints(pt1, 3, frameIdx), xyzPoints(pt2, 3, frameIdx)], ...
                          'b-', 'LineWidth', 3);
                end
            end

            % Set consistent axis limits and labels
            if exist('xRange', 'var')
                xlim(xRange); ylim(yRange); zlim(zRange);
            end
            xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
            title(sprintf('3D Human Motion - Frame %d', frameIdx));
            grid on;
            view(45, 15); % Set viewing angle

            % Capture frame
            frame = getframe(fig);
            writeVideo(writerObj, frame);

            if mod(frameCount, 50) == 0
                fprintf('Processed %d frames\n', frameCount);
            end
        end
    end

    close(writerObj);
    close(fig);

    fprintf('✓ 3D animation video saved as %s\n', videoFileName);
    fprintf('Total frames in video: %d\n', frameCount);
end


=== Creating 3D Animation Video ===
Cannot create file 3D_Human_Motion_Animation.mp4. Permission Denied.
 

% Create 3D animation video of human motion
if exist('xyzPoints', 'var') && ~isempty(xyzPoints)
    fprintf('\n=== Creating 3D Animation Video ===\n');

    [numKeypoints, ~, numFrames, ~] = size(xyzPoints);

    % Set up video writer
    videoFileName = '3D_Human_Motion_Animation.mp4';
    writerObj = VideoWriter(videoFileName, 'MPEG-4');
    writerObj.FrameRate = 30; % Adjust frame rate as needed
    writerObj.Quality = 95;
    open(writerObj);

    % Create figure for animation
    fig = figure('Position', [100, 100, 1200, 800], 'Color', 'white');

    % Define body connections (MediaPipe format)
    connections = [11, 12; 12, 24; 11, 23; 23, 24; ... % torso
                  11, 13; 13, 15; 12, 14; 14, 16; ... % arms
                  23, 25; 25, 27; 24, 26; 26, 28];   % legs

    % Find overall 3D bounds for consistent axis limits
    allValidPoints = xyzPoints(~isnan(xyzPoints));
    if ~isempty(allValidPoints)
        xRange = [min(xyzPoints(:, 1, :), [], 'all', 'omitnan'), max(xyzPoints(:, 1, :), [], 'all', 'omitnan')];
        yRange = [min(xyzPoints(:, 2, :), [], 'all', 'omitnan'), max(xyzPoints(:, 2, :), [], 'all', 'omitnan')];
        zRange = [min(xyzPoints(:, 3, :), [], 'all', 'omitnan'), max(xyzPoints(:, 3, :), [], 'all', 'omitnan')];

        % Add padding
        xPadding = (xRange(2) - xRange(1)) * 0.1;
        yPadding = (yRange(2) - yRange(1)) * 0.1;
        zPadding = (zRange(2) - zRange(1)) * 0.1;

        xRange = [xRange(1) - xPadding, xRange(2) + xPadding];
        yRange = [yRange(1) - yPadding, yRange(2) + yPadding];
        zRange = [zRange(1) - zPadding, zRange(2) + zPadding];
    end

    frameCount = 0;

    for frameIdx = 1:numFrames
        % Check if frame has valid 3D points
        validPoints = ~isnan(xyzPoints(:, 1, frameIdx));

        if sum(validPoints) >= 5 % Only animate frames with sufficient points
            frameCount = frameCount + 1;

            clf; % Clear figure

            % Plot 3D points
            scatter3(xyzPoints(validPoints, 1, frameIdx), ...
                     xyzPoints(validPoints, 2, frameIdx), ...
                     xyzPoints(validPoints, 3, frameIdx), ...
                     100, 'filled', 'MarkerFaceColor', 'red', 'MarkerEdgeColor', 'black');

            hold on;

            % Draw skeleton connections
            for i = 1:size(connections, 1)
                pt1 = connections(i, 1);
                pt2 = connections(i, 2);
                if ~isnan(xyzPoints(pt1, 1, frameIdx)) && ~isnan(xyzPoints(pt2, 1, frameIdx))
                    plot3([xyzPoints(pt1, 1, frameIdx), xyzPoints(pt2, 1, frameIdx)], ...
                          [xyzPoints(pt1, 2, frameIdx), xyzPoints(pt2, 2, frameIdx)], ...
                          [xyzPoints(pt1, 3, frameIdx), xyzPoints(pt2, 3, frameIdx)], ...
                          'b-', 'LineWidth', 3);
                end
            end

            % Set consistent axis limits and labels
            if exist('xRange', 'var')
                xlim(xRange); ylim(yRange); zlim(zRange);
            end
            xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
            title(sprintf('3D Human Motion - Frame %d', frameIdx));
            grid on;
            view(45, 15); % Set viewing angle

            % Capture frame
            frame = getframe(fig);
            writeVideo(writerObj, frame);

            if mod(frameCount, 50) == 0
                fprintf('Processed %d frames\n', frameCount);
            end
        end
    end

    close(writerObj);
    close(fig);

    fprintf('✓ 3D animation video saved as %s\n', videoFileName);
    fprintf('Total frames in video: %d\n', frameCount);
end


=== Creating 3D Animation Video ===
Processed 50 frames
Processed 100 frames
Processed 150 frames
✓ 3D animation video saved as 3D_Human_Motion_Animation.mp4
Total frames in video: 153


% Create side-by-side video with corrected 3D orientation
if exist('humanKeypoints', 'var') && exist('xyzPoints', 'var')
    fprintf('\n=== Creating Corrected 2D+3D Side-by-Side Video ===\n');

    videoFileName = '2D_3D_Corrected_SideBySide.mp4';
    writerObj = VideoWriter(videoFileName, 'MPEG-4');
    writerObj.FrameRate = 20;
    writerObj.Quality = 95;
    open(writerObj);

    fig = figure('Position', [100, 100, 1800, 800], 'Color', 'white');

    % Load original video for 2D background
    reader = VideoReader(videoFiles(1)); % Use first camera

    % Define body connections for better skeleton visualization
    connections = [
        % Head and torso
        1, 2;   % nose to neck (approximate)
        11, 12; % shoulders
        11, 23; % left shoulder to left hip
        12, 24; % right shoulder to right hip
        23, 24; % hips

        % Arms
        11, 13; % left shoulder to left elbow
        13, 15; % left elbow to left wrist
        12, 14; % right shoulder to right elbow
        14, 16; % right elbow to right wrist

        % Legs
        23, 25; % left hip to left knee
        25, 27; % left knee to left ankle
        24, 26; % right hip to right knee
        26, 28; % right knee to right ankle
    ];

    % Process frames
    frameCount = 0;
    for frameIdx = 1:min(numFrames, reader.NumFrames)
        validPoints3D = ~isnan(xyzPoints(:, 1, frameIdx));
        validPoints2D = ~isnan(humanKeypoints(:, 1, frameIdx, 1));

        if sum(validPoints3D) >= 8 && sum(validPoints2D) >= 8
            frameCount = frameCount + 1;
            clf;

            % LEFT SUBPLOT: Original 2D with keypoints overlay
            subplot(1, 2, 1);

            % Read and display original frame
            reader.CurrentTime = (frameIdx - 1) / reader.FrameRate;
            if hasFrame(reader)
                originalFrame = readFrame(reader);
                imshow(originalFrame);
                hold on;

                % Overlay 2D keypoints
                keypoints2D = squeeze(humanKeypoints(:, :, frameIdx, 1));
                validKp = ~isnan(keypoints2D(:, 1));

                % Plot keypoints
                scatter(keypoints2D(validKp, 1), keypoints2D(validKp, 2), ...
                       60, 'red', 'filled', 'MarkerEdgeColor', 'yellow', 'LineWidth', 2);

                % Draw 2D skeleton connections
                for i = 1:size(connections, 1)
                    pt1 = connections(i, 1);
                    pt2 = connections(i, 2);
                    if ~isnan(keypoints2D(pt1, 1)) && ~isnan(keypoints2D(pt2, 1))
                        plot([keypoints2D(pt1, 1), keypoints2D(pt2, 1)], ...
                             [keypoints2D(pt1, 2), keypoints2D(pt2, 2)], ...
                             'yellow', 'LineWidth', 3);
                    end
                end

                title(sprintf('2D MediaPipe Detection - Frame %d', frameIdx), ...
                      'FontSize', 14, 'FontWeight', 'bold');
            end

            % RIGHT SUBPLOT: Corrected 3D reconstruction
            subplot(1, 2, 2);

            % CORRECTION: Transform 3D points to proper orientation
            % The person appears to be lying down, so we need to rotate
            corrected3D = xyzPoints(:, :, frameIdx);

            % Apply rotation to make person stand upright
            % Rotate around X-axis by 90 degrees to make person vertical
            rotationAngle = 90; % degrees
            rotMatrix = [1, 0, 0; 
                        0, cosd(rotationAngle), -sind(rotationAngle);
                        0, sind(rotationAngle), cosd(rotationAngle)];

            for kpIdx = 1:size(corrected3D, 1)
                if ~isnan(corrected3D(kpIdx, 1))
                    corrected3D(kpIdx, :) = (rotMatrix * corrected3D(kpIdx, :)')';
                end
            end

            % Plot corrected 3D points
            validPoints3D_corrected = ~isnan(corrected3D(:, 1));

            scatter3(corrected3D(validPoints3D_corrected, 1), ...
                     corrected3D(validPoints3D_corrected, 2), ...
                     corrected3D(validPoints3D_corrected, 3), ...
                     100, 'filled', 'MarkerFaceColor', 'red', ...
                     'MarkerEdgeColor', 'black', 'LineWidth', 1);

            hold on;

            % Draw corrected 3D skeleton
            for i = 1:size(connections, 1)
                pt1 = connections(i, 1);
                pt2 = connections(i, 2);
                if ~isnan(corrected3D(pt1, 1)) && ~isnan(corrected3D(pt2, 1))
                    plot3([corrected3D(pt1, 1), corrected3D(pt2, 1)], ...
                          [corrected3D(pt1, 2), corrected3D(pt2, 2)], ...
                          [corrected3D(pt1, 3), corrected3D(pt2, 3)], ...
                          'blue', 'LineWidth', 4);
                end
            end

            % Set proper viewing angle and labels
            xlabel('X (mm)', 'FontSize', 12, 'FontWeight', 'bold');
            ylabel('Y (mm)', 'FontSize', 12, 'FontWeight', 'bold');
            zlabel('Z (mm)', 'FontSize', 12, 'FontWeight', 'bold');
            title(sprintf('3D Reconstruction (Corrected) - Frame %d', frameIdx), ...
                  'FontSize', 14, 'FontWeight', 'bold');

            grid on;
            axis equal;
            view(45, 15); % Good viewing angle for upright person

            % Set consistent axis limits for smooth animation
            if frameIdx == 1
                % Calculate overall bounds for consistent scaling
                allPoints = corrected3D(validPoints3D_corrected, :);
                if ~isempty(allPoints)
                    xRange = [min(allPoints(:, 1)) - 200, max(allPoints(:, 1)) + 200];
                    yRange = [min(allPoints(:, 2)) - 200, max(allPoints(:, 2)) + 200];
                    zRange = [min(allPoints(:, 3)) - 200, max(allPoints(:, 3)) + 200];
                end
            end

            if exist('xRange', 'var')
                xlim(xRange); ylim(yRange); zlim(zRange);
            end

            % Add overall title
            sgtitle(sprintf('Human Motion Analysis - Frame %d/%d', frameIdx, numFrames), ...
                   'FontSize', 16, 'FontWeight', 'bold');

            % Capture and write frame
            frame = getframe(fig);
            writeVideo(writerObj, frame);

            if mod(frameCount, 25) == 0
                fprintf('Processed %d frames\n', frameCount);
            end
        end
    end

    close(writerObj);
    close(fig);

    fprintf('✓ Corrected side-by-side video saved as %s\n', videoFileName);
    fprintf('Total frames in video: %d\n', frameCount);
end


=== Creating Corrected 2D+3D Side-by-Side Video ===
Warning: The video's width and height has been padded to be a multiple of two as required by the H.264 codec. 
Error using alternateGetframe
A valid figure or axes handle must be specified

Error in getframe (line 72)
    x = alternateGetframe(parentFig, offsetRect, scaledOffsetRect, includeDecorations, true, h, offsetRectSpecified);
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 

% Create corrected 3D animation video with 2D keypoints overlay
if exist('xyzPoints', 'var') && exist('humanKeypoints', 'var')
    fprintf('\n=== Creating Corrected 3D Animation with 2D Overlay ===\n');

    [numKeypoints, ~, numFrames, ~] = size(xyzPoints);

    % Set up video writer
    videoFileName = '3D_Human_Motion_Corrected_with_2D.mp4';
    writerObj = VideoWriter(videoFileName, 'MPEG-4');
    writerObj.FrameRate = 20; % Adjust frame rate as needed
    writerObj.Quality = 95;
    open(writerObj);

    % Create figure for animation
    fig = figure('Position', [100, 100, 1600, 800], 'Color', 'white');

    % Define body connections (MediaPipe format)
    connections = [11, 12; 12, 24; 11, 23; 23, 24; ... % torso
                  11, 13; 13, 15; 12, 14; 14, 16; ... % arms
                  23, 25; 25, 27; 24, 26; 26, 28];   % legs

    % Load original video for 2D overlay
    reader = VideoReader(videoFiles(1)); % Use first camera

    % CORRECTED: Apply 90-degree rotation to fix orientation
    % Rotation matrix for 90 degrees around Z-axis (to rotate right)
    rotationAngle = 90; % degrees
    rotMatrix = [cosd(rotationAngle), -sind(rotationAngle), 0;
                 sind(rotationAngle), cosd(rotationAngle), 0;
                 0, 0, 1];

    % Find overall 3D bounds for consistent axis limits (after rotation)
    allCorrectedPoints = [];
    for frameIdx = 1:numFrames
        framePoints = xyzPoints(:, :, frameIdx);
        validPoints = ~isnan(framePoints(:, 1));
        if any(validPoints)
            % Apply rotation to get corrected points
            correctedFrame = framePoints;
            for kpIdx = 1:size(framePoints, 1)
                if validPoints(kpIdx)
                    correctedFrame(kpIdx, :) = (rotMatrix * framePoints(kpIdx, :)')';
                end
            end
            allCorrectedPoints = [allCorrectedPoints; correctedFrame(validPoints, :)];
        end
    end

    if ~isempty(allCorrectedPoints)
        xRange = [min(allCorrectedPoints(:, 1)), max(allCorrectedPoints(:, 1))];
        yRange = [min(allCorrectedPoints(:, 2)), max(allCorrectedPoints(:, 2))];
        zRange = [min(allCorrectedPoints(:, 3)), max(allCorrectedPoints(:, 3))];

        % Add padding
        xPadding = (xRange(2) - xRange(1)) * 0.2;
        yPadding = (yRange(2) - yRange(1)) * 0.2;
        zPadding = (zRange(2) - zRange(1)) * 0.2;

        xRange = [xRange(1) - xPadding, xRange(2) + xPadding];
        yRange = [yRange(1) - yPadding, yRange(2) + yPadding];
        zRange = [zRange(1) - zPadding, zRange(2) + zPadding];
    end

    frameCount = 0;

    for frameIdx = 1:min(numFrames, reader.NumFrames)
        % Check if frame has valid 3D points
        validPoints3D = ~isnan(xyzPoints(:, 1, frameIdx));
        validPoints2D = ~isnan(humanKeypoints(:, 1, frameIdx, 1));

        if sum(validPoints3D) >= 5 && sum(validPoints2D) >= 5
            frameCount = frameCount + 1;

            clf; % Clear figure

            % LEFT SUBPLOT: Original video with 2D keypoints overlay
            subplot(1, 2, 1);

            % Read and display original frame
            reader.CurrentTime = (frameIdx - 1) / reader.FrameRate;
            if hasFrame(reader)
                originalFrame = readFrame(reader);
                imshow(originalFrame);
                hold on;

                % Overlay 2D keypoints
                keypoints2D = squeeze(humanKeypoints(:, :, frameIdx, 1));
                validKp = ~isnan(keypoints2D(:, 1));

                % Plot keypoints
                scatter(keypoints2D(validKp, 1), keypoints2D(validKp, 2), ...
                       80, 'red', 'filled', 'MarkerEdgeColor', 'yellow', 'LineWidth', 2);

                % Draw 2D skeleton connections
                for i = 1:size(connections, 1)
                    pt1 = connections(i, 1);
                    pt2 = connections(i, 2);
                    if ~isnan(keypoints2D(pt1, 1)) && ~isnan(keypoints2D(pt2, 1))
                        plot([keypoints2D(pt1, 1), keypoints2D(pt2, 1)], ...
                             [keypoints2D(pt1, 2), keypoints2D(pt2, 2)], ...
                             'yellow', 'LineWidth', 4);
                    end
                end

                title(sprintf('2D MediaPipe Detection - Frame %d', frameIdx), ...
                      'FontSize', 14, 'FontWeight', 'bold', 'Color', 'white');
            end

            % RIGHT SUBPLOT: Corrected 3D reconstruction
            subplot(1, 2, 2);

            % CORRECTED: Apply rotation to fix orientation (90 degrees right)
            corrected3D = xyzPoints(:, :, frameIdx);
            for kpIdx = 1:size(corrected3D, 1)
                if ~isnan(corrected3D(kpIdx, 1))
                    corrected3D(kpIdx, :) = (rotMatrix * corrected3D(kpIdx, :)')';
                end
            end

            % Plot corrected 3D points
            validPoints3D_corrected = ~isnan(corrected3D(:, 1));

            scatter3(corrected3D(validPoints3D_corrected, 1), ...
                     corrected3D(validPoints3D_corrected, 2), ...
                     corrected3D(validPoints3D_corrected, 3), ...
                     120, 'filled', 'MarkerFaceColor', 'red', ...
                     'MarkerEdgeColor', 'black', 'LineWidth', 2);

            hold on;

            % Draw corrected 3D skeleton with enhanced colors
            for i = 1:size(connections, 1)
                pt1 = connections(i, 1);
                pt2 = connections(i, 2);
                if ~isnan(corrected3D(pt1, 1)) && ~isnan(corrected3D(pt2, 1))
                    % Color code different body parts
                    if any(i == [1, 2, 3, 4]) % torso connections
                        lineColor = 'red';
                        lineWidth = 5;
                    elseif any(i == [5, 6, 7, 8]) % arm connections
                        lineColor = 'green';
                        lineWidth = 4;
                    else % leg connections
                        lineColor = 'blue';
                        lineWidth = 4;
                    end

                    plot3([corrected3D(pt1, 1), corrected3D(pt2, 1)], ...
                          [corrected3D(pt1, 2), corrected3D(pt2, 2)], ...
                          [corrected3D(pt1, 3), corrected3D(pt2, 3)], ...
                          'Color', lineColor, 'LineWidth', lineWidth);
                end
            end

            % Set proper viewing angle and labels
            xlabel('X (mm)', 'FontSize', 12, 'FontWeight', 'bold');
            ylabel('Y (mm)', 'FontSize', 12, 'FontWeight', 'bold');
            zlabel('Z (mm)', 'FontSize', 12, 'FontWeight', 'bold');
            title(sprintf('3D Reconstruction (Corrected) - Frame %d', frameIdx), ...
                  'FontSize', 14, 'FontWeight', 'bold');

            grid on;
            axis equal;
            view(45, 15); % Good viewing angle for corrected orientation

            % Set consistent axis limits for smooth animation
            if exist('xRange', 'var')
                xlim(xRange); ylim(yRange); zlim(zRange);
            end

            % Add overall title
            sgtitle(sprintf('Human Motion Analysis: 2D Detection + 3D Reconstruction - Frame %d/%d', ...
                   frameIdx, numFrames), 'FontSize', 16, 'FontWeight', 'bold');

            % Capture frame
            frame = getframe(fig);
            writeVideo(writerObj, frame);

            if mod(frameCount, 25) == 0
                fprintf('Processed %d frames\n', frameCount);
            end
        end
    end

    close(writerObj);
    close(fig);

    fprintf('✓ Corrected 3D animation with 2D overlay saved as %s\n', videoFileName);
    fprintf('Total frames in video: %d\n', frameCount);
end


=== Creating Corrected 3D Animation with 2D Overlay ===
Warning: The video's width and height has been padded to be a multiple of two as required by the H.264 codec. 
Error using alternateGetframe
A valid figure or axes handle must be specified

Error in getframe (line 72)
    x = alternateGetframe(parentFig, offsetRect, scaledOffsetRect, includeDecorations, true, h, offsetRectSpecified);
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 

% Pure 3D animation with corrected orientation
if exist('xyzPoints', 'var')
    fprintf('\n=== Creating Pure 3D Animation (Corrected Orientation) ===\n');

    videoFileName = '3D_Human_Motion_Pure_Corrected.mp4';
    writerObj = VideoWriter(videoFileName, 'MPEG-4');
    writerObj.FrameRate = 25;
    writerObj.Quality = 95;
    open(writerObj);

    fig = figure('Position', [100, 100, 1000, 800], 'Color', 'white');

    % Rotation matrix for 90 degrees around Z-axis (rotate right)
    rotationAngle = 90;
    rotMatrix = [cosd(rotationAngle), -sind(rotationAngle), 0;
                 sind(rotationAngle), cosd(rotationAngle), 0;
                 0, 0, 1];

    connections = [11, 12; 12, 24; 11, 23; 23, 24; ... % torso
                  11, 13; 13, 15; 12, 14; 14, 16; ... % arms
                  23, 25; 25, 27; 24, 26; 26, 28];   % legs

    frameCount = 0;

    for frameIdx = 1:numFrames
        validPoints = ~isnan(xyzPoints(:, 1, frameIdx));

        if sum(validPoints) >= 5
            frameCount = frameCount + 1;
            clf;

            % Apply rotation correction
            corrected3D = xyzPoints(:, :, frameIdx);
            for kpIdx = 1:size(corrected3D, 1)
                if ~isnan(corrected3D(kpIdx, 1))
                    corrected3D(kpIdx, :) = (rotMatrix * corrected3D(kpIdx, :)')';
                end
            end

            validCorrected = ~isnan(corrected3D(:, 1));

            % Plot 3D points
            scatter3(corrected3D(validCorrected, 1), ...
                     corrected3D(validCorrected, 2), ...
                     corrected3D(validCorrected, 3), ...
                     150, 'filled', 'MarkerFaceColor', 'red', ...
                     'MarkerEdgeColor', 'black', 'LineWidth', 2);

            hold on;

            % Draw skeleton
            for i = 1:size(connections, 1)
                pt1 = connections(i, 1);
                pt2 = connections(i, 2);
                if ~isnan(corrected3D(pt1, 1)) && ~isnan(corrected3D(pt2, 1))
                    plot3([corrected3D(pt1, 1), corrected3D(pt2, 1)], ...
                          [corrected3D(pt1, 2), corrected3D(pt2, 2)], ...
                          [corrected3D(pt1, 3), corrected3D(pt2, 3)], ...
                          'blue', 'LineWidth', 5);
                end
            end

            xlabel('X (mm)', 'FontSize', 14, 'FontWeight', 'bold');
            ylabel('Y (mm)', 'FontSize', 14, 'FontWeight', 'bold');
            zlabel('Z (mm)', 'FontSize', 14, 'FontWeight', 'bold');
            title(sprintf('3D Human Motion (Corrected Orientation) - Frame %d', frameIdx), ...
                  'FontSize', 16, 'FontWeight', 'bold');

            grid on;
            axis equal;
            view(45, 15);

            frame = getframe(fig);
            writeVideo(writerObj, frame);
        end
    end

    close(writerObj);
    close(fig);

    fprintf('✓ Pure 3D animation saved as %s\n', videoFileName);
end


=== Creating Pure 3D Animation (Corrected Orientation) ===
Error using alternateGetframe
A valid figure or axes handle must be specified

Error in getframe (line 72)
    x = alternateGetframe(parentFig, offsetRect, scaledOffsetRect, includeDecorations, true, h, offsetRectSpecified);
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 

% Create side-by-side video: 2D keypoints + corrected 3D view
if exist('xyzPoints', 'var') && exist('humanKeypoints', 'var')
    fprintf('\n=== Creating 2D + 3D Side-by-Side Video ===\n');

    videoFileName = '2D_3D_SideBySide_Corrected.mp4';
    writerObj = VideoWriter(videoFileName, 'MPEG-4');
    writerObj.FrameRate = 20;
    writerObj.Quality = 95;
    open(writerObj);

    fig = figure('Position', [100, 100, 1600, 800], 'Color', 'white');

    % Load original video
    reader = VideoReader(videoFiles(1));

    connections = [11, 12; 12, 24; 11, 23; 23, 24; ... % torso
                  11, 13; 13, 15; 12, 14; 14, 16; ... % arms
                  23, 25; 25, 27; 24, 26; 26, 28];   % legs

    frameCount = 0;

    for frameIdx = 1:min(numFrames, reader.NumFrames)
        validPoints3D = ~isnan(xyzPoints(:, 1, frameIdx));
        validPoints2D = ~isnan(humanKeypoints(:, 1, frameIdx, 1));

        if sum(validPoints3D) >= 5 && sum(validPoints2D) >= 5
            frameCount = frameCount + 1;
            clf;

            % LEFT: 2D keypoints on original video
            subplot(1, 2, 1);
            reader.CurrentTime = (frameIdx - 1) / reader.FrameRate;
            if hasFrame(reader)
                frame = readFrame(reader);
                imshow(frame);
                hold on;

                keypoints2D = squeeze(humanKeypoints(:, :, frameIdx, 1));
                validKp = ~isnan(keypoints2D(:, 1));

                scatter(keypoints2D(validKp, 1), keypoints2D(validKp, 2), ...
                       60, 'red', 'filled', 'MarkerEdgeColor', 'yellow', 'LineWidth', 2);

                for i = 1:size(connections, 1)
                    pt1 = connections(i, 1);
                    pt2 = connections(i, 2);
                    if ~isnan(keypoints2D(pt1, 1)) && ~isnan(keypoints2D(pt2, 1))
                        plot([keypoints2D(pt1, 1), keypoints2D(pt2, 1)], ...
                             [keypoints2D(pt1, 2), keypoints2D(pt2, 2)], ...
                             'yellow', 'LineWidth', 3);
                    end
                end

                title(sprintf('2D Detection - Frame %d', frameIdx), 'FontSize', 14);
            end

            % RIGHT: 3D with corrected view
            subplot(1, 2, 2);

            scatter3(xyzPoints(validPoints3D, 1, frameIdx), ...
                     xyzPoints(validPoints3D, 2, frameIdx), ...
                     xyzPoints(validPoints3D, 3, frameIdx), ...
                     100, 'filled', 'MarkerFaceColor', 'red', 'MarkerEdgeColor', 'black');

            hold on;

            for i = 1:size(connections, 1)
                pt1 = connections(i, 1);
                pt2 = connections(i, 2);
                if ~isnan(xyzPoints(pt1, 1, frameIdx)) && ~isnan(xyzPoints(pt2, 1, frameIdx))
                    plot3([xyzPoints(pt1, 1, frameIdx), xyzPoints(pt2, 1, frameIdx)], ...
                          [xyzPoints(pt1, 2, frameIdx), xyzPoints(pt2, 2, frameIdx)], ...
                          [xyzPoints(pt1, 3, frameIdx), xyzPoints(pt2, 3, frameIdx)], ...
                          'b-', 'LineWidth', 3);
                end
            end

            xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
            title(sprintf('3D Reconstruction - Frame %d', frameIdx), 'FontSize', 14);
            grid on;
            view(135, 15); % Corrected view: head up, legs down
            axis equal;

            frame = getframe(fig);
            writeVideo(writerObj, frame);
        end
    end

    close(writerObj);
    close(fig);

    fprintf('✓ Side-by-side video saved as %s\n', videoFileName);
end


=== Creating 2D + 3D Side-by-Side Video ===
Warning: The video's width and height has been padded to be a multiple of two as required by the H.264 codec. 
Error using alternateGetframe
A valid figure or axes handle must be specified

Error in getframe (line 72)
    x = alternateGetframe(parentFig, offsetRect, scaledOffsetRect, includeDecorations, true, h, offsetRectSpecified);
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 



% Create 3D animation video of human motion with corrected viewing angle
if exist('xyzPoints', 'var') && ~isempty(xyzPoints)
    fprintf('\n=== Creating 3D Animation Video ===\n');

    [numKeypoints, ~, numFrames, ~] = size(xyzPoints);

    % Set up video writer
    videoFileName = '3D_Human_Motion_Animation.mp4';
    writerObj = VideoWriter(videoFileName, 'MPEG-4');
    writerObj.FrameRate = 30; % Adjust frame rate as needed
    writerObj.Quality = 95;
    open(writerObj);

    % Create figure for animation
    fig = figure('Position', [100, 100, 1200, 800], 'Color', 'white');

    % Define body connections (MediaPipe format)
    connections = [11, 12; 12, 24; 11, 23; 23, 24; ... % torso
                  11, 13; 13, 15; 12, 14; 14, 16; ... % arms
                  23, 25; 25, 27; 24, 26; 26, 28];   % legs

    % Find overall 3D bounds for consistent axis limits
    allValidPoints = xyzPoints(~isnan(xyzPoints));
    if ~isempty(allValidPoints)
        xRange = [min(xyzPoints(:, 1, :), [], 'all', 'omitnan'), max(xyzPoints(:, 1, :), [], 'all', 'omitnan')];
        yRange = [min(xyzPoints(:, 2, :), [], 'all', 'omitnan'), max(xyzPoints(:, 2, :), [], 'all', 'omitnan')];
        zRange = [min(xyzPoints(:, 3, :), [], 'all', 'omitnan'), max(xyzPoints(:, 3, :), [], 'all', 'omitnan')];

        % Add padding
        xPadding = (xRange(2) - xRange(1)) * 0.1;
        yPadding = (yRange(2) - yRange(1)) * 0.1;
        zPadding = (zRange(2) - zRange(1)) * 0.1;

        xRange = [xRange(1) - xPadding, xRange(2) + xPadding];
        yRange = [yRange(1) - yPadding, yRange(2) + yPadding];
        zRange = [zRange(1) - zPadding, zRange(2) + zPadding];
    end

    frameCount = 0;

    for frameIdx = 1:numFrames
        % Check if frame has valid 3D points
        validPoints = ~isnan(xyzPoints(:, 1, frameIdx));

        if sum(validPoints) >= 5 % Only animate frames with sufficient points
            frameCount = frameCount + 1;

            clf; % Clear figure

            % Plot 3D points
            scatter3(xyzPoints(validPoints, 1, frameIdx), ...
                     xyzPoints(validPoints, 2, frameIdx), ...
                     xyzPoints(validPoints, 3, frameIdx), ...
                     100, 'filled', 'MarkerFaceColor', 'red', 'MarkerEdgeColor', 'black');

            hold on;

            % Draw skeleton connections
            for i = 1:size(connections, 1)
                pt1 = connections(i, 1);
                pt2 = connections(i, 2);
                if ~isnan(xyzPoints(pt1, 1, frameIdx)) && ~isnan(xyzPoints(pt2, 1, frameIdx))
                    plot3([xyzPoints(pt1, 1, frameIdx), xyzPoints(pt2, 1, frameIdx)], ...
                          [xyzPoints(pt1, 2, frameIdx), xyzPoints(pt2, 2, frameIdx)], ...
                          [xyzPoints(pt1, 3, frameIdx), xyzPoints(pt2, 3, frameIdx)], ...
                          'b-', 'LineWidth', 3);
                end
            end

            % Set consistent axis limits and labels
            if exist('xRange', 'var')
                xlim(xRange); ylim(yRange); zlim(zRange);
            end
            xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
            title(sprintf('3D Human Motion - Frame %d', frameIdx));
            grid on;
            view(135, 15); % CHANGED: Rotated view by 90 degrees (45+90=135) to make head up, legs down

            % Capture frame
            frame = getframe(fig);
            writeVideo(writerObj, frame);

            if mod(frameCount, 50) == 0
                fprintf('Processed %d frames\n', frameCount);
            end
        end
    end

    close(writerObj);
    close(fig);

    fprintf('✓ 3D animation video saved as %s\n', videoFileName);
    fprintf('Total frames in video: %d\n', frameCount);
end


=== Creating 3D Animation Video ===
Processed 50 frames
Invalid or deleted object.

Error in alternateGetframe

Error in alternateGetframe

Error in getframe (line 72)
    x = alternateGetframe(parentFig, offsetRect, scaledOffsetRect, includeDecorations, true, h, offsetRectSpecified);
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 



% Create consistent side-by-side video with proper synchronization
if exist('xyzPoints', 'var') && exist('humanKeypoints', 'var')
    fprintf('\n=== Creating Synchronized 2D+3D Video ===\n');

    videoFileName = '2D_3D_Synchronized_Fixed.mp4';
    writerObj = VideoWriter(videoFileName, 'MPEG-4');
    writerObj.FrameRate = 15; % Lower frame rate for consistency
    writerObj.Quality = 95;
    open(writerObj);

    fig = figure('Position', [100, 100, 1600, 800], 'Color', 'white');

    % Load original video
    reader = VideoReader(videoFiles(1));

    connections = [11, 12; 12, 24; 11, 23; 23, 24; ... % torso
                  11, 13; 13, 15; 12, 14; 14, 16; ... % arms
                  23, 25; 25, 27; 24, 26; 26, 28];   % legs

    % Apply same rotation as above
    rotationMatrix = [0, -1, 0; 1, 0, 0; 0, 0, 1];

    % Pre-rotate all 3D points for consistency
    rotatedXyzPoints = nan(size(xyzPoints));
    for frameIdx = 1:numFrames
        for kpIdx = 1:numKeypoints
            if ~isnan(xyzPoints(kpIdx, 1, frameIdx))
                originalPoint = squeeze(xyzPoints(kpIdx, :, frameIdx));
                rotatedPoint = rotationMatrix * originalPoint';
                rotatedXyzPoints(kpIdx, :, frameIdx) = rotatedPoint';
            end
        end
    end

    % Calculate consistent 3D bounds
    xRange = [min(rotatedXyzPoints(:, 1, :), [], 'all', 'omitnan'), max(rotatedXyzPoints(:, 1, :), [], 'all', 'omitnan')];
    yRange = [min(rotatedXyzPoints(:, 2, :), [], 'all', 'omitnan'), max(rotatedXyzPoints(:, 2, :), [], 'all', 'omitnan')];
    zRange = [min(rotatedXyzPoints(:, 3, :), [], 'all', 'omitnan'), max(rotatedXyzPoints(:, 3, :), [], 'all', 'omitnan')];

    xPadding = (xRange(2) - xRange(1)) * 0.1;
    yPadding = (yRange(2) - yRange(1)) * 0.1;
    zPadding = (zRange(2) - zRange(1)) * 0.1;

    xRange = [xRange(1) - xPadding, xRange(2) + xPadding];
    yRange = [yRange(1) - yPadding, yRange(2) + yPadding];
    zRange = [zRange(1) - zPadding, zRange(2) + zPadding];

    frameCount = 0;

    for frameIdx = 1:min(numFrames, reader.NumFrames)
        validPoints3D = ~isnan(rotatedXyzPoints(:, 1, frameIdx));
        validPoints2D = ~isnan(humanKeypoints(:, 1, frameIdx, 1));

        if sum(validPoints3D) >= 5 && sum(validPoints2D) >= 5
            frameCount = frameCount + 1;
            clf;

            % LEFT: 2D keypoints
            subplot(1, 2, 1);
            reader.CurrentTime = (frameIdx - 1) / reader.FrameRate;
            if hasFrame(reader)
                frame = readFrame(reader);
                imshow(frame);
                hold on;

                keypoints2D = squeeze(humanKeypoints(:, :, frameIdx, 1));
                validKp = ~isnan(keypoints2D(:, 1));

                scatter(keypoints2D(validKp, 1), keypoints2D(validKp, 2), ...
                       60, 'red', 'filled', 'MarkerEdgeColor', 'yellow', 'LineWidth', 2);

                for i = 1:size(connections, 1)
                    pt1 = connections(i, 1);
                    pt2 = connections(i, 2);
                    if ~isnan(keypoints2D(pt1, 1)) && ~isnan(keypoints2D(pt2, 1))
                        plot([keypoints2D(pt1, 1), keypoints2D(pt2, 1)], ...
                             [keypoints2D(pt1, 2), keypoints2D(pt2, 2)], ...
                             'yellow', 'LineWidth', 3);
                    end
                end

                title(sprintf('2D Detection - Frame %d', frameIdx), 'FontSize', 14, 'FontWeight', 'bold');
            end

            % RIGHT: Fixed 3D
            subplot(1, 2, 2);

            scatter3(rotatedXyzPoints(validPoints3D, 1, frameIdx), ...
                     rotatedXyzPoints(validPoints3D, 2, frameIdx), ...
                     rotatedXyzPoints(validPoints3D, 3, frameIdx), ...
                     100, 'filled', 'MarkerFaceColor', 'red', 'MarkerEdgeColor', 'black');

            hold on;

            for i = 1:size(connections, 1)
                pt1 = connections(i, 1);
                pt2 = connections(i, 2);
                if ~isnan(rotatedXyzPoints(pt1, 1, frameIdx)) && ~isnan(rotatedXyzPoints(pt2, 1, frameIdx))
                    plot3([rotatedXyzPoints(pt1, 1, frameIdx), rotatedXyzPoints(pt2, 1, frameIdx)], ...
                          [rotatedXyzPoints(pt1, 2, frameIdx), rotatedXyzPoints(pt2, 2, frameIdx)], ...
                          [rotatedXyzPoints(pt1, 3, frameIdx), rotatedXyzPoints(pt2, 3, frameIdx)], ...
                          'b-', 'LineWidth', 3);
                end
            end

            xlim(xRange); ylim(yRange); zlim(zRange);
            xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
            title(sprintf('3D Reconstruction (Fixed) - Frame %d', frameIdx), 'FontSize', 14, 'FontWeight', 'bold');
            grid on;
            view(45, 15);

            sgtitle(sprintf('Synchronized Human Motion Analysis - Frame %d', frameIdx), 'FontSize', 16, 'FontWeight', 'bold');

            frame = getframe(fig);
            writeVideo(writerObj, frame);

            if mod(frameCount, 25) == 0
                fprintf('Processed %d frames\n', frameCount);
            end
        end
    end

    close(writerObj);
    close(fig);

    fprintf('✓ Synchronized video saved as %s\n', videoFileName);
end


=== Creating Synchronized 2D+3D Video ===
Warning: The video's width and height has been padded to be a multiple of two as required by the H.264 codec. 
Error using alternateGetframe
A valid figure or axes handle must be specified

Error in getframe (line 72)
    x = alternateGetframe(parentFig, offsetRect, scaledOffsetRect, includeDecorations, true, h, offsetRectSpecified);
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 



% Create 3D animation video with properly rotated coordinates
if exist('xyzPoints', 'var') && ~isempty(xyzPoints)
    fprintf('\n=== Creating 3D Animation Video with Fixed Orientation ===\n');

    [numKeypoints, ~, numFrames, ~] = size(xyzPoints);

    % Set up video writer
    videoFileName = '3D_Human_Motion_Fixed_Orientation.mp4';
    writerObj = VideoWriter(videoFileName, 'MPEG-4');
    writerObj.FrameRate = 30;
    writerObj.Quality = 95;
    open(writerObj);

    % Create figure for animation
    fig = figure('Position', [100, 100, 1200, 800], 'Color', 'white');

    % Define body connections (MediaPipe format)
    connections = [11, 12; 12, 24; 11, 23; 23, 24; ... % torso
                  11, 13; 13, 15; 12, 14; 14, 16; ... % arms
                  23, 25; 25, 27; 24, 26; 26, 28];   % legs

    % FIXED: Apply proper rotation to make person upright
    % Rotate 90 degrees around Z-axis to make head up, legs down
    rotationMatrix = [0, -1, 0;   % Swap and negate Y to X
                      1,  0, 0;   % X becomes Y  
                      0,  0, 1];  % Z stays same

    % Apply rotation to all 3D points
    rotatedXyzPoints = nan(size(xyzPoints));
    for frameIdx = 1:numFrames
        for kpIdx = 1:numKeypoints
            if ~isnan(xyzPoints(kpIdx, 1, frameIdx))
                originalPoint = squeeze(xyzPoints(kpIdx, :, frameIdx));
                rotatedPoint = rotationMatrix * originalPoint';
                rotatedXyzPoints(kpIdx, :, frameIdx) = rotatedPoint';
            end
        end
    end

    % Find overall bounds for rotated points
    allValidPoints = rotatedXyzPoints(~isnan(rotatedXyzPoints));
    if ~isempty(allValidPoints)
        xRange = [min(rotatedXyzPoints(:, 1, :), [], 'all', 'omitnan'), max(rotatedXyzPoints(:, 1, :), [], 'all', 'omitnan')];
        yRange = [min(rotatedXyzPoints(:, 2, :), [], 'all', 'omitnan'), max(rotatedXyzPoints(:, 2, :), [], 'all', 'omitnan')];
        zRange = [min(rotatedXyzPoints(:, 3, :), [], 'all', 'omitnan'), max(rotatedXyzPoints(:, 3, :), [], 'all', 'omitnan')];

        % Add padding
        xPadding = (xRange(2) - xRange(1)) * 0.1;
        yPadding = (yRange(2) - yRange(1)) * 0.1;
        zPadding = (zRange(2) - zRange(1)) * 0.1;

        xRange = [xRange(1) - xPadding, xRange(2) + xPadding];
        yRange = [yRange(1) - yPadding, yRange(2) + yPadding];
        zRange = [zRange(1) - zPadding, zRange(2) + zPadding];
    end

    frameCount = 0;

    for frameIdx = 1:numFrames
        % Check if frame has valid 3D points
        validPoints = ~isnan(rotatedXyzPoints(:, 1, frameIdx));

        if sum(validPoints) >= 5
            frameCount = frameCount + 1;

            clf; % Clear figure

            % Plot rotated 3D points
            scatter3(rotatedXyzPoints(validPoints, 1, frameIdx), ...
                     rotatedXyzPoints(validPoints, 2, frameIdx), ...
                     rotatedXyzPoints(validPoints, 3, frameIdx), ...
                     100, 'filled', 'MarkerFaceColor', 'red', 'MarkerEdgeColor', 'black');

            hold on;

            % Draw skeleton connections with rotated points
            for i = 1:size(connections, 1)
                pt1 = connections(i, 1);
                pt2 = connections(i, 2);
                if ~isnan(rotatedXyzPoints(pt1, 1, frameIdx)) && ~isnan(rotatedXyzPoints(pt2, 1, frameIdx))
                    plot3([rotatedXyzPoints(pt1, 1, frameIdx), rotatedXyzPoints(pt2, 1, frameIdx)], ...
                          [rotatedXyzPoints(pt1, 2, frameIdx), rotatedXyzPoints(pt2, 2, frameIdx)], ...
                          [rotatedXyzPoints(pt1, 3, frameIdx), rotatedXyzPoints(pt2, 3, frameIdx)], ...
                          'b-', 'LineWidth', 3);
                end
            end

            % Set consistent axis limits and labels
            if exist('xRange', 'var')
                xlim(xRange); ylim(yRange); zlim(zRange);
            end
            xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
            title(sprintf('3D Human Motion (Head Up, Legs Down) - Frame %d', frameIdx));
            grid on;
            view(45, 15); % Standard view angle

            % Capture frame
            frame = getframe(fig);
            writeVideo(writerObj, frame);

            if mod(frameCount, 50) == 0
                fprintf('Processed %d frames\n', frameCount);
            end
        end
    end

    close(writerObj);
    close(fig);

    fprintf('✓ 3D animation with fixed orientation saved as %s\n', videoFileName);
    fprintf('Total frames in video: %d\n', frameCount);
end


=== Creating 3D Animation Video with Fixed Orientation ===
Processed 50 frames
Error using alternateGetframe
A valid figure or axes handle must be specified

Error in getframe (line 72)
    x = alternateGetframe(parentFig, offsetRect, scaledOffsetRect, includeDecorations, true, h, offsetRectSpecified);
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 



% Create 3D animation video with properly rotated coordinates
if exist('xyzPoints', 'var') && ~isempty(xyzPoints)
    fprintf('\n=== Creating 3D Animation Video with Fixed Orientation ===\n');

    [numKeypoints, ~, numFrames, ~] = size(xyzPoints);

    % Set up video writer
    videoFileName = '3D_Human_Motion_Fixed_Orientation.mp4';
    writerObj = VideoWriter(videoFileName, 'MPEG-4');
    writerObj.FrameRate = 30;
    writerObj.Quality = 95;
    open(writerObj);

    % Create figure for animation
    fig = figure('Position', [100, 100, 1200, 800], 'Color', 'white');

    % Define body connections (MediaPipe format)
    connections = [11, 12; 12, 24; 11, 23; 23, 24; ... % torso
                  11, 13; 13, 15; 12, 14; 14, 16; ... % arms
                  23, 25; 25, 27; 24, 26; 26, 28];   % legs

    % FIXED: Apply proper rotation to make person upright
    % Rotate 90 degrees around Z-axis to make head up, legs down
    rotationMatrix = [0, 0, -1;   % Swap and negate Y to X
                      1,  0, 0;   % X becomes Y  
                      0,  1, 0];  % Z stays same

    % Apply rotation to all 3D points
    rotatedXyzPoints = nan(size(xyzPoints));
    for frameIdx = 1:numFrames
        for kpIdx = 1:numKeypoints
            if ~isnan(xyzPoints(kpIdx, 1, frameIdx))
                originalPoint = squeeze(xyzPoints(kpIdx, :, frameIdx));
                rotatedPoint = rotationMatrix * originalPoint';
                rotatedXyzPoints(kpIdx, :, frameIdx) = rotatedPoint';
            end
        end
    end

    % Find overall bounds for rotated points
    allValidPoints = rotatedXyzPoints(~isnan(rotatedXyzPoints));
    if ~isempty(allValidPoints)
        xRange = [min(rotatedXyzPoints(:, 1, :), [], 'all', 'omitnan'), max(rotatedXyzPoints(:, 1, :), [], 'all', 'omitnan')];
        yRange = [min(rotatedXyzPoints(:, 2, :), [], 'all', 'omitnan'), max(rotatedXyzPoints(:, 2, :), [], 'all', 'omitnan')];
        zRange = [min(rotatedXyzPoints(:, 3, :), [], 'all', 'omitnan'), max(rotatedXyzPoints(:, 3, :), [], 'all', 'omitnan')];

        % Add padding
        xPadding = (xRange(2) - xRange(1)) * 0.1;
        yPadding = (yRange(2) - yRange(1)) * 0.1;
        zPadding = (zRange(2) - zRange(1)) * 0.1;

        xRange = [xRange(1) - xPadding, xRange(2) + xPadding];
        yRange = [yRange(1) - yPadding, yRange(2) + yPadding];
        zRange = [zRange(1) - zPadding, zRange(2) + zPadding];
    end

    frameCount = 0;

    for frameIdx = 1:numFrames
        % Check if frame has valid 3D points
        validPoints = ~isnan(rotatedXyzPoints(:, 1, frameIdx));

        if sum(validPoints) >= 5
            frameCount = frameCount + 1;

            clf; % Clear figure

            % Plot rotated 3D points
            scatter3(rotatedXyzPoints(validPoints, 1, frameIdx), ...
                     rotatedXyzPoints(validPoints, 2, frameIdx), ...
                     rotatedXyzPoints(validPoints, 3, frameIdx), ...
                     100, 'filled', 'MarkerFaceColor', 'red', 'MarkerEdgeColor', 'black');

            hold on;

            % Draw skeleton connections with rotated points
            for i = 1:size(connections, 1)
                pt1 = connections(i, 1);
                pt2 = connections(i, 2);
                if ~isnan(rotatedXyzPoints(pt1, 1, frameIdx)) && ~isnan(rotatedXyzPoints(pt2, 1, frameIdx))
                    plot3([rotatedXyzPoints(pt1, 1, frameIdx), rotatedXyzPoints(pt2, 1, frameIdx)], ...
                          [rotatedXyzPoints(pt1, 2, frameIdx), rotatedXyzPoints(pt2, 2, frameIdx)], ...
                          [rotatedXyzPoints(pt1, 3, frameIdx), rotatedXyzPoints(pt2, 3, frameIdx)], ...
                          'b-', 'LineWidth', 3);
                end
            end

            % Set consistent axis limits and labels
            if exist('xRange', 'var')
                xlim(xRange); ylim(yRange); zlim(zRange);
            end
            xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
            title(sprintf('3D Human Motion (Head Up, Legs Down) - Frame %d', frameIdx));
            grid on;
            view(45, 15); % Standard view angle

            % Capture frame
            frame = getframe(fig);
            writeVideo(writerObj, frame);

            if mod(frameCount, 50) == 0
                fprintf('Processed %d frames\n', frameCount);
            end
        end
    end

    close(writerObj);
    close(fig);

    fprintf('✓ 3D animation with fixed orientation saved as %s\n', videoFileName);
    fprintf('Total frames in video: %d\n', frameCount);
end


=== Creating 3D Animation Video with Fixed Orientation ===
Processed 50 frames
Error using alternateGetframe
A valid figure or axes handle must be specified

Error in getframe (line 72)
    x = alternateGetframe(parentFig, offsetRect, scaledOffsetRect, includeDecorations, true, h, offsetRectSpecified);
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 



% Create 3D animation video with properly rotated coordinates
if exist('xyzPoints', 'var') && ~isempty(xyzPoints)
    fprintf('\n=== Creating 3D Animation Video with Fixed Orientation ===\n');

    [numKeypoints, ~, numFrames, ~] = size(xyzPoints);

    % Set up video writer
    videoFileName = '3D_Human_Motion_Fixed_Orientation.mp4';
    writerObj = VideoWriter(videoFileName, 'MPEG-4');
    writerObj.FrameRate = 30;
    writerObj.Quality = 95;
    open(writerObj);

    % Create figure for animation
    fig = figure('Position', [100, 100, 1200, 800], 'Color', 'white');

    % Define body connections (MediaPipe format)
    connections = [11, 12; 12, 24; 11, 23; 23, 24; ... % torso
                  11, 13; 13, 15; 12, 14; 14, 16; ... % arms
                  23, 25; 25, 27; 24, 26; 26, 28];   % legs

    % FIXED: Apply proper rotation to make person upright
    % Rotate 90 degrees around Z-axis to make head up, legs down
    rotationMatrix = [0, -1, 0;   % Swap and negate Y to X
                      0,  0, 0;   % X becomes Y  
                      0,  1, 0];  % Z stays same

    % Apply rotation to all 3D points
    rotatedXyzPoints = nan(size(xyzPoints));
    for frameIdx = 1:numFrames
        for kpIdx = 1:numKeypoints
            if ~isnan(xyzPoints(kpIdx, 1, frameIdx))
                originalPoint = squeeze(xyzPoints(kpIdx, :, frameIdx));
                rotatedPoint = rotationMatrix * originalPoint';
                rotatedXyzPoints(kpIdx, :, frameIdx) = rotatedPoint';
            end
        end
    end

    % Find overall bounds for rotated points
    allValidPoints = rotatedXyzPoints(~isnan(rotatedXyzPoints));
    if ~isempty(allValidPoints)
        xRange = [min(rotatedXyzPoints(:, 1, :), [], 'all', 'omitnan'), max(rotatedXyzPoints(:, 1, :), [], 'all', 'omitnan')];
        yRange = [min(rotatedXyzPoints(:, 2, :), [], 'all', 'omitnan'), max(rotatedXyzPoints(:, 2, :), [], 'all', 'omitnan')];
        zRange = [min(rotatedXyzPoints(:, 3, :), [], 'all', 'omitnan'), max(rotatedXyzPoints(:, 3, :), [], 'all', 'omitnan')];

        % Add padding
        xPadding = (xRange(2) - xRange(1)) * 0.1;
        yPadding = (yRange(2) - yRange(1)) * 0.1;
        zPadding = (zRange(2) - zRange(1)) * 0.1;

        xRange = [xRange(1) - xPadding, xRange(2) + xPadding];
        yRange = [yRange(1) - yPadding, yRange(2) + yPadding];
        zRange = [zRange(1) - zPadding, zRange(2) + zPadding];
    end

    frameCount = 0;

    for frameIdx = 1:numFrames
        % Check if frame has valid 3D points
        validPoints = ~isnan(rotatedXyzPoints(:, 1, frameIdx));

        if sum(validPoints) >= 5
            frameCount = frameCount + 1;

            clf; % Clear figure

            % Plot rotated 3D points
            scatter3(rotatedXyzPoints(validPoints, 1, frameIdx), ...
                     rotatedXyzPoints(validPoints, 2, frameIdx), ...
                     rotatedXyzPoints(validPoints, 3, frameIdx), ...
                     100, 'filled', 'MarkerFaceColor', 'red', 'MarkerEdgeColor', 'black');

            hold on;

            % Draw skeleton connections with rotated points
            for i = 1:size(connections, 1)
                pt1 = connections(i, 1);
                pt2 = connections(i, 2);
                if ~isnan(rotatedXyzPoints(pt1, 1, frameIdx)) && ~isnan(rotatedXyzPoints(pt2, 1, frameIdx))
                    plot3([rotatedXyzPoints(pt1, 1, frameIdx), rotatedXyzPoints(pt2, 1, frameIdx)], ...
                          [rotatedXyzPoints(pt1, 2, frameIdx), rotatedXyzPoints(pt2, 2, frameIdx)], ...
                          [rotatedXyzPoints(pt1, 3, frameIdx), rotatedXyzPoints(pt2, 3, frameIdx)], ...
                          'b-', 'LineWidth', 3);
                end
            end

            % Set consistent axis limits and labels
            if exist('xRange', 'var')
                xlim(xRange); ylim(yRange); zlim(zRange);
            end
            xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
            title(sprintf('3D Human Motion (Head Up, Legs Down) - Frame %d', frameIdx));
            grid on;
            view(45, 15); % Standard view angle

            % Capture frame
            frame = getframe(fig);
            writeVideo(writerObj, frame);

            if mod(frameCount, 50) == 0
                fprintf('Processed %d frames\n', frameCount);
            end
        end
    end

    close(writerObj);
    close(fig);

    fprintf('✓ 3D animation with fixed orientation saved as %s\n', videoFileName);
    fprintf('Total frames in video: %d\n', frameCount);
end


=== Creating 3D Animation Video with Fixed Orientation ===
Error using ylim (line 37)
Limits must be a 2-element vector of increasing numeric values.
 



% Create 3D animation video with properly rotated coordinates
if exist('xyzPoints', 'var') && ~isempty(xyzPoints)
    fprintf('\n=== Creating 3D Animation Video with Fixed Orientation ===\n');

    [numKeypoints, ~, numFrames, ~] = size(xyzPoints);

    % Set up video writer
    videoFileName = '3D_Human_Motion_Fixed_Orientation.mp4';
    writerObj = VideoWriter(videoFileName, 'MPEG-4');
    writerObj.FrameRate = 30;
    writerObj.Quality = 95;
    open(writerObj);

    % Create figure for animation
    fig = figure('Position', [100, 100, 1200, 800], 'Color', 'white');

    % Define body connections (MediaPipe format)
    connections = [11, 12; 12, 24; 11, 23; 23, 24; ... % torso
                  11, 13; 13, 15; 12, 14; 14, 16; ... % arms
                  23, 25; 25, 27; 24, 26; 26, 28];   % legs

    % FIXED: Apply proper rotation to make person upright
    % Rotate 90 degrees around Z-axis to make head up, legs down
    rotationMatrix = [0, 0, 1;   % Swap and negate Y to X
                      1,  0, 0;   % X becomes Y  
                      0,  1, 0];  % Z stays same

    % Apply rotation to all 3D points
    rotatedXyzPoints = nan(size(xyzPoints));
    for frameIdx = 1:numFrames
        for kpIdx = 1:numKeypoints
            if ~isnan(xyzPoints(kpIdx, 1, frameIdx))
                originalPoint = squeeze(xyzPoints(kpIdx, :, frameIdx));
                rotatedPoint = rotationMatrix * originalPoint';
                rotatedXyzPoints(kpIdx, :, frameIdx) = rotatedPoint';
            end
        end
    end

    % Find overall bounds for rotated points
    allValidPoints = rotatedXyzPoints(~isnan(rotatedXyzPoints));
    if ~isempty(allValidPoints)
        xRange = [min(rotatedXyzPoints(:, 1, :), [], 'all', 'omitnan'), max(rotatedXyzPoints(:, 1, :), [], 'all', 'omitnan')];
        yRange = [min(rotatedXyzPoints(:, 2, :), [], 'all', 'omitnan'), max(rotatedXyzPoints(:, 2, :), [], 'all', 'omitnan')];
        zRange = [min(rotatedXyzPoints(:, 3, :), [], 'all', 'omitnan'), max(rotatedXyzPoints(:, 3, :), [], 'all', 'omitnan')];

        % Add padding
        xPadding = (xRange(2) - xRange(1)) * 0.1;
        yPadding = (yRange(2) - yRange(1)) * 0.1;
        zPadding = (zRange(2) - zRange(1)) * 0.1;

        xRange = [xRange(1) - xPadding, xRange(2) + xPadding];
        yRange = [yRange(1) - yPadding, yRange(2) + yPadding];
        zRange = [zRange(1) - zPadding, zRange(2) + zPadding];
    end

    frameCount = 0;

    for frameIdx = 1:numFrames
        % Check if frame has valid 3D points
        validPoints = ~isnan(rotatedXyzPoints(:, 1, frameIdx));

        if sum(validPoints) >= 5
            frameCount = frameCount + 1;

            clf; % Clear figure

            % Plot rotated 3D points
            scatter3(rotatedXyzPoints(validPoints, 1, frameIdx), ...
                     rotatedXyzPoints(validPoints, 2, frameIdx), ...
                     rotatedXyzPoints(validPoints, 3, frameIdx), ...
                     100, 'filled', 'MarkerFaceColor', 'red', 'MarkerEdgeColor', 'black');

            hold on;

            % Draw skeleton connections with rotated points
            for i = 1:size(connections, 1)
                pt1 = connections(i, 1);
                pt2 = connections(i, 2);
                if ~isnan(rotatedXyzPoints(pt1, 1, frameIdx)) && ~isnan(rotatedXyzPoints(pt2, 1, frameIdx))
                    plot3([rotatedXyzPoints(pt1, 1, frameIdx), rotatedXyzPoints(pt2, 1, frameIdx)], ...
                          [rotatedXyzPoints(pt1, 2, frameIdx), rotatedXyzPoints(pt2, 2, frameIdx)], ...
                          [rotatedXyzPoints(pt1, 3, frameIdx), rotatedXyzPoints(pt2, 3, frameIdx)], ...
                          'b-', 'LineWidth', 3);
                end
            end

            % Set consistent axis limits and labels
            if exist('xRange', 'var')
                xlim(xRange); ylim(yRange); zlim(zRange);
            end
            xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
            title(sprintf('3D Human Motion (Head Up, Legs Down) - Frame %d', frameIdx));
            grid on;
            view(45, 15); % Standard view angle

            % Capture frame
            frame = getframe(fig);
            writeVideo(writerObj, frame);

            if mod(frameCount, 50) == 0
                fprintf('Processed %d frames\n', frameCount);
            end
        end
    end

    close(writerObj);
    close(fig);

    fprintf('✓ 3D animation with fixed orientation saved as %s\n', videoFileName);
    fprintf('Total frames in video: %d\n', frameCount);
end

=== Creating 3D Animation Video with Fixed Orientation ===
Warning: No video frames were written to this file. The file may be invalid. 
> In VideoWriter/close (line 282)
In VideoWriter/delete (line 217) 
Processed 50 frames
Error using alternateGetframe
A valid figure or axes handle must be specified

Error in getframe (line 72)
    x = alternateGetframe(parentFig, offsetRect, scaledOffsetRect, includeDecorations, true, h, offsetRectSpecified);
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 



% Create 3D animation video with properly rotated coordinates
if exist('xyzPoints', 'var') && ~isempty(xyzPoints)
    fprintf('\n=== Creating 3D Animation Video with Fixed Orientation ===\n');

    [numKeypoints, ~, numFrames, ~] = size(xyzPoints);

    % Set up video writer
    videoFileName = '3D_Human_Motion_Fixed_Orientation.mp4';
    writerObj = VideoWriter(videoFileName, 'MPEG-4');
    writerObj.FrameRate = 30;
    writerObj.Quality = 95;
    open(writerObj);

    % Create figure for animation
    fig = figure('Position', [100, 100, 1200, 800], 'Color', 'white');

    % Define body connections (MediaPipe format)
    connections = [11, 12; 12, 24; 11, 23; 23, 24; ... % torso
                  11, 13; 13, 15; 12, 14; 14, 16; ... % arms
                  23, 25; 25, 27; 24, 26; 26, 28];   % legs

    % FIXED: Apply proper rotation to make person upright
    % Rotate 90 degrees around Z-axis to make head up, legs down
    rotationMatrix = [0, 0, -1;   % Swap and negate Y to X
                      -1,  0, 0;   % X becomes Y  
                      0,  1, 0];  % Z stays same

    % Apply rotation to all 3D points
    rotatedXyzPoints = nan(size(xyzPoints));
    for frameIdx = 1:numFrames
        for kpIdx = 1:numKeypoints
            if ~isnan(xyzPoints(kpIdx, 1, frameIdx))
                originalPoint = squeeze(xyzPoints(kpIdx, :, frameIdx));
                rotatedPoint = rotationMatrix * originalPoint';
                rotatedXyzPoints(kpIdx, :, frameIdx) = rotatedPoint';
            end
        end
    end

    % Find overall bounds for rotated points
    allValidPoints = rotatedXyzPoints(~isnan(rotatedXyzPoints));
    if ~isempty(allValidPoints)
        xRange = [min(rotatedXyzPoints(:, 1, :), [], 'all', 'omitnan'), max(rotatedXyzPoints(:, 1, :), [], 'all', 'omitnan')];
        yRange = [min(rotatedXyzPoints(:, 2, :), [], 'all', 'omitnan'), max(rotatedXyzPoints(:, 2, :), [], 'all', 'omitnan')];
        zRange = [min(rotatedXyzPoints(:, 3, :), [], 'all', 'omitnan'), max(rotatedXyzPoints(:, 3, :), [], 'all', 'omitnan')];

        % Add padding
        xPadding = (xRange(2) - xRange(1)) * 0.1;
        yPadding = (yRange(2) - yRange(1)) * 0.1;
        zPadding = (zRange(2) - zRange(1)) * 0.1;

        xRange = [xRange(1) - xPadding, xRange(2) + xPadding];
        yRange = [yRange(1) - yPadding, yRange(2) + yPadding];
        zRange = [zRange(1) - zPadding, zRange(2) + zPadding];
    end

    frameCount = 0;

    for frameIdx = 1:numFrames
        % Check if frame has valid 3D points
        validPoints = ~isnan(rotatedXyzPoints(:, 1, frameIdx));

        if sum(validPoints) >= 5
            frameCount = frameCount + 1;

            clf; % Clear figure

            % Plot rotated 3D points
            scatter3(rotatedXyzPoints(validPoints, 1, frameIdx), ...
                     rotatedXyzPoints(validPoints, 2, frameIdx), ...
                     rotatedXyzPoints(validPoints, 3, frameIdx), ...
                     100, 'filled', 'MarkerFaceColor', 'red', 'MarkerEdgeColor', 'black');

            hold on;

            % Draw skeleton connections with rotated points
            for i = 1:size(connections, 1)
                pt1 = connections(i, 1);
                pt2 = connections(i, 2);
                if ~isnan(rotatedXyzPoints(pt1, 1, frameIdx)) && ~isnan(rotatedXyzPoints(pt2, 1, frameIdx))
                    plot3([rotatedXyzPoints(pt1, 1, frameIdx), rotatedXyzPoints(pt2, 1, frameIdx)], ...
                          [rotatedXyzPoints(pt1, 2, frameIdx), rotatedXyzPoints(pt2, 2, frameIdx)], ...
                          [rotatedXyzPoints(pt1, 3, frameIdx), rotatedXyzPoints(pt2, 3, frameIdx)], ...
                          'b-', 'LineWidth', 3);
                end
            end

            % Set consistent axis limits and labels
            if exist('xRange', 'var')
                xlim(xRange); ylim(yRange); zlim(zRange);
            end
            xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
            title(sprintf('3D Human Motion (Head Up, Legs Down) - Frame %d', frameIdx));
            grid on;
            view(45, 15); % Standard view angle

            % Capture frame
            frame = getframe(fig);
            writeVideo(writerObj, frame);

            if mod(frameCount, 50) == 0
                fprintf('Processed %d frames\n', frameCount);
            end
        end
    end

    close(writerObj);
    close(fig);

    fprintf('✓ 3D animation with fixed orientation saved as %s\n', videoFileName);
    fprintf('Total frames in video: %d\n', frameCount);
end

=== Creating 3D Animation Video with Fixed Orientation ===
Error using alternateGetframe
A valid figure or axes handle must be specified

Error in getframe (line 72)
    x = alternateGetframe(parentFig, offsetRect, scaledOffsetRect, includeDecorations, true, h, offsetRectSpecified);
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 




% Create 3D animation video with properly rotated coordinates
if exist('xyzPoints', 'var') && ~isempty(xyzPoints)
    fprintf('\n=== Creating 3D Animation Video with Fixed Orientation ===\n');

    [numKeypoints, ~, numFrames, ~] = size(xyzPoints);

    % Set up video writer
    videoFileName = '3D_Human_Motion_Fixed_Orientation.mp4';
    writerObj = VideoWriter(videoFileName, 'MPEG-4');
    writerObj.FrameRate = 30;
    writerObj.Quality = 95;
    open(writerObj);

    % Create figure for animation
    fig = figure('Position', [100, 100, 1200, 800], 'Color', 'white');

    % Define body connections (MediaPipe format)
    connections = [11, 12; 12, 24; 11, 23; 23, 24; ... % torso
                  11, 13; 13, 15; 12, 14; 14, 16; ... % arms
                  23, 25; 25, 27; 24, 26; 26, 28];   % legs

    % FIXED: Apply proper rotation to make person upright
    % Rotate 90 degrees around Z-axis to make head up, legs down
    rotationMatrix = [0, 0, -1;   % Swap and negate Y to X
                      1,  0, 0;   % X becomes Y  
                      0,  -1, 0];  % Z stays same

    % Apply rotation to all 3D points
    rotatedXyzPoints = nan(size(xyzPoints));
    for frameIdx = 1:numFrames
        for kpIdx = 1:numKeypoints
            if ~isnan(xyzPoints(kpIdx, 1, frameIdx))
                originalPoint = squeeze(xyzPoints(kpIdx, :, frameIdx));
                rotatedPoint = rotationMatrix * originalPoint';
                rotatedXyzPoints(kpIdx, :, frameIdx) = rotatedPoint';
            end
        end
    end

    % Find overall bounds for rotated points
    allValidPoints = rotatedXyzPoints(~isnan(rotatedXyzPoints));
    if ~isempty(allValidPoints)
        xRange = [min(rotatedXyzPoints(:, 1, :), [], 'all', 'omitnan'), max(rotatedXyzPoints(:, 1, :), [], 'all', 'omitnan')];
        yRange = [min(rotatedXyzPoints(:, 2, :), [], 'all', 'omitnan'), max(rotatedXyzPoints(:, 2, :), [], 'all', 'omitnan')];
        zRange = [min(rotatedXyzPoints(:, 3, :), [], 'all', 'omitnan'), max(rotatedXyzPoints(:, 3, :), [], 'all', 'omitnan')];

        % Add padding
        xPadding = (xRange(2) - xRange(1)) * 0.1;
        yPadding = (yRange(2) - yRange(1)) * 0.1;
        zPadding = (zRange(2) - zRange(1)) * 0.1;

        xRange = [xRange(1) - xPadding, xRange(2) + xPadding];
        yRange = [yRange(1) - yPadding, yRange(2) + yPadding];
        zRange = [zRange(1) - zPadding, zRange(2) + zPadding];
    end

    frameCount = 0;

    for frameIdx = 1:numFrames
        % Check if frame has valid 3D points
        validPoints = ~isnan(rotatedXyzPoints(:, 1, frameIdx));

        if sum(validPoints) >= 5
            frameCount = frameCount + 1;

            clf; % Clear figure

            % Plot rotated 3D points
            scatter3(rotatedXyzPoints(validPoints, 1, frameIdx), ...
                     rotatedXyzPoints(validPoints, 2, frameIdx), ...
                     rotatedXyzPoints(validPoints, 3, frameIdx), ...
                     100, 'filled', 'MarkerFaceColor', 'red', 'MarkerEdgeColor', 'black');

            hold on;

            % Draw skeleton connections with rotated points
            for i = 1:size(connections, 1)
                pt1 = connections(i, 1);
                pt2 = connections(i, 2);
                if ~isnan(rotatedXyzPoints(pt1, 1, frameIdx)) && ~isnan(rotatedXyzPoints(pt2, 1, frameIdx))
                    plot3([rotatedXyzPoints(pt1, 1, frameIdx), rotatedXyzPoints(pt2, 1, frameIdx)], ...
                          [rotatedXyzPoints(pt1, 2, frameIdx), rotatedXyzPoints(pt2, 2, frameIdx)], ...
                          [rotatedXyzPoints(pt1, 3, frameIdx), rotatedXyzPoints(pt2, 3, frameIdx)], ...
                          'b-', 'LineWidth', 3);
                end
            end

            % Set consistent axis limits and labels
            if exist('xRange', 'var')
                xlim(xRange); ylim(yRange); zlim(zRange);
            end
            xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
            title(sprintf('3D Human Motion (Head Up, Legs Down) - Frame %d', frameIdx));
            grid on;
            view(45, 15); % Standard view angle

            % Capture frame
            frame = getframe(fig);
            writeVideo(writerObj, frame);

            if mod(frameCount, 50) == 0
                fprintf('Processed %d frames\n', frameCount);
            end
        end
    end

    close(writerObj);
    close(fig);

    fprintf('✓ 3D animation with fixed orientation saved as %s\n', videoFileName);
    fprintf('Total frames in video: %d\n', frameCount);
end

=== Creating 3D Animation Video with Fixed Orientation ===
Processed 50 frames
Processed 100 frames
Processed 150 frames
✓ 3D animation with fixed orientation saved as 3D_Human_Motion_Fixed_Orientation.mp4
Total frames in video: 153
