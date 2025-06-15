function humanKeypoints = loadStereoKeypoints(keypointFiles)
    % Load keypoints for stereo camera setup (2 cameras)
    
    humanKeypoints = [];
    numCameras = 2;
    
    fprintf('=== Loading Stereo Keypoints ===\n');
    
    for camIdx = 1:numCameras
        jsonFile = keypointFiles{camIdx};
        
        if isfile(jsonFile)
            fprintf('Loading %s...', jsonFile);
            
            try
                % Read JSON file
                fid = fopen(jsonFile, 'r');
                raw = fread(fid, inf);
                str = char(raw');
                fclose(fid);
                
                data = jsondecode(str);
                
                % Handle numeric array format
                if isnumeric(data)
                    [numFramesInFile, numKeypoints, ~] = size(data);
                    
                    if isempty(humanKeypoints)
                        humanKeypoints = nan(numKeypoints, 2, numFramesInFile, numCameras);
                        fprintf('\n  Initialized array: [%d keypoints, 2 coords, %d frames, %d cameras]\n', ...
                               numKeypoints, numFramesInFile, numCameras);
                    end
                    
                    validKeypointsCount = 0;
                    for frameIdx = 1:numFramesInFile
                        for kpIdx = 1:numKeypoints
                            x = data(frameIdx, kpIdx, 1);
                            y = data(frameIdx, kpIdx, 2);
                            visibility = data(frameIdx, kpIdx, 3);
                            
                            if visibility > 0.3 && ~isnan(x) && ~isnan(y) && x > 0 && y > 0
                                humanKeypoints(kpIdx, 1, frameIdx, camIdx) = x;
                                humanKeypoints(kpIdx, 2, frameIdx, camIdx) = y;
                                validKeypointsCount = validKeypointsCount + 1;
                            end
                        end
                    end
                    
                    fprintf(' ✓ %d valid keypoints loaded\n', validKeypointsCount);
                end
                
            catch ME
                fprintf(' ✗ Error: %s\n', ME.message);
            end
        else
            fprintf('✗ File not found: %s\n', jsonFile);
        end
    end
    
    % Report overall statistics
    if ~isempty(humanKeypoints)
        totalValid = sum(~isnan(humanKeypoints(:)));
        totalPossible = numel(humanKeypoints);
        successRate = (totalValid / totalPossible) * 100;
        
        fprintf('\n✓ Overall stereo keypoint loading: %.1f%% (%d/%d points)\n', ...
               successRate, totalValid, totalPossible);
        
        save('loaded_stereo_keypoints.mat', 'humanKeypoints');
    else
        fprintf('\n✗ No keypoints loaded. Check MediaPipe processing.\n');
    end
end
