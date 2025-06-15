function processedXyzPoints = stereoPostProcessing(xyzPoints)
    % Enhanced post-processing pipeline for stereo 3D points
    
    fprintf('=== Enhanced Stereo Post-Processing Pipeline ===\n');
    
    % Step 1: Enhanced temporal smoothing
    fprintf('Applying enhanced temporal smoothing...\n');
    smoothedPoints = enhancedTemporalSmoothing(xyzPoints, 7);
    
    % Step 2: Aggressive gap filling
    fprintf('Applying aggressive gap filling...\n');
    filledPoints = aggressiveGapFilling(smoothedPoints, 5);
    
    % Step 3: Smart outlier removal
    fprintf('Applying smart outlier removal...\n');
    cleanedPoints = smartOutlierRemoval(filledPoints);
    
    % Step 4: Biomechanical constraint enforcement
    fprintf('Applying biomechanical constraints...\n');
    constrainedPoints = applyBiomechanicalConstraints(cleanedPoints);
    
    processedXyzPoints = constrainedPoints;
    
    % Report improvements
    originalValid = sum(~isnan(xyzPoints(:)));
    finalValid = sum(~isnan(processedXyzPoints(:)));
    improvement = ((finalValid - originalValid) / numel(xyzPoints)) * 100;
    
    fprintf('\n✓ Enhanced stereo post-processing results:\n');
    fprintf('  Original valid points: %d\n', originalValid);
    fprintf('  Final valid points: %d\n', finalValid);
    fprintf('  Improvement: %.1f%%\n', improvement);
    
    save('enhanced_processed_stereo_points.mat', 'processedXyzPoints');
end

function smoothedXyzPoints = enhancedTemporalSmoothing(xyzPoints, windowSize)
    % Enhanced temporal smoothing with adaptive window size
    [numKeypoints, numCoords, numFrames] = size(xyzPoints);
    smoothedXyzPoints = xyzPoints;
    
    for kpIdx = 1:numKeypoints
        for coordIdx = 1:numCoords
            timeSeries = squeeze(xyzPoints(kpIdx, coordIdx, :));
            validFrames = ~isnan(timeSeries);
            
            if sum(validFrames) > windowSize
                validData = timeSeries(validFrames);
                validIndices = find(validFrames);
                
                % Adaptive window size based on data density
                adaptiveWindow = min(windowSize, floor(length(validData) / 3));
                adaptiveWindow = max(3, adaptiveWindow); % Minimum window size
                
                if length(validData) > adaptiveWindow
                    try
                        % Use Savitzky-Golay filter with adaptive parameters
                        polyOrder = min(3, adaptiveWindow - 1);
                        smoothedData = sgolayfilt(validData, polyOrder, adaptiveWindow);
                        smoothedXyzPoints(kpIdx, coordIdx, validIndices) = smoothedData;
                    catch
                        % Fallback to robust moving median
                        smoothedData = movmedian(validData, adaptiveWindow);
                        smoothedXyzPoints(kpIdx, coordIdx, validIndices) = smoothedData;
                    end
                end
            end
        end
    end
end

function filledXyzPoints = aggressiveGapFilling(xyzPoints, maxGapSize)
    % Aggressive gap filling with multiple interpolation methods
    [numKeypoints, numCoords, numFrames] = size(xyzPoints);
    filledXyzPoints = xyzPoints;
    
    for kpIdx = 1:numKeypoints
        for coordIdx = 1:numCoords
            timeSeries = squeeze(xyzPoints(kpIdx, coordIdx, :));
            validFrames = find(~isnan(timeSeries));
            
            if length(validFrames) > 2
                gaps = diff(validFrames);
                gapIndices = find(gaps > 1 & gaps <= maxGapSize);
                
                for gapIdx = gapIndices'
                    startFrame = validFrames(gapIdx);
                    endFrame = validFrames(gapIdx + 1);
                    
                    if isscalar(startFrame) && isscalar(endFrame) && endFrame > startFrame + 1
                        gapSize = endFrame - startFrame - 1;
                        
                        if gapSize <= maxGapSize
                            interpFrames = (startFrame+1):(endFrame-1);
                            
                            if ~isempty(interpFrames)
                                % Choose interpolation method based on gap size and context
                                if gapSize <= 2
                                    % Linear for small gaps
                                    startValue = timeSeries(startFrame);
                                    endValue = timeSeries(endFrame);
                                    interpValues = interp1([startFrame, endFrame], ...
                                                         [startValue, endValue], interpFrames);
                                else
                                    % Spline for larger gaps with more context
                                    contextFrames = max(1, startFrame-2):min(numFrames, endFrame+2);
                                    contextData = timeSeries(contextFrames);
                                    validContext = ~isnan(contextData);
                                    
                                    if sum(validContext) >= 4
                                        try
                                            interpValues = interp1(contextFrames(validContext), ...
                                                                 contextData(validContext), ...
                                                                 interpFrames, 'spline');
                                        catch
                                            % Fallback to linear
                                            startValue = timeSeries(startFrame);
                                            endValue = timeSeries(endFrame);
                                            interpValues = interp1([startFrame, endFrame], ...
                                                                 [startValue, endValue], interpFrames);
                                        end
                                    else
                                        % Linear fallback
                                        startValue = timeSeries(startFrame);
                                        endValue = timeSeries(endFrame);
                                        interpValues = interp1([startFrame, endFrame], ...
                                                             [startValue, endValue], interpFrames);
                                    end
                                end
                                
                                filledXyzPoints(kpIdx, coordIdx, interpFrames) = interpValues;
                            end
                        end
                    end
                end
            end
        end
    end
end

function cleanedPoints = smartOutlierRemoval(xyzPoints)
    % Smart outlier removal with biomechanical awareness
    [numKeypoints, numCoords, numFrames] = size(xyzPoints);
    cleanedPoints = xyzPoints;
    
    for kpIdx = 1:numKeypoints
        for coordIdx = 1:numCoords
            timeSeries = squeeze(xyzPoints(kpIdx, coordIdx, :));
            validFrames = ~isnan(timeSeries);
            
            if sum(validFrames) > 15 % Need sufficient data
                validData = timeSeries(validFrames);
                validIndices = find(validFrames);
                
                % Multi-criteria outlier detection
                outliers = false(size(validData));
                
                % 1. Statistical outliers (3-sigma rule)
                meanVal = mean(validData);
                stdVal = std(validData);
                statOutliers = abs(validData - meanVal) > 3 * stdVal;
                outliers = outliers | statOutliers;
                
                % 2. Velocity-based outliers
                if length(validData) > 5
                    velocities = diff(validData);
                    meanVel = mean(velocities);
                    stdVel = std(velocities);
                    velOutliers = [false; abs(velocities - meanVel) > 3 * stdVel];
                    outliers = outliers | velOutliers;
                end
                
                % 3. Acceleration-based outliers
                if length(validData) > 10
                    accelerations = diff(validData, 2);
                    meanAcc = mean(accelerations);
                    stdAcc = std(accelerations);
                    accOutliers = [false; false; abs(accelerations - meanAcc) > 3 * stdAcc];
                    outliers = outliers | accOutliers;
                end
                
                % 4. Median-based robust outlier detection
                medianVal = median(validData);
                mad = median(abs(validData - medianVal));
                robustOutliers = abs(validData - medianVal) > 3 * mad;
                outliers = outliers | robustOutliers;
                
                % Conservative outlier removal (only remove if multiple criteria agree)
                finalOutliers = sum([statOutliers, velOutliers, accOutliers, robustOutliers], 2) >= 2;
                
                % Mark outliers as NaN
                cleanedPoints(kpIdx, coordIdx, validIndices(finalOutliers)) = NaN;
            end
        end
    end
end

function constrainedPoints = applyBiomechanicalConstraints(xyzPoints)
    % Apply biomechanical constraints to improve data quality
    constrainedPoints = xyzPoints;
    
    % Define realistic human body segment lengths (in mm)
    constraints = struct();
    constraints.upperArm = [200, 450];    % shoulder to elbow
    constraints.forearm = [200, 400];     % elbow to wrist
    constraints.thigh = [350, 550];       % hip to knee
    constraints.shin = [300, 500];        % knee to ankle
    constraints.torso = [400, 800];       % shoulder to hip
    
    % MediaPipe keypoint pairs for body segments
    segments = struct();
    segments.rightUpperArm = [12, 14];
    segments.rightForearm = [14, 16];
    segments.leftUpperArm = [11, 13];
    segments.leftForearm = [13, 15];
    segments.rightThigh = [24, 26];
    segments.rightShin = [26, 28];
    segments.leftThigh = [23, 25];
    segments.leftShin = [25, 27];
    segments.torsoRight = [12, 24];
    segments.torsoLeft = [11, 23];
    
    numFrames = size(xyzPoints, 3);
    
    for frameIdx = 1:numFrames
        segmentNames = fieldnames(segments);
        
        for segIdx = 1:length(segmentNames)
            segmentName = segmentNames{segIdx};
            keypoints = segments.(segmentName);
            
            pt1 = squeeze(xyzPoints(keypoints(1), :, frameIdx));
            pt2 = squeeze(xyzPoints(keypoints(2), :, frameIdx));
            
            if ~any(isnan([pt1, pt2]))
                distance = norm(pt2 - pt1);
                
                % Determine constraint based on segment type
                if contains(segmentName, 'UpperArm')
                    validRange = constraints.upperArm;
                elseif contains(segmentName, 'Forearm')
                    validRange = constraints.forearm;
                elseif contains(segmentName, 'Thigh')
                    validRange = constraints.thigh;
                elseif contains(segmentName, 'Shin')
                    validRange = constraints.shin;
                elseif contains(segmentName, 'torso')
                    validRange = constraints.torso;
                else
                    continue;
                end
                
                % Apply constraints with some tolerance
                tolerance = 0.3; % 30% tolerance
                minDist = validRange(1) * (1 - tolerance);
                maxDist = validRange(2) * (1 + tolerance);
                
                if distance < minDist || distance > maxDist
                    % Mark points as invalid if constraint is violated
                    constrainedPoints(keypoints(1), :, frameIdx) = NaN;
                    constrainedPoints(keypoints(2), :, frameIdx) = NaN;
                end
            end
        end
    end
end
