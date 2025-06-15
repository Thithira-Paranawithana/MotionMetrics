function rotatedXyzPoints = applyStereoRotation(xyzPoints, rotationMatrix)
    % Apply coordinate rotation for proper orientation in stereo setup
    
    fprintf('Applying coordinate rotation for proper orientation...\n');
    
    [numKeypoints, ~, numFrames] = size(xyzPoints);
    rotatedXyzPoints = nan(size(xyzPoints));
    
    rotatedCount = 0;
    totalCount = 0;
    
    for frameIdx = 1:numFrames
        for kpIdx = 1:numKeypoints
            if ~isnan(xyzPoints(kpIdx, 1, frameIdx))
                totalCount = totalCount + 1;
                
                originalPoint = squeeze(xyzPoints(kpIdx, :, frameIdx));
                rotatedPoint = rotationMatrix * originalPoint';
                rotatedXyzPoints(kpIdx, :, frameIdx) = rotatedPoint';
                
                rotatedCount = rotatedCount + 1;
            end
        end
    end
    
    fprintf('✓ Coordinate rotation applied to %d/%d points\n', rotatedCount, totalCount);
    
    save('rotated_stereo_points.mat', 'rotatedXyzPoints', 'rotationMatrix');
end
