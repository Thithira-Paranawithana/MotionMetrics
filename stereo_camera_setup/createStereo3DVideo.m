function createStereo3DVideo(rotatedXyzPoints, videoFiles, stereoParams)
    % Create multiple types of 3D visualization videos
    
    fprintf('=== Creating Multiple 3D Animation Videos ===\n');
    
    % Define body connections with color coding
    connections = struct();
    connections.torso = [11, 12; 12, 24; 11, 23; 23, 24]; % Red - Core
    connections.arms = [11, 13; 13, 15; 12, 14; 14, 16];  % Green - Arms  
    connections.legs = [23, 25; 25, 27; 24, 26; 26, 28];  % Blue - Legs
    connections.head = [1, 11; 1, 12];                    % Yellow - Head connections
    
    % Calculate bounds for all videos
    xRange = [min(rotatedXyzPoints(:, 1, :), [], 'all', 'omitnan'), ...
              max(rotatedXyzPoints(:, 1, :), [], 'all', 'omitnan')];
    yRange = [min(rotatedXyzPoints(:, 2, :), [], 'all', 'omitnan'), ...
              max(rotatedXyzPoints(:, 2, :), [], 'all', 'omitnan')];
    zRange = [min(rotatedXyzPoints(:, 3, :), [], 'all', 'omitnan'), ...
              max(rotatedXyzPoints(:, 3, :), [], 'all', 'omitnan')];
    
    padding = 0.25;
    xPadding = (xRange(2) - xRange(1)) * padding;
    yPadding = (yRange(2) - yRange(1)) * padding;
    zPadding = (zRange(2) - zRange(1)) * padding;
    
    bounds = struct();
    bounds.x = [xRange(1) - xPadding, xRange(2) + xPadding];
    bounds.y = [yRange(1) - yPadding, yRange(2) + yPadding];
    bounds.z = [zRange(1) - zPadding, zRange(2) + zPadding];
    
    % Create different video types
    fprintf('Creating Video 1: Dynamic Rotating View...\n');
    createVideoType(rotatedXyzPoints, connections, bounds, stereoParams, 'dynamic', '1_Dynamic_Rotating_3D.mp4');
    
    fprintf('Creating Video 2: Static Front View...\n');
    createVideoType(rotatedXyzPoints, connections, bounds, stereoParams, 'front', '2_Static_Front_3D.mp4');
    
    fprintf('Creating Video 3: Side Profile View...\n');
    createVideoType(rotatedXyzPoints, connections, bounds, stereoParams, 'side', '3_Side_Profile_3D.mp4');
    
    fprintf('Creating Video 4: Top-Down View...\n');
    createVideoType(rotatedXyzPoints, connections, bounds, stereoParams, 'topdown', '4_TopDown_3D.mp4');
    
    fprintf('Creating Video 5: Multi-Angle View...\n');
    createVideoType(rotatedXyzPoints, connections, bounds, stereoParams, 'multiangle', '5_MultiAngle_3D.mp4');
    
    fprintf('Creating Video 6: Limb Focus View...\n');
    createVideoType(rotatedXyzPoints, connections, bounds, stereoParams, 'limbfocus', '6_LimbFocus_3D.mp4');
    
    fprintf('✓ All 6 video types created successfully!\n');
end

function createVideoType(rotatedXyzPoints, connections, bounds, stereoParams, mode, filename)
    % Create specific video type based on mode
    
    writerObj = VideoWriter(filename, 'MPEG-4');
    writerObj.FrameRate = 30;
    writerObj.Quality = 95;
    open(writerObj);
    
    % Figure setup based on mode
    if strcmp(mode, 'multiangle')
        fig = figure('Position', [100, 100, 1600, 1200], 'Color', 'black');
    else
        fig = figure('Position', [100, 100, 1200, 900], 'Color', 'black');
    end
    
    numFrames = size(rotatedXyzPoints, 3);
    frameCount = 0;
    
    % Color schemes
    colors = struct();
    colors.joints = [1, 0.3, 0.3];      % Bright red for joints
    colors.torso = [1, 0.4, 0.4];       % Red for torso
    colors.arms = [0.3, 1, 0.3];        % Green for arms
    colors.legs = [0.3, 0.5, 1];        % Blue for legs
    colors.head = [1, 1, 0.3];          % Yellow for head
    
    for frameIdx = 1:numFrames
        validPoints = ~isnan(rotatedXyzPoints(:, 1, frameIdx));
        
        if sum(validPoints) >= 3
            frameCount = frameCount + 1;
            
            clf;
            
            switch mode
                case 'dynamic'
                    createDynamicView(rotatedXyzPoints, connections, colors, bounds, frameIdx, frameCount, stereoParams, validPoints);
                    
                case 'front'
                    createStaticView(rotatedXyzPoints, connections, colors, bounds, frameIdx, validPoints, 0, 0, 'Front View');
                    
                case 'side'
                    createStaticView(rotatedXyzPoints, connections, colors, bounds, frameIdx, validPoints, 90, 0, 'Side Profile');
                    
                case 'topdown'
                    createStaticView(rotatedXyzPoints, connections, colors, bounds, frameIdx, validPoints, 0, 90, 'Top-Down View');
                    
                case 'multiangle'
                    createMultiAngleView(rotatedXyzPoints, connections, colors, bounds, frameIdx, validPoints);
                    
                case 'limbfocus'
                    createLimbFocusView(rotatedXyzPoints, connections, colors, bounds, frameIdx, validPoints, frameCount);
            end
            
            frame = getframe(fig);
            writeVideo(writerObj, frame);
        end
    end
    
    close(writerObj);
    close(fig);
    
    fprintf('  ✓ %s created (%d frames)\n', filename, frameCount);
end

function createDynamicView(xyzPoints, connections, colors, bounds, frameIdx, frameCount, stereoParams, validPoints)
    % Dynamic rotating view with enhanced visuals
    
    plotSkeleton(xyzPoints, connections, colors, frameIdx, validPoints);
    
    xlim(bounds.x); ylim(bounds.y); zlim(bounds.z);
    
    % Dynamic rotation
    baseAzimuth = 45;
    baseElevation = 15;
    rotationSpeed = 1.5;
    dynamicAzimuth = baseAzimuth + (frameCount * rotationSpeed);
    elevationVariation = 8 * sin(frameCount * 0.05);
    dynamicElevation = baseElevation + elevationVariation;
    
    view(dynamicAzimuth, dynamicElevation);
    
    setupAxes('Dynamic Rotating View', frameIdx, stereoParams, validPoints);
end

function createStaticView(xyzPoints, connections, colors, bounds, frameIdx, validPoints, azimuth, elevation, titleStr)
    % Static view from specific angle
    
    plotSkeleton(xyzPoints, connections, colors, frameIdx, validPoints);
    
    xlim(bounds.x); ylim(bounds.y); zlim(bounds.z);
    view(azimuth, elevation);
    
    setupAxes(titleStr, frameIdx, [], validPoints);
end

function createMultiAngleView(xyzPoints, connections, colors, bounds, frameIdx, validPoints)
    % Four different angles in one video
    
    views = struct();
    views.front = [0, 0, 'Front'];
    views.side = [90, 0, 'Side'];
    views.back = [180, 0, 'Back'];
    views.top = [0, 90, 'Top'];
    
    viewNames = fieldnames(views);
    
    for i = 1:4
        subplot(2, 2, i);
        
        plotSkeleton(xyzPoints, connections, colors, frameIdx, validPoints);
        
        xlim(bounds.x); ylim(bounds.y); zlim(bounds.z);
        
        viewData = views.(viewNames{i});
        view(viewData(1), viewData(2));
        
        title(sprintf('%s - Frame %d', viewData{3}, frameIdx), ...
              'Color', 'white', 'FontSize', 12, 'FontWeight', 'bold');
        
        xlabel('X (mm)', 'Color', 'white', 'FontSize', 10);
        ylabel('Y (mm)', 'Color', 'white', 'FontSize', 10);
        zlabel('Z (mm)', 'Color', 'white', 'FontSize', 10);
        
        grid on;
        set(gca, 'GridColor', [0.3, 0.3, 0.3], 'GridAlpha', 0.6);
        set(gca, 'Color', 'black', 'XColor', 'white', 'YColor', 'white', 'ZColor', 'white');
    end
    
    sgtitle(sprintf('Multi-Angle 3D Human Motion - Frame %d', frameIdx), ...
            'Color', 'white', 'FontSize', 16, 'FontWeight', 'bold');
end

function createLimbFocusView(xyzPoints, connections, colors, bounds, frameIdx, validPoints, frameCount)
    % Focus on different limbs in sequence
    
    % Cycle through different focus areas
    focusArea = mod(floor(frameCount / 30), 4) + 1; % Change every 30 frames
    
    plotSkeleton(xyzPoints, connections, colors, frameIdx, validPoints);
    
    switch focusArea
        case 1 % Upper body focus
            xlim([bounds.x(1), bounds.x(2)]);
            ylim([bounds.y(1) + (bounds.y(2)-bounds.y(1))*0.3, bounds.y(2)]);
            zlim(bounds.z);
            focusTitle = 'Upper Body Focus';
            
        case 2 % Lower body focus  
            xlim([bounds.x(1), bounds.x(2)]);
            ylim([bounds.y(1), bounds.y(1) + (bounds.y(2)-bounds.y(1))*0.7]);
            zlim(bounds.z);
            focusTitle = 'Lower Body Focus';
            
        case 3 % Left side focus
            xlim([bounds.x(1), bounds.x(1) + (bounds.x(2)-bounds.x(1))*0.7]);
            ylim(bounds.y);
            zlim(bounds.z);
            focusTitle = 'Left Side Focus';
            
        case 4 % Right side focus
            xlim([bounds.x(1) + (bounds.x(2)-bounds.x(1))*0.3, bounds.x(2)]);
            ylim(bounds.y);
            zlim(bounds.z);
            focusTitle = 'Right Side Focus';
    end
    
    view(45, 15);
    setupAxes(focusTitle, frameIdx, [], validPoints);
end

function plotSkeleton(xyzPoints, connections, colors, frameIdx, validPoints)
    % Plot enhanced skeleton with color coding
    
    % Plot joints with glow effect
    scatter3(xyzPoints(validPoints, 1, frameIdx), ...
             xyzPoints(validPoints, 2, frameIdx), ...
             xyzPoints(validPoints, 3, frameIdx), ...
             120, 'filled', 'MarkerFaceColor', colors.joints, ...
             'MarkerEdgeColor', [1, 1, 1], 'LineWidth', 2, ...
             'MarkerFaceAlpha', 0.9);
    
    hold on;
    
    % Draw skeleton connections with color coding
    bodyParts = fieldnames(connections);
    for partIdx = 1:length(bodyParts)
        partName = bodyParts{partIdx};
        partConnections = connections.(partName);
        
        for i = 1:size(partConnections, 1)
            pt1 = partConnections(i, 1);
            pt2 = partConnections(i, 2);
            
            if ~isnan(xyzPoints(pt1, 1, frameIdx)) && ~isnan(xyzPoints(pt2, 1, frameIdx))
                % Main colored line
                plot3([xyzPoints(pt1, 1, frameIdx), xyzPoints(pt2, 1, frameIdx)], ...
                      [xyzPoints(pt1, 2, frameIdx), xyzPoints(pt2, 2, frameIdx)], ...
                      [xyzPoints(pt1, 3, frameIdx), xyzPoints(pt2, 3, frameIdx)], ...
                      'Color', colors.(partName), 'LineWidth', 5);
                
                % White glow effect
                plot3([xyzPoints(pt1, 1, frameIdx), xyzPoints(pt2, 1, frameIdx)], ...
                      [xyzPoints(pt1, 2, frameIdx), xyzPoints(pt2, 2, frameIdx)], ...
                      [xyzPoints(pt1, 3, frameIdx), xyzPoints(pt2, 3, frameIdx)], ...
                      'Color', [1, 1, 1], 'LineWidth', 2);
            end
        end
    end
end

function setupAxes(titleStr, frameIdx, stereoParams, validPoints)
    % Setup axes, labels, and overlays
    
    xlabel('X (mm)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'white');
    ylabel('Y (mm)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'white');
    zlabel('Z (mm)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'white');
    
    title(sprintf('%s - Frame %d', titleStr, frameIdx), ...
          'FontSize', 14, 'FontWeight', 'bold', 'Color', 'white');
    
    grid on;
    set(gca, 'GridColor', [0.3, 0.3, 0.3], 'GridAlpha', 0.6);
    set(gca, 'Color', 'black', 'XColor', 'white', 'YColor', 'white', 'ZColor', 'white');
    
    % Add info overlay
    if ~isempty(stereoParams)
        dim = [0.02, 0.95, 0.3, 0.05];
        annotation('textbox', dim, 'String', ...
                  sprintf('Baseline: %.0f mm | Points: %d/33', ...
                          norm(stereoParams.PoseCamera2.Translation), sum(validPoints)), ...
                  'FitBoxToText', 'on', 'BackgroundColor', [0, 0, 0, 0.7], ...
                  'Color', 'white', 'FontSize', 10, 'FontWeight', 'bold');
    end
end
