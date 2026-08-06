clear; clc; close all;
rng(0, 'twister');   % Fix random sampling for repeatable plane fitting and map generation
warning off;
%% ==================== 0. Parameter Settings ====================
plyFile = 'C:\Users\Administrator\Desktop\Thesis C\code\fused _Cloud.ply';   % Point cloud path
robotModelFolder = 'C:\Users\Administrator\Desktop\Thesis C\code\a1';        % Folder containing robot STL files

% Map parameters
robotBaseHeight = 0.3;      % Nominal body height (m)
res = 0.15;                 % Grid resolution (m)
maxSlopeDeg = 50;           % Maximum traversable slope (deg)
maxStepHeight = 0.3;        % Maximum traversable step height (m)
minSupportArea = 0.005;     % Minimum support area (m^2)

% Colour-based obstacle detection parameters
useColorForMapping = true;   % Enable colour-assisted obstacle detection
greenThreshold = -0.02;      % Vegetation index ExG = 2*G - R - B; values above this are treated as grass
flatSlopeDeg = 5;            % Maximum slope for classification as "flat" (manholes are approximately level)
flatStepThreshold = 0.05;    % Maximum step height for classification as "flat" (manhole level with grass)

% Navigation parameters
vAvg = 0.35;                % Average speed (m/s)
Ts = 0.1;                   % Control period (s)
maxSimTime = 120;           % Maximum simulation time (s)

% MPC parameters
p = 10;                     % Prediction horizon (1.0 s)
m = 3;                      % Control horizon (0.3 s)
Q = diag([15, 15, 3, 1]);   % State weights [x, y, theta, v]
R = diag([0.1, 0.1]);       % Control weights [a, omega]
Rd = diag([1.0, 1.0]);      % Control-rate weights
maxAccel = 0.5;             % Maximum acceleration (m/s^2)
maxDecel = 0.5;             % Maximum deceleration
maxSteer = 0.5;             % Maximum steering angular rate (rad/s)

% Perception parameters
camRange = 5.0;             % Perception range (m)
camFOV = deg2rad(100);      % Half field of view (total FOV: 200 deg)

%% ==================== 1. Point Cloud Loading and Ground Levelling ====================
fprintf('Loading point cloud...\n');
ptCloud = pcread(plyFile);

% Extract colour information, if available
hasColor = isprop(ptCloud, 'Color') && ~isempty(ptCloud.Color);
if hasColor
    rawColors = double(ptCloud.Color) / 255;   % [0,1]
else
    rawColors = repmat([0.6, 0.6, 0.6], ptCloud.Count, 1);
end

% --- Colour-based ground-plane fitting using green grass ---
fprintf('Fitting the ground plane using green grass...\n');
if hasColor
    ExG = 2*rawColors(:,2) - rawColors(:,1) - rawColors(:,3);
    greenMask = ExG > (mean(ExG) + 0.2*std(ExG));
else
    greenMask = true(ptCloud.Count, 1);
end

groundPts = double(ptCloud.Location(greenMask, :));
if size(groundPts,1) < 100
    warning('Too few green points; fitting the plane using the full point cloud.');
    groundPts = double(ptCloud.Location);
end

maxDistance = 0.05;
[~, inliers] = pcfitplane(pointCloud(groundPts), maxDistance);
planePts = groundPts(inliers, :);
[coeff, ~, latent] = pca(planePts);
n = coeff(:,3)';
if n * [0;0;1] < 0
    n = -n;
end
fprintf('Estimated ground-plane normal: [%.4f, %.4f, %.4f]\n', n);

z_axis = [0, 0, 1];
v = cross(n, z_axis);
if norm(v) < 1e-6
    R_ground = eye(3);
else
    v = v / norm(v);
    angle = acos(dot(n, z_axis));
    K = [0, -v(3), v(2);
         v(3), 0, -v(1);
        -v(2), v(1), 0];
    R_ground = eye(3) + sin(angle)*K + (1-cos(angle))*K*K;
end

T_ground = [R_ground, [0;0;0]; [0,0,0,1]];
ptCloud = pctransform(ptCloud, affinetform3d(T_ground));
fprintf('Ground-plane levelling completed.\n');

T_flip = [1  0  0  0;
          0 -1  0  0;
          0  0 -1  0;
          0  0  0  1];
ptCloud = pctransform(ptCloud, affinetform3d(T_flip));

xyz = double(ptCloud.Location);
x = xyz(:,1);
y = xyz(:,2);
z = xyz(:,3) - min(xyz(:,3));

if hasColor
    rawColors = double(ptCloud.Color) / 255;
else
    rawColors = repmat([0.6, 0.6, 0.6], size(xyz,1), 1);
end

fprintf('Final point-cloud bounds: X[%.2f,%.2f] Y[%.2f,%.2f] Z[%.2f,%.2f]\n', ...
    min(x), max(x), min(y), max(y), min(z), max(z));

%% ==================== 2. Traversability Grid-Map Generation (with Colour) ====================
fprintf('Building traversability map with colour assistance...\n');

dsCloud = pcdownsample(pointCloud([x,y,z], 'Color', rawColors), 'gridAverage', 0.05);
xyz_ds = double(dsCloud.Location);
colors_ds = double(dsCloud.Color);

xLimits = [min(x), max(x)];
yLimits = [min(y), max(y)];

[occGrid, elevMapSmooth, slopeMap, stepHeightMap] = buildOccupancyMap_withColor( ...
    xyz_ds, colors_ds, res, maxSlopeDeg, maxStepHeight, minSupportArea, ...
    useColorForMapping, greenThreshold, flatSlopeDeg, flatStepThreshold, ...
    xLimits, yLimits);

fprintf('Map size: %d x %d, traversable ratio: %.1f%%\n', ...
    size(occGrid,1), size(occGrid,2), sum(~occGrid(:))/numel(occGrid)*100);

occGridInflated = imdilate(occGrid, strel('disk', ceil(0.3/res)));

perceptionCloudDS = pcdownsample(pointCloud([x,y,z], 'Color', rawColors), ...
    'gridAverage', 0.1);

perceptionPoints = double(perceptionCloudDS.Location);
perceptionColors = double(perceptionCloudDS.Color);

%% ==================== Figure 5: Terrain-Mapping Results ====================
figTerrain = figure( ...
    'Name', 'Figure 5: Terrain Mapping and Traversability', ...
    'Color', 'w', ...
    'Position', [100, 100, 1200, 850]);

tlTerrain = tiledlayout(figTerrain, 2, 2, ...
    'Padding', 'compact', ...
    'TileSpacing', 'compact');

% (a) Ground-levelled coloured point cloud
axT1 = nexttile(tlTerrain, 1);
pcshow(dsCloud, 'Parent', axT1, 'BackgroundColor', 'w');
view(axT1, 42, 25);
xlabel(axT1, 'X (m)');
ylabel(axT1, 'Y (m)');
zlabel(axT1, 'Z (m)');
title(axT1, '(a) Ground-levelled point cloud');
grid(axT1, 'on');

% (b) Local slope map
axT2 = nexttile(tlTerrain, 2);
imagesc(axT2, xLimits, yLimits, slopeMap);
set(axT2, 'YDir', 'normal');
axis(axT2, 'equal', 'tight');
xlabel(axT2, 'X (m)');
ylabel(axT2, 'Y (m)');
title(axT2, '(b) Local slope map (limit: 50 deg)');
colormap(axT2, turbo);
clim(axT2, [0, maxSlopeDeg]);
cbSlope = colorbar(axT2);
cbSlope.Label.String = 'Slope (deg)';

% (c) Local step-height map
axT3 = nexttile(tlTerrain, 3);
imagesc(axT3, xLimits, yLimits, stepHeightMap);
set(axT3, 'YDir', 'normal');
axis(axT3, 'equal', 'tight');
xlabel(axT3, 'X (m)');
ylabel(axT3, 'Y (m)');
title(axT3, '(c) Local step-height map (limit: 0.30 m)');
colormap(axT3, parula);
clim(axT3, [0, maxStepHeight]);
cbStep = colorbar(axT3);
cbStep.Label.String = 'Step height (m)';

% (d) Final grid-based traversability map
axT4 = nexttile(tlTerrain, 4);
traversabilityDisplay = double(~occGrid);
imagesc(axT4, xLimits, yLimits, traversabilityDisplay);
set(axT4, 'YDir', 'normal');
axis(axT4, 'equal', 'tight');
xlabel(axT4, 'X (m)');
ylabel(axT4, 'Y (m)');
title(axT4, '(d) Traversability map for A* planning');

% Orange = non-traversable; blue = traversable
colormap(axT4, [0.90, 0.40, 0.10;
                0.20, 0.55, 0.85]);
cbTrav = colorbar(axT4, 'Ticks', [0, 1], ...
    'TickLabels', {'Blocked', 'Traversable'});
cbTrav.Label.String = 'Planning state';

set(findall(figTerrain, '-property', 'FontName'), 'FontName', 'Arial');
set(findall(figTerrain, '-property', 'FontSize'), 'FontSize', 11);

exportgraphics(figTerrain, 'Figure5_Terrain_Mapping.tiff', ...
    'Resolution', 600);

%% ==================== 3. Interactive Start and Goal Selection ====================
figSelect = figure('Name','Select Start and Goal');

imagesc(xLimits, yLimits, ~occGridInflated);
colormap(gray);
axis equal tight;
set(gca, 'YDir', 'normal');

xlabel('X (m)');
ylabel('Y (m)');
title(['White = traversable, black = obstacle', newline, ...
       'Click the start (green), then the goal (red), and press Enter to confirm']);

hold on;

[x1, y1] = ginput(1);
plot(x1, y1, 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');

[x2, y2] = ginput(1);
plot(x2, y2, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');

pause(0.5);
close(figSelect);

startPos = [x1, y1];
goalPos = [x2, y2];

startGrid = worldToGrid(startPos, xLimits, yLimits, res);
goalGrid = worldToGrid(goalPos, xLimits, yLimits, res);

if occGridInflated(startGrid(1), startGrid(2)) || ...
        occGridInflated(goalGrid(1), goalGrid(2))
    error('The start or goal lies on an obstacle. Run again and select a white area.');
end

%% ==================== 3B. Inflation-Radius Sensitivity Analysis ====================
% Same start and goal; only the obstacle-inflation radius is changed.
inflationRadii = [0.00; 0.15; 0.30];
nCases = numel(inflationRadii);

pathFound = false(nCases,1);
pathLength = nan(nCases,1);
traversableRatioAfterInflation = nan(nCases,1);
minimumClearance = nan(nCases,1);

% Distance from each grid-cell centre to the nearest original blocked cell.
% This is used to quantify the clearance of each A* route.
distanceToObstacle = bwdist(occGrid) * res;

fprintf('\n===== Inflation-Radius Sensitivity Analysis =====\n');

for i = 1:nCases
    radius = inflationRadii(i);

    if radius == 0
        occGridTest = occGrid;
    else
        inflationCells = ceil(radius / res);
        occGridTest = imdilate(occGrid, strel('disk', inflationCells));
    end

    traversableRatioAfterInflation(i) = ...
        100 * sum(~occGridTest(:)) / numel(occGridTest);

    if occGridTest(startGrid(1), startGrid(2)) || ...
            occGridTest(goalGrid(1), goalGrid(2))
        fprintf('Inflation radius %.2f m: start or goal blocked.\n', radius);
        continue;
    end

    testPathGrid = AStarSearch(~occGridTest, startGrid, goalGrid);

    if isempty(testPathGrid)
        fprintf('Inflation radius %.2f m: no feasible path found.\n', radius);
        continue;
    end

    pathFound(i) = true;

    testPathWorld = gridToWorld(testPathGrid, xLimits, yLimits, res);
    pathLength(i) = sum(sqrt(sum(diff(testPathWorld).^2, 2)));

    linearIdx = sub2ind(size(occGrid), ...
        testPathGrid(:,1), testPathGrid(:,2));
    minimumClearance(i) = min(distanceToObstacle(linearIdx));

    fprintf(['Inflation radius %.2f m: path found, length = %.2f m, ' ...
             'traversable ratio = %.1f%%, minimum clearance = %.2f m\n'], ...
             radius, pathLength(i), ...
             traversableRatioAfterInflation(i), minimumClearance(i));
end

inflationResults = table(inflationRadii, pathFound, pathLength, ...
    traversableRatioAfterInflation, minimumClearance, ...
    'VariableNames', {'InflationRadius_m', 'PathFound', 'PathLength_m', ...
    'TraversableRatioAfterInflation_percent', 'MinimumClearance_m'});

disp(inflationResults);
writetable(inflationResults, 'Table4_Inflation_Sensitivity.csv');

fprintf('Table 4 sensitivity results exported: Table4_Inflation_Sensitivity.csv\n');
%% ==================== 4. A* Global Path Planning ====================
fprintf('Searching for an A* path...\n');

pathGrid = AStarSearch(~occGridInflated, startGrid, goalGrid);

if isempty(pathGrid)
    error('No feasible path found. Adjust the start and goal points.');
end

refPath = gridToWorld(pathGrid, xLimits, yLimits, res);

fprintf('Global path length: %.2f m, number of path points: %d\n', ...
    sum(sqrt(sum(diff(refPath).^2,2))), size(refPath,1));

%% ==================== 5. Smooth Reference-Trajectory Generation ====================
refTraj = generateSmoothTrajectory(refPath, vAvg, Ts);

fprintf('Number of reference-trajectory points: %d\n', size(refTraj,1));

%% ==================== Figure 6: Global Path-Planning Results ====================
figPath = figure( ...
    'Name', 'Figure 6: Safety-Aware Global Path Planning', ...
    'Color', 'w', ...
    'Position', [120, 120, 1200, 520]);

tlPath = tiledlayout(figPath, 1, 2, ...
    'Padding', 'compact', ...
    'TileSpacing', 'compact');

% (a) Inflated planning map with A* path and smooth reference
axP1 = nexttile(tlPath, 1);

planningDisplay = double(~occGridInflated);
imagesc(axP1, xLimits, yLimits, planningDisplay);
set(axP1, 'YDir', 'normal');
axis(axP1, 'equal', 'tight');
hold(axP1, 'on');

colormap(axP1, [0.90, 0.40, 0.10;    % Blocked
                0.20, 0.55, 0.85]);  % Traversable

plot(axP1, refPath(:,1), refPath(:,2), ':k.', ...
    'LineWidth', 1.0, 'MarkerSize', 7);

plot(axP1, refTraj(:,1), refTraj(:,2), '-', ...
    'Color', [0.75, 0.00, 0.60], 'LineWidth', 2.0);

plot(axP1, startPos(1), startPos(2), 'o', ...
    'MarkerSize', 9, ...
    'MarkerFaceColor', [0.10, 0.65, 0.20], ...
    'MarkerEdgeColor', 'k');

plot(axP1, goalPos(1), goalPos(2), 's', ...
    'MarkerSize', 9, ...
    'MarkerFaceColor', [0.85, 0.15, 0.15], ...
    'MarkerEdgeColor', 'k');

xlabel(axP1, 'X (m)');
ylabel(axP1, 'Y (m)');
title(axP1, '(a) Inflated planning map and global route');

cbPlan = colorbar(axP1, 'Ticks', [0, 1], ...
    'TickLabels', {'Blocked', 'Traversable'});
cbPlan.Label.String = 'Planning state';

legend(axP1, {'A* grid path', 'Smoothed reference', 'Start', 'Goal'}, ...
    'Location', 'southoutside', ...
    'NumColumns', 2, ...
    'FontSize', 9);

% (b) 3-D view of the planned route
axP2 = nexttile(tlPath, 2);

pcshow(dsCloud, 'Parent', axP2, 'BackgroundColor', 'w');
hold(axP2, 'on');

zAStar = interpElev(elevMapSmooth, xLimits, yLimits, res, refPath);
zReference = interpElev(elevMapSmooth, xLimits, yLimits, res, refTraj(:,1:2));

hAStar3D = plot3(axP2, refPath(:,1), refPath(:,2), zAStar + 0.05, ':k.', ...
    'LineWidth', 1.0, 'MarkerSize', 7);

hReference3D = plot3(axP2, refTraj(:,1), refTraj(:,2), zReference + 0.08, '-', ...
    'Color', [0.75, 0.00, 0.60], 'LineWidth', 2.0);

hStart3D = plot3(axP2, startPos(1), startPos(2), zAStar(1) + 0.10, 'o', ...
    'MarkerSize', 9, ...
    'MarkerFaceColor', [0.10, 0.65, 0.20], ...
    'MarkerEdgeColor', 'k');

hGoal3D = plot3(axP2, goalPos(1), goalPos(2), zAStar(end) + 0.10, 's', ...
    'MarkerSize', 9, ...
    'MarkerFaceColor', [0.85, 0.15, 0.15], ...
    'MarkerEdgeColor', 'k');

view(axP2, 42, 25);
grid(axP2, 'on');
xlabel(axP2, 'X (m)');
ylabel(axP2, 'Y (m)');
zlabel(axP2, 'Z (m)');
title(axP2, '(b) Three-dimensional view of planned route');

legend(axP2, [hAStar3D, hReference3D, hStart3D, hGoal3D], ...
    {'A* grid path', 'Smoothed reference', 'Start', 'Goal'}, ...
    'Location', 'southoutside', ...
    'NumColumns', 2, ...
    'FontSize', 9);

set(findall(figPath, '-property', 'FontName'), 'FontName', 'Arial');
set(findall(figPath, '-property', 'FontSize'), 'FontSize', 11);

exportgraphics(figPath, 'Figure6_Global_Path_Planning.tiff', ...
    'Resolution', 600);
%% ==================== 6. Robot-State Initialisation ====================
theta0 = atan2(refPath(2,2)-refPath(1,2), ...
               refPath(2,1)-refPath(1,1));

robotPose = [refPath(1,:), theta0, 0];
trajLog = robotPose;
u_prev = [0; 0];

% Logs for MPC evaluation
uLog = zeros(0, 2);          % [acceleration, yaw rate]
refIdxLog = zeros(0, 1);     % Nearest reference index at each step
%% ==================== 7. Load Full Quadruped Robot Model ====================
fprintf('Loading full quadruped robot model...\n');

if exist(robotModelFolder, 'dir')
    [robotPartList, robotPartFaces, robotPartColors] = ...
        loadAssembledRobot(robotModelFolder);
    showRobotModel = true;
else
    showRobotModel = false;
end

%% ==================== 8. Visualisation Window ====================
fig = figure( ...
    'Name','Quadruped Autonomous Navigation + First-Person View', ...
    'Position',[100,80,1250,760], ...
    'Color','w');

t = tiledlayout(fig, 2, 3, ...
    'Padding','compact', ...
    'TileSpacing','compact');

% Subplot 1: 2-D map
ax1 = nexttile(t, 1);

imagesc(xLimits, yLimits, ~occGrid);
colormap(gray);
axis equal tight;

set(gca, 'YDir', 'normal', 'Color', 'w');
hold on;

plot(refPath(:,1), refPath(:,2), 'b--', 'LineWidth', 1.5);

hRob = plot(robotPose(1), robotPose(2), ...
    'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b');

hTraj = plot(robotPose(1), robotPose(2), ...
    'r-', 'LineWidth', 1.5);

hRef = plot(refTraj(1,1), refTraj(1,2), ...
    'g.', 'MarkerSize', 8);

hPerception2D = scatter(ax1, [], [], 8, 'filled', ...
    'MarkerFaceAlpha', 0.3, 'CData', []);

title('Global Path and Perceived Point Cloud');
xlabel('X (m)');
ylabel('Y (m)');

legend(ax1, {'Global path','Robot','Trajectory', 'Reference','Local cloud'}, ...
       'Location','southoutside',  'NumColumns',3, 'FontSize',8);

% Subplot 2: speed
ax2 = nexttile(t, 2);
set(gca,'Color','w');

hVel = animatedline('Color', 'b', 'LineWidth', 1.5);

title('Speed Profile');
xlabel('Time (s)');
ylabel('Speed (m/s)');
grid on;

% Subplot 3: elevation
ax3 = nexttile(t, 3);
set(gca,'Color','w');

hElev = animatedline('Color', 'g', 'LineWidth', 1.5);

title('Elevation Change');
xlabel('Time (s)');
ylabel('Elevation (m)');
grid on;

% Subplot 4: 3-D view
ax4 = nexttile(t, 4);

pcshow(dsCloud, 'Parent', ax4, 'BackgroundColor', 'w');
set(ax4, 'Color', 'w');
hold(ax4, 'on');

hTraj3D = plot3(trajLog(:,1), trajLog(:,2), ...
    robotBaseHeight*ones(1,1), 'r-', 'LineWidth', 2);

hPerception3D = scatter3(ax4, [], [], [], 6, 'filled', ...
    'MarkerFaceAlpha', 0.5, 'CData', []);

title('3-D View');
view(3);
axis equal;

xlabel('X');
ylabel('Y');
zlabel('Z');

if showRobotModel
    hRobotPatches = drawAssembledRobot(ax4, robotPartList, ...
        robotPartFaces, robotPartColors, robotPose, robotBaseHeight);
else
    hRobotMarker = plot3(robotPose(1), robotPose(2), ...
        robotBaseHeight, 'bo', 'MarkerSize', 8, ...
        'MarkerFaceColor', 'b');
end

camva(ax4, 8);

% Subplot 5: first-person view
ax5 = nexttile(t, 5);

set(ax5, 'Color', 'w');
hold(ax5, 'on');

title('First-Person View (Ahead of Robot)');
xlabel('X');
ylabel('Y');
zlabel('Z');

view(ax5, 3);
axis(ax5, 'normal');
grid(ax5, 'on');

% Empty placeholder handle; redrawn during each loop iteration
hFirstPerson = gobjects(0);

% Subplot 6: MPC predicted-trajectory display
ax6 = nexttile(t, 6);

set(ax6, 'Color', 'w');
hold(ax6, 'on');

title('MPC Predicted Trajectory');
xlabel('X (m)');
ylabel('Y (m)');

axis equal;
grid on;

% Draw the global path as the background
hGlobalPath6 = plot(ax6, refPath(:,1), refPath(:,2), ...
    'b--', 'LineWidth', 1);

% Initialise predicted-trajectory handles with NaN placeholders
hPredTraj = plot(ax6, nan, nan, 'm-o', ...
    'LineWidth', 1.5, 'MarkerSize', 5);

hPredStart = plot(ax6, nan, nan, 'mo', ...
    'MarkerSize', 8, 'LineWidth', 2);

% Show the current robot position
hRob6 = plot(ax6, robotPose(1), robotPose(2), ...
    'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'b');

% Show the current reference point
hRef6 = plot(ax6, nan, nan, ...
    'g.', 'MarkerSize', 10);

legend(ax6, [hGlobalPath6, hPredTraj, hRob6], ...
    {'Global path', 'MPC prediction', 'Robot'}, ...
    'Location','southoutside', ...
    'NumColumns',3, ...
    'FontSize',8);

set([ax1, ax2, ax3, ax4, ax5, ax6], 'FontSize', 10);

%% ==================== 9. Main Simulation Loop ====================
fprintf('Simulation started. Robot is moving...\n');

tic;

for k = 1:(maxSimTime / Ts)

    % 9.1 Find the nearest reference point
    dists = vecnorm(refTraj(:,1:2) - robotPose(1:2), 2, 2);
    [~, idx] = min(dists);

    idxEnd = min(idx + p, size(refTraj, 1));
    refBlock = refTraj(idx:idxEnd, :);

    if size(refBlock,1) < p+1
        numPad = p+1 - size(refBlock,1);
        padBlock = repmat(refBlock(end,:), numPad, 1);
        padBlock(:,4) = linspace(refBlock(end,4), 0, numPad);
        refBlock = [refBlock; padBlock];
    end

    % 9.2 Solve the MPC problem
    [u, predStates] = MPC_Solve(robotPose, refBlock, u_prev, ...
        Ts, p, m, Q, R, Rd, maxAccel, maxDecel, maxSteer);

    uLog(end+1, :) = u';
refIdxLog(end+1, 1) = idx;

    % 9.3 State update
    xk = robotPose';

    k1 = RobotModel(xk, u);
    k2 = RobotModel(xk + Ts/2*k1, u);
    k3 = RobotModel(xk + Ts/2*k2, u);
    k4 = RobotModel(xk + Ts*k3, u);

    xk_new = xk + Ts/6 * (k1 + 2*k2 + 2*k3 + k4);

    robotPose = xk_new';
    robotPose(3) = wrapToPi(robotPose(3));

    % Position clipping
    margin = res;

    robotPose(1) = max(xLimits(1)+margin, ...
        min(xLimits(2)-margin, robotPose(1)));

    robotPose(2) = max(yLimits(1)+margin, ...
        min(yLimits(2)-margin, robotPose(2)));

    % Elevation lookup
    elev = interpElev(elevMapSmooth, xLimits, yLimits, ...
        res, robotPose(1:2));

    % Log trajectory
    trajLog(end+1,:) = robotPose;

    % Extract local perceived point cloud
    dx = perceptionPoints(:,1) - robotPose(1);
    dy = perceptionPoints(:,2) - robotPose(2);

    dist = sqrt(dx.^2 + dy.^2);
    angle = wrapToPi(atan2(dy, dx) - robotPose(3));

    inRange = dist < camRange & abs(angle) < camFOV/2;

    localPts = perceptionPoints(inRange, :);
    localColors = perceptionColors(inRange, :);

    % Update 2-D perception points
    if ~isempty(localPts)
        set(hPerception2D, ...
            'XData', localPts(:,1), ...
            'YData', localPts(:,2), ...
            'CData', localColors);
    else
        set(hPerception2D, ...
            'XData', [], ...
            'YData', [], ...
            'CData', []);
    end

    % Update 3-D perception points
    if ~isempty(localPts)
        set(hPerception3D, ...
            'XData', localPts(:,1), ...
            'YData', localPts(:,2), ...
            'ZData', localPts(:,3), ...
            'CData', localColors);
    else
        set(hPerception3D, ...
            'XData', [], ...
            'YData', [], ...
            'ZData', [], ...
            'CData', []);
    end

    % Update first-person point cloud
    delete(hFirstPerson);

    if ~isempty(localPts)
        try
            hFirstPerson = pcshow( ...
                pointCloud(localPts, 'Color', localColors), ...
                'Parent', ax5, ...
                'MarkerSize', 30);
        catch
            hFirstPerson = scatter3(ax5, ...
                localPts(:,1), localPts(:,2), localPts(:,3), ...
                6, 'g', 'filled', 'MarkerFaceAlpha', 0.8);
        end
    else
        hFirstPerson = gobjects(0);
    end

    % Update the first-person camera
    eyeHeight = robotBaseHeight + elev + 0.15;

    campos(ax5, [robotPose(1), robotPose(2), eyeHeight]);

    camtarget(ax5, ...
        [robotPose(1) + camRange*cos(robotPose(3)), ...
         robotPose(2) + camRange*sin(robotPose(3)), ...
         eyeHeight]);

    camva(ax5, 50);

    % Update other graphics
    set(hRob, 'XData', robotPose(1), 'YData', robotPose(2));

    set(hTraj, ...
        'XData', trajLog(:,1), ...
        'YData', trajLog(:,2));

    set(hRef, ...
        'XData', refBlock(1,1), ...
        'YData', refBlock(1,2));

    addpoints(hVel, k*Ts, robotPose(4));
    addpoints(hElev, k*Ts, elev);

    % Update 3-D trajectory height
    allElev = interpElev(elevMapSmooth, xLimits, yLimits, ...
        res, trajLog(:,1:2));

    set(hTraj3D, ...
        'XData', trajLog(:,1), ...
        'YData', trajLog(:,2), ...
        'ZData', allElev + robotBaseHeight);

    % Update robot model
    if showRobotModel
        updateAssembledRobot(hRobotPatches, robotPartList, ...
            robotPose, elev + robotBaseHeight);
    else
        set(hRobotMarker, ...
            'XData', robotPose(1), ...
            'YData', robotPose(2), ...
            'ZData', elev + robotBaseHeight);
    end

    camtarget(ax4, [robotPose(1), robotPose(2), robotBaseHeight]);

    % Update MPC predicted trajectory
    if exist('predStates', 'var') && ~isempty(predStates)
        set(hPredTraj, ...
            'XData', predStates(:,1), ...
            'YData', predStates(:,2));

        set(hPredStart, ...
            'XData', predStates(1,1), ...
            'YData', predStates(1,2));
    end

    set(hRob6, ...
        'XData', robotPose(1), ...
        'YData', robotPose(2));

    set(hRef6, ...
        'XData', refBlock(1,1), ...
        'YData', refBlock(1,2));

    % Adjust subplot 6 limits to follow the robot
    xlim(ax6, robotPose(1) + [-3, 3]);
    ylim(ax6, robotPose(2) + [-3, 3]);

    drawnow limitrate;

    if norm(robotPose(1:2) - goalPos) < 0.4
        fprintf('Success: goal reached in %.2f s.\n', k*Ts);
        break;
    end
end

if k == floor(maxSimTime/Ts)
    fprintf('Simulation timed out before reaching the goal.\n');
end

fprintf('Actual computation time: %.2f s\n', toc);

%% ==================== Figure 7: MPC Tracking Results ====================
executedLog = trajLog(2:end, :);
referenceLog = refTraj(refIdxLog, :);

tLog = (1:size(executedLog, 1))' * Ts;

positionError = hypot( ...
    executedLog(:,1) - referenceLog(:,1), ...
    executedLog(:,2) - referenceLog(:,2));

rmseError = sqrt(mean(positionError.^2));
maxError = max(positionError);
completionTime = tLog(end);

fprintf('\n===== MPC Tracking Metrics =====\n');
fprintf('RMSE position error: %.4f m\n', rmseError);
fprintf('Maximum position error: %.4f m\n', maxError);
fprintf('Trajectory completion time: %.2f s\n', completionTime);

trackingMetrics = table(rmseError, maxError, completionTime, ...
    'VariableNames', {'RMSE_Position_Error_m', ...
                      'Maximum_Position_Error_m', ...
                      'Completion_Time_s'});

writetable(trackingMetrics, 'Table3_MPC_Tracking_Metrics.csv');

figMPC = figure( ...
    'Name', 'Figure 7: MPC Tracking Results', ...
    'Color', 'w', ...
    'Position', [120, 120, 1200, 850]);

tlMPC = tiledlayout(figMPC, 2, 2, ...
    'Padding', 'compact', ...
    'TileSpacing', 'compact');

% (a) Reference trajectory and executed trajectory
axM1 = nexttile(tlMPC, 1);
plot(axM1, referenceLog(:,1), referenceLog(:,2), '-', ...
    'Color', [0.75, 0.00, 0.60], 'LineWidth', 2.0);
hold(axM1, 'on');

plot(axM1, executedLog(:,1), executedLog(:,2), '--', ...
    'Color', [0.10, 0.10, 0.10], 'LineWidth', 1.4);

plot(axM1, executedLog(1,1), executedLog(1,2), 'o', ...
    'MarkerSize', 8, ...
    'MarkerFaceColor', [0.10, 0.65, 0.20], ...
    'MarkerEdgeColor', 'k');

plot(axM1, executedLog(end,1), executedLog(end,2), 's', ...
    'MarkerSize', 8, ...
    'MarkerFaceColor', [0.85, 0.15, 0.15], ...
    'MarkerEdgeColor', 'k');

axis(axM1, 'equal');
grid(axM1, 'on');
xlabel(axM1, 'X (m)');
ylabel(axM1, 'Y (m)');
title(axM1, '(a) Reference and executed trajectories');
legend(axM1, {'Reference trajectory', 'Executed trajectory', ...
    'Start', 'Final position'}, ...
    'Location', 'best');

% (b) Position-tracking error
axM2 = nexttile(tlMPC, 2);
plot(axM2, tLog, positionError, ...
    'Color', [0.85, 0.20, 0.20], 'LineWidth', 1.5);
grid(axM2, 'on');
xlabel(axM2, 'Time (s)');
ylabel(axM2, 'Position error (m)');
title(axM2, '(b) Position-tracking error');

% (c) Speed tracking
axM3 = nexttile(tlMPC, 3);
plot(axM3, tLog, referenceLog(:,4), '-', ...
    'Color', [0.75, 0.00, 0.60], 'LineWidth', 2.0);
hold(axM3, 'on');

plot(axM3, tLog, executedLog(:,4), '--', ...
    'Color', [0.10, 0.10, 0.10], 'LineWidth', 1.4);

grid(axM3, 'on');
xlabel(axM3, 'Time (s)');
ylabel(axM3, 'Speed (m/s)');
title(axM3, '(c) Speed tracking');
legend(axM3, {'Reference speed', 'Executed speed'}, ...
    'Location', 'best');

% (d) Constrained MPC control inputs
axM4 = nexttile(tlMPC, 4);

yyaxis(axM4, 'left');
plot(axM4, tLog, uLog(:,1), ...
    'Color', [0.10, 0.35, 0.80], 'LineWidth', 1.4);
hold(axM4, 'on');
yline(axM4, maxAccel, '--k', 'HandleVisibility', 'off');
yline(axM4, -maxDecel, '--k', 'HandleVisibility', 'off');
ylabel(axM4, 'Acceleration (m/s^2)');

yyaxis(axM4, 'right');
plot(axM4, tLog, uLog(:,2), ...
    'Color', [0.85, 0.20, 0.20], 'LineWidth', 1.4);
hold(axM4, 'on');
yline(axM4, maxSteer, '--k', 'HandleVisibility', 'off');
yline(axM4, -maxSteer, '--k', 'HandleVisibility', 'off');
ylabel(axM4, 'Yaw rate (rad/s)');

grid(axM4, 'on');
xlabel(axM4, 'Time (s)');
title(axM4, '(d) MPC control inputs (limits: +/-0.50)');

set(findall(figMPC, '-property', 'FontName'), 'FontName', 'Arial');
set(findall(figMPC, '-property', 'FontSize'), 'FontSize', 11);

exportgraphics(figMPC, 'Figure7_MPC_Tracking_Results.tiff', ...
    'Resolution', 600);

disp('Figure 7 and Table 3 metrics were exported successfully.');

%% ==================== Core Function Definitions ====================

% Map construction with colour-based vegetation and manhole detection
function [occGrid, elevSmooth, slope, stepHeight] = buildOccupancyMap_withColor( ...
    xyz, colors, res, maxSlopeDeg, maxStep, minArea, useColor, ...
    greenThresh, flatSlopeDeg, flatStepThresh, xLims, yLims)

    x = xyz(:,1);
    y = xyz(:,2);
    z = xyz(:,3);

    xEdges = xLims(1):res:xLims(2);
    yEdges = yLims(1):res:yLims(2);

    [Xq, Yq] = meshgrid( ...
        xEdges(1:end-1)+res/2, ...
        yEdges(1:end-1)+res/2);

    % Geometric processing
    F = scatteredInterpolant(x, y, z, 'linear', 'none');
    elevMap = F(Xq, Yq);

    nanMask = isnan(elevMap);
    if any(nanMask)
        elevMap(nanMask) = 100;
    end

    elevMap = filloutliers(elevMap, 'linear', 'mean');
    elevSmooth = imgaussfilt(elevMap, 1.0);

    [dzdx, dzdy] = gradient(elevSmooth, res);

    slope = atan(sqrt(dzdx.^2 + dzdy.^2)) * 180/pi;

    stepX = abs(diff(elevSmooth, 1, 2));
    stepX = [stepX, stepX(:,end)];

    stepY = abs(diff(elevSmooth, 1, 1));
    stepY = [stepY; stepY(end,:)];

    stepHeight = max(stepX, stepY);

    [N,~] = histcounts2(y, x, yEdges, xEdges);
    density = N * res^2;

    passable = (slope <= maxSlopeDeg) & ...
               (stepHeight <= maxStep) & ...
               (density >= minArea);

    % Colour-assisted obstacle detection
    if useColor && ~isempty(colors)

        maxRow = length(yEdges)-1;
        maxCol = length(xEdges)-1;

        % Calculate grid coordinates and clamp them to map bounds
        col = round((x - xLims(1)) / res) + 1;
        row = round((y - yLims(1)) / res) + 1;

        col = max(1, min(col, maxCol));
        row = max(1, min(row, maxRow));

        ExG = 2*colors(:,2) - colors(:,1) - colors(:,3);

        sumExG = accumarray([row, col], ExG, [maxRow, maxCol]);
        count = accumarray([row, col], 1, [maxRow, maxCol]);

        count(count == 0) = 1;
        avgExG = sumExG ./ count;

        isGreen = avgExG > greenThresh;
        isFlat = (slope < flatSlopeDeg) & ...
                 (stepHeight < flatStepThresh);

        % Non-green, flat regions are treated as manholes or roads
        isManhole = ~isGreen & isFlat;

        passable = passable & ~isManhole;
    end

    % Morphological post-processing
    passable = imclose(passable, strel('disk', 3));
    passable = bwareaopen(passable, 10);

    occGrid = ~passable;
end

% Coordinate conversion with boundary protection
function grid = worldToGrid(worldPos, xLim, yLim, res)

    col = round((worldPos(1) - xLim(1)) / res) + 1;
    row = round((worldPos(2) - yLim(1)) / res) + 1;

    maxRow = ceil((yLim(2) - yLim(1)) / res);
    maxCol = ceil((xLim(2) - xLim(1)) / res);

    row = max(1, min(row, maxRow));
    col = max(1, min(col, maxCol));

    grid = [row, col];
end

function world = gridToWorld(gridIdx, xLim, yLim, res)

    world = [xLim(1) + (gridIdx(:,2)-0.5)*res, ...
             yLim(1) + (gridIdx(:,1)-0.5)*res];
end

% A* search
function path = AStarSearch(passableGrid, start, goal)

    [rows, cols] = size(passableGrid);

    start = round(start);
    goal = round(goal);

    if ~passableGrid(start(1), start(2)) || ...
            ~passableGrid(goal(1), goal(2))
        path = [];
        return;
    end

    openList = [start(1), start(2), 0, norm(start-goal)];
    closed = false(rows, cols);
    parent = zeros(rows, cols, 2);

    while ~isempty(openList)

        [~, idx] = min(openList(:,4));

        current = openList(idx,:);
        openList(idx,:) = [];

        if current(1) == goal(1) && current(2) == goal(2)

            path = [];

            while true
                path = [current(1:2); path];

                if current(1) == start(1) && current(2) == start(2)
                    break;
                end

                parentNode = squeeze(parent(current(1), current(2), :))';
                current = [parentNode(1), parentNode(2), 0, 0];
            end

            return;
        end

        closed(current(1), current(2)) = true;

        neighbors = [ ...
            current(1)-1, current(2);
            current(1)+1, current(2);
            current(1), current(2)-1;
            current(1), current(2)+1;
            current(1)-1, current(2)-1;
            current(1)-1, current(2)+1;
            current(1)+1, current(2)-1;
            current(1)+1, current(2)+1];

        for i = 1:size(neighbors,1)

            r = neighbors(i,1);
            c = neighbors(i,2);

            if r < 1 || r > rows || c < 1 || c > cols || ...
                    ~passableGrid(r,c) || closed(r,c)
                continue;
            end

            g = current(3) + norm([r,c] - current(1:2));
            h = norm([r,c] - goal);
            f = g + h;

            existing = find(openList(:,1)==r & openList(:,2)==c);

            if isempty(existing)
                openList(end+1,:) = [r, c, g, f];
                parent(r,c,:) = current(1:2);

            elseif g < openList(existing,3)
                openList(existing,3) = g;
                openList(existing,4) = f;
                parent(r,c,:) = current(1:2);
            end
        end
    end

    path = [];
end

% Smooth reference trajectory
function refTraj = generateSmoothTrajectory(pathXY, vAvg, Ts)

    diffPath = diff(pathXY);
    dists = sqrt(sum(diffPath.^2, 2));

    keep = [true; dists > 1e-6];
    pathXY = pathXY(keep, :);

    if size(pathXY,1) < 2
        refTraj = [pathXY, 0, 0];
        return;
    end

    s = [0; cumsum(sqrt(sum(diff(pathXY).^2, 2)))];
    totalLen = s(end);
    totalTime = totalLen / vAvg;

    t = (0:Ts:totalTime)';

    x_smooth = spline(s, pathXY(:,1), t * vAvg);
    y_smooth = spline(s, pathXY(:,2), t * vAvg);

    theta = atan2(gradient(y_smooth, Ts), gradient(x_smooth, Ts));

    v_ref = vAvg * ones(size(theta));

    nAcc = min(ceil(1.0 / Ts), floor(length(v_ref)/4));

    if nAcc > 1
        v_ref(1:nAcc) = linspace(0, vAvg, nAcc);
        v_ref(end-nAcc+1:end) = linspace(vAvg, 0, nAcc);
    end

    refTraj = [x_smooth, y_smooth, theta, v_ref];
end

% Robot kinematics
function dx = RobotModel(x, u)

    dx = [x(4)*cos(x(3));
          x(4)*sin(x(3));
          u(2);
          u(1)];
end

% MPC solver
function [u_opt, predStates] = MPC_Solve( ...
    x0, ref, up, Ts, p, m, Q, R, Rd, amax, dmax, wmax)

    n_u = 2;

    u0 = repmat(up, m, 1);

    lb = [-dmax*ones(m,1);
          -wmax*ones(m,1)];

    ub = [amax*ones(m,1);
          wmax*ones(m,1)];

    options = optimoptions('fmincon', ...
        'Display', 'off', ...
        'Algorithm', 'interior-point', ...
        'MaxIterations', 80);

    fun = @(u) MPC_Cost(u, x0, ref, Ts, p, m, Q, R, Rd, up);

    u_all = fmincon(fun, u0, [], [], [], [], lb, ub, [], options);

    u_opt = u_all(1:n_u);

    % Calculate predicted states for visualisation
    u_seq = reshape(u_all, n_u, m);
    u_seq = [u_seq, repmat(u_seq(:,end), 1, p-m)];

    x = x0';

    predStates = zeros(p+1, 4);
    predStates(1,:) = x';

    for k = 1:p
        x = x + Ts * RobotModel(x, u_seq(:,k));
        x(3) = wrapToPi(x(3));
        predStates(k+1,:) = x';
    end
end

function J = MPC_Cost(u, x0, ref, Ts, p, m, Q, R, Rd, up)

    x = x0';

    u = reshape(u, 2, m);
    u = [u, repmat(u(:,end), 1, p-m)];

    J = 0;

    for k = 1:p
        x = x + Ts * RobotModel(x, u(:,k));
        x(3) = wrapToPi(x(3));

        e = [x(1)-ref(k,1);
             x(2)-ref(k,2);
             wrapToPi(x(3)-ref(k,3));
             x(4)-ref(k,4)];

        J = J + e' * Q * e;
    end

    for k = 1:m
        J = J + u(:,k)' * R * u(:,k);
    end

    J = J + (u(:,1)-up)' * Rd * (u(:,1)-up);

    for k = 2:m
        J = J + (u(:,k)-u(:,k-1))' * Rd * (u(:,k)-u(:,k-1));
    end
end

% Elevation lookup
function elev = interpElev(elevMap, xLim, yLim, res, pos)

    if size(pos,1) > 1

        elev = zeros(size(pos,1),1);

        for i = 1:size(pos,1)
            g = worldToGrid(pos(i,:), xLim, yLim, res);
            elev(i) = elevMap(g(1), g(2));
        end

    else
        g = worldToGrid(pos, xLim, yLim, res);
        elev = elevMap(g(1), g(2));
    end
end

% Full quadruped robot-model loading
function [partList, partFaces, partColors] = loadAssembledRobot(folder)

    function [pts, faces] = readSTL(filename)

        tri = stlread(filename);

        if isa(tri, 'triangulation')
            pts = tri.Points;
            faces = tri.ConnectivityList;
        else
            pts = tri.vertices;
            faces = tri.faces;
        end
    end

    [trunk_pts, trunk_faces] = readSTL(fullfile(folder, 'trunk.STL'));
    [hip_pts, hip_faces] = readSTL(fullfile(folder, 'hip.STL'));

    [thighL_pts, thighL_faces] = ...
        readSTL(fullfile(folder, 'thigh_left.STL'));

    [thighR_pts, thighR_faces] = ...
        readSTL(fullfile(folder, 'thigh_right.STL'));

    [calf_pts, calf_faces] = readSTL(fullfile(folder, 'calf.STL'));

    calfR_pts = calf_pts * diag([1, -1, 1]);

    mounts = [ ...
         0.18,  0.08, 0;
         0.18, -0.08, 0;
        -0.05,  0.08, 0;
        -0.05, -0.08, 0];

    thighTopZ = max(thighL_pts(:,3));
    calfTopZ = max(calf_pts(:,3));

    thighL_pts = thighL_pts - [0,0,thighTopZ];
    thighR_pts = thighR_pts - [0,0,thighTopZ];

    calf_pts = calf_pts - [0,0,calfTopZ];
    calfR_pts = calfR_pts - [0,0,calfTopZ];

    partList = {};
    partFaces = {};
    partColors = {};

    partList{1} = trunk_pts;
    partFaces{1} = trunk_faces;
    partColors{1} = [0.8 0.2 0.2];

    hipCol = [0.3 0.3 0.3];
    thighCol = [0.5 0.5 0.5];
    calfCol = [0.7 0.7 0.7];

    for leg = 1:4

        mnt = mounts(leg,:);

        partList{end+1} = hip_pts + mnt;
        partFaces{end+1} = hip_faces;
        partColors{end+1} = hipCol;

        if leg == 1 || leg == 3
            thighPts = thighL_pts;
            thighF = thighL_faces;

            calfPts = calf_pts;
            calfF = calf_faces;
        else
            thighPts = thighR_pts;
            thighF = thighR_faces;

            calfPts = calfR_pts;
            calfF = calf_faces;
        end

        partList{end+1} = thighPts + mnt;
        partFaces{end+1} = thighF;
        partColors{end+1} = thighCol;

        kneeOffset = [0, 0, min(thighPts(:,3))];

        partList{end+1} = calfPts + mnt + kneeOffset;
        partFaces{end+1} = calfF;
        partColors{end+1} = calfCol;
    end
end

function h = drawAssembledRobot( ...
    ax, partList, partFaces, partColors, pose, baseHeight)

    R = [cos(pose(3)), -sin(pose(3)), 0;
         sin(pose(3)),  cos(pose(3)), 0;
         0,             0,            1];

    allPts = [];

    for i = 1:length(partList)
        allPts = [allPts; (R * partList{i}')'];
    end

    minZ = min(allPts(:,3));
    z_offset = baseHeight - minZ;

    h = cell(size(partList));

    for i = 1:length(partList)

        pts = (R * partList{i}')' + ...
            [pose(1), pose(2), z_offset];

        h{i} = patch(ax, ...
            'Vertices', pts, ...
            'Faces', partFaces{i}, ...
            'FaceColor', partColors{i}, ...
            'EdgeColor', 'none', ...
            'FaceAlpha', 0.9);
    end
end

function updateAssembledRobot(h, partList, pose, baseHeight)

    R = [cos(pose(3)), -sin(pose(3)), 0;
         sin(pose(3)),  cos(pose(3)), 0;
         0,             0,            1];

    allPts = [];

    for i = 1:length(partList)
        allPts = [allPts; (R * partList{i}')'];
    end

    minZ = min(allPts(:,3));
    z_offset = baseHeight - minZ;

    for i = 1:length(partList)

        pts = (R * partList{i}')' + ...
            [pose(1), pose(2), z_offset];

        set(h{i}, 'Vertices', pts);
    end
end