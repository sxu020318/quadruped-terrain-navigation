%% Figure 4: RGB-D reconstruction process
clear; clc; close all;

% Use RGB and depth images from the SAME frame
rgbFile = 'C:\Users\Administrator\Desktop\project\thesis B\RGBD_Reconstruction\RGBD_Reconstruction\output-img\rgb_1289.png';
depthFile = ['C:\Users\Administrator\Desktop\project\thesis B\RGBD_Reconstruction\RGBD_Reconstruction\output-depth\depth_1289.png'];

% Camera intrinsics used in this study
fx = 320;
fy = 320;
cx = 160;
cy = 120;

% Read RGB and depth images
rgb = imread(rgbFile);
depthRaw = imread(depthFile);

if ndims(depthRaw) == 3
    depthRaw = depthRaw(:,:,1);
end

% Convert depth from mm to m
depth = double(depthRaw) / 1000;
depth(depthRaw == 0) = NaN;

% Ensure RGB and depth images have the same resolution
if size(rgb,1) ~= size(depth,1) || size(rgb,2) ~= size(depth,2)
    rgb = imresize(rgb, [size(depth,1), size(depth,2)]);
end

% Back-project valid depth pixels to a coloured point cloud
[h, w] = size(depth);
[u, v] = meshgrid(1:w, 1:h);

X = (u - cx) .* depth / fx;
Y = (v - cy) .* depth / fy;
Z = depth;

valid = isfinite(Z) & Z > 0.30 & Z < 5.00;

points = [X(valid), Y(valid), Z(valid)];
colours = reshape(rgb, [], 3);
colours = colours(valid(:), :);

% Randomly reduce the number of displayed points
maxPoints = 25000;
if size(points,1) > maxPoints
    idx = randperm(size(points,1), maxPoints);
    points = points(idx,:);
    colours = colours(idx,:);
end

%% Create Figure 4
figRGBD = figure( ...
    'Name', 'Figure 4: RGB-D Point-Cloud Reconstruction', ...
    'Color', 'w', ...
    'Position', [100, 100, 1300, 430]);

tl = tiledlayout(figRGBD, 1, 3, ...
    'Padding', 'compact', ...
    'TileSpacing', 'compact');

% (a) RGB frame
ax1 = nexttile(tl, 1);
imshow(rgb, 'Parent', ax1);
title(ax1, '(a) RGB frame');

% (b) Colourised depth map
ax2 = nexttile(tl, 2);
imagesc(ax2, depth);
axis(ax2, 'image');
axis(ax2, 'off');
set(ax2, 'YDir', 'reverse');
colormap(ax2, turbo);

validDepth = depth(isfinite(depth));
if ~isempty(validDepth)
    clim(ax2, prctile(validDepth, [2, 98]));
end

cbDepth = colorbar(ax2);
cbDepth.Label.String = 'Depth (m)';
title(ax2, '(b) Colourised depth map');

% (c) Coloured local point cloud
ax3 = nexttile(tl, 3);
scatter3(ax3, points(:,1), points(:,2), points(:,3), ...
    4, double(colours)/255, 'filled');

axis(ax3, 'equal');
grid(ax3, 'on');
view(ax3, 42, 22);
xlabel(ax3, 'X (m)');
ylabel(ax3, 'Y (m)');
zlabel(ax3, 'Camera depth, Z_c (m)');
title(ax3, '(c) Coloured local point cloud in camera frame');

set(findall(figRGBD, '-property', 'FontName'), 'FontName', 'Arial');
set(findall(figRGBD, '-property', 'FontSize'), 'FontSize', 11);

exportgraphics(figRGBD, 'Figure4_RGBD_Reconstruction.tiff', ...
    'Resolution', 600);

disp('Figure 4 exported successfully: Figure4_RGBD_Reconstruction.tiff');