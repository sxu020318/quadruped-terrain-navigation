# Quadruped Terrain-Aware Navigation

This repository contains the MATLAB implementation used for the thesis
“Control and Planning of a Quadruped Robot Operating on Uneven Terrain”.

## Files

- `main1.m`: Ground alignment, terrain-traversability mapping, A* planning,
  obstacle-inflation sensitivity analysis and planar MPC tracking.
- `main2.m`: RGB-D reconstruction visualisation.
- `a1/`: Unitree A1 STL components used for robot visualisation.

## Required data

The fused RGB-D point cloud (`fused_Cloud.ply`) and output img and depth are not included because of its
file size. It can be accessed at:

https://unsw-my.sharepoint.com/:f:/g/personal/z5530899_ad_unsw_edu_au/IgCzHniuR7DTT7eKm2TeSitcASiZh3UErsUgy9bUc1fexkg?e=ckjh9C


Before running `main1.m`, update `plyFile` and `robotModelFolder` near the
start of the script to match your local folders.
