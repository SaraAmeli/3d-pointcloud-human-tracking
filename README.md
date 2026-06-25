# Human Tracking in 3D Point Clouds

## Overview

This project implements a complete pipeline for detecting,
tracking, and analyzing human motion in sequential 3D point cloud
(PCD) data.

The pipeline consists of:

1. Binary PCD parsing
2. Voxel downsampling
3. DBSCAN clustering
4. Kalman Filter tracking
5. Hungarian data association
6. Motion analysis and visualization

---

## Pipeline

PCD Files
    ↓
Downsampling
    ↓
DBSCAN Detection
    ↓
Centroid Extraction
    ↓
Kalman Filter Prediction
    ↓
Hungarian Assignment
    ↓
Trajectory & Velocity Estimation

---

## Installation

pip install -r requirements.txt

---

## Usage

python scripts/run_tracking.py

python scripts/generate_plots.py

---

## Results

### Trajectory

![trajectory](results/trajectory.png)

### Velocity

![velocity](results/velocity.png)

### ID Timeline

![timeline](results/id_timeline.png)

---

## Tracking Model

State Vector:

x = [x,y,z,vx,vy,vz]^T

Motion Model:

x_k = F x_{k-1}

Measurement:

z_k = H x_k

Kalman Filter:
Prediction → Update → Association

---

## Future Work

- Multi-human tracking
- HDBSCAN clustering
- UKF / Particle Filter
- Sensor fusion
- ROS integration
