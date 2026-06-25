# tracker on mapHumanOnly
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from src.pcd_loader import load_point_cloud
from src.preprocessing import downsample
from src.detection import detect_clusters
from src.tracker import SimpleTracker

data_dir = "mapHumanOnly"
out_dir = "tracking_results"
os.makedirs(out_dir, exist_ok=True)

cfg = {
    'voxel_size': 0.03,
    'dbscan_eps': 0.25,
    'dbscan_min_pts': 30,
    'max_distance': 1.0,
    'max_age_frames': 30
}

# List & sort files
files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pcd')]
files_ts = sorted([(f, parse_timestamp(f)) for f in files],
                  key=lambda x: x[1])

tracker = SimpleTracker(
    max_distance=cfg['max_distance'],
    max_age=cfg['max_age_frames']
)

results = []

for frame_id, (fname, ts) in enumerate(files_ts):

    pcd = load_point_cloud(
        os.path.join(data_dir, fname)
    )

    pcd = downsample(
        pcd,
        cfg['voxel_size']
    )

    clusters = detect_clusters(
        pcd,
        eps=cfg['dbscan_eps'],
        min_points=cfg['dbscan_min_pts']
    )

    detections = [
        {
            'centroid': c['centroid'],
            'n_points': c['n_points']
        }
        for c in clusters
    ]

    outs = tracker.step(
        detections,
        ts
    )

    results.append({
        'frame_id': frame_id,
        'timestamp_ms': int(ts),
        'detections': outs
    })