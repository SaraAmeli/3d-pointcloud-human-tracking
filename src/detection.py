import numpy as np
from sklearn.cluster import DBSCAN

def detect_clusters(pcd, eps=0.25, min_points=30):
    pts = np.asarray(pcd.points)
    if pts.size==0:
        return []
    clustering = DBSCAN(eps=eps, min_samples=min_points).fit(pts)
    labels = clustering.labels_
    clusters = []
    for lab in set(labels):
        if lab==-1:
            continue
        inds = np.where(labels==lab)[0]
        centroid = pts[inds].mean(axis=0)
        clusters.append({'indices': inds, 'centroid': centroid.tolist(), 'n_points': int(len(inds))})
    clusters.sort(key=lambda x: -x['n_points'])
    return clusters