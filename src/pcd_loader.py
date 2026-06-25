# Helper functions
import open3d as o3d
import numpy as np
import re

def parse_timestamp(filename):
    """Extract timestamp from filename like 'occupied_1234ms.pcd'"""
    m = re.search(r'(\d+)ms', filename)
    return int(m.group(1)) if m else 0

# Replace load_point_cloud with robust binary PCD reader + wrapper
import struct, os, re, numpy as np, textwrap
import open3d as o3d

def read_pcd_binary_custom(path, verbose=False):
    with open(path, 'rb') as f:
        header_lines = []
        while True:
            line = f.readline().decode('utf-8', errors='replace')
            if not line:
                break
            header_lines.append(line.strip())
            if line.strip().lower().startswith('data '):
                break
        header = '\n'.join(header_lines)
        def get_header_val(key):
            for L in header_lines:
                if L.upper().startswith(key):
                    return L.split(maxsplit=1)[1].strip()
            return None
        fields = get_header_val('FIELDS'); sizes = get_header_val('SIZE'); types = get_header_val('TYPE')
        counts = get_header_val('COUNT'); points = get_header_val('POINTS'); data = get_header_val('DATA')
        fields = fields.split(); sizes = list(map(int, sizes.split())); types = types.split()
        counts = list(map(int, counts.split())) if counts else [1]*len(fields)
        n_points = int(points) if points else None
        data = data.lower() if data else 'binary'
        bytes_per_point = sum(s*c for s,c in zip(sizes, counts))
        bin_data = f.read()
        if n_points is not None:
            expected = n_points * bytes_per_point
            if len(bin_data) < expected:
                n_points = len(bin_data) // bytes_per_point
        else:
            n_points = len(bin_data) // bytes_per_point
        buf = np.frombuffer(bin_data, dtype=np.uint8)
        total_expected = n_points * bytes_per_point
        buf = buf[:total_expected]
        pts_raw = buf.reshape((n_points, bytes_per_point))
        offset = 0
        x = np.zeros(n_points, dtype=np.float32)
        y = np.zeros(n_points, dtype=np.float32)
        z = np.zeros(n_points, dtype=np.float32)
        rgb = None
        for fi, (field, size, typ, count) in enumerate(zip(fields, sizes, types, counts)):
            field_bytes = size * count
            sl = pts_raw[:, offset:offset+field_bytes].copy()
            offset += field_bytes
            if field in ('x','y','z'):
                vals = sl.view(np.float32).reshape(-1)
                if field == 'x': x = vals.copy()
                if field == 'y': y = vals.copy()
                if field == 'z': z = vals.copy()
            elif field.lower() in ('rgb','rgba','r','g','b'):
                if size == 4:
                    try:
                        uvals = sl.view(np.uint32).reshape(-1)
                        rgb = uvals.copy()
                    except:
                        try:
                            fvals = sl.view(np.float32).reshape(-1)
                            rgb = fvals.view(np.uint32).copy()
                        except:
                            rgb = None
                else:
                    rgb = None
            else:
                continue
        points_xyz = np.vstack([x,y,z]).T
        colors = None
        if rgb is not None:
            u = np.asarray(rgb, dtype=np.uint32)
            r = ((u >> 16) & 0xFF).astype(np.uint8)
            g = ((u >> 8) & 0xFF).astype(np.uint8)
            b = (u & 0xFF).astype(np.uint8)
            colors = np.vstack([r,g,b]).T.astype(np.float32) / 255.0
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_xyz)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd, {'n_points': n_points, 'has_colors': colors is not None, 'fields': fields}

def load_point_cloud(path):
    pcd, info = read_pcd_binary_custom(path)
    return pcd


def downsample(pcd, voxel_size=0.03):
    return pcd.voxel_down_sample(voxel_size) if voxel_size>0 else pcd

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
