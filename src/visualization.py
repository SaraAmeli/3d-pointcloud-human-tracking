# Cell B — Plot speed over time (Figure 1)
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

plt.rcParams.update({'font.size': 12})

times = timestamps
speed_series = speeds.copy()

# optional smoothing (set sigma to 0 to disable)
sigma = 1.0  # adjust: 0 = no smoothing, 1~2 gentle smoothing
masked = ~np.isnan(speed_series)
if masked.sum() > 2 and sigma > 0:
    # apply smoothing only where there are values; create a full-length smoothed array by interpolating NaNs
    # linear interp to fill NaNs for smoothing
    s = speed_series.copy()
    idx = np.where(masked)[0]
    s = np.interp(np.arange(len(s)), idx, speed_series[idx])
    s_smooth = gaussian_filter1d(s, sigma=sigma)
else:
    s_smooth = speed_series

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(times, speed_series, marker='o', linestyle='-', label='Instantaneous speed', alpha=0.6)
ax.plot(times, s_smooth, linestyle='-', linewidth=2, label='Smoothed speed', color='tab:orange')

# show confidence shading if available
if not np.isnan(confidences).all():
    conf = np.nan_to_num(confidences, nan=0.0)
    # scale confidence to alpha
    alpha = np.clip(conf, 0.05, 1.0)
    ax.fill_between(times, 0, np.nan_to_num(s_smooth, nan=0.0), alpha=0.06, facecolor='tab:orange')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Speed (m/s)')
ax.set_title('Figure 1 — Speed over time (modal ID {})'.format(modal_id))
ax.grid(True)
ax.legend()
plt.tight_layout()

out_path = res_dir / "velocity.png"
plt.savefig(out_path, dpi=200)
print("Saved Figure 1 to", out_path)
plt.show()


# Cell C — Plot trajectory (Figure 2)
import matplotlib.pyplot as plt
from matplotlib import cm

xy = centroids[:, :2]
valid = ~np.isnan(xy[:,0])
if valid.sum() == 0:
    print("No valid centroid positions to plot.")
else:
    xs = xy[valid,0]
    ys = xy[valid,1]
    ts_valid = timestamps[valid]
    # color by time (normalized)
    norm_t = (ts_valid - ts_valid.min()) / max(1e-6, (ts_valid.max()-ts_valid.min()))
    cmap = cm.get_cmap('viridis')

    fig, ax = plt.subplots(figsize=(6,6))
    sc = ax.scatter(xs, ys, c=norm_t, cmap=cmap, s=30, edgecolors='k', linewidths=0.2)
    # connect as a line (trajectory)
    ax.plot(xs, ys, linewidth=1, alpha=0.7)
    # mark start and end
    ax.scatter(xs[0], ys[0], s=80, color='green', marker='D', label='start')
    ax.scatter(xs[-1], ys[-1], s=80, color='red', marker='X', label='end')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Figure 2 — Top-down trajectory (modal ID {})'.format(modal_id))
    ax.axis('equal')
    ax.grid(True)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized time (start → end)')
    ax.legend()
    plt.tight_layout()

    out_path = res_dir / "trajectory.png"
    plt.savefig(out_path, dpi=200)
    print("Saved Figure 2 to", out_path)
    plt.show()


# produce ID timeline and stats plots from tracking_results/detections.jsonl
import json, numpy as np, matplotlib.pyplot as plt, os
from pathlib import Path
from itertools import groupby
from matplotlib import cm

res_dir = Path("tracking_results")
det_file = res_dir / "detections.jsonl"
assert det_file.exists(), f"File not found: {det_file}"

# --- Load frames ---
frames = [json.loads(line) for line in open(det_file, "r")]
N = len(frames)
timestamps = np.array([fr['timestamp_ms'] for fr in frames], dtype=float) / 1000.0

# --- Count IDs and pick modal ID ---
id_counts = {}
for fr in frames:
    for d in fr['detections']:
        id_counts[d['human_id']] = id_counts.get(d['human_id'], 0) + 1

if not id_counts:
    raise RuntimeError("No detections found in detections.jsonl")

modal_id = max(id_counts, key=id_counts.get)
all_ids = sorted(id_counts.keys())

# --- Build chosen-per-frame array and other stats ---
chosen = np.full((N,), np.nan)
num_detections = np.zeros(N, dtype=int)
present_modal = np.zeros(N, dtype=bool)

for i, fr in enumerate(frames):
    dets = fr['detections']
    num_detections[i] = len(dets)
    if len(dets) == 0:
        continue
    ids = [d['human_id'] for d in dets]
    if modal_id in ids:
        chosen[i] = modal_id
        present_modal[i] = True
    else:
        chosen[i] = ids[0]  # fallback pick first detection

# fragments: contiguous segments where modal is present
fragments = 0
prev = False
for v in present_modal:
    if v and not prev:
        fragments += 1
    prev = v

# approx id switches: count changes in chosen id between consecutive non-nan frames
id_switches = 0
last = None
for val in chosen:
    if np.isnan(val):
        continue
    if last is None:
        last = val
        continue
    if val != last:
        id_switches += 1
        last = val

frames_with_modal = int(present_modal.sum())
distinct_ids = len(all_ids)
frames_total = N

stats = {
    'frames_total': frames_total,
    'frames_with_modal': frames_with_modal,
    'fragments_modal': fragments,
    'approx_id_switches': id_switches,
    'distinct_ids': distinct_ids
}

print("Computed stats:")
for k,v in stats.items():
    print(f"  {k}: {v}")

# --- Figure A: ID timeline ---
# --- Figure A: ID timeline (ONLY) ---
plt.rcParams.update({'font.size': 12})
unique_ids = all_ids
id_to_idx = {uid: idx for idx, uid in enumerate(unique_ids)}
chosen_idx = np.full_like(chosen, -1.0)
for i, v in enumerate(chosen):
    if not np.isnan(v):
        chosen_idx[i] = id_to_idx[int(v)]

n_colors = max(8, len(unique_ids))
cmap = plt.get_cmap('tab20', n_colors)

fig, ax = plt.subplots(figsize=(12, 2.8))
x = np.arange(N)

# draw colored squares for each frame
for idx in range(len(unique_ids)):
    mask = chosen_idx == idx
    if mask.any():
        ax.scatter(x[mask], np.zeros_like(x[mask]) + 0.5, c=[cmap(idx)], marker='s', s=64, label=f"ID {unique_ids[idx]}")
# draw no-detection frames in light gray
mask_nan = chosen_idx == -1
if mask_nan.any():
    ax.scatter(x[mask_nan], np.zeros_like(x[mask_nan]) + 0.5, c='lightgray', marker='s', s=64, label='No detection')

# overlay modal presence step plot
ax2 = ax.twinx()
ax2.plot(x, present_modal.astype(int), drawstyle='steps-mid', color='tab:orange', linewidth=2, label='Modal present (1/0)')
ax2.set_ylim(-0.1, 1.1)
ax2.set_yticks([])

ax.set_yticks([])
ax.set_xlim(-1, N)
ax.set_xlabel('Frame index')
ax.set_title(f'Figure A — ID timeline (modal ID = {modal_id})')

# Build combined legend (limit to first 12 ids to avoid huge legend)
handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
combined_handles = handles1[:12] + handles2
combined_labels = labels1[:12] + labels2
ax.legend(combined_handles, combined_labels, bbox_to_anchor=(1.01, 1), loc='upper left', ncol=1, fontsize=9)

plt.tight_layout()
outA = res_dir / "id_timeline.png"
plt.savefig(outA, dpi=200, bbox_inches='tight')
plt.show()
plt.close(fig)
print("Saved:", outA)


# src/visualization.py

import matplotlib.pyplot as plt
import numpy as np


def plot_summary_metrics(stats, save_path):
    metrics = [
        stats['fragments_modal'],
        stats['frames_with_modal'],
        stats['distinct_ids'],
        stats['approx_id_switches']
    ]

    labels = [
        'Fragments',
        'Frames with modal',
        'Distinct IDs',
        'Approx ID switches'
    ]

    plt.figure(figsize=(7,5))

    bars = plt.bar(labels, metrics)

    plt.title("Tracking Summary Metrics")
    plt.ylabel("Count")

    for bar, val in zip(bars, metrics):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            val,
            str(val),
            ha='center'
        )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_detections_per_frame(num_detections, save_path):

    plt.figure(figsize=(10,4))

    plt.plot(
        np.arange(len(num_detections)),
        num_detections
    )

    plt.xlabel("Frame Index")
    plt.ylabel("Number of Detections")
    plt.title("Detections per Frame")

    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()