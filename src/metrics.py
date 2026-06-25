import numpy as np
from itertools import groupby

def compute_metrics(results):
"""
Compute overall tracking metrics.

Parameters
----------
results : list
    Tracker output loaded from detections.jsonl.

Returns
-------
dict
    Dictionary containing tracking statistics.
"""

frame_count = len(results)

frames_with_detection = sum(
    1 for r in results
    if len(r["detections"]) > 0
)

total_detections = sum(
    len(r["detections"])
    for r in results
)

metrics = {
    "frame_count": frame_count,
    "track_completeness_percent":
        100.0 * frames_with_detection / max(1, frame_count),

    "avg_detections_per_frame":
        total_detections / max(1, frame_count)
}

# Modal-ID consistency estimate
id_counts = {}

for r in results:
    for d in r["detections"]:
        hid = d["human_id"]
        id_counts[hid] = id_counts.get(hid, 0) + 1

if id_counts:
    modal_id = max(id_counts, key=id_counts.get)

    metrics["modal_id"] = int(modal_id)

    metrics["id_consistency_rate_percent"] = (
        100.0 * id_counts[modal_id] /
        max(1, frame_count)
    )
else:
    metrics["modal_id"] = None
    metrics["id_consistency_rate_percent"] = 0.0

return metrics

def compute_id_statistics(results):
"""
Compute ID stability statistics without ground truth.

Returns
-------
dict
    ID tracking statistics.
"""

id_counts = {}

for r in results:
    for d in r["detections"]:
        hid = d["human_id"]
        id_counts[hid] = id_counts.get(hid, 0) + 1

if not id_counts:
    return {
        "modal_id": None,
        "frames_total": len(results),
        "frames_with_modal": 0,
        "fragments": 0,
        "distinct_ids": 0,
        "approx_id_switches": 0
    }

modal_id = max(id_counts, key=id_counts.get)

# Presence sequence of modal ID
modal_present = []

for r in results:
    present = any(
        d["human_id"] == modal_id
        for d in r["detections"]
    )
    modal_present.append(present)

# Count fragments
fragments = sum(
    1
    for key, group in groupby(modal_present)
    if key
)

# Approximate ID switches
chosen_ids = []

for r in results:
    if len(r["detections"]) == 0:
        chosen_ids.append(None)
        continue

    ids = [d["human_id"] for d in r["detections"]]

    if modal_id in ids:
        chosen_ids.append(modal_id)
    else:
        chosen_ids.append(ids[0])

id_switches = 0
previous = None

for cid in chosen_ids:
    if cid is None:
        continue

    if previous is None:
        previous = cid
        continue

    if cid != previous:
        id_switches += 1

    previous = cid

stats = {
    "modal_id": int(modal_id),
    "frames_total": len(results),
    "frames_with_modal": int(sum(modal_present)),
    "fragments": int(fragments),
    "distinct_ids": int(len(id_counts)),
    "approx_id_switches": int(id_switches)
}

return stats


def compute_id_statistics(results):
"""
Compute ID stability statistics without ground truth.

Returns
-------
dict
    ID tracking statistics.
"""

id_counts = {}

for r in results:
    for d in r["detections"]:
        hid = d["human_id"]
        id_counts[hid] = id_counts.get(hid, 0) + 1

if not id_counts:
    return {
        "modal_id": None,
        "frames_total": len(results),
        "frames_with_modal": 0,
        "fragments": 0,
        "distinct_ids": 0,
        "approx_id_switches": 0
    }

modal_id = max(id_counts, key=id_counts.get)

# Presence sequence of modal ID
modal_present = []

for r in results:
    present = any(
        d["human_id"] == modal_id
        for d in r["detections"]
    )
    modal_present.append(present)

# Count fragments
fragments = sum(
    1
    for key, group in groupby(modal_present)
    if key
)

# Approximate ID switches
chosen_ids = []

for r in results:
    if len(r["detections"]) == 0:
        chosen_ids.append(None)
        continue

    ids = [d["human_id"] for d in r["detections"]]

    if modal_id in ids:
        chosen_ids.append(modal_id)
    else:
        chosen_ids.append(ids[0])

id_switches = 0
previous = None

for cid in chosen_ids:
    if cid is None:
        continue

    if previous is None:
        previous = cid
        continue

    if cid != previous:
        id_switches += 1

    previous = cid

stats = {
    "modal_id": int(modal_id),
    "frames_total": len(results),
    "frames_with_modal": int(sum(modal_present)),
    "fragments": int(fragments),
    "distinct_ids": int(len(id_counts)),
    "approx_id_switches": int(id_switches)
}

return stats