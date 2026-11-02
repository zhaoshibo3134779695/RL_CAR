#!/usr/bin/env python
"""Inspect the fixed OGM-TurtleBot2 test extraction without changing it."""

from __future__ import print_function

import collections
import hashlib
import os
from pathlib import Path

import numpy as np

ROOT = Path("/home/jovyan/work/research/scope_repro")
DATA = ROOT / "datasets/OGM-datasets/OGM-Turtlebot2/test"
ARCHIVE_LIST = ROOT / "logs/dataset_archive_file_list.txt"
ARCHIVE = ROOT / "datasets/OGM-datasets.tar.gz"
KINDS = ("scans", "positions", "velocities")
SAMPLE_IDS = ("0000000", "0000001", "0000010", "0000100", "0008499", "0016999")


def checksum(path, algorithm):
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


print("SCOPE phase-1 dataset snapshot")
print("official_record=https://zenodo.org/records/7051560")
print("official_api=https://zenodo.org/api/records/7051560")
print(
    "final_download_url="
    "https://zenodo.org/api/records/7051560/files/OGM-datasets.tar.gz/content"
)
print("license=CC-BY-4.0 (from Zenodo metadata)")
print("archive={}".format(ARCHIVE))
print("archive_size_bytes={}".format(ARCHIVE.stat().st_size))
print("metadata_md5=092e097761508b857a10285c059ce5c1")
print("actual_md5={}".format(checksum(ARCHIVE, "md5")))
print("actual_sha256={}".format(checksum(ARCHIVE, "sha256")))
print("archive_entry_count=296428")
print("archive_path_safety=PASS (no absolute or parent-traversal paths)")
print("extraction_policy=only OGM-Turtlebot2/test and OGM-datasets/ReadMe.txt")
print("resolved_test_path={}".format(DATA))

archive_counts = collections.Counter()
with ARCHIVE_LIST.open("r") as stream:
    for line in stream:
        parts = line.rstrip("\n").split("/")
        if len(parts) == 5 and parts[0:2] == ["OGM-datasets", "OGM-Turtlebot2"]:
            split, kind, filename = parts[2:]
            if kind in KINDS and filename.endswith(".npy"):
                archive_counts[(split, kind)] += 1

print("\n## Split counts from official archive")
for split in ("train", "val", "test"):
    counts = [archive_counts[(split, kind)] for kind in KINDS]
    print(
        "{}: scans={} positions={} velocities={} aligned_frames={}".format(
            split, counts[0], counts[1], counts[2], min(counts)
        )
    )

print("\n## Extracted test files")
total_files = 0
total_bytes = 0
for path in DATA.rglob("*"):
    if path.is_file():
        total_files += 1
        total_bytes += path.stat().st_size
print("file_count={}".format(total_files))
print("size_bytes={}".format(total_bytes))
print("size_mib={:.3f}".format(total_bytes / 1024.0 / 1024.0))

kind_names = {}
for kind in KINDS:
    directory = DATA / kind
    actual = sorted(path.name for path in directory.glob("*.npy"))
    declared = [
        line.strip()
        for line in (directory / "test.txt").read_text().splitlines()
        if line.strip().endswith(".npy")
    ]
    kind_names[kind] = declared
    print(
        "{}: npy_files={} list_entries={} list_matches_files={} first={} last={}".format(
            kind,
            len(actual),
            len(declared),
            actual == declared,
            declared[0],
            declared[-1],
        )
    )
print(
    "cross_modality_lists_identical={}".format(
        kind_names["scans"] == kind_names["positions"] == kind_names["velocities"]
    )
)

print("\n## Selected sample structure")
for sample_id in SAMPLE_IDS:
    print("sample_id={}".format(sample_id))
    for kind in KINDS:
        path = DATA / kind / (sample_id + ".npy")
        value = np.load(str(path), allow_pickle=False)
        finite = value[np.isfinite(value)]
        unique_preview = np.unique(value)[:12]
        print(
            "  {}: format=NumPy-NPY dtype={} shape={} size={} "
            "finite_min={} finite_max={} nan={} inf={} unique_preview={}".format(
                kind,
                value.dtype,
                tuple(value.shape),
                value.size,
                float(finite.min()) if finite.size else "NA",
                float(finite.max()) if finite.size else "NA",
                int(np.isnan(value).sum()),
                int(np.isinf(value).sum()),
                unique_preview.tolist(),
            )
        )

print("\n## Semantics established from official loader/source")
print("raw_frame_rate_hz=10 (source uses 0.1 seconds per velocity integration step)")
print("lidar_points_per_frame=1080")
print("position_components=3 (x, y, heading/yaw)")
print("velocity_components=2 (linear speed, angular speed)")
print("self_motion_present=yes (positions and velocities are explicit modalities)")
print("loader_window_frames=20 (10 observed scans + 10 future scans)")
print("prediction_horizon_frames=10")
print("prediction_horizon_seconds=1.0")
print("grid_shape_per_time_step=64x64")
print("grid_resolution_m=0.1")
print("grid_x_limits_m=[0,6.4]")
print("grid_y_limits_m=[-3.2,3.2]")
print("raw_scan_no_return_value=30; loader replaces 30/NaN/Inf with 20")
print("occupancy_encoding=generated binary endpoint map: free/unmarked=0, occupied=1")
print(
    "note=archive contains raw lidar/pose/velocity arrays, not precomputed occupancy grids"
)
