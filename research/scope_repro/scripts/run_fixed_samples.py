#!/usr/bin/env python
"""Numerically instrumented SO-SCOPE inference for fixed consecutive windows.

This is a reproduction harness, not an official metric implementation. It uses
the official model, checkpoint, occupancy mapping code, inputs and 10-step
autoregressive loop while saving numeric arrays that the official visualization
demo does not save.
"""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
import math
import os
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

IMG_SIZE = 64
SEQ_LEN = 10
POINTS = 1080
MAP_X_LIMIT = [0, 6.4]
MAP_Y_LIMIT = [-3.2, 3.2]
RESOLUTION = 0.1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--start", type=int, default=0)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values, percent):
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * percent / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def read_names(data, kind):
    listing = data / kind / "test.txt"
    return [
        line.strip()
        for line in listing.read_text().splitlines()
        if line.strip().endswith(".npy")
    ]


def load_window(data, names_by_kind, start):
    scans = np.zeros((SEQ_LEN * 2, POINTS), dtype=np.float64)
    positions = np.zeros((SEQ_LEN * 2, 3), dtype=np.float64)
    velocities = np.zeros((SEQ_LEN * 2, 2), dtype=np.float64)
    for offset in range(SEQ_LEN * 2):
        index = start + offset
        scans[offset] = np.load(
            str(data / "scans" / names_by_kind["scans"][index]), allow_pickle=False
        )
        positions[offset] = np.load(
            str(data / "positions" / names_by_kind["positions"][index]),
            allow_pickle=False,
        )
        velocities[offset] = np.load(
            str(data / "velocities" / names_by_kind["velocities"][index]),
            allow_pickle=False,
        )
    scans[np.isnan(scans) | np.isinf(scans) | (scans == 30)] = 20.0
    positions[np.isnan(positions) | np.isinf(positions)] = 0.0
    velocities[np.isnan(velocities) | np.isinf(velocities)] = 0.0
    return (
        torch.from_numpy(scans.astype(np.float32)).unsqueeze(0),
        torch.from_numpy(positions.astype(np.float32)).unsqueeze(0),
        torch.from_numpy(velocities.astype(np.float32)).unsqueeze(0),
    )


def infer_window(model, local_map_class, device, scans, positions, velocities):
    scans = scans.to(device)
    positions = positions.to(device)
    velocities = velocities.to(device)
    batch_size = scans.size(0)

    mask_grid_map = local_map_class(
        X_lim=MAP_X_LIMIT,
        Y_lim=MAP_Y_LIMIT,
        resolution=RESOLUTION,
        p=0.5,
        size=[batch_size, SEQ_LEN],
        device=device,
    )
    zeros = torch.zeros(batch_size, SEQ_LEN, device=device)
    distances = scans[:, SEQ_LEN:]
    angles = torch.linspace(
        -(135 * np.pi / 180),
        135 * np.pi / 180,
        distances.shape[-1],
        device=device,
    )
    distances_x, distances_y = mask_grid_map.lidar_scan_xy(
        distances, angles, zeros, zeros, zeros
    )
    mask_binary_maps = mask_grid_map.discretize(distances_x, distances_y)

    prediction_maps = torch.zeros(
        SEQ_LEN, 1, IMG_SIZE, IMG_SIZE, device=device
    )
    for prediction_step in range(SEQ_LEN):
        input_grid_map = local_map_class(
            X_lim=MAP_X_LIMIT,
            Y_lim=MAP_Y_LIMIT,
            resolution=RESOLUTION,
            p=0.5,
            size=[batch_size, SEQ_LEN],
            device=device,
        )
        observed_position = positions[:, SEQ_LEN - 1]
        latest_velocity = velocities[:, SEQ_LEN - 1]
        horizon = prediction_step + 1
        predicted_origin = input_grid_map.origin_pose_prediction(
            latest_velocity, observed_position, horizon, [0, 0, 0]
        )
        x_odom, y_odom, theta_odom = input_grid_map.robot_coordinate_transform(
            positions[:, :SEQ_LEN], predicted_origin
        )
        distances = scans[:, :SEQ_LEN]
        angles = torch.linspace(
            -(135 * np.pi / 180),
            135 * np.pi / 180,
            distances.shape[-1],
            device=device,
        )
        distances_x, distances_y = input_grid_map.lidar_scan_xy(
            distances, angles, x_odom, y_odom, theta_odom
        )
        inputs_samples = input_grid_map.discretize(
            distances_x, distances_y
        ).unsqueeze(2)
        for _ in range(horizon):
            prediction = model(inputs_samples)
            prediction = prediction.reshape(-1, 1, 1, IMG_SIZE, IMG_SIZE)
            inputs_samples = torch.cat([inputs_samples[:, 1:], prediction], dim=1)
        prediction_maps[prediction_step, 0] = prediction.squeeze()

    return prediction_maps[:, 0], mask_binary_maps[0]


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    arrays_dir = args.output / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.repo / "scripts"))
    from local_occ_grid_map import LocalMap
    from model import so_scope

    names_by_kind = {kind: read_names(args.data, kind) for kind in ("scans", "positions", "velocities")}
    if not (
        names_by_kind["scans"]
        == names_by_kind["positions"]
        == names_by_kind["velocities"]
    ):
        raise RuntimeError("modality file lists are not aligned")
    end_exclusive = args.start + args.count + (SEQ_LEN * 2) - 1
    if end_exclusive > len(names_by_kind["scans"]):
        raise RuntimeError("not enough consecutive frames for requested windows")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.repo / "model/so_scope_model.pth"
    model = so_scope(input_channels=1, latent_dim=512, output_channels=1).to(device)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    load_result = model.load_state_dict(checkpoint["model"])
    model.eval()

    torch.manual_seed(1337)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1337)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    print("device={}".format(device))
    print("torch={}".format(torch.__version__))
    print("torch_cuda={}".format(torch.version.cuda))
    print("checkpoint={}".format(checkpoint_path))
    print("checkpoint_sha256={}".format(sha256(checkpoint_path)))
    print("load_state_dict={}".format(load_result))
    print("window_count={}".format(args.count))

    records = []
    run_started = time.perf_counter()
    with torch.inference_mode():
        for sample_number, start in enumerate(
            range(args.start, args.start + args.count)
        ):
            wall_started = time.perf_counter()
            scans, positions, velocities = load_window(
                args.data, names_by_kind, start
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            compute_started = time.perf_counter()
            prediction, target = infer_window(
                model, LocalMap, device, scans, positions, velocities
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            compute_seconds = time.perf_counter() - compute_started
            prediction_np = prediction.detach().cpu().numpy().astype(np.float32)
            target_np = target.detach().cpu().numpy().astype(np.float32)
            diagnostic_mse = float(np.mean((prediction_np - target_np) ** 2))
            sample_id = "{:07d}".format(start)
            output_path = arrays_dir / ("sample_{}.npz".format(sample_id))
            np.savez_compressed(
                str(output_path),
                prediction=prediction_np,
                target_mask=target_np,
                start_frame=np.asarray(start, dtype=np.int64),
            )
            wall_seconds = time.perf_counter() - wall_started
            record = {
                "sample_id": sample_id,
                "start_frame": start,
                "prediction_shape": list(prediction_np.shape),
                "target_shape": list(target_np.shape),
                "prediction_min": float(prediction_np.min()),
                "prediction_max": float(prediction_np.max()),
                "prediction_nan": int(np.isnan(prediction_np).sum()),
                "prediction_inf": int(np.isinf(prediction_np).sum()),
                "target_values": np.unique(target_np).tolist(),
                "diagnostic_unweighted_mse_not_official": diagnostic_mse,
                "compute_seconds": compute_seconds,
                "wall_seconds_including_io_and_save": wall_seconds,
                "output_sha256": sha256(output_path),
            }
            records.append(record)
            print(
                "sample={} compute_seconds={:.6f} wall_seconds={:.6f} "
                "diagnostic_mse={:.9f}".format(
                    sample_id, compute_seconds, wall_seconds, diagnostic_mse
                ),
                flush=True,
            )

    total_seconds = time.perf_counter() - run_started
    compute_values = [record["compute_seconds"] for record in records]
    wall_values = [record["wall_seconds_including_io_and_save"] for record in records]
    prediction_nan = sum(record["prediction_nan"] for record in records)
    prediction_inf = sum(record["prediction_inf"] for record in records)
    summary = {
        "status": "PASS",
        "scope": "instrumented fixed-window compatibility inference",
        "official_metric_implementation": False,
        "repo_branch": "compat/modern-pytorch",
        "repo_base_commit": "dc9792a58514d5b1f648b4b3fa03392f674aac5a",
        "device": str(device),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "sample_count": len(records),
        "sample_ids": [record["sample_id"] for record in records],
        "total_seconds": total_seconds,
        "throughput_fps": len(records) / total_seconds,
        "compute_mean_seconds": statistics.mean(compute_values),
        "compute_p50_seconds": percentile(compute_values, 50),
        "compute_p95_seconds": percentile(compute_values, 95),
        "wall_mean_seconds": statistics.mean(wall_values),
        "wall_p50_seconds": percentile(wall_values, 50),
        "wall_p95_seconds": percentile(wall_values, 95),
        "prediction_nan_total": prediction_nan,
        "prediction_inf_total": prediction_inf,
        "gpu_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "gpu_peak_reserved_bytes": (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        ),
        "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "diagnostic_unweighted_mse_mean_not_official": statistics.mean(
            [
                record["diagnostic_unweighted_mse_not_official"]
                for record in records
            ]
        ),
        "wmse": None,
        "ssim": None,
        "ospa": None,
        "metric_note": (
            "The official so-scope repository implements none of WMSE, SSIM or "
            "OSPA in its demo. The unweighted MSE here is a diagnostic only."
        ),
    }

    (args.output / "sample_ids.txt").write_text(
        "\n".join(summary["sample_ids"]) + "\n"
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    with (args.output / "per_sample.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "sample_id",
                "start_frame",
                "compute_seconds",
                "wall_seconds_including_io_and_save",
                "diagnostic_unweighted_mse_not_official",
                "prediction_min",
                "prediction_max",
                "prediction_nan",
                "prediction_inf",
                "output_sha256",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(records)
    print("summary={}".format(args.output / "summary.json"))
    print("overall_status=PASS")


if __name__ == "__main__":
    main()
