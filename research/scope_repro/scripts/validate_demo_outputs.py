#!/usr/bin/env python
"""Validate coverage and readability of the compatibility demo JPEG outputs."""

from __future__ import print_function

import argparse
import json
import re
import sys
from pathlib import Path

from PIL import Image


def indexed_files(root, prefix):
    pattern = re.compile(r"^{}([0-9]+)\.jpg$".format(re.escape(prefix)))
    result = {}
    for path in root.glob("{}*.jpg".format(prefix)):
        match = pattern.match(path.name)
        if match:
            result[int(match.group(1))] = path
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=17000)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    expected_ids = set(range(args.expected_count))
    masks = indexed_files(args.output, "mask")
    predictions = indexed_files(args.output, "pred")
    mask_ids = set(masks)
    prediction_ids = set(predictions)
    failures = []
    if mask_ids != expected_ids:
        failures.append(
            "mask IDs differ: missing={}, extra={}".format(
                sorted(expected_ids - mask_ids)[:20],
                sorted(mask_ids - expected_ids)[:20],
            )
        )
    if prediction_ids != expected_ids:
        failures.append(
            "prediction IDs differ: missing={}, extra={}".format(
                sorted(expected_ids - prediction_ids)[:20],
                sorted(prediction_ids - expected_ids)[:20],
            )
        )

    modes = set()
    dimensions = set()
    unreadable = []
    total_bytes = 0
    for path in list(masks.values()) + list(predictions.values()):
        total_bytes += path.stat().st_size
        try:
            with Image.open(str(path)) as image:
                modes.add(image.mode)
                dimensions.add(tuple(image.size))
                image.verify()
        except Exception as error:
            unreadable.append("{}: {!r}".format(path.name, error))
    if unreadable:
        failures.append("unreadable JPEG files={}".format(len(unreadable)))

    decode_logs = sorted(args.output.glob("01_decode_dev_shard_*.log"))
    if not decode_logs and (args.output / "01_decode_dev.log").is_file():
        decode_logs = [args.output / "01_decode_dev.log"]
    logged_ids = []
    for path in decode_logs:
        for line in path.read_text(errors="replace").splitlines():
            stripped = line.strip()
            if stripped.isdigit():
                logged_ids.append(int(stripped))
    logged_id_set = set(logged_ids)
    if logged_id_set != expected_ids or len(logged_ids) != args.expected_count:
        failures.append(
            "decode logs do not cover each ID exactly once: entries={} unique={}".format(
                len(logged_ids), len(logged_id_set)
            )
        )

    summary = {
        "status": "PASS" if not failures else "FAIL",
        "expected_count": args.expected_count,
        "mask_file_count": len(masks),
        "prediction_file_count": len(predictions),
        "decode_log_count": len(decode_logs),
        "decode_logged_id_count": len(logged_ids),
        "decode_logged_unique_id_count": len(logged_id_set),
        "jpeg_modes": sorted(modes),
        "jpeg_dimensions": [list(size) for size in sorted(dimensions)],
        "unreadable_jpeg_count": len(unreadable),
        "output_bytes": total_bytes,
        "failures": failures,
        "unreadable_examples": unreadable[:20],
        "numeric_nan_inf_check": (
            "not representable in JPEG; performed separately on fixed numeric arrays"
        ),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
