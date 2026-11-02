#!/usr/bin/env python
"""Compare locally reproduced fixed-window arrays with bundled expectations."""

from __future__ import print_function

import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()

    expected_files = sorted((args.expected / "arrays").glob("sample_*.npz"))
    actual_files = sorted((args.actual / "arrays").glob("sample_*.npz"))
    expected_names = [path.name for path in expected_files]
    actual_names = [path.name for path in actual_files]
    failures = []
    if expected_names != actual_names:
        failures.append(
            "file list differs: expected {} files, actual {} files".format(
                len(expected_names), len(actual_names)
            )
        )

    by_name = {path.name: path for path in actual_files}
    largest_difference = 0.0
    for expected_path in expected_files:
        actual_path = by_name.get(expected_path.name)
        if actual_path is None:
            continue
        with np.load(str(expected_path), allow_pickle=False) as expected, np.load(
            str(actual_path), allow_pickle=False
        ) as actual:
            if not np.array_equal(expected["start_frame"], actual["start_frame"]):
                failures.append("{}: start_frame differs".format(expected_path.name))
            if not np.array_equal(expected["target_mask"], actual["target_mask"]):
                failures.append("{}: target_mask must be exact".format(expected_path.name))
            expected_prediction = expected["prediction"]
            actual_prediction = actual["prediction"]
            if expected_prediction.shape != actual_prediction.shape:
                failures.append("{}: prediction shape differs".format(expected_path.name))
                continue
            difference = float(
                np.max(np.abs(expected_prediction - actual_prediction))
            )
            largest_difference = max(largest_difference, difference)
            if not np.allclose(
                expected_prediction,
                actual_prediction,
                atol=args.atol,
                rtol=args.rtol,
                equal_nan=False,
            ):
                failures.append(
                    "{}: prediction outside tolerance; max_abs_difference={}".format(
                        expected_path.name, difference
                    )
                )

    print("expected_files={}".format(len(expected_files)))
    print("actual_files={}".format(len(actual_files)))
    print("atol={}".format(args.atol))
    print("rtol={}".format(args.rtol))
    print("largest_prediction_absolute_difference={}".format(largest_difference))
    print("failures={}".format(len(failures)))
    for failure in failures:
        print("FAIL: {}".format(failure))
    if failures:
        return 1
    print("PASS: file IDs/shapes and target masks match exactly; predictions match tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
