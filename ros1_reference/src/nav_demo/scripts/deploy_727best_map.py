#!/usr/bin/env python3
"""Deploy 727best while preserving the existing navigation coordinates."""

from pathlib import Path
import shutil

import cv2
import yaml


ROOT = Path(__file__).resolve().parents[1]
MAP_DIR = ROOT / "map"
SOURCE = Path("/home/mowen/Desktop/727best.png")
SOURCE_COPY = MAP_DIR / "727best_source.png"
MAP_IMAGE = MAP_DIR / "727best_map.png"
MAP_YAML = MAP_DIR / "727best_map.yaml"

REFERENCE_SOURCE = MAP_DIR / "724best2_source.png"
REFERENCE_MAP = MAP_DIR / "724best2_map.png"
REFERENCE_YAML = MAP_DIR / "724best2_map.yaml"

DARK_THRESHOLD = 200
BINARY_THRESHOLD = 127
REMOVED_TOP_ROW = 6


def dark_bounds(image):
    dark_y, dark_x = (image < DARK_THRESHOLD).nonzero()
    if len(dark_x) == 0:
        raise RuntimeError("No map lines found")
    return (
        int(dark_x.min()),
        int(dark_y.min()),
        int(dark_x.max()) + 1,
        int(dark_y.max()) + 1,
    )


def main():
    source = cv2.imread(str(SOURCE), cv2.IMREAD_GRAYSCALE)
    reference_source = cv2.imread(str(REFERENCE_SOURCE), cv2.IMREAD_GRAYSCALE)
    reference_map = cv2.imread(str(REFERENCE_MAP), cv2.IMREAD_GRAYSCALE)
    if source is None:
        raise RuntimeError("Unable to read {}".format(SOURCE))
    if reference_source is None or reference_map is None:
        raise RuntimeError("Unable to read the 724best2 reference map")

    left, top, right, bottom = dark_bounds(source)
    old_left, old_top, old_right, old_bottom = dark_bounds(reference_source)
    if left != old_left or top != old_top:
        raise RuntimeError(
            "727best moved the preserved top-left anchor: "
            "new=({}, {}) old=({}, {})".format(
                left, top, old_left, old_top
            )
        )
    if right < old_right or bottom < old_bottom:
        raise RuntimeError("727best is smaller than the reference map")

    cropped = source[top:bottom, left:right]
    if cropped.shape[0] <= REMOVED_TOP_ROW:
        raise RuntimeError("727best crop is unexpectedly short")
    cropped = cv2.vconcat(
        [
            cropped[:REMOVED_TOP_ROW],
            cropped[REMOVED_TOP_ROW + 1:],
        ]
    )
    _, binary = cv2.threshold(
        cropped, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY
    )

    with REFERENCE_YAML.open("r", encoding="utf-8") as stream:
        reference_metadata = yaml.safe_load(stream)
    resolution = float(reference_metadata["resolution"])
    old_origin = [float(value) for value in reference_metadata["origin"]]

    width_extension = binary.shape[1] - reference_map.shape[1]
    height_extension = binary.shape[0] - reference_map.shape[0]
    origin = [
        old_origin[0],
        old_origin[1] - height_extension * resolution,
        old_origin[2],
    ]

    MAP_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(SOURCE), str(SOURCE_COPY))
    if not cv2.imwrite(str(MAP_IMAGE), binary):
        raise RuntimeError("Unable to write {}".format(MAP_IMAGE))

    metadata = {
        "image": MAP_IMAGE.name,
        "resolution": resolution,
        "origin": origin,
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
    }
    with MAP_YAML.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(metadata, stream, sort_keys=False)

    print(
        "Deployed {}x{} map at {:.3f} m/pixel".format(
            binary.shape[1], binary.shape[0], resolution
        )
    )
    print(
        "Extended right by {} px ({:.3f} m), down by {} px ({:.3f} m)".format(
            width_extension,
            width_extension * resolution,
            height_extension,
            height_extension * resolution,
        )
    )
    print("Origin: [{:.6f}, {:.6f}, {:.1f}]".format(*origin))


if __name__ == "__main__":
    main()
