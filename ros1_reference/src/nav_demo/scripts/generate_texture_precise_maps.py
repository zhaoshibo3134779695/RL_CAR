#!/usr/bin/env python3
"""Trace centimeter-grid navigation maps from the proportional field image."""

import math
from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
MAP_DIR = ROOT / "map"
CONFIG_DIR = ROOT / "config"
TEXTURE = Path("/home/mowen/Pictures/map_texture.png")
USER_FORBIDDEN = MAP_DIR / "722109_source.png"

OLD_WIDTH = 96
OLD_HEIGHT = 78
OLD_RESOLUTION = 0.049743
OLD_ORIGIN = (-2.486458, -1.856923)

# Centers of the printed field border in map_texture.png.
TEXTURE_LEFT = 19.0
TEXTURE_RIGHT = 681.0
TEXTURE_TOP = 13.0
TEXTURE_BOTTOM = 539.0

WIDTH = 475
HEIGHT = 380
USER_WIDTH = 481
USER_HEIGHT = 386
RESOLUTION = 0.01
LINE_WIDTH = 3
INITIAL_POSE = (-2.188, 1.296422513659279)


def guide_to_texture_x(x):
    return TEXTURE_LEFT + x * (TEXTURE_RIGHT - TEXTURE_LEFT) / 95.0


def guide_to_texture_y(y):
    return TEXTURE_TOP + (y - 1.0) * (TEXTURE_BOTTOM - TEXTURE_TOP) / 76.0


def texture_to_output_x(x):
    return 1.0 + (x - TEXTURE_LEFT) * 472.0 / (
        TEXTURE_RIGHT - TEXTURE_LEFT
    )


def texture_to_output_y(y):
    return 1.0 + (y - TEXTURE_TOP) * 377.0 / (
        TEXTURE_BOTTOM - TEXTURE_TOP
    )


class TextureProjection:
    def __init__(self):
        texture = cv2.imread(str(TEXTURE), cv2.IMREAD_COLOR)
        if texture is None:
            raise RuntimeError("Unable to read {}".format(TEXTURE))
        self.gray = cv2.cvtColor(texture, cv2.COLOR_BGR2GRAY)
        self.dark = self.gray < 150

        guide = cv2.imread(str(MAP_DIR / "7.15.png"), cv2.IMREAD_GRAYSCALE)
        if guide is None or guide.shape != (OLD_HEIGHT, OLD_WIDTH):
            raise RuntimeError("Unexpected 7.15.png dimensions")
        self.guide = guide < 128
        self.offsets = self._build_offsets()
        self.offset_points = np.array(list(self.offsets), dtype=np.float64)
        self.offset_values = np.array(
            [self.offsets[tuple(point)] for point in self.offset_points],
            dtype=np.float64,
        )

    def _nearest_dark_pixel(self, x, y, radius=12):
        predicted_x = guide_to_texture_x(x)
        predicted_y = guide_to_texture_y(y)
        x0 = max(0, int(math.floor(predicted_x - radius)))
        x1 = min(self.gray.shape[1], int(math.ceil(predicted_x + radius + 1)))
        y0 = max(0, int(math.floor(predicted_y - radius)))
        y1 = min(self.gray.shape[0], int(math.ceil(predicted_y + radius + 1)))
        ys, xs = np.nonzero(self.dark[y0:y1, x0:x1])
        if len(xs) == 0:
            return 0.0, 0.0, False

        xs = xs + x0
        ys = ys + y0
        distance_squared = (
            (xs - predicted_x) ** 2 + (ys - predicted_y) ** 2
        )
        valid = distance_squared <= radius * radius
        if not np.any(valid):
            return 0.0, 0.0, False
        xs = xs[valid]
        ys = ys[valid]
        distance_squared = distance_squared[valid]
        index = int(np.argmin(distance_squared))
        return (
            float(xs[index] - predicted_x),
            float(ys[index] - predicted_y),
            True,
        )

    def _build_offsets(self):
        raw = {}
        ys, xs = np.nonzero(self.guide)
        for y, x in zip(ys, xs):
            raw[(x, y)] = self._nearest_dark_pixel(x, y)

        smoothed = {}
        for (x, y), (_, _, valid) in raw.items():
            nearby = []
            for neighbor_y in range(max(0, y - 2), min(OLD_HEIGHT, y + 3)):
                for neighbor_x in range(max(0, x - 2), min(OLD_WIDTH, x + 3)):
                    value = raw.get((neighbor_x, neighbor_y))
                    if value is None or not value[2]:
                        continue
                    dx, dy, _ = value
                    if abs(dx) <= 10.0 and abs(dy) <= 10.0:
                        nearby.append((dx, dy))
            if nearby:
                dx = float(np.median([value[0] for value in nearby]))
                dy = float(np.median([value[1] for value in nearby]))
            elif valid:
                dx, dy, _ = raw[(x, y)]
            else:
                dx = dy = 0.0
            smoothed[(x, y)] = (dx, dy)
        return smoothed

    def interpolated_offset(self, x, y):
        delta = self.offset_points - np.array((x, y))
        distance_squared = np.sum(delta * delta, axis=1)
        nearest = np.argpartition(distance_squared, min(15, len(distance_squared) - 1))[:16]
        nearest_distance = distance_squared[nearest]
        weights = 1.0 / np.maximum(nearest_distance, 0.25)
        values = self.offset_values[nearest]
        weighted = np.sum(values * weights[:, None], axis=0) / np.sum(weights)
        return float(weighted[0]), float(weighted[1])

    def project(self, x, y):
        if (int(x), int(y)) in self.offsets and x == int(x) and y == int(y):
            dx, dy = self.offsets[(int(x), int(y))]
        else:
            dx, dy = self.interpolated_offset(x, y)
        texture_x = guide_to_texture_x(x) + dx
        texture_y = guide_to_texture_y(y) + dy
        return texture_to_output_x(texture_x), texture_to_output_y(texture_y)


def render_map(projection, source_name, output_name):
    source = cv2.imread(str(MAP_DIR / source_name), cv2.IMREAD_GRAYSCALE)
    if source is None or source.shape != (OLD_HEIGHT, OLD_WIDTH):
        raise RuntimeError("Unexpected source map: {}".format(source_name))
    occupied = source < 128
    points = {}
    ys, xs = np.nonzero(occupied)
    for y, x in zip(ys, xs):
        output_x, output_y = projection.project(float(x), float(y))
        points[(x, y)] = (int(round(output_x)), int(round(output_y)))

    skeleton = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for (x, y), point in points.items():
        skeleton[point[1], point[0]] = 255
        for dx, dy in ((1, 0), (0, 1), (1, 1), (-1, 1)):
            neighbor = points.get((x + dx, y + dy))
            if neighbor is not None:
                cv2.line(skeleton, point, neighbor, 255, 1, cv2.LINE_8)

    # The measured outer-edge size is exactly 475 x 380 cm.
    cv2.rectangle(skeleton, (1, 1), (473, 378), 255, 1, cv2.LINE_8)
    occupied_mask = cv2.dilate(
        (skeleton != 0).astype(np.uint8),
        np.ones((LINE_WIDTH, LINE_WIDTH), dtype=np.uint8),
    )
    output = np.where(occupied_mask != 0, 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(MAP_DIR / output_name), output):
        raise RuntimeError("Unable to write {}".format(output_name))


def render_forbidden_from_texture(output_name):
    """Draw a clean vector trace of the course printed in map_texture.png."""
    cleaned = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    def point(source_x, source_y):
        return np.array(
            [texture_to_output_x(source_x), texture_to_output_y(source_y)],
            dtype=np.float64,
        )

    def draw_polyline(source_points):
        output_points = np.rint(
            np.array([point(x, y) for x, y in source_points])
        ).astype(np.int32)
        cv2.polylines(cleaned, [output_points], False, 255, 1, cv2.LINE_8)

    def draw_filled_rectangle(top_left, bottom_right):
        first = np.rint(point(*top_left)).astype(np.int32)
        second = np.rint(point(*bottom_right)).astype(np.int32)
        cv2.rectangle(cleaned, tuple(first), tuple(second), 255, -1, cv2.LINE_8)

    def cubic(p0, p1, p2, p3, count=48):
        values = []
        for value in np.linspace(0.0, 1.0, count):
            inverse = 1.0 - value
            values.append(
                inverse ** 3 * p0
                + 3.0 * inverse ** 2 * value * p1
                + 3.0 * inverse * value ** 2 * p2
                + value ** 3 * p3
            )
        return values

    # Straight area boundaries visible at the upper-left of the supplied map.
    draw_polyline([(110, 13), (110, 86)])
    draw_polyline([(99, 459), (99, 86), (294, 86)])

    # Left course edge and the two rectangular battle-area recesses.
    draw_polyline(
        [
            (294, 154),
            (166, 154),
            (166, 234),
            (251, 234),
            (251, 294),
            (166, 294),
            (166, 375),
            (251, 375),
            (251, 437),
            (166, 437),
            (166, 539),
        ]
    )

    # Inner edge: a smooth entry curve, central turning bay and bottom divider.
    inner = [point(294, 86), point(294, 190)]
    inner.extend(
        cubic(point(294, 190), point(294, 244), point(333, 274), point(382, 274))
    )
    inner.extend(
        cubic(point(382, 274), point(430, 274), point(459, 301), point(459, 340))
    )
    inner.extend(
        cubic(point(459, 340), point(459, 365), point(438, 379), point(414, 379))
    )
    inner.extend([point(329, 379), point(329, 539)])
    cv2.polylines(
        cleaned,
        [np.rint(np.array(inner)).astype(np.int32)],
        False,
        255,
        1,
        cv2.LINE_8,
    )

    # Upper and right course edge. All non-curved sections are exact H/V lines.
    draw_polyline(
        [
            (294, 139),
            (615, 139),
            (615, 235),
            (530, 235),
            (530, 294),
            (615, 294),
            (615, 375),
            (530, 375),
            (530, 437),
            (615, 437),
            (615, 474),
            (404, 474),
        ]
    )
    draw_polyline([(365, 139), (365, 204)])

    # Outer S bend. Its endpoints are the lower edge of the upper-right area
    # and the left edge of the lower-right area in the supplied field drawing.
    outer = cubic(
        point(365, 204), point(438, 204), point(500, 239), point(530, 294)
    )
    outer.extend(
        cubic(point(530, 294), point(555, 351), point(522, 431), point(404, 474))
    )
    cv2.polylines(
        cleaned,
        [np.rint(np.array(outer)).astype(np.int32)],
        False,
        255,
        1,
        cv2.LINE_8,
    )

    # Four straight rectangular field blocks shown in the source drawing.
    draw_filled_rectangle((225, 245), (243, 289))
    draw_filled_rectangle((225, 386), (243, 430))
    draw_filled_rectangle((542, 245), (560, 289))
    draw_filled_rectangle((542, 386), (560, 430))

    # Exact measured physical perimeter: 475 cm by 380 cm.
    cv2.rectangle(cleaned, (1, 1), (473, 378), 255, 1, cv2.LINE_8)
    occupied = cv2.dilate(
        (cleaned != 0).astype(np.uint8),
        np.ones((LINE_WIDTH, LINE_WIDTH), dtype=np.uint8),
    )
    output = np.where(occupied != 0, 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(MAP_DIR / output_name), output):
        raise RuntimeError("Unable to write {}".format(output_name))


def render_complete_forbidden(projection, output_name):
    """Preserve every edge in 7.15.png while lightly smoothing its raster trace."""
    render_map(projection, "7.15.png", output_name)
    path = MAP_DIR / output_name
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    occupied = (image < 128).astype(np.uint8) * 255
    blurred = cv2.GaussianBlur(occupied, (3, 3), 0.8)
    centerline = cv2.ximgproc.thinning(
        np.where(blurred >= 70, 255, 0).astype(np.uint8)
    )
    cv2.rectangle(centerline, (1, 1), (473, 378), 255, 1, cv2.LINE_8)
    occupied = cv2.dilate(
        (centerline != 0).astype(np.uint8),
        np.ones((LINE_WIDTH, LINE_WIDTH), dtype=np.uint8),
    )
    output = np.where(occupied != 0, 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), output):
        raise RuntimeError("Unable to write {}".format(output_name))


def render_user_forbidden(output_name):
    """Normalize the user's finished drawing without changing its geometry."""
    source = cv2.imread(str(USER_FORBIDDEN), cv2.IMREAD_GRAYSCALE)
    if source is None or source.shape != (559, 703):
        raise RuntimeError("Unexpected user forbidden map: {}".format(USER_FORBIDDEN))

    # All supplied dimensions are measured between the inner yellow-line
    # edges. Add 3 cm on both sides around the 475 x 380 cm clear field.
    source_corners = np.float32(
        [[19.0, 13.0], [681.0, 13.0], [681.0, 539.0], [19.0, 539.0]]
    )
    output_corners = np.float32(
        [[1.0, 1.0], [479.0, 1.0], [479.0, 384.0], [1.0, 384.0]]
    )
    transform = cv2.getPerspectiveTransform(source_corners, output_corners)
    normalized = cv2.warpPerspective(
        source,
        transform,
        (USER_WIDTH, USER_HEIGHT),
        flags=cv2.INTER_AREA,
        borderValue=255,
    )

    # Build a temporary centered 3 cm trace only to recover the enclosed
    # regions from the user's drawing.
    thresholded = np.where(normalized < 210, 255, 0).astype(np.uint8)
    centerline = cv2.ximgproc.thinning(thresholded)
    cv2.rectangle(centerline, (1, 1), (479, 384), 255, 1, cv2.LINE_8)
    centered = cv2.dilate(
        (centerline != 0).astype(np.uint8),
        np.ones((LINE_WIDTH, LINE_WIDTH), dtype=np.uint8),
    )

    # The large exterior free component is the route. Draw each enclosed
    # region's 3 cm boundary inward, so line width never consumes the route.
    free = (centered == 0).astype(np.uint8)
    component_count, free_labels, free_stats, _ = cv2.connectedComponentsWithStats(
        free, connectivity=4
    )
    enclosed = [
        label
        for label in range(1, component_count)
        if free_stats[label, cv2.CC_STAT_AREA] >= 1000
    ]
    enclosed.sort(
        key=lambda label: free_stats[label, cv2.CC_STAT_AREA], reverse=True
    )
    if len(enclosed) != 3:
        raise RuntimeError("Unexpected free-space topology in {}".format(USER_FORBIDDEN))
    route_label = enclosed[0]

    corrected = np.zeros((USER_HEIGHT, USER_WIDTH), dtype=np.uint8)
    corrected[:3, :] = 1
    corrected[-3:, :] = 1
    corrected[:, :3] = 1
    corrected[:, -3:] = 1
    for label in enclosed:
        if label == route_label:
            continue
        region = (free_labels == label).astype(np.uint8)
        distance = cv2.distanceTransform(region, cv2.DIST_L2, 5)
        boundary = (distance > 0.0) & (distance <= 3.0)
        if (
            free_stats[label, cv2.CC_STAT_TOP]
            + free_stats[label, cv2.CC_STAT_HEIGHT]
            >= USER_HEIGHT - 3
        ):
            boundary[USER_HEIGHT - 6:, :] = False
        if free_stats[label, cv2.CC_STAT_LEFT] < USER_WIDTH // 2:
            boundary = cv2.warpAffine(
                boundary.astype(np.uint8),
                np.float32([[1, 0, -6], [0, 1, 0]]),
                (USER_WIDTH, USER_HEIGHT),
                flags=cv2.INTER_NEAREST,
                borderValue=0,
            ).astype(bool)
        corrected[boundary] = 1

    # Preserve the independent upper-left divider. Moving it and the complete
    # left region 6 cm left makes both consecutive start corridors 48 cm.
    occupied_count, occupied_labels, occupied_stats, _ = cv2.connectedComponentsWithStats(
        centered, connectivity=8
    )
    divider_candidates = [
        label
        for label in range(1, occupied_count)
        if occupied_stats[label, cv2.CC_STAT_AREA] >= 100
        and occupied_stats[label, cv2.CC_STAT_LEFT] < USER_WIDTH // 2
        and occupied_stats[label, cv2.CC_STAT_WIDTH] < USER_WIDTH // 2
    ]
    if len(divider_candidates) != 1:
        raise RuntimeError("Unable to isolate upper-left divider")
    divider = (occupied_labels == divider_candidates[0]).astype(np.uint8)
    divider = cv2.warpAffine(
        divider,
        np.float32([[1, 0, -7], [0, 1, 0]]),
        (USER_WIDTH, USER_HEIGHT),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    corrected[divider != 0] = 1

    # The left enclosed region meets the bottom perimeter through two vertical
    # lines. Extend those lines after converting the boundary to its inner edge.
    bottom_probe = corrected[350]
    starts = np.flatnonzero(bottom_probe & ~np.r_[False, bottom_probe[:-1]])
    ends = np.flatnonzero(bottom_probe & ~np.r_[bottom_probe[1:], False])
    for start, end in zip(starts, ends):
        if 3 < start < USER_WIDTH - 3:
            corrected[350:USER_HEIGHT - 3, start:end + 1] = 1

    def move_horizontal(y0, y1, x0, x1, offset):
        segment = corrected[y0:y1 + 1, x0:x1 + 1].copy()
        corrected[y0:y1 + 1, x0:x1 + 1] = 0
        corrected[y0 + offset:y1 + offset + 1, x0:x1 + 1] |= segment

    # The straight recess-to-recess passages are 46 cm after edge conversion;
    # moving each opposing edge outward by 1 cm makes their inner gap 48 cm.
    straight_edges = [
        (158, 160, 99, 168, -1),
        (207, 209, 99, 168, 1),
        (261, 263, 99, 166, -1),
        (310, 312, 99, 166, 1),
        (160, 162, 366, 430, -1),
        (209, 211, 366, 430, 1),
        (263, 265, 366, 430, -1),
        (312, 314, 366, 430, 1),
    ]
    for edge in straight_edges:
        move_horizontal(*edge)

    # Reconnect one-pixel corner gaps introduced by the exact edge shifts.
    corrected = cv2.morphologyEx(
        corrected,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )

    output = np.where(corrected != 0, 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(MAP_DIR / output_name), output):
        raise RuntimeError("Unable to write {}".format(output_name))


def old_world_to_source_pixel(x, y):
    column = (x - OLD_ORIGIN[0]) / OLD_RESOLUTION - 0.5
    row = OLD_HEIGHT - 0.5 - (y - OLD_ORIGIN[1]) / OLD_RESOLUTION
    return column, row


def calculate_origin(projection):
    old_x, old_y = old_world_to_source_pixel(*INITIAL_POSE)
    new_x, new_y = projection.project(old_x, old_y)
    return (
        INITIAL_POSE[0] - (new_x + 0.5) * RESOLUTION,
        INITIAL_POSE[1] - (HEIGHT - new_y - 0.5) * RESOLUTION,
    )


def old_world_to_new_world(projection, origin, x, y):
    old_x, old_y = old_world_to_source_pixel(x, y)
    new_x, new_y = projection.project(old_x, old_y)
    return (
        origin[0] + (new_x + 0.5) * RESOLUTION,
        origin[1] + (HEIGHT - new_y - 0.5) * RESOLUTION,
    )


def write_map_yaml(image_name, yaml_name, origin):
    data = {
        "image": str((MAP_DIR / image_name).resolve()),
        "resolution": RESOLUTION,
        "origin": [round(origin[0], 6), round(origin[1], 6), 0.0],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
    }
    with open(MAP_DIR / yaml_name, "w") as stream:
        yaml.safe_dump(data, stream, sort_keys=False)


def transform_waypoints(projection, origin):
    with open(CONFIG_DIR / "7.20nicepoint.yaml") as stream:
        route = yaml.safe_load(stream)
    for waypoint in route["waypoints"]:
        position = waypoint["position"]
        x, y = old_world_to_new_world(
            projection, origin, position["x"], position["y"]
        )
        position["x"] = round(x, 12)
        position["y"] = round(y, 12)
    with open(CONFIG_DIR / "7.20nicepoint_texture_precise.yaml", "w") as stream:
        yaml.safe_dump(route, stream, sort_keys=False)


def main():
    projection = TextureProjection()
    origin = calculate_origin(projection)
    render_map(projection, "map7_aligned.png", "map7_texture_precise.png")
    render_user_forbidden("forbidden_texture_precise.png")
    render_user_forbidden("722109_map.png")
    write_map_yaml("map7_texture_precise.png", "map7_texture_precise.yaml", origin)
    write_map_yaml(
        "forbidden_texture_precise.png",
        "forbidden_texture_precise.yaml",
        origin,
    )
    write_map_yaml("722109_map.png", "722109_map.yaml", origin)
    transform_waypoints(projection, origin)
    print("field inner edges: 475 x 380 cm")
    print("map raster: 481 x 386 cm")
    print("line width: 3 cm")
    print("origin: [{:.6f}, {:.6f}, 0]".format(*origin))


if __name__ == "__main__":
    main()
