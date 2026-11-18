#!/usr/bin/env python3

import copy
import math
import unittest

from yellow_track_avoidance.compact_obstacles import (
    CompactSceneError,
    build_compact_obstacle_scene,
    sample_filled_disc_points,
    select_teb_obstacles,
)


def boundary_payload(generation=3):
    return {
        "scene_generation": generation,
        "frame_id": "odom",
        "boundary_pair_valid": True,
        "left_base_m": [[0.0, 0.455], [1.85, 0.455]],
        "right_base_m": [[0.0, -0.455], [1.85, -0.455]],
    }


def disc_status(generation=3):
    return {
        "scene_generation": generation,
        "fixed_frame": "odom",
        "confirmed_positions_frozen": True,
        "costmap_writes_enabled": True,
        "fixed_transform_available": True,
        "memory_count": 2,
        "active_memory_count": 2,
        "disc_count": 2,
        "costmap_disc_count": 2,
        "active_tracks": [
            {
                "memory_id": 7,
                "fixed_xy": [0.65, -0.15],
                "confirmed": True,
            },
            {
                "memory_id": 2,
                "fixed_xy": [1.14, 0.22],
                "confirmed": True,
            },
        ],
    }


class CompactObstacleSceneTest(unittest.TestCase):
    def test_collision_guard_does_not_change_physical_gap_measurement(self):
        guarded = build_compact_obstacle_scene(
            boundary_payload(),
            disc_status(),
            disc_radius_m=0.05,
            disc_collision_radius_m=0.08,
            enable_passage_closures=True,
        )
        circles = [item for item in guarded.obstacles if item.kind == "circle"]
        self.assertEqual([item.radius_m for item in circles], [0.08, 0.08])

        baseline = build_compact_obstacle_scene(
            boundary_payload(),
            disc_status(),
            disc_radius_m=0.05,
            enable_passage_closures=True,
        )
        self.assertEqual(
            [
                (item.left_gap_m, item.right_gap_m)
                for item in guarded.passage_decisions
            ],
            [
                (item.left_gap_m, item.right_gap_m)
                for item in baseline.passage_decisions
            ],
        )

    def test_builds_two_inner_edge_lines_and_two_circles(self):
        scene = build_compact_obstacle_scene(boundary_payload(), disc_status())
        self.assertEqual(scene.scene_generation, 3)
        self.assertEqual(scene.frame_id, "odom")
        self.assertEqual([item.kind for item in scene.obstacles], [
            "line", "line", "circle", "circle"
        ])
        self.assertEqual(len(scene.obstacles), 4)
        for point in scene.obstacles[0].points:
            self.assertAlmostEqual(point[1], 0.440, places=12)
        for point in scene.obstacles[1].points:
            self.assertAlmostEqual(point[1], -0.440, places=12)
        self.assertEqual(
            [item.obstacle_id for item in scene.obstacles[2:]], [2002, 2007]
        )
        self.assertTrue(
            all(math.isclose(item.radius_m, 0.05) for item in scene.obstacles[2:])
        )

    def test_builds_every_confirmed_disc_in_collected_set(self):
        status = disc_status()
        status["active_tracks"].append(
            {
                "memory_id": 9,
                "fixed_xy": [1.45, -0.08],
                "confirmed": True,
            }
        )
        for field in (
            "memory_count",
            "active_memory_count",
            "disc_count",
            "costmap_disc_count",
        ):
            status[field] = 3

        scene = build_compact_obstacle_scene(boundary_payload(), status)
        circles = [item for item in scene.obstacles if item.kind == "circle"]
        self.assertEqual(len(circles), 3)
        self.assertEqual(
            [item.obstacle_id for item in circles], [2002, 2007, 2009]
        )

    def test_samples_complete_filled_disc_geometry_for_sbpl(self):
        scene = build_compact_obstacle_scene(boundary_payload(), disc_status())
        points = sample_filled_disc_points(scene, resolution_m=0.005)
        self.assertEqual(len(points), 634)
        xy = {(round(point[0], 6), round(point[1], 6)) for point in points}
        for center in ((1.14, 0.22), (0.65, -0.15)):
            self.assertIn((round(center[0], 6), round(center[1], 6)), xy)
            self.assertIn((round(center[0] + 0.05, 6), round(center[1], 6)), xy)
            self.assertIn((round(center[0] - 0.05, 6), round(center[1], 6)), xy)
            self.assertIn((round(center[0], 6), round(center[1] + 0.05, 6)), xy)
            self.assertIn((round(center[0], 6), round(center[1] - 0.05, 6)), xy)

    def test_rejects_invalid_full_disc_cloud_resolution(self):
        scene = build_compact_obstacle_scene(boundary_payload(), disc_status())
        for value in (0.0, -0.01, float("nan")):
            with self.subTest(value=value):
                with self.assertRaises(CompactSceneError):
                    sample_filled_disc_points(scene, resolution_m=value)

    def test_offsets_slanted_boundaries_toward_drivable_space(self):
        boundary = boundary_payload()
        boundary["left_base_m"] = [[0.0, 1.0], [1.0, 2.0]]
        boundary["right_base_m"] = [[1.0, 0.0], [2.0, 1.0]]
        scene = build_compact_obstacle_scene(boundary, disc_status())
        offset = 0.015 / math.sqrt(2.0)
        self.assertAlmostEqual(scene.obstacles[0].points[0][0], offset)
        self.assertAlmostEqual(scene.obstacles[0].points[0][1], 1.0 - offset)
        self.assertAlmostEqual(scene.obstacles[1].points[0][0], 1.0 - offset)
        self.assertAlmostEqual(scene.obstacles[1].points[0][1], offset)

    def test_stage1_closes_only_the_narrower_side_of_each_disc(self):
        scene = build_compact_obstacle_scene(
            boundary_payload(),
            disc_status(),
            enable_passage_closures=True,
            vehicle_width_m=0.33,
            passage_total_clearance_m=0.02,
        )
        self.assertEqual(len(scene.passage_decisions), 2)
        self.assertEqual(
            [(item.pass_side, item.blocked_side) for item in scene.passage_decisions],
            [("left", "right"), ("right", "left")],
        )
        closures = [item for item in scene.obstacles if item.obstacle_id >= 3000]
        self.assertEqual(len(closures), 2)
        self.assertTrue(all(item.kind == "line" for item in closures))
        self.assertTrue(all(len(item.points) == 2 for item in closures))

    def test_center_disc_keeps_both_globally_feasible_sides_open(self):
        discs = disc_status()
        # This disc is slightly off centre: the two physical gaps are unequal,
        # but both still accommodate the body and the complete 50 mm allowance.
        discs["active_tracks"][0]["fixed_xy"] = [0.65, 0.009]
        # Keep the second disc near one edge so exactly one semantic closure is
        # still generated for the combined middle-plus-edge scenario.
        discs["active_tracks"][1]["fixed_xy"] = [1.14, 0.22]
        scene = build_compact_obstacle_scene(
            boundary_payload(),
            discs,
            enable_passage_closures=True,
            vehicle_width_m=0.33,
            passage_total_clearance_m=0.05,
            side_tie_tolerance_m=0.01,
        )
        decisions = {item.memory_id: item for item in scene.passage_decisions}
        self.assertEqual(decisions[7].pass_side, "either")
        self.assertEqual(decisions[7].blocked_side, "none")
        self.assertEqual(decisions[7].closure, ())
        self.assertEqual(decisions[2].pass_side, "right")
        closures = [item for item in scene.obstacles if item.obstacle_id >= 3000]
        self.assertEqual(len(closures), 1)

    def test_teb_can_exclude_global_topology_closures(self):
        scene = build_compact_obstacle_scene(
            boundary_payload(),
            disc_status(),
            enable_passage_closures=True,
        )
        self.assertEqual(len(scene.obstacles), 6)
        local_obstacles = select_teb_obstacles(
            scene, include_passage_closures=False
        )
        self.assertEqual(len(local_obstacles), 4)
        self.assertEqual(
            [item.kind for item in local_obstacles],
            ["line", "line", "circle", "circle"],
        )

    def test_small_gaps_are_diagnostic_and_do_not_reject_the_scene(self):
        boundary = boundary_payload()
        # 0.63 m yellow-centre spacing gives 0.60 m inner-edge clear width.
        boundary["left_base_m"] = [[0.0, 0.315], [1.85, 0.315]]
        boundary["right_base_m"] = [[0.0, -0.315], [1.85, -0.315]]
        discs = disc_status()
        discs["active_tracks"][0]["fixed_xy"] = [0.65, -0.02]
        discs["active_tracks"][1]["fixed_xy"] = [1.14, 0.02]
        scene = build_compact_obstacle_scene(
            boundary,
            discs,
            enable_passage_closures=True,
            vehicle_width_m=0.33,
            passage_total_clearance_m=0.02,
            side_tie_tolerance_m=0.0,
        )
        self.assertEqual(len(scene.passage_decisions), 2)
        self.assertTrue(
            all(
                max(item.left_gap_m, item.right_gap_m) < item.required_gap_m
                for item in scene.passage_decisions
            )
        )
        self.assertTrue(all(item.closure for item in scene.passage_decisions))

    def test_rejects_partial_stale_or_unfrozen_world_models(self):
        cases = []
        stale = disc_status(generation=4)
        cases.append(stale)
        one_disc = disc_status()
        one_disc["costmap_disc_count"] = 1
        cases.append(one_disc)
        unfrozen = disc_status()
        unfrozen["confirmed_positions_frozen"] = False
        cases.append(unfrozen)
        duplicate = disc_status()
        duplicate["active_tracks"][1]["memory_id"] = 7
        cases.append(duplicate)
        invalid = disc_status()
        invalid["active_tracks"][0]["fixed_xy"] = [float("nan"), 0.0]
        cases.append(invalid)
        for status in cases:
            with self.subTest(status=status):
                with self.assertRaises(CompactSceneError):
                    build_compact_obstacle_scene(boundary_payload(), status)

    def test_input_payloads_are_not_mutated(self):
        boundary = boundary_payload()
        discs = disc_status()
        original_boundary = copy.deepcopy(boundary)
        original_discs = copy.deepcopy(discs)
        build_compact_obstacle_scene(boundary, discs)
        self.assertEqual(boundary, original_boundary)
        self.assertEqual(discs, original_discs)


if __name__ == "__main__":
    unittest.main()
