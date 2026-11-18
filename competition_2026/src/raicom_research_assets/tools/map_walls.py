"""Extrude occupied Nav2 trinary map cells without smoothing the boundary."""
import argparse
import struct
from xml.etree import ElementTree as ET
from math import cos, sin
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


def spans(row):
    changes = np.flatnonzero(np.diff(row, prepend=False, append=False))
    return zip(changes[::2], changes[1::2])


def write_walls(map_yaml, output, height, template, world_output):
    config = yaml.safe_load(map_yaml.read_text())
    with Image.open(map_yaml.parent / config['image']) as image:
        gray = np.asarray(image.convert('L'), dtype=float) / 255.0
    probability = gray if config['negate'] else 1.0 - gray
    # PNG rows start at the top. The map origin is the lower-left cell corner.
    occupied = np.flipud(probability > config['occupied_thresh'])
    x_origin, y_origin, yaw = config['origin']
    resolution = config['resolution']
    vertices = {}
    triangles = []

    def quad(*corners):
        a, b, c, d = [vertices.setdefault(point, len(vertices) + 1) for point in corners]
        triangles.extend(((a, b, c), (a, c, d)))

    padded = np.pad(occupied, ((1, 1), (0, 0)))
    for y, row in enumerate(occupied):
        for left, right in spans(row):
            quad((left,y,1), (right,y,1), (right,y+1,1), (left,y+1,1))
            quad((left,y,0), (left,y+1,0), (right,y+1,0), (right,y,0))
            quad((left,y,0), (left,y,1), (left,y+1,1), (left,y+1,0))
            quad((right,y,0), (right,y+1,0), (right,y+1,1), (right,y,1))
        for left, right in spans(row & ~padded[y]):
            quad((left,y,0), (right,y,0), (right,y,1), (left,y,1))
        for left, right in spans(row & ~padded[y+2]):
            quad((left,y+1,0), (left,y+1,1), (right,y+1,1), (right,y+1,0))

    grid_points = np.array(list(vertices), dtype=float)
    xs, ys, levels = grid_points.T
    points = np.column_stack((x_origin + resolution * (cos(yaw)*xs - sin(yaw)*ys),
                              y_origin + resolution * (sin(yaw)*xs + cos(yaw)*ys),
                              height * levels))
    faces = points[np.array(triangles) - 1]
    normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    records = np.empty(len(faces), dtype=[('normal', '<f4', 3), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
    records['normal'], records['vertices'], records['attribute'] = normals, faces, 0
    with output.open('wb') as stream:
        stream.write(b'727best occupancy walls in map coordinates'.ljust(80, b' '))
        stream.write(struct.pack('<I', len(faces)))
        stream.write(records.tobytes())

    # A compound of boxes keeps concave corridors open and uses native physics primitives.
    ET.register_namespace('gz', 'http://gazebosim.org/schema')
    world = ET.parse(template)
    link = world.find("./world/model[@name='walls_727best']/link")
    open_runs = {}
    count = 0
    for y, row in enumerate(np.vstack([occupied, np.zeros((1, occupied.shape[1]), dtype=bool)])):
        current = set(spans(row))
        for left, right in sorted(open_runs.keys() - current):
            start = open_runs.pop((left, right))
            cx, cy = 0.5 * (left + right) * resolution, 0.5 * (start + y) * resolution
            wx, wy = x_origin + cos(yaw)*cx - sin(yaw)*cy, y_origin + sin(yaw)*cx + cos(yaw)*cy
            collision = ET.SubElement(link, 'collision', name=f'wall_{count}')
            count += 1
            ET.SubElement(collision, 'pose').text = f'{wx:.8f} {wy:.8f} {height/2:.8f} 0 0 {yaw:.8f}'
            box = ET.SubElement(ET.SubElement(collision, 'geometry'), 'box')
            ET.SubElement(box, 'size').text = f'{(right-left)*resolution:.8f} {(y-start)*resolution:.8f} {height:.8f}'
        for run in current:
            open_runs.setdefault(run, y)
    ET.indent(world)
    world.write(world_output, encoding='utf-8', xml_declaration=True)



def write_motion_world(static_world, output, navigation_config, height):
    world = ET.parse(static_world)
    scene = world.find('world')
    ET.SubElement(scene, 'gravity').text = '0 0 0'
    car = scene.find("model[@name='mowen']")
    car.find('static').text = 'false'
    body = car.find('link')
    # The velocity-driven planar body uses the original navigation footprint.
    params = yaml.safe_load(navigation_config.read_text())['competition_reference']['ros__parameters']
    vertices = np.array(params['nav_costmap_footprint_xy']).reshape(-1, 2)
    left, bottom = vertices.min(axis=0)
    right, top = vertices.max(axis=0)
    collision = ET.SubElement(body, 'collision', name='navigation_footprint')
    ET.SubElement(collision, 'pose').text = f'{(right+left)/2} {(top+bottom)/2} {height/2} 0 0 0'
    box = ET.SubElement(ET.SubElement(collision, 'geometry'), 'box')
    ET.SubElement(box, 'size').text = f'{right-left} {top-bottom} {height}'
    velocity = ET.SubElement(car, 'plugin', filename='gz-sim-velocity-control-system',
                             name='gz::sim::systems::VelocityControl')
    ET.SubElement(velocity, 'topic').text = '/model/mowen/route_velocity'
    ET.indent(world)
    world.write(output, encoding='utf-8', xml_declaration=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('map_yaml', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('template', type=Path)
    parser.add_argument('world_output', type=Path)
    parser.add_argument('--height', required=True, type=float)
    parser.add_argument('--motion-world', type=Path)
    parser.add_argument('--navigation-config', type=Path)
    args = parser.parse_args()
    if args.height <= 0:
        parser.error('Wall height must be positive')
    write_walls(args.map_yaml, args.output, args.height, args.template, args.world_output)
    if args.motion_world:
        write_motion_world(args.world_output, args.motion_world, args.navigation_config, args.height)
