"""Collision checks between the saved navigation rectangle and occupied map cells."""
from math import ceil, cos, floor, hypot, sin

import numpy as np
import yaml
from PIL import Image


class FootprintMap:
    def __init__(self, map_file, footprint):
        data = yaml.safe_load(map_file.read_text())
        self.ox, self.oy, origin_yaw = data['origin']
        if origin_yaw != 0.0:
            raise ValueError('The competition map must be axis aligned.')
        self.res = data['resolution']
        with Image.open(map_file.parent / data['image']) as image:
            gray = np.asarray(image.convert('L'), dtype=float) / 255.0
        probability = gray if data['negate'] else 1 - gray
        self.blocked = np.flipud(probability >= data['free_thresh'])
        vertices = np.asarray(footprint).reshape(-1, 2)
        lower, upper = vertices.min(axis=0), vertices.max(axis=0)
        if not np.allclose(lower, -upper) or len(set(map(tuple, vertices))) != 4 or not np.all(np.abs(vertices) == upper):
            raise ValueError('Expected the centered rectangular competition footprint.')
        self.hx, self.hy = upper
        self.radius = hypot(self.hx, self.hy)

    def intersects(self, x, y, yaw):
        # Separating axis test on grid X/Y and the two vehicle axes, including cell area.
        c, s = cos(yaw), sin(yaw)
        ax, ay = abs(c)*self.hx + abs(s)*self.hy, abs(s)*self.hx + abs(c)*self.hy
        left = floor((x - ax - self.ox) / self.res)
        right = floor((x + ax - self.ox) / self.res)
        bottom = floor((y - ay - self.oy) / self.res)
        top = floor((y + ay - self.oy) / self.res)
        height, width = self.blocked.shape
        if left < 0 or bottom < 0 or right >= width or top >= height:
            return True
        rows, cols = np.nonzero(self.blocked[bottom:top+1, left:right+1])
        if not rows.size:
            return False
        dx = self.ox + (cols + left + 0.5) * self.res - x
        dy = self.oy + (rows + bottom + 0.5) * self.res - y
        pad = 0.5 * self.res * (abs(c) + abs(s))
        return bool(np.any((np.abs(dx*c + dy*s) <= self.hx + pad) &
                           (np.abs(-dx*s + dy*c) <= self.hy + pad)))

    def impending_contact(self, pose, v, w, horizon):
        x, y, yaw = pose
        steps = max(1, ceil((abs(v) + self.radius*abs(w))*horizon / (self.res/2)))
        dt = horizon/steps
        for _ in range(steps):
            midpoint = yaw + w*dt/2
            x += v*dt*cos(midpoint)
            y += v*dt*sin(midpoint)
            yaw += w*dt
            if self.intersects(x, y, yaw):
                return True
        return False
