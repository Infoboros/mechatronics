from math import cos, sin

import numpy as np
from PyQt6.QtGui import QVector3D

from models.model import Model


class Border(Model):
    CIRCLE_RADIANS = 6.29

    UP = 0.3
    UP_RADIUS = 0.5

    DOWN = -0.2
    DOWN_RADIUS = 1.0

    def def_color_const(self):
        self.C1 = (0.0, 0.0)
        self.C2 = (0.5, 0.95)
        self.C3 = (1.0, 0.0)

    def __init__(self):
        self.edge_count = 3
        self.def_color_const()

    @staticmethod
    def get_point_by_angle(angle, radius, y) -> (float, float):
        return (
            radius * cos(angle),
            y,
            radius * sin(angle),
            1.0
        )

    def get_polygons(self) -> [[float]]:
        polygons = []

        # НИЗ
        polygons += self.get_two_polygons(
            (-1, -1, 1),
            (1, -1, 1),
            (1, -1, -1)
        )
        polygons += self.get_two_polygons(
            (1, -1, -1),
            (-1, -1, -1),
            (-1, -1, 1)
        )

        # БЭК
        polygons += self.get_two_polygons(
            (-1, 1, -1),
            (1, 1, -1),
            (1, -1, -1)
        )
        polygons += self.get_two_polygons(
            (1, -1, -1),
            (-1, -1, -1),
            (-1, 1, -1)
        )

        # Право
        polygons += self.get_two_polygons(
            (1, -1, 1),
            (1, 1, 1),
            (1, 1, -1)
        )
        polygons += self.get_two_polygons(
            (1, 1, -1),
            (1, -1, -1),
            (1, -1, 1)
        )

        return polygons
