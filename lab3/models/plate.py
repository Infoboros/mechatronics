from math import cos, sin

import numpy as np
from PyQt6.QtGui import QVector3D

from models.model import Model


class Plate(Model):
    CIRCLE_RADIANS = 6.29

    UP = 0.1
    UP_RADIUS = 0.7

    DOWN = -0.1
    DOWN_RADIUS = 0.3

    def def_color_const(self):
        self.C1 = (0.0, 0.0)
        self.C2 = (0.5, 0.95)
        self.C3 = (1.0, 0.0)

    def __init__(self):
        self.edge_count = 10
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
        bottom_polygons = []
        upper_polygons = []

        step = self.CIRCLE_RADIANS / float(self.edge_count)

        for index in range(self.edge_count):
            start_angle_up = step * index
            end_angle_up = start_angle_up + step

            start_angle_down = (start_angle_up + end_angle_up) / 2.0
            end_angle_down = start_angle_down + step

            start_up = self.get_point_by_angle(start_angle_up, self.UP_RADIUS, self.UP)
            end_up = self.get_point_by_angle(end_angle_up, self.UP_RADIUS, self.UP)

            start_down = self.get_point_by_angle(start_angle_down, self.DOWN_RADIUS, self.DOWN)
            end_down = self.get_point_by_angle(end_angle_down, self.DOWN_RADIUS, self.DOWN)

            bottom_polygons.append(np.array(
                self.add_norm_to_polygon(
                    start_down, self.C3,
                    end_down, self.C1,
                    (0.0, self.DOWN, 0.0, 1.0), self.C2
                )
            ))

            bottom_polygons.append(np.array(
                self.add_norm_to_polygon(
                    (0.0, self.DOWN, 0.0, 1.0), self.C2,
                    end_down, self.C3,
                    start_down, self.C1
                )
            ))

            upper_polygons.append(np.array(
                self.add_norm_to_polygon(
                    start_up, self.C3,
                    end_up, self.C1,
                    start_down, self.C2
                )
            ))
            upper_polygons.append(np.array(
                self.add_norm_to_polygon(
                    end_down, self.C1,
                    start_down, self.C1,
                    end_up, self.C1
                )
            ))

            upper_polygons.append(np.array(
                self.add_norm_to_polygon(
                    start_down, self.C2,
                    end_up, self.C3,
                    start_up, self.C1
                )
            ))
            upper_polygons.append(np.array(
                self.add_norm_to_polygon(
                    end_up, self.C1,
                    start_down, self.C1,
                    end_down, self.C1
                )
            ))

        return bottom_polygons + upper_polygons
