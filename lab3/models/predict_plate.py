from math import cos, sin

import numpy as np
from PyQt6.QtGui import QVector3D, QMatrix4x4

from models.model import Model


class PredictPlate(Model):
    CIRCLE_RADIANS = 6.29

    UP = 0.02
    UP_RADIUS = 0.02

    DOWN = -0.01
    DOWN_RADIUS = 0.01

    def def_color_const(self):
        self.C1 = (0.0, 0.0)
        self.C2 = (0.5, 0.95)
        self.C3 = (1.0, 0.0)

    def __init__(self, center):
        self.edge_count = 10
        self.center = center

        self.center_matrix = QMatrix4x4()
        self.center_matrix.setToIdentity()
        self.center_matrix.translate(*center)

        self.def_color_const()

    @staticmethod
    def get_point_by_angle(angle, radius, y) -> (float, float):
        return (
            radius * cos(angle),
            y,
            radius * sin(angle),
            1.0
        )

    def transfer_to_center(self, point):
        vector = self.center_matrix * QVector3D(point[0], point[1], point[2])
        return vector.x(), vector.y(), vector.z(), 1.0

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
                    self.transfer_to_center(start_down), self.C3,
                    self.transfer_to_center(end_down), self.C1,
                    self.transfer_to_center((0.0, self.DOWN, 0.0, 1.0)), self.C2
                )
            ))

            bottom_polygons.append(np.array(
                self.add_norm_to_polygon(
                    self.transfer_to_center((0.0, self.DOWN, 0.0, 1.0)), self.C2,
                    self.transfer_to_center(end_down), self.C3,
                    self.transfer_to_center(start_down), self.C1
                )
            ))

            upper_polygons.append(np.array(
                self.add_norm_to_polygon(
                    self.transfer_to_center(start_up), self.C3,
                    self.transfer_to_center(end_up), self.C1,
                    self.transfer_to_center(start_down), self.C2
                )
            ))
            upper_polygons.append(np.array(
                self.add_norm_to_polygon(
                    self.transfer_to_center(end_down), self.C1,
                    self.transfer_to_center(start_down), self.C1,
                    self.transfer_to_center(end_up), self.C1
                )
            ))

            upper_polygons.append(np.array(
                self.add_norm_to_polygon(
                    self.transfer_to_center(start_down), self.C2,
                    self.transfer_to_center(end_up), self.C3,
                    self.transfer_to_center(start_up), self.C1
                )
            ))
            upper_polygons.append(np.array(
                self.add_norm_to_polygon(
                    self.transfer_to_center(end_up), self.C1,
                    self.transfer_to_center(start_down), self.C1,
                    self.transfer_to_center(end_down), self.C1
                )
            ))

        return bottom_polygons + upper_polygons
