from math import cos, sin

import numpy as np
from PyQt6.QtGui import QVector3D

from models.model import Model


class PredictPoint(Model):
    R = 0.01

    def __init__(self, center):
        self.center = center
        self.C1 = (0.0, 0.0)
        self.C2 = (0.5, 0.95)
        self.C3 = (1.0, 0.0)

    def get_polygons(self) -> [[float]]:
        x, y, z = self.center
        a = (x, y - self.R, z - self.R)
        b = (x + self.R, y - self.R, z + self.R)
        c = (x - self.R, y - self.R, z + self.R)
        d = (x, y + self.R, z)
        polygons = []

        polygons += self.get_two_polygons(a, b, c)
        polygons += self.get_two_polygons(a, b, d)
        polygons += self.get_two_polygons(b, c, d)
        polygons += self.get_two_polygons(c, a, d)
        return polygons
