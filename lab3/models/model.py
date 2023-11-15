from abc import ABC, abstractmethod

import numpy as np
from math import sqrt


class Model(ABC):

    @staticmethod
    def sub_lists(a: [float], b: [float]):
        return [
            a_item - b[index]
            for index, a_item in enumerate(a)
        ]

    @staticmethod
    def get_norm(a: [float], b: [float], c: [float]):
        v1 = Model.sub_lists(a, b)
        v2 = Model.sub_lists(b, c)

        N = [
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0]
        ]

        norma = sqrt(
            pow(v1[1] * v2[2] - v1[2] * v2[1], 2) +
            pow(v1[2] * v2[0] - v1[0] * v2[2], 2) +
            pow(v1[0] * v2[1] - v1[1] * v2[0], 2)
        )
        return tuple([
            n / norma
            for n in N
        ])

    @staticmethod
    def add_norm_to_polygon(*polygon: [[float]]) -> [float]:
        normal = Model.get_norm(*polygon[::2])

        return \
                polygon[0] + polygon[1] + normal + \
                polygon[2] + polygon[3] + normal + \
                polygon[4] + polygon[5] + normal

    @abstractmethod
    def get_polygons(self) -> [[float]]:
        raise NotImplemented()

    def get_vao_list(self, ctx, prog) -> []:
        return [
            ctx.simple_vertex_array(
                prog,
                ctx.buffer(polygon.astype(f'f4').tobytes()),
                ['vert', 'tex_coord', 'normal']
            )
            for polygon in self.get_polygons()
        ]

    def get_two_polygons(self, a, b, c):
        return [
            np.array(
                self.add_norm_to_polygon(
                    (*a, 1.0), self.C3,
                    (*b, 1), self.C1,
                    (*c, 1.0), self.C2
                )
            ),
            np.array(
                self.add_norm_to_polygon(
                    (*c, 1.0), self.C2,
                    (*b, 1), self.C1,
                    (*a, 1.0), self.C3,
                )
            )
        ]
