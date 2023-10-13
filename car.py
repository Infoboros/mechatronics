from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QMatrix4x4, QPainter, QColor

from math import pi, cos, sin

from map import Map
from settings import START_POSITION_CAR, CAR_WIDTH, CAR_LENGTH, WHEEL_WIDTH, WHEEL_LENGTH


class Car:
    t = 1
    MAP_DIV = 100

    MOMENT_STEP = 1
    MOMENT_REDUCTION_STEP = MOMENT_STEP / 2
    MAX_M = 350
    FRICTION_CONTACT_RADIUS = WHEEL_WIDTH / 2

    # Масса машины
    M = 1000
    # Масса колеса
    W = 20
    # Радиус колеса
    R = 0.5
    # Ширина колеи
    L = 2.5

    G = 9.8

    def __init__(self, map: Map, trace_color: str, with_abs: bool = False):
        self.map = map
        self.trace_color = trace_color
        self.with_abs = with_abs
        self.body, self.wheels, self.left_wheel_center, self.right_wheel_center = self.__init_model()

        self.ml, self.mr = 0, 0
        self.x, self.y = START_POSITION_CAR
        self.x *= self.MAP_DIV
        self.y *= self.MAP_DIV
        self.alfa = 0

        self.vfl, self.vfr, self.vbl, self.vbr = 0, 0, 0, 0
        self.ipsilon = 0
        self.u = 0
        self.w = 0

        # 0 - покой
        # 1 - скольжение
        # 2 - качение
        self.l_state = 0
        self.r_state = 0

    def _moment_reduction(self):
        self.ml = 0
        self.mr = 0

    def next_x(self):
        return self.x + self.t * self.u * cos(self.ipsilon)

    def next_y(self):
        return self.y + self.t * self.u * sin(self.ipsilon)

    def next_alfa(self):
        return self.alfa + self.t * self.w

    def _get_uk(self, center=None):
        if center is None:
            projected_center = (
                        self.get_project_matrix() * ((self.right_wheel_center + self.left_wheel_center) / 2)).toPoint()
        else:
            projected_center = (self.get_project_matrix() * center).toPoint()

        return self.map.get_resistance(projected_center.x(), projected_center.y())

    def _get_I(self, m=None, center=None):
        if m is None:
            m = (self.ml + self.mr) / 2
        uk = self._get_uk(center)
        k = 1
        if uk * (self.M / 4 + self.W) * self.R * self.G < m:
            k = uk + 0.5
        if self.with_abs:
            k = 1
        return (k * self.M / 4 + self.W / 2) * self.R ** 2

    def get_mk(self, center=None):
        uk = self._get_uk(center)
        if self.with_abs:
            uk = 0.5
        return self.M / 4 * self.G * self.R * (uk + 0.7)

    @staticmethod
    def sign(value):
        if value > 0:
            return 1
        if value < 0:
            return -1
        return 0

    def next_vbl(self):
        new_vbl = self.vbl + (self.ml - self.sign(self.vbl) * self.get_mk(self.left_wheel_center)) \
                  / self._get_I(self.ml, self.left_wheel_center) \
                  * self.t
        if new_vbl * self.vbl < 0:
            return 0
        return new_vbl

    def next_vbr(self):
        new_vbr = self.vbr + (self.mr - self.sign(self.vbr) * self.get_mk(self.right_wheel_center)) \
                  / self._get_I(self.mr, self.right_wheel_center) \
                  * self.t
        if new_vbr * self.vbr < 0:
            return 0
        return new_vbr

    def next_impsilon(self):
        return self.ipsilon + (self.vbl - self.vbr) * self.t * self.R / (2 * pi * self.L)

    def next_u(self):
        new_u = self.u + self.R * self.t * \
                (self.vbr if abs(self.vbl) > abs(self.vbr) else self.vbl)

        # TODO
        if abs(self.u):
            new_u -= \
                self.sign(self.u) * self.get_mk() / self._get_I() * self.t / 100

        if new_u * self.u < 0:
            self.w = 0
            return 0
        return new_u

    def next_w(self, prev_ipsilon):
        new_w = self.w + (self.ipsilon - prev_ipsilon) / self.t

        # TODO
        if abs(self.w):
            new_w -= self.sign(self.w) * 0.1

        if new_w * self.w < 0:
            self.ipsilon = self.alfa
            return 0
        return new_w

    def step_car(self):
        self.x = self.next_x()
        self.y = self.next_y()
        self.alfa = self.next_alfa()

        self.vbl = self.next_vbl()
        self.vbr = self.next_vbr()

        prev_ipsilon = self.ipsilon
        self.ipsilon = self.next_impsilon()

        self.u = self.next_u()
        self.w = self.next_w(prev_ipsilon)

        self.map.set_trace(
            self.get_center(),
            self.trace_color
        )
        self._moment_reduction()

    def get_project_matrix(self) -> QMatrix4x4:
        a = QMatrix4x4()
        a.setToIdentity()
        a.rotate(self.alfa * 180 / pi, 0, 0, 1)

        t = QMatrix4x4()
        t.setToIdentity()
        t.translate(self.x / self.MAP_DIV, self.y / self.MAP_DIV)
        return t * a

    @staticmethod
    def __init_model():
        rotate = QMatrix4x4()
        rotate.setToIdentity()
        rotate.rotate(90, 0, 0, 1)

        def get_polygon(x, y, w, l) -> [QPointF]:
            return [
                rotate * QPointF(x, y),
                rotate * QPointF(x + w, y),
                rotate * QPointF(x + w, y + l),
                rotate * QPointF(x, y + l),
            ]

        yield get_polygon(
            -CAR_WIDTH / 2,
            -CAR_LENGTH / 2,
            CAR_WIDTH,
            CAR_LENGTH
        )
        yield [
            get_polygon(
                -CAR_WIDTH / 2 - WHEEL_WIDTH,
                -CAR_LENGTH / 2 - WHEEL_LENGTH / 2,
                WHEEL_WIDTH,
                WHEEL_LENGTH
            ),
            get_polygon(
                CAR_WIDTH / 2,
                -CAR_LENGTH / 2 - WHEEL_LENGTH // 2,
                WHEEL_WIDTH,
                WHEEL_LENGTH
            ),
            get_polygon(
                CAR_WIDTH / 2,
                CAR_LENGTH / 2 - WHEEL_LENGTH / 2,
                WHEEL_WIDTH,
                WHEEL_LENGTH
            ),
            get_polygon(
                -CAR_WIDTH / 2 - WHEEL_WIDTH,
                CAR_LENGTH / 2 - WHEEL_LENGTH / 2,
                WHEEL_WIDTH,
                WHEEL_LENGTH
            )
        ]
        yield rotate * QPointF(
            -CAR_WIDTH / 2 - WHEEL_WIDTH / 2,
            CAR_LENGTH / 2
        )
        yield rotate * QPointF(
            CAR_WIDTH / 2 + WHEEL_WIDTH / 2,
            CAR_LENGTH / 2
        )

    def map_point(self, matrix: QMatrix4x4, point: QPointF) -> QPointF:
        return matrix * point

    def map_polygon(self, matrix: QMatrix4x4, points: [QPointF]) -> [QPointF]:
        return list(
            map(
                lambda point: self.map_point(matrix, point),
                points
            )
        )

    def draw(self, painter: QPainter):
        matrix = self.get_project_matrix()

        painter.setBrush(QColor(self.trace_color))
        painter.drawPolygon(
            self.map_polygon(matrix, self.body)
        )

        painter.setBrush(QColor('black'))
        painter.drawPolygon(
            self.map_polygon(matrix, self.wheels[0])
        )
        painter.drawPolygon(
            self.map_polygon(matrix, self.wheels[1])
        )

        painter.setBrush(QColor('blue'))
        painter.drawPolygon(
            self.map_polygon(matrix, self.wheels[2])
        )
        painter.drawPolygon(
            self.map_polygon(matrix, self.wheels[3])
        )

        painter.setBrush(QColor('red'))
        painter.drawEllipse(
            self.map_point(matrix, self.left_wheel_center),
            self.FRICTION_CONTACT_RADIUS,
            self.FRICTION_CONTACT_RADIUS
        )
        painter.drawEllipse(
            self.map_point(matrix, self.right_wheel_center),
            self.FRICTION_CONTACT_RADIUS,
            self.FRICTION_CONTACT_RADIUS
        )

    def forward(self):
        self.ml = self.MAX_M
        self.mr = self.MAX_M

    def back(self):
        self.ml = -self.MAX_M
        self.mr = -self.MAX_M

    def right(self):
        self.mr = -self.MAX_M / 10

    def left(self):
        self.ml = - self.MAX_M / 10

    def get_center(self) -> QPointF:
        return self.get_project_matrix() * QPointF(0., 0.)
