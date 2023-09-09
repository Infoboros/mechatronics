from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QMatrix4x4, QPainter, QColor

from map import Map
from settings import START_POSITION_CAR, CAR_WIDTH, CAR_LENGTH, WHEEL_WIDTH, WHEEL_LENGTH


class Car:
    MAX_SPEED = 1000
    MAX_ANGEL = 500
    FRICTION_CONTACT_RADIUS = WHEEL_WIDTH / 2

    def __init__(self, map: Map, trace_color: str):
        self.map = map
        self.trace_color = trace_color
        self.body, self.wheels, self.left_wheel_center, self.right_wheel_center = self.__init_model()

        self.projection = self.__init_matrix()

        self.direction = 0
        self.angel_direction = 0

    def left_wheel(self):
        if self.direction < self.MAX_SPEED:
            self.direction += 20
        if self.angel_direction < self.MAX_ANGEL:
            self.angel_direction += 20

    def right_wheel(self):
        if self.direction < self.MAX_SPEED:
            self.direction += 20
        if self.angel_direction > -self.MAX_ANGEL:
            self.angel_direction -= 20

    def reverse_wheel(self):
        if self.direction > -self.MAX_SPEED:
            self.direction -= 50

    def friction_force(self):
        left_friction_point = self.map_point(self.left_wheel_center).toPoint()
        right_friction_point = self.map_point(self.right_wheel_center).toPoint()

        left_friction_c = self.map.get_resistance(
            left_friction_point.x(),
            left_friction_point.y()
        )
        right_friction_c = self.map.get_resistance(
            right_friction_point.x(),
            right_friction_point.y()
        )

        friction_c_avg = (
                                 (left_friction_c + right_friction_c) / 2
                                 +
                                 self.map.RESISTANCE_RANGE
                         ) \
                         / \
                         2 * self.map.RESISTANCE_RANGE

        def abs_diff(value, inc):
            if value > 0:
                return value - inc
            if value < 0:
                return value + inc
            return value

        new_angel = abs_diff(self.angel_direction, 1)
        new_angel += right_friction_c / 100
        new_angel -= left_friction_c / 100

        return \
            abs_diff(self.direction, 5 * friction_c_avg), \
            new_angel

    def step(self):
        self.direction, self.angel_direction = self.friction_force()

        self.projection.translate(0, -self.direction / 500.)
        self.projection.rotate(self.angel_direction / 500., 0, 0, 1)

        self.map.set_trace(
            self.get_center(),
            self.trace_color
        )

    @staticmethod
    def __init_matrix() -> QMatrix4x4:
        matrix = QMatrix4x4()
        matrix.setToIdentity()
        matrix.translate(*START_POSITION_CAR)
        return matrix

    @staticmethod
    def __init_model():
        width_offset = -CAR_WIDTH / 2
        length_offset = -CAR_LENGTH / 2

        def get_polygon(x, y, w, l) -> [QPointF]:
            return [
                QPointF(x, y),
                QPointF(x + w, y),
                QPointF(x + w, y + l),
                QPointF(x, y + l),
            ]

        yield get_polygon(
            width_offset,
            length_offset,
            CAR_WIDTH,
            CAR_LENGTH
        )
        yield [
            get_polygon(
                width_offset - WHEEL_WIDTH,
                length_offset - WHEEL_LENGTH // 2,
                WHEEL_WIDTH,
                WHEEL_LENGTH
            ),
            get_polygon(
                width_offset + CAR_WIDTH,
                length_offset - WHEEL_LENGTH // 2,
                WHEEL_WIDTH,
                WHEEL_LENGTH
            ),
            get_polygon(
                width_offset - WHEEL_WIDTH,
                length_offset + CAR_LENGTH - WHEEL_LENGTH // 2,
                WHEEL_WIDTH,
                WHEEL_LENGTH
            ),
            get_polygon(
                width_offset + CAR_WIDTH,
                length_offset + CAR_LENGTH - WHEEL_LENGTH // 2,
                WHEEL_WIDTH,
                WHEEL_LENGTH
            )
        ]
        yield QPointF(
            width_offset - WHEEL_WIDTH / 2,
            length_offset + CAR_LENGTH
        )
        yield QPointF(
            width_offset + CAR_WIDTH + WHEEL_WIDTH / 2,
            length_offset + CAR_LENGTH
        )

    def map_point(self, point: QPointF) -> QPointF:
        return self.projection * point

    def map_polygon(self, points: [QPointF]) -> [QPointF]:
        return list(map(self.map_point, points))

    def draw(self, painter: QPainter):
        painter.setBrush(QColor('purple'))
        painter.drawPolygon(
            self.map_polygon(self.body)
        )

        painter.setBrush(QColor('black'))
        painter.drawPolygon(
            self.map_polygon(self.wheels[0])
        )
        painter.drawPolygon(
            self.map_polygon(self.wheels[1])
        )

        painter.setBrush(QColor('blue'))
        painter.drawPolygon(
            self.map_polygon(self.wheels[2])
        )
        painter.drawPolygon(
            self.map_polygon(self.wheels[3])
        )

        painter.setBrush(QColor('red'))
        painter.drawEllipse(
            self.map_point(self.left_wheel_center),
            self.FRICTION_CONTACT_RADIUS,
            self.FRICTION_CONTACT_RADIUS
        )
        painter.drawEllipse(
            self.map_point(self.right_wheel_center),
            self.FRICTION_CONTACT_RADIUS,
            self.FRICTION_CONTACT_RADIUS
        )

    def forward(self):
        self.left_wheel()
        self.right_wheel()

    def back(self):
        self.reverse_wheel()

    def right(self):
        self.left_wheel()

    def left(self):
        self.right_wheel()

    def get_center(self) -> QPointF:
        return self.projection.map(QPointF(0., 0.))
