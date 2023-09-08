from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QMatrix4x4, QPainter, QColor

from settings import START_POSITION_CAR, CAR_WIDTH, CAR_LENGTH, WHEEL_WIDTH, WHEEL_LENGTH


class Car:
    MAX_SPEED = 1000
    MAX_ANGEL = 500

    def __init__(self):
        self.body, self.wheels = self.__init_model()

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

    @staticmethod
    def friction_force(value: int):
        if value > 0:
            return value - 5
        if value < 0:
            return value + 5
        return value

    def step(self):
        self.direction = self.friction_force(self.direction)
        self.angel_direction = self.friction_force(self.angel_direction)

        self.projection.translate(0, -self.direction / 500.)
        self.projection.rotate(self.angel_direction / 500., 0, 0, 1)

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

        return get_polygon(
            width_offset,
            length_offset,
            CAR_WIDTH,
            CAR_LENGTH
        ), [
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

    def map_polygon(self, points: [QPointF]) -> [QPointF]:
        return [
            self.projection * point
            for point in points
        ]

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

    def forward(self):
        self.left_wheel()
        self.right_wheel()

    def back(self):
        self.reverse_wheel()

    def right(self):
        self.left_wheel()

    def left(self):
        self.right_wheel()
