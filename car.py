from PyQt6.QtCore import QRectF, QPoint, QRect, QPointF
from PyQt6.QtGui import QMatrix3x3, QMatrix4x4, QPainter, QColor, QPolygon, QPolygonF

from settings import START_POSITION_CAR, CAR_WIDTH, CAR_LENGTH, WHEEL_WIDTH, WHEEL_LENGTH


class Car:
    def __init__(self):
        self.body, self.wheels = self.__init_model()

        self.projection = self.__init_matrix()

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
        self.projection.translate(0, -0.1)

    def back(self):
        self.projection.translate(0, 0.1)

    def right(self):
        self.projection.rotate(0.1, 0, 0, 1)

    def left(self):
        self.projection.rotate(-0.1, 0, 0, 1)
