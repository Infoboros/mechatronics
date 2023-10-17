from random import random

from PyQt6.QtCore import QPointF, QPoint
from PyQt6.QtGui import QImage, QPixmap, QColor, QPainter

from settings import MAP_WIDTH, MAP_HEIGHT, GRADIENT_RESISTANCE_FILL, WHEEL_LENGTH


class Map:
    BRUSH_RADIUS = 50
    BRUSH_STEP = 0.1

    RESISTANCE_RANGE = 1.0
    RESISTANCE_CENTER = 0.0

    def __init__(self):
        self.break_points = []
        self.w = MAP_WIDTH
        self.h = MAP_HEIGHT
        self._map = {
            x: ({
                y: self.RESISTANCE_CENTER for y in range(self.h)
            })
            for x in range(self.w)
        }
        self._image = QImage(MAP_WIDTH, MAP_HEIGHT, QImage.Format.Format_ARGB32)
        if GRADIENT_RESISTANCE_FILL:
            self.init_gradient()

    def set_resistance(self, x: int, y: int, resistance: float):
        if resistance > self.RESISTANCE_RANGE:
            resistance = self.RESISTANCE_RANGE
        if resistance < -self.RESISTANCE_RANGE:
            resistance = -self.RESISTANCE_RANGE

        self._map[x][y] = resistance

        if resistance > 0:
            green_part = 255 - 255 * resistance / self.RESISTANCE_RANGE
            red_part = 255
        else:
            green_part = 255
            red_part = 255 + 255 * resistance / self.RESISTANCE_RANGE

        self._image.setPixel(x, y, QColor(int(red_part), int(green_part), 0).rgb())

    def get_resistance(self, x: int, y: int) -> float:
        try:
            return self._map[x][y] + self.RESISTANCE_RANGE / 2
        except KeyError:
            return 0

    def get_pixmap(self) -> QPixmap:
        return QPixmap.fromImage(self._image)

    def init_gradient(self):
        x_inc = self.RESISTANCE_RANGE * 2 / self.w
        y_inc = self.RESISTANCE_RANGE * 2 / self.h

        x_res = -self.RESISTANCE_RANGE
        for x in range(self.w):
            y_res = -self.RESISTANCE_RANGE
            for y in range(self.h):
                self.set_resistance(x, y, y_res)
                y_res += y_inc
            x_res += x_inc

    def init_zero(self):
        for x in range(self.w):
            for y in range(self.h):
                self.set_resistance(x, y, 0.5)

    def brush(self, x: int, y: int):
        start_x = 0 if x < self.BRUSH_RADIUS else x - self.BRUSH_RADIUS
        end_x = self.w if x + self.BRUSH_RADIUS > self.w else x + self.BRUSH_RADIUS

        start_y = 0 if y < self.BRUSH_RADIUS else y - self.BRUSH_RADIUS
        end_y = self.h if y + self.BRUSH_RADIUS > self.h else y + self.BRUSH_RADIUS
        for x in range(start_x, end_x):
            for y in range(start_y, end_y):
                resistance = self.get_resistance(x, y)
                if (resistance + self.BRUSH_STEP) > self.RESISTANCE_RANGE:
                    self.set_resistance(x, y, -self.RESISTANCE_RANGE)
                else:
                    self.set_resistance(x, y, resistance + self.BRUSH_STEP)

    def set_trace(self, point: QPointF, color: str = 'black'):
        self._image.setPixel(
            point.toPoint(),
            QColor(color).rgb()
        )

    def paint_break_points(self, painter: QPainter):
        painter.setBrush(QColor('purple'))
        for break_point, *_ in self.break_points:
            painter.drawEllipse(
                break_point,
                WHEEL_LENGTH,
                WHEEL_LENGTH
            )

    def add_break_point(self, point: QPoint, ml, mr):
        self.break_points.append((point, ml, mr))

    def clean_break_points(self):
        self.break_points = []
