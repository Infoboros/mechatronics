import typing
from PyQt6 import QtGui
from PyQt6.QtCore import QRect, QPoint, Qt, QRectF
from PyQt6.QtGui import QPainter, QBrush, QPen, QColor, QPixmap
from PyQt6.QtWidgets import QMainWindow
import numpy as np
from PIL import Image

from map import Map
from settings import MAP_WIDTH, MAP_HEIGHT


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Модель тележки")
        self.setFixedWidth(MAP_WIDTH)
        self.setFixedHeight(MAP_HEIGHT)

        self.map = Map()

    def paint_resistance(self, painter: QPainter):
        pixmap = self.map.get_pixmap()
        painter.drawTiledPixmap(
            QRect(0, 0, MAP_WIDTH, MAP_HEIGHT),
            pixmap,
            QPoint(0, 0)
        )

    def paintEvent(self, e):
        painter = QPainter(self)
        self.paint_resistance(painter)

    def mouseReleaseEvent(self, a0: typing.Optional[QtGui.QMouseEvent]) -> None:
        point = a0.pos()
        self.map.brush(point.x(), point.y())
        self.update()
