import typing
from PyQt6 import QtGui
from PyQt6.QtCore import QRect, QPoint, Qt
from PyQt6.QtGui import QPainter, QBrush, QPen, QColor
from PyQt6.QtWidgets import QMainWindow

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
        painter.setPen(QPen(Qt.GlobalColor.transparent, 0,
                            Qt.PenStyle.SolidLine))

        for x in range(MAP_WIDTH - 1):
            for y in range(MAP_HEIGHT - 1):
                color = self.map.get_color(x, y)
                painter.setBrush(
                    QBrush(
                        QColor(*color),
                        Qt.BrushStyle.SolidPattern)
                )
                painter.drawRect(
                    QRect(
                        QPoint(x - 1, y - 1),
                        QPoint(x, y)
                    )
                )

    def paintEvent(self, e):
        painter = QPainter(self)
        self.paint_resistance(painter)

    def mouseReleaseEvent(self, a0: typing.Optional[QtGui.QMouseEvent]) -> None:
        point = a0.pos()
        self.map.brush(point.x(), point.y())
        self.update()
