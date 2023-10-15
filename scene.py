import time
import typing
from datetime import datetime
from time import sleep

from PyQt6 import QtGui
from PyQt6.QtCore import QRect, QPoint, Qt, QRectF, QThread, QEventLoop
from PyQt6.QtGui import QPainter, QBrush, QPen, QColor, QPixmap
from PyQt6.QtWidgets import QMainWindow
import numpy as np
from PIL import Image

from car import Car
from kalman import KalmanCar
from map import Map
from ml import get_agent, CarEnv
from settings import MAP_WIDTH, MAP_HEIGHT, FRAME_TIME
from trace import TraceCar


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Модель тележки")
        self.setFixedWidth(MAP_WIDTH)
        self.setFixedHeight(MAP_HEIGHT)

        self.map = Map()
        self.cars = []

        self.run = False
        self.active_events = {
            'forward': False,
            'right': False,
            'left': False,
            'back': False
        }

    def activate_event(self, event: str):
        self.active_events[event] = True

    def deactivate_event(self, event: str):
        self.active_events[event] = False

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
        self.map.paint_break_points(painter)
        for car in self.cars:
            if type(car) is CarEnv:
                car.car.draw(painter)
            else:
                car.draw(painter)

    def mouseReleaseEvent(self, mouse_event: typing.Optional[QtGui.QMouseEvent]) -> None:
        point = mouse_event.pos()
        button = mouse_event.button()

        if button is Qt.MouseButton.LeftButton:
            self.map.brush(point.x(), point.y())
        elif button is Qt.MouseButton.RightButton:
            self.map.add_break_point(point)

        self.update()

    def start(self):
        self.run = True
        prev_frame = datetime.now()
        while self.run:
            if (datetime.now() - prev_frame).microseconds < FRAME_TIME:
                continue
            prev_frame = datetime.now()
            for car in self.cars:
                if type(car) is CarEnv:
                    car = car.car
                car.step_car()
                if self.active_events['forward']:
                    car.forward()

                if self.active_events['left']:
                    car.left()

                if self.active_events['right']:
                    car.right()

                if self.active_events['back']:
                    car.back()

            self.update()
            QEventLoop().processEvents(QEventLoop.ProcessEventsFlag.AllEvents)

    def keyPressEvent(self, a0: typing.Optional[QtGui.QKeyEvent]) -> None:
        key = a0.key()
        if self.run:
            if key == 16777234:
                self.activate_event('left')
            elif key == 16777235:
                self.activate_event('forward')
            elif key == 16777236:
                self.activate_event('right')
            elif key == 16777237:
                self.activate_event('back')
        # r
        if key == 82:
            self.cars.append(
                Car(self.map, 'red', True)
            )
            self.update()
        # t
        if key == 84:
            self.cars.append(
                TraceCar(self.map, 'black', True)
            )
            self.update()
        # y
        if key == 89:
            self.cars.append(
                TraceCar(self.map, 'blue')
            )
            self.update()
        # u
        if key == 85:
            self.cars.append(
                KalmanCar(self.map, 'purple')
            )
            self.update()
        # z
        if key == 90:
            self.cars = []
        # space
        if key == 32:
            self.start()
        # b
        if key == 66:
            self.map.clean_break_points()

        # m
        if key == 77:
            self.ml_init()
        # s
        if key == 83:
            self.ml_training()

    def keyReleaseEvent(self, a0: typing.Optional[QtGui.QKeyEvent]) -> None:
        key = a0.key()
        if key == 16777234:
            self.deactivate_event('left')
        elif key == 16777235:
            self.deactivate_event('forward')
        elif key == 16777236:
            self.deactivate_event('right')
        elif key == 16777237:
            self.deactivate_event('back')

    def closeEvent(self, a0: typing.Optional[QtGui.QCloseEvent]) -> None:
        self.run = False

    def _get_ml_env(self):
        car_env = CarEnv(self.map, 'yellow', self)
        self.cars.append(car_env)
        return car_env

    def ml_init(self):
        self.agent, self.actor = get_agent(self._get_ml_env())

    def ml_training(self):
        # Обучим процесс на nb_steps шагах,
        # nb_max_episode_steps ограничивает количество шагов в одном эпизоде
        self.agent.fit(
            self._get_ml_env(),
            nb_steps=10000000,
            visualize=True,
            verbose=1,
            nb_max_episode_steps=10000,
            log_interval=10,
            # action_repetition=10
        )
