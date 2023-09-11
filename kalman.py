from random import random

from map import Map
from trace import TraceCar


class KalmanCar(TraceCar):
    SENSOR_VARIANCE = 1 ** 2
    FRICTION_DIRECTION_VARIANCE = 5 ** 2
    FRICTION_ANGEL_VARIANCE = 0.01 ** 2

    def __init__(self, map: Map, trace_color: str):
        super().__init__(map, trace_color)
        self.Ekd = self.SENSOR_VARIANCE
        self.Eka = self.SENSOR_VARIANCE

    def kalman(self, Xnew, Xsensor, variance, Ek):
        Ek = self.SENSOR_VARIANCE * (Ek + variance) / (Ek + variance + self.SENSOR_VARIANCE)
        k = Ek / self.SENSOR_VARIANCE
        return k * Xsensor + (1 - k) * Xnew, Ek

    def friction_force(self):
        self.with_abs = True
        abs_direction, abs_angel = super().friction_force()

        self.with_abs = False
        force_direction, force_angel = super().friction_force()

        def get_sensor_data(value):
            return value + (random() * self.SENSOR_VARIANCE * 2 - self.SENSOR_VARIANCE)

        sensor_direction, sensor_angel = get_sensor_data(abs_direction), get_sensor_data(abs_angel)

        kalman_direction, self.Ekd = self.kalman(force_direction, sensor_direction, self.FRICTION_DIRECTION_VARIANCE,
                                                 self.Ekd)
        kalman_angel, self.Eka = self.kalman(force_angel, sensor_angel, self.FRICTION_ANGEL_VARIANCE, self.Eka)

        return kalman_direction, kalman_angel
