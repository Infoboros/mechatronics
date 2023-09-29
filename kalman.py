from random import random

from map import Map
from trace import TraceCar


class KalmanCar(TraceCar):
    k = 0.9
    SENSOR_VARIANCE = 0.05 ** 2
    FRICTION_DIRECTION_VARIANCE = 0.5 ** 2
    FRICTION_ANGEL_VARIANCE = 0.5 ** 2

    def __init__(self, map: Map, trace_color: str):
        super().__init__(map, trace_color)
        self.Ekvbl = self.FRICTION_DIRECTION_VARIANCE
        self.Ekvbr = self.FRICTION_DIRECTION_VARIANCE

        self.Ekipsilon = self.FRICTION_ANGEL_VARIANCE
        self.Eku = self.FRICTION_DIRECTION_VARIANCE
        self.Ekw = self.FRICTION_ANGEL_VARIANCE

    def kalman(self, Xnew, Xsensor, variance, Ek):
        Ek = self.SENSOR_VARIANCE * (Ek + variance) / (Ek + variance + self.SENSOR_VARIANCE)
        k = Ek / self.SENSOR_VARIANCE
        return k * Xsensor + (1 - k) * Xnew, Ek

    @staticmethod
    def get_sensor_data(value, variance=None):
        if variance is None:
            variance = KalmanCar.SENSOR_VARIANCE
        if value:
            return value + (random() * variance * 2 - variance)
        return value

    def next_vbl(self):
        self.with_abs = True
        abs_vbl = self.get_sensor_data(super().next_vbl())
        self.with_abs = False
        vbl = super().next_vbl()

        new_vbl, self.Ekvbl = self.kalman(vbl, abs_vbl, self.FRICTION_DIRECTION_VARIANCE, self.Ekvbl)
        return new_vbl

    def next_vbr(self):
        self.with_abs = True
        abs_vbr = self.get_sensor_data(super().next_vbr(), 1e-23)
        self.with_abs = False
        vbr = super().next_vbr()

        new_vbr, self.Ekvbr = self.kalman(vbr, abs_vbr, self.FRICTION_DIRECTION_VARIANCE, self.Ekvbr)
        return new_vbr

    def next_u(self):
        self.with_abs = True
        abs_u = self.get_sensor_data(super().next_u())
        self.with_abs = False
        u = super().next_u()

        new_u, self.Eku = self.kalman(u, abs_u, self.FRICTION_DIRECTION_VARIANCE, self.Eku)
        return new_u


    def next_w(self, prev):
        self.with_abs = True
        abs_w = self.get_sensor_data(super().next_w(prev))
        self.with_abs = False
        w = super().next_w(prev)

        new_w, self.Ekw = self.kalman(w, abs_w, self.FRICTION_DIRECTION_VARIANCE, self.Ekw)
        return new_w