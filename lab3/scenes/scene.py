import numpy as np
from random import random, gauss
from ModernGL import VertexArray
from PyQt6 import QtGui
from PyQt6.QtGui import QMatrix4x4, QVector3D
from matplotlib import pyplot as plt

from ai.ai import get_x, start, generate_f, umpf, ann_train, sigmoid, layers, abs_value
from models.border import Border
from models.plate import Plate
from models.point import Point
from models.predict_plate import PredictPlate
from models.predict_point import PredictPoint
from scenes.default_scene import DefaultScene


class Scene(DefaultScene):
    # Количество образцов данных при обучении ИНС
    A = 3

    # количество временных отсчетов при обучении ИНС
    T = 12

    # Количество нейронов сети
    N = 12

    # период обновления входных данных
    Tu = 4

    # период обновления выходных данных
    Tp = 2

    # гамма, пороговое значение для функции управления при обучении ИНС.
    gm = 20

    def __init__(self, screen):
        super().__init__(
            screen,
            [
                "ИРТСИК. Лабораторная работа №3",
                "R - перезапуск",
                "S - шаг симуляции",
                "B - скрыть границы"
            ]
        )
        self.weight_list = [np.array([[0.2 * random() + 0.9 for _ in range(self.N)] for _ in range(self.N)]) for _
                            in
                            range(self.T)]
        self.bias_list = [np.array([0 for _ in range(self.N)]) for _ in range(self.T)]
        self.reset()
        self.show_border = True

    def get_vaoes(self) -> [VertexArray]:
        vaoes = []
        if self.show_border:
            vaoes += self.border.get_vao_list(self.ctx, self.prog)

        for point in self.optimal_path:
            vaoes += Point(point).get_vao_list(self.ctx, self.prog)
        for point in self.predict_path:
            vaoes += PredictPoint(point).get_vao_list(self.ctx, self.prog)

        vaoes += self.predict_plate.get_vao_list(self.ctx, self.prog)
        vaoes += self.optimal_plate.get_vao_list(self.ctx, self.prog)
        return vaoes

    def point_to_window(self, points):
        matrix = QMatrix4x4()
        matrix.setToIdentity()
        matrix.translate(-1, -1, -1)
        matrix.scale(2.0, 2.0, 2.0)
        vectors = [
            matrix * QVector3D(*point)
            for point in points
        ]
        return [
            (vector.x(), vector.y(), vector.z())
            for vector in vectors
        ]

    def get_optimal_path(self):
        return self.point_to_window(
            get_x(self.T)
        )

    def get_predict_path(self):
        npdata = np.array([[start(i) for i in range(self.T)] for _ in range(self.A)])
        X = get_x(self.T + 2)
        F = generate_f(self.T, self.A)
        labels_dict = {}
        for l in range(1, self.T + 1):
            labels_dict[l - 1] = umpf(l, F, X, self.gm)

        self.weight_list, self.bias_list = ann_train(npdata, labels_dict, self.weight_list, self.bias_list,
                                                     sigmoid, 0.05, 100)
        Ld = layers(npdata, self.weight_list, self.bias_list, sigmoid)
        """ 
        Ld содержит информацию о T шагах движения аппарата. 
        Для каждого шага имеется матрица: ее строки соответствуют 
        экземплярам данных (то есть разным местам или временам испытаний),
        а столбцы - показаниям датчиков в этом испытании.
        Визуализируем абсолютную величину отклонения аппарата от 
        желаемой траектории в каждый момент времени в каждом испытании.
        """
        _data = []
        for a in range(self.A):
            dx = [np.array(Ld[t][a][6:9]) - X[t] for t in range(self.T)]
            abs_dx = list(map(abs_value, dx))
            _data.append(abs_dx)
            # colors = ("red", "green", "blue")
        for a in range(self.A):
            plt.plot(range(self.T), _data[a], label=str(a))
        plt.xlabel("Шаги")
        plt.ylabel("Отклонения")
        plt.title("Движения аппарата в разных испытаниях")
        plt.legend(loc=1)
        plt.grid()
        plt.show()
        return self.point_to_window(
            [Ld[t][2][6:9] for t in range(self.T)]
        )

    def reset(self):
        self.border = Border()

        self.optimal_path = self.get_optimal_path()
        self.optimal_plate = Plate(self.optimal_path[0])

        self.predict_path = self.get_predict_path()
        self.predict_plate = Plate(self.predict_path[0])

        self.update()

    def step_by_path(self):
        self.optimal_plate = Plate(self.optimal_path[0])
        self.optimal_path = self.optimal_path[1:]

        self.predict_plate = PredictPlate(self.predict_path[0])
        self.predict_path = self.predict_path[1:]

        if not (self.optimal_path and self.predict_path):
            self.reset()

        self.update()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
        # r
        if key == 82:
            self.reset()
        # s
        elif key == 83:
            self.step_by_path()
        # b
        elif key == 66:
            self.show_border = not self.show_border
            self.update()
        print(key)
        super().keyPressEvent(event)
