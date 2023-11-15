from ModernGL import VertexArray
from PyQt6 import QtGui

from models.border import Border
from models.plate import Plate
from scenes.default_scene import DefaultScene


class Scene(DefaultScene):

    def __init__(self, screen):
        super().__init__(
            screen,
            [
                "ИРТСИК. Лабораторная работа №3"
            ]
        )
        self.plate = Plate()
        self.border = Border()

    def get_vaoes(self) -> [VertexArray]:
        return self.plate.get_vao_list(self.ctx, self.prog) + self.border.get_vao_list(self.ctx, self.prog)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        super().keyPressEvent(event)
