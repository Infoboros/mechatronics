from OpenGL import GL
from PyQt6.QtWidgets import QApplication

from scenes.scene import Scene

if __name__ == '__main__':
    app = QApplication([])
    screens = app.screens()

    widget = Scene(
        screens[-1]
    )
    widget.show()
    app.exec()
