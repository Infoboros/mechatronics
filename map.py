from settings import MAP_WIDTH, MAP_HEIGHT, GRADIENT_RESISTANCE_FILL


class Map:
    BRUSH_RADIUS = 50
    BRUSH_STEP = 0.1

    RESISTANCE_RANGE = 1.0
    RESISTANCE_CENTER = 0.0

    def __init__(self):
        self.w = MAP_WIDTH - 1
        self.h = MAP_HEIGHT - 1
        self._map = {
            x: ({
                y: self.RESISTANCE_CENTER for y in range(self.h)
            })
            for x in range(self.w)
        }
        self._colors = {
            x: ({
                y: (255, 255, 255) for y in range(self.h)
            })
            for x in range(self.w)
        }
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

        self._colors[x][y] = int(red_part), int(green_part), 0

    def get_resistance(self, x: int, y: int) -> float:
        return self._map[x][y]

    def get_color(self, x: int, y: int) -> float:
        return self._colors[x][y]

    def init_gradient(self):
        x_inc = self.RESISTANCE_RANGE * 2 / self.w
        y_inc = self.RESISTANCE_RANGE * 2 / self.h

        x_res = -self.RESISTANCE_RANGE
        for x in range(self.w):
            y_res = -self.RESISTANCE_RANGE
            for y in range(self.h):
                self.set_resistance(x, y, (x_res + y_res) / 2)
                y_res += y_inc
            x_res += x_inc

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
