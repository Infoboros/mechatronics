from car import Car
from map import Map


class TraceCar(Car):

    def __init__(self, map: Map, trace_color: str, with_abs: bool = False):
        super().__init__(map, trace_color, with_abs)
        self.steps = \
            ['left'] * 183 + \
            [''] * 100 + \
            ['forward'] * 200 + \
            [''] * 80 + \
            ['right'] * 365 + \
            [''] * 80 + \
            ['forward'] * 50 + \
            [''] * 130 + \
            ['left'] * 365 + \
            [''] * 100 + \
            ['forward'] * 170 + \
            [''] * 80 + \
            ['right'] * 365 + \
            [''] * 80 + \
            ['forward'] * 30

    def step(self):
        super().step()
        if not self.steps:
            return

        step = self.steps[0]
        if step == 'forward':
            self.forward()
        elif step == 'back':
            self.back()
        elif step == 'left':
            self.left()
        elif step == 'right':
            self.right()

        self.steps = self.steps[1:]
