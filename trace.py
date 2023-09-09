from car import Car
from map import Map


class TraceCar(Car):

    def __init__(self, map: Map, trace_color: str, with_abs: bool = False):
        super().__init__(map, trace_color, with_abs)
        self.steps = [
                         'forward'
                     ] * 100 + \
                     [''] * 300 + \
                     [
                         'right', ''
                     ] * 109 + \
                     [''] * 300 + \
                     [
                         'forward'
                     ] * 100 + \
                     [''] * 300 + \
                     [
                         'left', ''
                     ] * 109 + \
                     [''] * 300 + \
                     [
                         'forward'
                     ] * 100 + \
                     [''] * 300 + \
                     [
                         'right', ''
                     ] * 109 + \
                     [''] * 300 + \
                     [
                         'forward'
                     ] * 100 + \
                     [''] * 300 + \
                     [
                         'left', ''
                     ] * 109 + \
                     [''] * 300 + \
                     [
                         'forward'
                     ] * 100 + \
                     [''] * 300 + \
                     [
                         'right', ''
                     ] * 109 + \
                     [''] * 300 + \
                     [
                         'forward'
                     ] * 100 + \
                     [''] * 300 + \
                     [
                         'left', ''
                     ] * 109 + \
                     [''] * 300 + \
                     [''] * 300 + \
                     [
                         'forward'
                     ] * 100 + \
                     [''] * 300 + \
                     [
                         'right', ''
                     ] * 109 + \
                     [''] * 300 + \
                     [
                         'forward'
                     ] * 100 + \
                     [''] * 300 + \
                     [
                         'left', ''
                     ] * 109 + \
                     [''] * 300 + \
                     [''] * 300 + \
                     [
                         'forward'
                     ] * 100 + \
                     [''] * 300 + \
                     [
                         'right', ''
                     ] * 109 + \
                     [''] * 300 + \
                     [
                         'forward'
                     ] * 100 + \
                     [''] * 300 + \
                     [
                         'left', ''
                     ] * 109 + \
                     [''] * 300
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
