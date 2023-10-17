from car import Car


class GenerateCar(Car):

    def step_car(self):
        with open('dataset.txt', 'a') as ds:
            ds.write(
                ';'.join([
                    str(el)
                    for el in [self.x,
                               self.y,
                               self.alfa,
                               self.vbl,
                               self.vbr,
                               self.u,
                               self.w,
                               self.ipsilon,
                               self.sign(self.vbl) * self.get_mk(self.left_wheel_center),
                               self.sign(self.vbr) * self.get_mk(self.right_wheel_center),
                               self.ml,
                               self.mr
                               ]
                ]) + '\n'
            )
        super().step_car()
