import random

import numpy as np
import rl.core as krl
import rl.callbacks
from PyQt6.QtCore import QEventLoop
from keras import Sequential, Input, Model
from keras.src.layers import Flatten, Dense, Activation, Concatenate, Normalization, BatchNormalization
from keras.src.optimizers import Adam
from math import sqrt, cos, sin, pi
from numpy import arccos
from rl.agents import DDPGAgent
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.random import GaussianWhiteNoiseProcess, OrnsteinUhlenbeckProcess

from car import Car
from map import Map
from settings import START_POSITION_CAR, MAP_WIDTH, MAP_HEIGHT


class ActionSpace(krl.Space):
    def __init__(self):
        self.shape = (4,)

    def sample(self, seed=None):
        if seed:
            random.seed(seed)
        return np.array(
            [
                random.random()
                for _ in range(4)
            ]
        )

    def contains(self, x):
        return len(x) == 4


class ObservationSpace(krl.Space):
    def __init__(self):
        self.shape = (12,)  #

    def sample(self, seed=None):
        pass

    def contains(self, x):
        pass


class CarEnv(krl.Env):
    # TODO
    reward_range = (-75, 75)  # (-np.inf, np.inf)

    def __init__(self, map: Map, trace_color: str, scene):
        self.map = map
        self.trace_color = trace_color
        self.scene = scene
        self.car = Car(map, trace_color)
        self.action_space = ActionSpace()
        self.path = self.map.break_points
        self.observation_space = ObservationSpace()
        self.reset()

    def observe_area(self):
        params = (
            self.car.x,
            self.car.y,
            self.car.alfa,
            self.car.vbl,
            self.car.vbr,
            self.car.u,
            self.car.w,
            self.car.ipsilon,
            self.car.sign(self.car.vbl) * self.car.get_mk(self.car.left_wheel_center),
            self.car.sign(self.car.vbr) * self.car.get_mk(self.car.right_wheel_center),
            self.map.break_points[0].x() if self.map.break_points else 0.,
            self.map.break_points[0].y() if self.map.break_points else 0.,
        )
        area = np.zeros(12)
        for index, el in enumerate(params):
            area[index] = el
        return area

    def reset(self):
        self.car = Car(self.map, self.trace_color)
        self.map.break_points = self.path
        return self.observe_area()

    def step(self, action):
        prev_x, prev_y = self.car.x, self.car.y
        if action == 0:
            self.car.forward()
        elif action == 1:
            self.car.left()
        elif action == 2:
            self.car.right()
        elif action == 3:
            self.car.back()
        self.car.step_car()

        if not self.map.break_points:
            done = True
            reward = 100
        elif (self.car.x < 0) or (self.car.x > MAP_WIDTH * self.car.MAP_DIV) or (self.car.y < 0) or (
                self.car.y > MAP_HEIGHT * self.car.MAP_DIV):
            done = True
            reward = -50
        else:
            point = self.map.break_points[0]
            distance = sqrt(
                (point.x() - self.car.x / self.car.MAP_DIV) ** 2 + (point.y() - self.car.y / self.car.MAP_DIV) ** 2)
            prev_distance = sqrt(
                (point.x() - prev_x / self.car.MAP_DIV) ** 2 + (point.y() - prev_y / self.car.MAP_DIV) ** 2)
            if distance < 50:
                self.map.break_points = self.map.break_points[1:]
            if prev_distance > distance:
                reward = 0
                tdv_x, tdv_y = (point.x() - self.car.x / self.car.MAP_DIV), (point.y() - self.car.y / self.car.MAP_DIV)
                idv_x, idv_y = cos(self.car.ipsilon), sin(self.car.ipsilon)
                diff_angel = np.arccos(
                    (idv_x * tdv_x + idv_y * tdv_y) / sqrt(tdv_x ** 2 + tdv_y ** 2) / sqrt(idv_x ** 2 + idv_y ** 2))
                reward += (pi - diff_angel) * (50 / pi)
            elif abs(prev_distance - distance) < 1:
                reward = -50
            else:
                reward = -25
            done = False


        self.scene.update()
        QEventLoop().processEvents(QEventLoop.ProcessEventsFlag.AllEvents)
        return self.observe_area(), reward, done, {}

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass


def get_agent(env):
    nb_actions = env.action_space.shape[0]

    input = Input(shape=(1, env.observation_space.shape[0]))
    x = Flatten()(input)
    x = Dense(16, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    output = Dense(nb_actions, activation='linear')(x)
    model = Model(inputs=input, outputs=output)
    print(model.summary())


    # Keras-RL предоставляет нам класс, rl.memory.SequentialMemory
    # где хранится "опыт" агента:
    memory = SequentialMemory(limit=100000, window_length=1)

    # random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0.)
    # Создаем agent из класса DDPGAgent
    # agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
    #                   memory=memory,
    #                   random_process=random_process, gamma=.99, target_model_update=1)
    # agent = DQNAgent(model=actor, memory=memory, nb_actions=nb_actions)
    # agent.compile(Adam())
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=10000)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['accuracy'])
    return dqn, model

