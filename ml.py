import random

import numpy as np
import rl.core as krl
import rl.callbacks
from PyQt6.QtCore import QEventLoop
from keras import Sequential, Input, Model
from keras.src.layers import Flatten, Dense, Activation, Concatenate, Normalization
from keras.src.optimizers import Adam
from math import sqrt
from rl.agents import DDPGAgent
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.random import GaussianWhiteNoiseProcess, OrnsteinUhlenbeckProcess

from car import Car
from map import Map
from settings import START_POSITION_CAR, MAP_WIDTH


class ActionSpace(krl.Space):
    def __init__(self):
        self.shape = (5,)

    def sample(self, seed=None):
        if seed:
            random.seed(seed)
        return np.array(
            [
                random.random()
                for _ in range(5)
            ]
        )

    def contains(self, x):
        return len(x) == 5


class ObservationSpace(krl.Space):
    def __init__(self):
        self.shape = (10,)  #

    def sample(self, seed=None):
        pass

    def contains(self, x):
        pass


class CarEnv(krl.Env):
    # TODO
    reward_range = (0, 1000)  # (-np.inf, np.inf)

    def __init__(self, map: Map, trace_color: str, scene):
        self.map = map
        self.trace_color = trace_color
        self.scene = scene
        self.car = Car(map, trace_color)
        self.action_space = ActionSpace()
        self.observation_space = ObservationSpace()
        self.reset()

    def observe_area(self):
        params = (
            self.car.x,
            self.car.y,
            self.car.alfa,
            self.car.vbl,
            self.car.vbr,
            self.car.ipsilon,
            self.car.u,
            self.car.w,
            self.car.sign(self.car.vbl) * self.car.get_mk(self.car.left_wheel_center),
            self.car.sign(self.car.vbr) * self.car.get_mk(self.car.right_wheel_center)
        )
        area = np.zeros(10)
        for index, el in enumerate(params):
            area[index] = el
        return area

    def reset(self):
        self.car = Car(self.map, self.trace_color)
        return self.observe_area()

    def step(self, action):
        index_action = list(action).index(max(action))
        print(action)
        if index_action == 0:
            self.car.forward()
        elif index_action == 1:
            self.car.back()
        elif index_action == 2:
            self.car.left()
        elif index_action == 3:
            self.car.right()
        self.car.step_car()

        if not self.map.break_points:
            done = True
            reward = np.inf
        else:
            point = self.map.break_points[0]
            reward = 1000 - sqrt(
                (point.x() - self.car.x / self.car.MAP_DIV) ** 2 + (point.y() - self.car.y / self.car.MAP_DIV) ** 2)
            done = False

        if reward > 900:
            self.map.break_points = self.map.break_points[1:]
        if reward < 0:
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

    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    actor.add(Dense(32))
    actor.add(Activation('sigmoid'))
    actor.add(Dense(16))
    actor.add(Activation('sigmoid'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('softmax'))
    print(actor.summary())

    # Построим модель критика. Подаем среду и действие, получаем награду
    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(8)(x)
    x = Activation('relu')(x)
    x = Dense(16)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    print(critic.summary())

    # Keras-RL предоставляет нам класс, rl.memory.SequentialMemory
    # где хранится "опыт" агента:
    memory = SequentialMemory(limit=100000, window_length=1)

    random_process = GaussianWhiteNoiseProcess(size=nb_actions, mu=0., sigma=.2)
    # Создаем agent из класса DDPGAgent
    # agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
    #                   memory=memory,
    #                   random_process=random_process, gamma=.99, target_model_update=1e-3)
    agent = DQNAgent(model=actor, memory=memory, nb_actions=nb_actions)
    agent.compile(Adam(learning_rate=0.1))

    return agent, actor
