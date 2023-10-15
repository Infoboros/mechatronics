import gym
import numpy as np
import random
import asyncio

from keras.src.initializers.initializers import RandomUniform
from tensorflow import device
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.losses import MeanSquaredError
from collections import deque

env = gym.make("Acrobot-v1") # Загружаем среду
state = env.reset() # Получаем текущие состояние

REPLAY_MEMORY_SIZE = 50_000 # максимальное количиство данных для обучение ии
MIN_REPLAY_MEMORY_SIZE = 1000 # минимальное количиство данных для обучение ии
UPDATE_TARGET_EVERY = 5 # Через сколько будут обновлятся веса model_target
NUM_EPIZODS = 10 # Всево эпизодоа
max_steps = 10_000 # максимальное количиство шагов в эпизоде


class DQNAgent:

    def __init__(self):
        self.N_action = 2
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.decay_epsilon = -0.0005
        self.gamma = 0.98
        self.update_target = 0

    def create_model(self):
        model = Sequential([
            Dense(100, activation='relu', kernel_initializer=RandomUniform(minval=-0.3, maxval=0.3)),
            Dense(50, activation='relu', kernel_initializer=RandomUniform(minval=-0.3, maxval=0.3)),
            Dense(2, activation='linear')
        ])
        model.compile(optimizer='adam')
        return model

    def update_replay_memory(self, state, state_next, reward, action, done):
        self.memory.append([state, state_next, reward, action, done])

    def act(self, state):
        if random.random() < self.epsilon:
            return random.choice(range(3))
        else:
            action = self.model.predict(state)
            return np.argmax(action)

    def epsilon_minval(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = np.exp(self.decay_epsilon - self.epsilon)
        else:
            self.epsilon = self.epsilon_min

    async def train(self):
        if len(self.memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        queue = asyncio.Queue()
        minibatch = random.sample(self.memory, len(self.memory))
        task = []

        for state, state_next, reward, action, done in minibatch:
            task.append(self.train_precess(state, state_next, reward, action, done))

        await queue.join()
        await asyncio.gather(*task, return_exceptions=True)

        self.update_target += 1

    def update_weights_target_model(self):
        if self.update_target == UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.update_target = 0

    async def train_precess(self, state, state_next, reward, action, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.model.predict(np.array(state)))
            target_q = self.target_model.predict(np.array(state_next))
            target_q[action] = target
            await self.model.fit(np.array(state), np.array(target_q), batch_size=len(self.memory), verbose=0)

agent = DQNAgent()
victory = 0

with device('/gpu:0'):
  for epizod in range(NUM_EPIZODS):
      state = env.reset()

      for step in range(max_steps):
        action = agent.act(state)
        if action > 2:action = 2
        observation, reward, done, info = env.step(int(action))
        print(f'action = {action}, step = {step}')
        agent.update_replay_memory(state, observation, reward, action, done)
        agent.train()
        agent.update_weights_target_model()
        agent.epsilon_minval()

        state = observation
        print(f'reward = {reward}')

        if done:
          victory += 1
          print(f'victory = {victory}')
          break
