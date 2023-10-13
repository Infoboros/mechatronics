import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent


env = gym.make('CartPole-v0')


model = Sequential()
model.add(Dense(32, input_shape=(4,), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='linear'))


agent = DQNAgent(model=model, nb_actions=2, enable_double_dqn=True, enable_dueling_network=True)


agent.compile(Adam(lr=0.001), metrics=['mae'])
agent.fit(env, nb_steps=5000, visualize=False, verbose=2)


predictions = agent.predict(env)