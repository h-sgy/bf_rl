import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from stable_baselines.common.env_checker import check_env
from env import CustomEnv
from generate_data import GenerateData, Logger

import pandas as pd
import numpy as np
import datetime as dt

# env = CustomEnv()
# # It will check your custom environment and output additional warnings if needed
# check_env(env)


logger = Logger()

generateData = GenerateData( logger, './executions')
df = generateData.executions()
# df = generateData.run()

y = 2021
m = 8
d = 3
start = dt.datetime(y, m, d, 0, 0)
end = dt.datetime(y, m, d, 23, 59) + dt.timedelta(days=1)

print(df[start.strftime("%Y/%m/%d %H:%M") : end.strftime("%Y/%m/%d %H:%M")]["size"])
# print(df)
# env = gym.make('CartPole-v1')
# # Optional: PPO2 requires a vectorized environment to run
# # the env is now wrapped automatically when passing it to the constructor
# # env = DummyVecEnv([lambda: env])

# model = PPO2(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=10000)

# obs = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()