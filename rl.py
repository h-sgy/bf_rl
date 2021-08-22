
import warnings
warnings.simplefilter('ignore', FutureWarning)
import gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback

from env import CustomEnv
from generate_data import GenerateData, Logger

import pandas as pd
import numpy as np
import datetime as dt
import os
import talib as ta

# # It will check your custom environment and output additional warnings if needed








logger = Logger()

path = "executions/executions.pkl"
if os.path.exists(path):
  df = pd.read_pickle(path)
else:
  generateData = GenerateData( logger, './executions')
  generateData.timescale = '500ms'
  df = generateData.run()
  df.to_pickle(path)


# df = generateData.run()
df['open'] = df['open'].fillna(method='ffill')
df['high'] = df['high'].fillna(method='ffill')
df['low'] = df['low'].fillna(method='ffill')
df['close'] = df['close'].fillna(method='ffill')
df['volume'] = df['volume'].fillna(method='ffill')
mfi = ta.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
df.insert(0, 'mfi', mfi)
mom = ta.MOM(df['close'], timeperiod=10)
df.insert(0, 'mom', mfi)

env = CustomEnv(df)
# check_env(env)
# y = 2021
# m = 8
# d = 3
# start = dt.datetime(y, m, d, 0, 0)
# end = dt.datetime(y, m, d, 23, 59) + dt.timedelta(days=1)

# print(df[start.strftime("%Y/%m/%d %H:%M") : end.strftime("%Y/%m/%d %H:%M")])
# time = df[-1:].index[0]
# print(time)
# print(df[time.strftime("%Y/%m/%d %H:%M")])
# print(df[time.strftime("%Y/%m/%d %H:%M")][-1:]['price'])
# print(df[-1:].index[0] + dt.timedelta(minutes=1))
# env = gym.make('CartPole-v1')
# # Optional: PPO2 requires a vectorized environment to run
# # the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

# model = PPO2(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=10000)

# obs = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()

# ----------------------------------------

# Parallel environments
log_dir = './logs/'
env = Monitor(env, log_dir, allow_early_resets=True)
# env = DummyVecEnv([lambda: env])
env = make_vec_env(lambda: env)
model = PPO("MlpPolicy", env, verbose=1)
eval_callback = EvalCallback(env, best_model_save_path='./model/',
                             log_path='./logs/', eval_freq=500,
                             deterministic=True, render=False)
model.learn(total_timesteps=25000, callback=eval_callback)
# model.save("model/btc_rl")

del model # remove to demonstrate saving and loading

model = PPO.load("model/best_model")

obs = env.reset()
while True:
  action, _states = model.predict(obs, deterministic=True)
  obs, rewards, done, info = env.step(action)
  # env.render()
  if done:
    print("done")
    print(info)
    break
    