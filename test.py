
import warnings
warnings.simplefilter('ignore', FutureWarning)
import gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy

from env import CustomEnv
from generate_data import GenerateData, Logger

import pandas as pd
import numpy as np
import datetime as dt
import os
import talib as ta
import pytz

logger = Logger()

path = "executions/test/test.pkl"
if os.path.exists(path):
  df = pd.read_pickle(path)
else:
  generateData = GenerateData( logger, './executions/test')
  generateData.timescale = '1s'
  df = generateData.run()
  df.to_pickle(path)

env = CustomEnv(df)

# Parallel environments
log_dir = './logs/'
model_dir = './model/'
env = Monitor(env, log_dir, allow_early_resets=True)
# env = DummyVecEnv([lambda: env])
env = make_vec_env(lambda: env)
# model.save("model/btc_rl")
model = PPO.load(model_dir + "best_model.pkl")

obs = env.reset()
while True:
  action, _states = model.predict(obs, deterministic=True)
  obs, rewards, done, info = env.step(action)
  env.render()
  if done:
    print("done")
    print(info)
    break
    