
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

# # It will check your custom environment and output additional warnings if needed








logger = Logger()

path = "executions/train/executions.pkl"
if os.path.exists(path):
  df = pd.read_pickle(path)
else:
  generateData = GenerateData( logger, './executions/train')
  generateData.timescale = '1s'
  df = generateData.run()
  df.to_pickle(path)


# df = generateData.run()


# sma atr mom mfi open high low close volume buy_volume sell_volume exec_count buy_exec_count sell_exec_count buy_value sell_value total_value latency

# hige_top = (df['high']-df['close'])/(df['high']-df['low'])*100
# df.insert(0, 'hige_top', hige_top.interpolate(limit_direction='both'))
# hige_bottom = (df['high']-df['open'])/(df['high']-df['low'])*100
# df.insert(0, 'hige_bottom', hige_bottom.interpolate(limit_direction='both'))

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
model_dir = './model/'
# env = Monitor(env, log_dir, allow_early_resets=True)
# env = DummyVecEnv([lambda: env])
env = make_vec_env(lambda: env)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

eval_callback = EvalCallback(env, best_model_save_path=model_dir,
                             log_path='./logs/', eval_freq=1000,
                             deterministic=True, render=False)

checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/',
                                         name_prefix='rl_model')

best_mean_reward = -np.inf # ベスト平均報酬
nupdates = 0 # 更新数
# 更新毎に呼ばれるコールバック
def callback(_locals, _globals):
   global nupdates
   global best_mean_reward
   # print('callback:', nupdates)

   # 10更新毎
   if (nupdates + 1) % 100 == 0:
       # 平均エピソード長、平均報酬の取得
       x, y = ts2xy(load_results(log_dir), 'timesteps')
       if len(x) > 0:
           # 最近10件の平均報酬
           mean_reward = np.mean(y[-100:])

           # 平均報酬がベスト報酬以上の時はエージェントを保存
           update_model = mean_reward > best_mean_reward
           if update_model:
               best_mean_reward = mean_reward
               _locals['self'].save(model_dir + 'best_model.pkl')

           # ログ
           print("time: {}, nupdates: {}, mean: {:.2f}, best_mean: {:.2f}, model_update: {}".format(
               dt.datetime.now(pytz.timezone('Asia/Tokyo')),
               nupdates, mean_reward, best_mean_reward, update_model))
   nupdates += 1
   return True

model.learn(total_timesteps=100000, callback=callback)
# model.save("model/btc_rl")

del model # remove to demonstrate saving and loading

# model = PPO.load("model/best_model")
model = PPO.load(model_dir + "best_model.pkl")

obs = env.reset()
while True:
  action, _states = model.predict(obs, deterministic=True)
  obs, rewards, done, info = env.step(action)
  # env.render()
  if done:
    print("done")
    print(info)
    break
    