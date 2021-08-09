import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from stable_baselines.common.env_checker import check_env
from env import CustomEnv
from generate_candle import GenerateHLOC, Logger

import pandas as pd
import numpy as np
import datetime as dt

# env = CustomEnv()
# # It will check your custom environment and output additional warnings if needed
# check_env(env)


logger = Logger()

generate_ohlc = GenerateHLOC( logger, './executions')
df = generate_ohlc.run()
# df = pd.DataFrame(ohlc)
# df = df.set_index("exec_date") 
print(df['2021-08-01 00:00' : '2021-08-01 23:00'])

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