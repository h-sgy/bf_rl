import gym
import numpy as np
from enum import Enum
from gym import spaces
from generate_data import GenerateData, Logger
import datetime as dt
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.preprocessing import MinMaxScaler



class Actions(Enum):
    Flat = 0
    Buy = 1
    Sell = 2

class Positions(Enum):
    Flat = 0
    Short = 1
    Long = 2

class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, df):
    super(CustomEnv, self).__init__()
    self._window_size = 600
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = spaces.Discrete(3)
    # Example for using image as input:
    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(600, 7), dtype=np.float32)
    self._done = False
    self._position = None
    self.executions = df
    self._current_tick = None
    self._last_trade_tick = None

    self._start_tick = self._window_size
    self._end_tick = self.executions.shape[0] - self._window_size
    self._total_reward = None
    self._total_profit = None
    self._position_history = None
    self._first_rendering = None
    self.prices, self.signal_features = self._process_data()

    self.trade_fee = 500

  def reset(self):
    self._done = False
    self._position = Positions.Flat
    self._current_tick = self._window_size
    self._last_trade_tick = self._current_tick - self._window_size
    self._total_reward = 0.
    self._total_profit = 0.  # unit
    self._first_rendering = True
    self._position_history = (self._window_size * [None]) + [self._position]
    return self._observe()

  def step(self, action):
    self._done = False
    self._current_tick = self._current_tick + self._window_size

    if self._current_tick == self._end_tick:
      self._done = True
      
    step_reward = self._calculate_reward(action)
    self._total_reward += step_reward

    self._update_profit(action)
    
    if self._total_profit < -100000:
      self._done = True

    if action == 0:  # flat
      self._position = Positions.Flat
    elif action == 1:  # buy
      self._position = Positions.Long
      self._last_trade_tick = self._current_tick
    elif action == 2:  # sell
      self._position = Positions.Short
      self._last_trade_tick = self._current_tick

    self._position_history.append(self._position)
    observation = self._observe()
    info = dict(
        total_reward = self._total_reward,
        total_profit = self._total_profit,
        position = self._position.value
    )
    return observation, step_reward, self._done, info

  def _calculate_reward(self, action):
    # 報酬を返す。
    step_reward = 0  # pip

    trade = False
    if ((action == Actions.Buy.value and self._position == Positions.Short) or
        (action == Actions.Sell.value and self._position == Positions.Long)):
        trade = True

    if trade:
        current_price = self.executions.iloc[self._current_tick]['close']
        last_trade_price = self.executions.iloc[self._last_trade_tick]['close']
        # price_diff = current_price - last_trade_price - self.trade_fee
        # diff = round(price_diff/100000, 1)
        # if price_diff > 0:
        #   step_reward += 1.0 + diff
        # else:
        #   step_reward += -1.0 + diff
        price_diff = current_price - last_trade_price
        if self._position == Positions.Short:
            step_reward += (-price_diff + self.trade_fee ) / 10000
        elif self._position == Positions.Long:
            step_reward += (price_diff - self.trade_fee) / 10000

    else:
      step_reward += -0.01

    return step_reward

  def _observe(self):
    executions = self.executions[self._current_tick - self._window_size : self._current_tick].fillna(0)
    observation = [np.append(exe[8:12], [exe[1],exe[2],exe[15]]) for exe in executions.values]
    mms = MinMaxScaler()
    observation = mms.fit_transform(observation)
    return np.array(observation)

  def render(self, mode='human'):

      def _plot_position(position, tick):
          color = None
          if position == Positions.Short:
              color = 'red'
          elif position == Positions.Long:
              color = 'green'
          elif position == Positions.Flat:
              color = 'gray'
          if color:
              plt.scatter(tick, self.prices[tick], color=color)

      if self._first_rendering:
          self._first_rendering = False
          plt.cla()
          plt.plot(self.prices)
          start_position = self._position_history[self._start_tick]
          _plot_position(start_position, self._start_tick)

      _plot_position(self._position, self._current_tick)

      plt.suptitle(
          "Total Reward: %.6f" % self._total_reward + ' ~ ' +
          "Total Profit: %.6f" % self._total_profit
      )

      plt.pause(0.01)

  def close (self):
    return

  def _process_data(self):
      prices = self.executions.fillna(method='ffill').loc[:, 'close'].to_numpy()

      diff = np.insert(np.diff(prices), 0, 0)
      signal_features = np.column_stack((prices, diff))

      return prices, signal_features

  def _update_profit(self, action):
    trade = False
    if ((action == Actions.Buy.value and self._position == Positions.Short) or
        (action == Actions.Sell.value and self._position == Positions.Long)):
        trade = True

    if trade or self._done:
      current_price = self.prices[self._current_tick]
      last_trade_price = self.prices[self._last_trade_tick]

      if self._position == Positions.Short:
        self._total_profit += (last_trade_price - current_price - self.trade_fee)

      elif self._position == Positions.Long:
        self._total_profit += (current_price - last_trade_price - self.trade_fee)

  def zscore(x, axis = None):
      xmean = x.mean(axis=axis, keepdims=True)
      xstd  = np.std(x, axis=axis, keepdims=True)
      zscore = (x-xmean)/xstd
      return zscore