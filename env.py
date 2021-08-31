import gym
import numpy as np
from enum import Enum
from gym import spaces
from generate_data import GenerateData, Logger
import datetime as dt
import matplotlib.pyplot as plt
import scipy.stats
# from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import random
import talib as ta



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
    self._window_size = 300
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = spaces.Discrete(3)
    # Example for using image as input:
    self.executions = df
    self.scaler_mixmax = preprocessing.MinMaxScaler()
    self.prices, self.signal_features = self._process_data()
    self.observation_space = spaces.Box(
      low=0,
      high=1,
      shape=(self._window_size, self.signal_features.shape[1]),
      dtype=np.float16,
    )
    self._done = False
    self._position = None
    self._current_tick = None
    self._last_trade_tick = None

    self._start_tick = self._window_size
    self._end_tick = self.executions.shape[0] - self._window_size
    self._total_reward = None
    self._total_profit = None
    self._position_history = None
    self._first_rendering = None

    self._pos = None

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
    self._pos = 0
    return self._observe()

  def step(self, action):
    # self.trade_fee = random.randrange(250, 500, 1)
    self._done = False
    self._current_tick += 60

    if self._current_tick == self._end_tick:
      self._done = True
      
    step_reward = self._calculate_reward(action)
    self._total_reward += step_reward

    self._update_profit(action)
    
    if action == Actions.Flat.value:  # flat
      pass
    elif action == Actions.Buy.value and self._pos <= 0:  # buy
      self._pos += 1
    elif action == Actions.Sell.value and self._pos >= 0:  # sell
      self._pos -= 1

    if self._pos == 0 :
      self._position = Positions.Flat
    elif self._pos > 0:
      self._position = Positions.Long
      self._last_trade_tick = self._current_tick
    elif self._pos < 0:
      self._position = Positions.Short
      self._last_trade_tick = self._current_tick

    self._position_history.append(self._position)
    observation = self._observe()
    info = dict(
        total_reward = self._total_reward,
        total_profit = self._total_profit,
        position = self._position.value,
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
        current_price = self.prices[self._current_tick]
        last_trade_price = self.prices[self._last_trade_tick]
        price_diff = current_price - last_trade_price
        if self._position == Positions.Short:
            step_reward += 1
        elif self._position == Positions.Long:
            step_reward += 1

        step_reward += 0.01

        

    # else:
    #   step_reward += -0.01

    return step_reward

  def _observe(self):
    return self.signal_features[(self._current_tick-self._window_size):self._current_tick]

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
      open = self.executions['open'].interpolate(limit_direction='both')
      high = self.executions['high'].interpolate(limit_direction='both')
      low = self.executions['low'].interpolate(limit_direction='both')
      close = self.executions['close'].interpolate(limit_direction='both')
      volume = self.executions['volume'].interpolate(limit_direction='both')
      exec_count = self.executions['exec_count'].interpolate(limit_direction='both')
      latency = self.executions['latency'].interpolate(limit_direction='both')
      hige_top = (high-close)/(high-low)*100
      hige_top = hige_top.fillna(0).to_numpy()
      hige_bottom = (high-open)/(high-low)*100
      hige_bottom = hige_bottom.fillna(0).to_numpy()

      prices = close.to_numpy()

      # prices = self.executions.fillna(method='ffill').loc[:, 'close'].to_numpy()


      # mfi = ta.MFI(high, low, close, volume, timeperiod=14).interpolate(limit_direction='both').to_numpy()
      # mom = ta.MOM(df['close'], timeperiod=10)
      # df.insert(0, 'mom', mom)


      # # atr = ta.ATR(df['high'], df['low'], df['close'], timeperiod=10).interpolate(limit_direction='both')
      # # df.insert(0, 'atr', atr)

      sma = ta.SMA(close).interpolate(limit_direction='both').to_numpy()

        
      diff = np.insert(np.diff(prices), 0, 0)

      diff = self.scaler_mixmax.fit_transform(diff.reshape(-1,1))
      prices = self.scaler_mixmax.fit_transform(prices.reshape(-1,1))
      signal_features = np.column_stack((prices, diff))
      # signal_features = np.column_stack((signal_features, mfi))
      # signal_features = np.column_stack((signal_features, sma))
      # signal_features = np.column_stack((signal_features, exec_count))
      # signal_features = np.column_stack((signal_features, latency))
      # signal_features = np.column_stack((signal_features, hige_top))
      # signal_features = np.column_stack((signal_features, hige_bottom))
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
