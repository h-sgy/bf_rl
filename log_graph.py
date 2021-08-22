import pandas as pd
import matplotlib.pyplot as plt

# ログファイルの読み込み(報酬,エピソード,経過時間)
df = pd.read_csv('./logs/monitor.csv', names=['r', 'l', 't'])
df = df.drop(range(2))  # 先頭２行を排除

# 報酬グラフの表示
x = range(len(df['r']))
y = df['r'].astype(float)
plt.plot(x, y)
plt.xlabel('episode')
plt.ylabel('reward')
plt.show()