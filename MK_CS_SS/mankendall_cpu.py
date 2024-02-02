import pandas as pd
import numpy as np
from scipy.stats import kendalltau, norm, linregress
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from matplotlib.ticker import FuncFormatter

def format_percent(x, pos):
    """パーセント表記にフォーマットする関数"""
    return f"{x:.1f}%"

def format_time(x, pos):
    """時間表記にフォーマットする関数"""
    hours, remainder = divmod(x, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}"

total_time = 34 * 3600
# CSVファイルのパス
csv_file = 'output3.csv'

# CSVファイルを読み込む
data = pd.read_csv(csv_file, header=None).iloc[:, 2] / 2

# Mann-Kendall テストを実行
tau, p_value = kendalltau(data, np.arange(len(data)))

# Cox-Stuart テストを実行
def cox_stuart_test(data):
    m = len(data) // 2
    signs = np.sign(data[m+1:] - data[:m])
    pos_signs = np.count_nonzero(signs == 1)
    neg_signs = np.count_nonzero(signs == -1)
    n = len(data) - m
    p_value = 2 * min(pos_signs, neg_signs) / n
    return p_value

def modified_cox_stuart_test(x):
    n = len(x)
    signs = np.zeros(n//2)
    for i in range(n//2):
        signs[i] = np.sign(x[i] - x[n//2 + i])
    S = np.sum(signs)
    Z = (S + 1) / ( np.sqrt(n / 2) )
    print(Z)
    p_value = (1 - norm.cdf(np.abs(Z)))
    return p_value

cox_stuart_p_value = modified_cox_stuart_test(data)

print(f"Mann-Kendall p-value: {p_value}")
print(f"Cox-Stuart p-value: {cox_stuart_p_value}")

if p_value < 0.05:
    print("Mann-Kendall: Null hypothesis rejected.")
else:
    print("Mann-Kendall: Null hypothesis not rejected.")

if cox_stuart_p_value < 0.05:
    print("Cox-Stuart: Null hypothesis rejected.")
else:
    print("Cox-Stuart: Null hypothesis not rejected.")

# 時系列データを作成（0からデータ数までの連番）
time_series = range(len(data))

# センズ傾斜を計算
result = linregress(time_series, data)

# センズ傾斜の値を取得
sen_slope = result.slope

# 結果を表示
print(f"Sen's Slope: {sen_slope:.10f}")

from matplotlib.ticker import MultipleLocator, FuncFormatter

# 時間軸を生成（5秒ごとのデータと仮定）
time = [i * 0.5 for i in range(len(data))]

# プロット
plt.figure(figsize=(8, 6))  # グラフのサイズを設定

# 散布図をプロット
plt.scatter(time, data)

# グラフの設定
plt.xlabel('Time[hour]')  # x軸ラベル
plt.ylabel('CPU usage [%]')  # y軸ラベル
title='Order CPU usage'  # グラフタイトル

print(title)

plt.xticks(np.arange(0, total_time + 1, 3600 * 8))
plt.gca().xaxis.set_major_formatter(FuncFormatter(format_time))

plt.gca().yaxis.set_major_formatter(FuncFormatter(format_percent))
plt.yticks(np.arange(0, 101, 20))

# グラフを表示
plt.grid(True)
plt.tight_layout()
plt.show()