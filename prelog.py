import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

# CSVファイルからデータを読み取る
input_file = "outputlog_final_cleaned.csv"

total_faults = []  # フォールト総数のリスト

with open(input_file, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    
    # ヘッダーがない場合、スキップしない
    # next(csv_reader)  # ヘッダ行をスキップ

    for row in csv_reader:
        # データを整数に変換
        total_fault = int(row[5])
        total_faults.append(total_fault)

# フォールト総数から瞬間フォールト数を導出
instant_faults = [total_faults[i] - total_faults[i-1] for i in range(1, len(total_faults))]

# グラフにプロット
plt.figure(figsize=(8, 6))

# X軸を適切な単位で表示する
def format_hour(x, pos):
    hours = int(x * 5 / 3600)
    return f"{hours}"

plt.plot(instant_faults, color='r')
plt.xlabel("Time[hour]")
plt.ylabel("Number of Faults")

# X軸の表示を適切な単位で設定
plt.gca().xaxis.set_major_locator(MultipleLocator(3600 * 4 / 5))  # 4時間ごとにLocatorを設定
plt.gca().xaxis.set_major_formatter(FuncFormatter(format_hour))  # フォーマットを指定

plt.grid(True)
plt.show()
