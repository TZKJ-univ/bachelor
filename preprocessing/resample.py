import pandas as pd

# inputファイルの読み込み
input_file = 'output1.csv'
df = pd.read_csv(input_file, header=None)

# 2列目と7列目を取得
selected_columns = df.iloc[:, [2, 6]]

# 5個ごとに平均を計算
averages = selected_columns.groupby(selected_columns.index // 10).mean()

# outputファイルの書き込み
output_file = 'output4.csv'
averages.to_csv(output_file, header=False, index=False)
