import pandas as pd

# 1つ目のCSVファイルを読み込む
df1 = pd.read_csv('output1.csv')

# 2つ目のCSVファイルを読み込む
df2 = pd.read_csv('output2.csv')

result = pd.DataFrame({
    '0列目': df1.iloc[:, 2] + df2.iloc[:, 2],
    '1列目': df1.iloc[:, 3] + df2.iloc[:, 3]
})

# 結果を新しいCSVファイルに書き出す
result.to_csv('output.csv', index=False, header=False)