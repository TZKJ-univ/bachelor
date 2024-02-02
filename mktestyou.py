import pandas as pd

# Step 1: 0から数えて6列目のデータを読み込み、0以外のデータを1に変換
outputlog_df = pd.read_csv('outputlog_final_cleaned.csv', header=None, delimiter=',')
outputlog_df.iloc[:, 5] = outputlog_df.iloc[:, 5].apply(lambda x: 1 if x != 0 else 0)

# 0列目に変換したデータをkaikiyou.csvに保存
outputlog_df.iloc[:, [5, 5]].to_csv('kaikiyou.csv', header=None, index=False)

# Step 2: output3.csvの0から数えて2列目と5列目のデータを読み込み
output3_df = pd.read_csv('output3.csv', header=None, delimiter=',', usecols=[2, 6])

# kaikiyou.csvに新しい列として追加
kaikiyou_df = pd.read_csv('kaikiyou.csv', header=None)
kaikiyou_df[1] = output3_df.iloc[:, 0]
kaikiyou_df[2] = output3_df.iloc[:, 1]

# Step 3: kaikiyou.csvに新しい列として40を追加
kaikiyou_df[3] = 40

# kaikiyou.csvを保存（新しい列が追加された状態）
kaikiyou_df.to_csv('testyou.csv', header=None, index=False)