import pandas as pd

# ファイル読み込み（header=Noneでヘッダーを無視）
df_en = pd.read_csv("160tps_en.csv", header=None)
df_non = pd.read_csv("160tps_non.csv", header=None)
df_ord = pd.read_csv("160tps_ord.csv", header=None)

# 60tps_all.csvの作成
df_all = pd.DataFrame()

# 1〜5列目
df_all[[1, 2, 3, 4, 5]] = df_en.iloc[:, :5]

# 6, 7列目
df_all[[6, 7]] = df_non.iloc[:, 1:3]

# 8, 9列目
df_all[[8, 9]] = df_ord.iloc[:, 1:3]

# ファイル書き込み
df_all.to_csv("160tps_all.csv", index=False, header=None)
