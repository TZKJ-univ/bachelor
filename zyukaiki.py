import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor  
from sklearn.metrics import mean_squared_error, r2_score

# 140tps_en.csvの読み込み
tps_en_df = pd.read_csv('60tps_ord.csv', header=None)

# ロジスティック回帰のモデルを構築
X_tps_en = tps_en_df.iloc[:, 1:]  # 1列目以降を特徴量として使います
y_tps_en = tps_en_df.iloc[:, 0]   # 0列目を目的変数にします

model_tps_en_logistic = LogisticRegression()
model_tps_en_logistic.fit(X_tps_en, y_tps_en)

# 予測結果を追加
tps_en_df['Logistic_Predictions'] = model_tps_en_logistic.predict(X_tps_en)

# 他の回帰モデル（ここではRandom Forest）を構築
model_tps_en_rf = RandomForestRegressor()  # 適切なパラメータを設定してください
model_tps_en_rf.fit(X_tps_en, y_tps_en)

# 予測結果を追加
tps_en_df['RF_Predictions'] = model_tps_en_rf.predict(X_tps_en)

# 結果を保存
tps_en_df.to_csv('60tps_ord_with_predictions.csv', header=None, index=False)