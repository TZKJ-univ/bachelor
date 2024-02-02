import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB


# 学習データ1の読み込み
train_data1_df = pd.read_csv('200tps_ord.csv', header=None)

# 学習データ2の読み込み
train_data2_df = pd.read_csv('140tps_ord.csv', header=None)

# 学習データ3の読み込み
train_data3_df = pd.read_csv('100tps_ord.csv', header=None)

# 学習データ4の読み込み
train_data4_df = pd.read_csv('80tps_ord.csv', header=None)

train_data5_df = pd.read_csv('60tps_ord.csv', header=None)

train_data6_df = pd.read_csv('180tps_ord.csv', header=None)

train_data7_df = pd.read_csv('160tps_ord.csv', header=None)

# 両方の学習データを結合
combined_train_data_df = pd.concat([train_data1_df, train_data2_df, train_data3_df, train_data4_df, train_data5_df, train_data6_df, train_data7_df], ignore_index=True)

# ロジスティック回帰のモデルを構築
X_train = combined_train_data_df.iloc[:, 1:]  # 1列目以降を特徴量として使います
y_train = combined_train_data_df.iloc[:, 0]   # 0列目を目的変数にします

model_tps_en_logistic = LogisticRegression()
model_tps_en_logistic.fit(X_train, y_train)

# 予測結果を追加（確率をそのまま使用）
combined_train_data_df['Logistic_Predictions'] = model_tps_en_logistic.predict_proba(X_train)[:, 1]

# Random Forestモデルの構築
model_tps_en_rf = RandomForestRegressor()  # 適切なパラメータを設定してください
model_tps_en_rf.fit(X_train, y_train)

# 予測結果を追加
combined_train_data_df['RF_Predictions'] = model_tps_en_rf.predict(X_train)

# Gradient Boostingモデルの構築
model_tps_en_gb = GradientBoostingRegressor()  # 適切なパラメータを設定してください
model_tps_en_gb.fit(X_train, y_train)

# 予測結果を追加
combined_train_data_df['GB_Predictions'] = model_tps_en_gb.predict(X_train)

# 秒数を時間に変換
combined_train_data_df['Time'] = combined_train_data_df.iloc[:, -1] / 3600  # 1時間 = 3600秒

# テストデータの読み込み
test_data_df = pd.read_csv('120tps_ord.csv', header=None)

# 特徴量と目的変数の取得
X_test = test_data_df.iloc[:, 1:]
y_test = test_data_df.iloc[:, 0]

# ロジスティック回帰モデルの評価
logistic_predictions = model_tps_en_logistic.predict_proba(X_test)[:, 1]
mse_logistic = mean_squared_error(y_test, logistic_predictions)
r2_logistic = r2_score(y_test, logistic_predictions)

# Random Forestモデルの評価
rf_predictions = model_tps_en_rf.predict(X_test)
mse_rf = mean_squared_error(y_test, rf_predictions)
r2_rf = r2_score(y_test, rf_predictions)

# Gradient Boostingモデルの評価
gb_predictions = model_tps_en_gb.predict(X_test)
mse_gb = mean_squared_error(y_test, gb_predictions)
r2_gb = r2_score(y_test, gb_predictions)

# 結果をCSVに保存
result_df = pd.DataFrame({
    'Time': test_data_df.iloc[:, -1] / 3600,
    'Actual': y_test,
    'Logistic_Predictions': logistic_predictions,
    'RF_Predictions': rf_predictions,
    'GB_Predictions': gb_predictions,
})

result_df.to_csv('test_results.csv', index=False)

# グラフ表示
plt.figure(figsize=(8, 6))

# ロジスティック回帰モデルの予測結果
plt.subplot(1, 3, 1)
plt.scatter(result_df['Time'], result_df['Actual'], label='Actual', marker='o')
plt.scatter(result_df['Time'], result_df['Logistic_Predictions'], label='Logistic Predictions', marker='o')
plt.title('Logistic Regression Model')
plt.xlabel('Time (hours)')
plt.ylabel('Value')
plt.legend()

# Random Forestモデルの予測結果
plt.subplot(1, 3, 2)
plt.scatter(result_df['Time'], result_df['Actual'], label='Actual', marker='o')
plt.scatter(result_df['Time'], result_df['RF_Predictions'], label='RF Predictions', marker='o')
plt.title('Random Forest Model')
plt.xlabel('Time (hours)')
plt.ylabel('Value')
plt.legend()

# Gradient Boostingモデルの予測結果
plt.subplot(1, 3, 3)
plt.scatter(result_df['Time'], result_df['Actual'], label='Actual', marker='o')
plt.scatter(result_df['Time'], result_df['GB_Predictions'], label='GB Predictions', marker='o')
plt.title('Gradient Boosting Model')
plt.xlabel('Time (hours)')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()

# 結果の表示
print('ロジスティック回帰モデル:')
print(f'Mean Squared Error (MSE): {mse_logistic:.4f}')
print(f'R^2 Score: {r2_logistic:.4f}')

print('\nRandom Forestモデル:')
print(f'Mean Squared Error (MSE): {mse_rf:.4f}')
print(f'R^2 Score: {r2_rf:.4f}')

print('\nGradient Boostingモデル:')
print(f'Mean Squared Error (MSE): {mse_gb:.4f}')
print(f'R^2 Score: {r2_gb:.4f}')
