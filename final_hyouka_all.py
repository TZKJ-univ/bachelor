import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# 学習データ1の読み込み
train_data1_df = pd.read_csv('60tps_all.csv', header=None)

# 学習データ2の読み込み
train_data2_df = pd.read_csv('80tps_all.csv', header=None)

# 学習データ3の読み込み
train_data3_df = pd.read_csv('100tps_all.csv', header=None)

# 学習データ4の読み込み
train_data4_df = pd.read_csv('120tps_all.csv', header=None)

train_data5_df = pd.read_csv('140tps_all.csv', header=None)

train_data6_df = pd.read_csv('160tps_all.csv', header=None)

train_data7_df = pd.read_csv('180tps_all.csv', header=None)

train_data8_df = pd.read_csv('200tps_all.csv', header=None)

datasets = [train_data1_df, train_data2_df, train_data3_df, train_data4_df, train_data5_df, train_data6_df, train_data7_df, train_data8_df]
sumAUC = 0
sumMSE = 0

fig, axes = plt.subplots(2, 4, figsize=(15, 8))
axes = axes.flatten()

# テストデータの読み込み
for i, ax in enumerate(axes):
    
    test_data_df = datasets[i]
    combined_train_data_df = pd.concat([datasets[j] for j in range(len(datasets)) if datasets[j] is not test_data_df], ignore_index=True)
    # ロジスティック回帰のモデルを構築
    X_train = combined_train_data_df.iloc[:, 1:]  # 1列目以降を特徴量として使います
    y_train = combined_train_data_df.iloc[:, 0]   # 0列目を目的変数にします
    model_tps_en_logistic = LogisticRegression()
    model_tps_en_logistic.fit(X_train, y_train)
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
    combined_train_data_df['Time'] = combined_train_data_df.iloc[:, -1] / 3600  # 1時間 = 3600秒
    # 特徴量と目的変数の取得
    X_test = test_data_df.iloc[:, 1:]
    y_test = test_data_df.iloc[:, 0]
    # ロジスティック回帰モデルの評価
    logistic_predictions = model_tps_en_logistic.predict_proba(X_test)[:, 1]
    # Random Forestモデルの評価
    rf_predictions = model_tps_en_rf.predict(X_test)
    # Gradient Boostingモデルの評価
    gb_predictions = model_tps_en_gb.predict(X_test)


    result_df = pd.DataFrame({
        'Time': test_data_df.iloc[:, -5] / 3600,
        'Actual': y_test,
        'Logistic_Predictions': logistic_predictions,
        'RF_Predictions': rf_predictions,
        'GB_Predictions': gb_predictions,
    })
    result_df.to_csv('test_results.csv', index=False)
    # グラフ表示
    
    ax.plot(result_df['Time'], result_df['Actual'], label='Actual', marker='o',color='red')
    ax.scatter(result_df['Time'], result_df['Logistic_Predictions'], label='Logistic Predictions', marker='o')
    ax.scatter(result_df['Time'], result_df['RF_Predictions'], label='RF Predictions', marker='o')
    ax.scatter(result_df['Time'], result_df['GB_Predictions'], label='GB Predictions', marker='o')
    ax.set_title(f'Test Dataset {i + 1}')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Value')
    ax.legend()
    
plt.tight_layout()
plt.show()