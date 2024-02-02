import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt

# 学習データの読み込み
train_data1_df = pd.read_csv('60tps_en.csv', header=None)
train_data2_df = pd.read_csv('70tps_en.csv', header=None)
train_data3_df = pd.read_csv('80tps_en.csv', header=None)
train_data4_df = pd.read_csv('90tps_en.csv', header=None)
train_data5_df = pd.read_csv('100tps_en.csv', header=None)
train_data6_df = pd.read_csv('110tps_en.csv', header=None)
train_data7_df = pd.read_csv('120tps_en.csv', header=None)
train_data8_df = pd.read_csv('130tps_en.csv', header=None)
train_data9_df = pd.read_csv('140tps_en.csv', header=None)
train_data10_df = pd.read_csv('150tps_en.csv', header=None)
train_data11_df = pd.read_csv('160tps_en.csv', header=None)
train_data12_df = pd.read_csv('170tps_en.csv', header=None)
train_data13_df = pd.read_csv('180tps_en.csv', header=None)
train_data14_df = pd.read_csv('190tps_en.csv', header=None)
train_data15_df = pd.read_csv('200tps_en.csv', header=None)

datasets = [train_data1_df, train_data2_df, train_data3_df, train_data4_df, train_data5_df, train_data6_df, train_data7_df, train_data8_df, train_data9_df, train_data10_df, train_data11_df, train_data12_df, train_data13_df, train_data14_df, train_data15_df]

sumAUC = 0
sumMSE = 0

fig, axes = plt.subplots(3, 5, figsize=(15, 8))
axes = axes.flatten()

time_diffs_gb = []  # Gradient Boostingの時間差を格納するリスト
time_diffs_logistic = []  # Logistic Regressionの時間差を格納するリスト
time_diffs_rf = []  # Random Forestの時間差を格納するリスト

# テストデータの読み込み
for i, ax in enumerate(axes):
    print(i)
    
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
        'Time': test_data_df.iloc[:, -1] / 3600,
        'Actual': y_test,
        'Logistic_Predictions': logistic_predictions,
        'RF_Predictions': rf_predictions,
        'GB_Predictions': gb_predictions,
    })
    
    # 最大予測値のプロット
    max_pred_time_gb = result_df.loc[result_df['GB_Predictions'].idxmax(), 'Time']
    max_pred_time_logistic = result_df.loc[result_df['Logistic_Predictions'].idxmax(), 'Time']
    max_pred_time_rf = result_df.loc[result_df['RF_Predictions'].idxmax(), 'Time']
    

    # 最大予測値の時間と実測値の時間の差を計算してリストに追加
    time_diff_gb = max_pred_time_gb - result_df.loc[result_df['Actual'].idxmax(), 'Time']
    time_diff_logistic = max_pred_time_logistic - result_df.loc[result_df['Actual'].idxmax(), 'Time']
    time_diff_rf = max_pred_time_rf - result_df.loc[result_df['Actual'].idxmax(), 'Time']
    
    time_diffs_gb.append(time_diff_gb)
    time_diffs_logistic.append(time_diff_logistic)
    time_diffs_rf.append(time_diff_rf)

# 時間差の平均を計算
average_time_diff_gb = sum(time_diffs_gb) / len(time_diffs_gb)
average_time_diff_logistic = sum(time_diffs_logistic) / len(time_diffs_logistic)
average_time_diff_rf = sum(time_diffs_rf) / len(time_diffs_rf)

# 時間差の平均を表示
print(f'Average Time Difference for GB: {average_time_diff_gb:.2f} hours')
print(f'Average Time Difference for Logistic: {average_time_diff_logistic:.2f} hours')
print(f'Average Time Difference for RF: {average_time_diff_rf:.2f} hours')

