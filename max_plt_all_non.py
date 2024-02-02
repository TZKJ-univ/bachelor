import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt

# 学習データの読み込み
train_data1_df = pd.read_csv('60tps_non.csv', header=None)
train_data2_df = pd.read_csv('70tps_non.csv', header=None)
train_data3_df = pd.read_csv('80tps_non.csv', header=None)
train_data4_df = pd.read_csv('90tps_non.csv', header=None)
train_data5_df = pd.read_csv('100tps_non.csv', header=None)
train_data6_df = pd.read_csv('110tps_non.csv', header=None)
train_data7_df = pd.read_csv('120tps_non.csv', header=None)
train_data8_df = pd.read_csv('130tps_non.csv', header=None)
train_data9_df = pd.read_csv('140tps_non.csv', header=None)
train_data10_df = pd.read_csv('150tps_non.csv', header=None)
train_data11_df = pd.read_csv('160tps_non.csv', header=None)
train_data12_df = pd.read_csv('170tps_non.csv', header=None)
train_data13_df = pd.read_csv('180tps_non.csv', header=None)
train_data14_df = pd.read_csv('190tps_non.csv', header=None)
train_data15_df = pd.read_csv('200tps_non.csv', header=None)

datasets = [train_data1_df, train_data2_df, train_data3_df, train_data4_df, train_data5_df, train_data6_df, train_data7_df, train_data8_df, train_data9_df, train_data10_df, train_data11_df, train_data12_df, train_data13_df, train_data14_df, train_data15_df]

time_diffs_gb = []  
time_diffs_logistic = []  
time_diffs_rf = []  

# データの保存先
scatter_points = {
    'Test Data Index': [],
    'First Failure (Actual)': [],
    'Max Prediction GB': [],
    'Max Prediction Logistic': [],
    'Max Prediction RF': []
}

# テストデータの読み込み
for i, test_data_df in enumerate(datasets):
    print(i)
    combined_train_data_df = pd.concat([datasets[j] for j in range(len(datasets)) if datasets[j] is not test_data_df], ignore_index=True)
    
    # ロジスティック回帰のモデルを構築
    X_train = combined_train_data_df.iloc[:, 1:]
    y_train = combined_train_data_df.iloc[:, 0]
    model_tps_en_logistic = LogisticRegression()
    model_tps_en_logistic.fit(X_train, y_train)
    combined_train_data_df['Logistic_Predictions'] = model_tps_en_logistic.predict_proba(X_train)[:, 1]
    
    # Random Forestモデルの構築
    model_tps_en_rf = RandomForestRegressor()
    model_tps_en_rf.fit(X_train, y_train)
    combined_train_data_df['RF_Predictions'] = model_tps_en_rf.predict(X_train)
    
    # Gradient Boostingモデルの構築
    model_tps_en_gb = GradientBoostingRegressor()
    model_tps_en_gb.fit(X_train, y_train)
    combined_train_data_df['GB_Predictions'] = model_tps_en_gb.predict(X_train)
    combined_train_data_df['Time'] = combined_train_data_df.iloc[:, -1] / 3600  
    
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

    # 最初のActualが1になった時刻
    first_1_index = result_df['Actual'].eq(1).idxmax()
    scatter_points['Test Data Index'].append(i + 1)
    scatter_points['First Failure (Actual)'].append(result_df.loc[first_1_index, 'Time'])

    # 最大予測値の時刻
    max_pred_time_gb = result_df.loc[result_df['GB_Predictions'].idxmax(), 'Time']
    max_pred_time_logistic = result_df.loc[result_df['Logistic_Predictions'].idxmax(), 'Time']
    max_pred_time_rf = result_df.loc[result_df['RF_Predictions'].idxmax(), 'Time']

    scatter_points['Max Prediction GB'].append(max_pred_time_gb)
    scatter_points['Max Prediction Logistic'].append(max_pred_time_logistic)
    scatter_points['Max Prediction RF'].append(max_pred_time_rf)

# 保存したデータをCSVに出力
scatter_points_df = pd.DataFrame(scatter_points)
scatter_points_df.to_csv('scatter_points.csv', index=False)

# CSVからデータを読み込んで折れ線グラフに表示
scatter_points_df = pd.read_csv('scatter_points.csv')

# グラフ描画
plt.figure(figsize=(10, 6))

# 折れ線グラフ
plt.plot(scatter_points_df['Test Data Index']*10+50, scatter_points_df['First Failure (Actual)'], marker='o', label='First Failure (Actual)', color='red')
plt.plot(scatter_points_df['Test Data Index']*10+50, scatter_points_df['Max Prediction Logistic'], marker='x', label='Max Prediction Logistic', color='blue')
plt.plot(scatter_points_df['Test Data Index']*10+50, scatter_points_df['Max Prediction RF'], marker='x', label='Max Prediction RF', color='orange')
plt.plot(scatter_points_df['Test Data Index']*10+50, scatter_points_df['Max Prediction GB'], marker='x', label='Max Prediction GB', color='green')
# グラフにラベルを追加
plt.xlabel('Test Data (TPS)')
plt.ylabel('Time [hour]')
plt.legend()

# グラフを表示
plt.show()

