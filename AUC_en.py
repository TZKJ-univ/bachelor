import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

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
sum_auc_logistic = 0
sum_auc_rf = 0
sum_auc_gb = 0

# AUCスコアを保存するリスト
auc_logistic_list = []
auc_rf_list = []
auc_gb_list = []

# テストデータの読み込み
for i, test_data_df in enumerate(datasets):
    test_data_df = datasets[i]
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
    auc_logistic = roc_auc_score(y_test, logistic_predictions)

    # Random Forestモデルの評価
    rf_predictions = model_tps_en_rf.predict(X_test)
    auc_rf = roc_auc_score(y_test, rf_predictions)

    # Gradient Boostingモデルの評価
    gb_predictions = model_tps_en_gb.predict(X_test)
    auc_gb = roc_auc_score(y_test, gb_predictions)

    sum_auc_logistic += auc_logistic
    sum_auc_rf += auc_rf
    sum_auc_gb += auc_gb

    # AUCスコアをリストに追加
    auc_logistic_list.append(auc_logistic)
    auc_rf_list.append(auc_rf)
    auc_gb_list.append(auc_gb)

    # 結果の表示
    print(f'{i*10 + 60}TPS AUC:')
    print(f'Logistic Regression AUC: {auc_logistic:.4f}')
    print(f'Random Forest AUC: {auc_rf:.4f}')
    print(f'Gradient Boosting AUC: {auc_gb:.4f}\n')

# 平均AUCの表示
average_auc_logistic = sum_auc_logistic / len(datasets)
average_auc_rf = sum_auc_rf / len(datasets)
average_auc_gb = sum_auc_gb / len(datasets)

print(f'Average AUC for Logistic Regression: {average_auc_logistic:.4f}')
print(f'Average AUC for Random Forest: {average_auc_rf:.4f}')
print(f'Average AUC for Gradient Boosting: {average_auc_gb:.4f}')

# グラフのプロット
tps_values = [60, 70, 80, 90, 100, 110, 120,130, 140, 150, 160, 170, 180, 190, 200]

plt.figure(figsize=(10, 6))

plt.plot(tps_values, auc_logistic_list, label='Logistic Regression', marker='o')
plt.plot(tps_values, auc_rf_list, label='Random Forest', marker='o')
plt.plot(tps_values, auc_gb_list, label='Gradient Boosting', marker='o')

plt.xlabel('Test data (TPS)')
plt.ylabel('AUC Score')
plt.legend()
plt.grid(True)
plt.show()

# モデル名とAUCスコアを格納するリスト
models = ['Logistic Regression', 'Random Forest', 'Gradient Boosting']
auc_scores = [auc_logistic_list, auc_rf_list, auc_gb_list]

# AUCスコアの表を作成
# AUCスコアの表を作成
auc_table = pd.DataFrame({'Model': models, **{f'TPS={tps}': auc_scores[i] for i, tps in enumerate(tps_values) if i < len(auc_scores)}})

# LaTeX形式で表を表示
print(auc_table.to_latex(index=False))