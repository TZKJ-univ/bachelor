import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score, recall_score
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
sum_auc_logistic = 0
sum_auc_rf = 0
sum_auc_gb = 0
sum_recall_logistic = 0
sum_recall_rf = 0
sum_recall_gb = 0

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
    result_df.to_csv(f'test_results_dataset_{i + 1}.csv', index=False)

    # AUCの計算
    auc_logistic = roc_auc_score(y_test, logistic_predictions)
    auc_rf = roc_auc_score(y_test, rf_predictions)
    auc_gb = roc_auc_score(y_test, gb_predictions)

    # 再現率の計算
    recall_logistic = recall_score(y_test, (logistic_predictions >= 0.5).astype(int))
    recall_rf = recall_score(y_test, (rf_predictions >= 0.5).astype(int))
    recall_gb = recall_score(y_test, (gb_predictions >= 0.5).astype(int))
    
    sum_auc_logistic += auc_logistic
    sum_auc_rf += auc_rf
    sum_auc_gb += auc_gb
    sum_recall_logistic += recall_logistic
    sum_recall_rf += recall_rf
    sum_recall_gb += recall_gb

    # 結果の表示
    print(f'Test Dataset {i + 1} AUC:')
    print(f'Logistic Regression AUC: {auc_logistic:.4f}')
    print(f'Random Forest AUC: {auc_rf:.4f}')
    print(f'Gradient Boosting AUC: {auc_gb:.4f}')
    print(f'Test Dataset {i + 1} Recall:')
    print(f'Logistic Regression Recall: {recall_logistic:.4f}')
    print(f'Random Forest Recall: {recall_rf:.4f}')
    print(f'Gradient Boosting Recall: {recall_gb:.4f}\n')

# 平均AUCの表示
average_auc_logistic = sum_auc_logistic / len(datasets)
average_auc_rf = sum_auc_rf / len(datasets)
average_auc_gb = sum_auc_gb / len(datasets)
average_recall_logistic = sum_recall_logistic / len(datasets)
average_recall_rf = sum_recall_rf / len(datasets)
average_recall_gb = sum_recall_gb / len(datasets)

print(f'Average AUC for Logistic Regression: {average_auc_logistic:.4f}')
print(f'Average AUC for Random Forest: {average_auc_rf:.4f}')
print(f'Average AUC for Gradient Boosting: {average_auc_gb:.4f}')
print(f'Average Recall for Logistic Regression: {average_recall_logistic:.4f}')
print(f'Average Recall for Random Forest: {average_recall_rf:.4f}')
print(f'Average Recall for Gradient Boosting: {average_recall_gb:.4f}')
