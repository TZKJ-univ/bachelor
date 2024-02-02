import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

# データを読み込む
df = pd.read_csv('/Users/tozawa/Library/CloudStorage/OneDrive-東京都公立大学法人/caliper result/test_results.csv')

# 実測値と各モデルの連続的な予測値を取得
actual_values = df['Actual']
logistic_predictions = df['Logistic_Predictions']
rf_predictions = df['RF_Predictions']
gb_predictions = df['GB_Predictions']

# 連続的な予測をクラスに変換する閾値を選択
threshold = 0.5  # 例: 0.5　

# 連続的な予測をバイナリに変換
logistic_predictions_binary = (logistic_predictions > threshold).astype(int)
rf_predictions_binary = (rf_predictions > threshold).astype(int)
gb_predictions_binary = (gb_predictions > threshold).astype(int)

# 適合率、再現率、F1スコア
logistic_precision = precision_score(actual_values, logistic_predictions_binary)
logistic_recall = recall_score(actual_values, logistic_predictions_binary)
logistic_f1 = f1_score(actual_values, logistic_predictions_binary)

rf_precision = precision_score(actual_values, rf_predictions_binary)
rf_recall = recall_score(actual_values, rf_predictions_binary)
rf_f1 = f1_score(actual_values, rf_predictions_binary)

gb_precision = precision_score(actual_values, gb_predictions_binary)
gb_recall = recall_score(actual_values, gb_predictions_binary)
gb_f1 = f1_score(actual_values, gb_predictions_binary)

# AUC-ROC曲線
logistic_auc_roc = roc_auc_score(actual_values, logistic_predictions)
rf_auc_roc = roc_auc_score(actual_values, rf_predictions)
gb_auc_roc = roc_auc_score(actual_values, gb_predictions)

# 結果の表示
print("ロジスティック回帰:")
print("適合率:", logistic_precision)
print("再現率:", logistic_recall)
print("F1スコア:", logistic_f1)
print("AUC-ROC曲線の面積:", logistic_auc_roc)
print("\n")

print("ランダムフォレスト:")
print("適合率:", rf_precision)
print("再現率:", rf_recall)
print("F1スコア:", rf_f1)
print("AUC-ROC曲線の面積:", rf_auc_roc)
print("\n")

print("勾配ブースティング:")
print("適合率:", gb_precision)
print("再現率:", gb_recall)
print("F1スコア:", gb_f1)
print("AUC-ROC曲線の面積:", gb_auc_roc)
print("\n")

# AUC-ROC曲線のプロット
fpr_logistic, tpr_logistic, _ = roc_curve(actual_values, logistic_predictions)
fpr_rf, tpr_rf, _ = roc_curve(actual_values, rf_predictions)
fpr_gb, tpr_gb, _ = roc_curve(actual_values, gb_predictions)

plt.figure(figsize=(10, 8))
plt.plot(fpr_logistic, tpr_logistic, color='orange', lw=2, label='Logistic Regression (AUC = %0.2f)' % logistic_auc_roc)
plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label='Random Forest (AUC = %0.2f)' % rf_auc_roc)
plt.plot(fpr_gb, tpr_gb, color='blue', lw=2, label='Gradient Boosting (AUC = %0.2f)' % gb_auc_roc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
