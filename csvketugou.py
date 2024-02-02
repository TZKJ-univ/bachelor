import csv
import os

# 既存のCSVファイル名と新しいデータのCSVファイル名
existing_file = 'existing_data.csv'  # 既存のCSVファイル名
new_data_file = 'outputlog_final.csv'  # 新しいデータのCSVファイル名

# 既存のCSVファイルが存在するか確認
if os.path.exists(existing_file):
    # 既存のCSVファイルを読み込む
    with open(existing_file, 'r', newline='') as existing_csv:
        existing_data = list(csv.reader(existing_csv))
else:
    # 既存のCSVファイルが存在しない場合、新しいデータをそのままコピー
    with open(new_data_file, 'r', newline='') as new_data_csv:
        new_data = list(csv.reader(new_data_csv))
        existing_data = new_data

# 新しいデータを読み込む
with open(new_data_file, 'r', newline='') as new_data_csv:
    new_data = list(csv.reader(new_data_csv))

# 新しいデータの列数と既存のデータの列数を取得
new_data_columns = len(new_data[0])
existing_data_columns = len(existing_data[0])

# 新しいデータのi列目を既存のデータに追加
for i in range(len(new_data)):
    if i < len(existing_data):
        existing_data[i].append(new_data[i][5])
    else:
        # 既存のデータの行数が足りない場合、新しい行を追加
        existing_data.append([''] * (existing_data_columns - 1) + [new_data[i][5]])

# 更新されたデータを既存のCSVファイルに書き込む
with open(existing_file, 'w', newline='') as existing_csv:
    writer = csv.writer(existing_csv)
    writer.writerows(existing_data)

print(f'既存のCSVファイルに新しいデータの3列目のデータを追加しました。')
