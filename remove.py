import pandas as pd

def remove_rows(csv_file, num_rows):
    # CSVファイルを読み込む
    df = pd.read_csv(csv_file,header=None)

    # 指定した行数を削除
    df = df.iloc[num_rows:]

    # 削除後のデータを新しいCSVファイルに保存
    df.to_csv('output2.csv', index=False, header=None)

# 使用例
csv_file_path = 'output1.csv'  # ご自身のCSVファイルのパスに変更してください
num_rows_to_remove = 3780 # 削除する行数を指定してください

remove_rows(csv_file_path, num_rows_to_remove)

def remove_rows1(csv_file, target_rows):
    # CSVファイルを読み込む（ヘッダーがない場合はheader=Noneを指定）
    df = pd.read_csv(csv_file, header=None)

    # 削除後のデータを新しいCSVファイルに保存（ヘッダーなしの場合はheader=Falseを指定）
    df.to_csv('output3.csv', index=False, header=False)

# 使用例
csv_file_path = 'output2.csv'  # ご自身のCSVファイルのパスに変更してください
target_rows = 317880  # 指定する行数を指定してください

remove_rows1(csv_file_path, target_rows)
