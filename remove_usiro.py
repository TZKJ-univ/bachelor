import pandas as pd

def remove_rows(csv_file, target_rows):
    # CSVファイルを読み込む（ヘッダーがない場合はheader=Noneを指定）
    df = pd.read_csv(csv_file, header=None)

    # 指定した行数になるように後ろから削除
    current_rows = len(df)
    if target_rows >= current_rows:
        print("指定した行数がデータ行数以上です。全ての行が削除されます。")
        df = pd.DataFrame()  # 空のデータフレームを作成
    else:
        df = df.iloc[:-(current_rows-target_rows)]

    # 削除後のデータを新しいCSVファイルに保存（ヘッダーなしの場合はheader=Falseを指定）
    df.to_csv('output3.csv', index=False, header=False)

# 使用例
csv_file_path = 'output2.csv'  # ご自身のCSVファイルのパスに変更してください
target_rows = 79820  # 指定する行数を指定してください

remove_rows(csv_file_path, target_rows)
