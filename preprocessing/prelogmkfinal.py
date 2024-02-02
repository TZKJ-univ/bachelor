import re

# ログファイルのパスと出力ファイルのパス
log_file_path = "txt/result_log_all_180tps_mac.txt"
output_file_path = "output.txt"

# "Submitt" を含む行を抽出して出力ファイルに書き込む
with open(log_file_path, "r") as input_file, open(output_file_path, "w") as output_file:
    for line in input_file:
        if "Submitt" in line:
            output_file.write(line)

# 入力ファイル名と出力ファイル名
input_file = "output.txt"
output_file = "outputlog.csv"

# スペース区切りのテキストファイルをカンマ区切りのCSVファイルに変換
with open(input_file, 'r') as f, open(output_file, 'w', newline='') as output:
    for line in f:
        # スペースで行を分割し、カンマで結合
        data = line.strip().split()
        data = [data[0]] + data[11:]
        csv_line = ','.join(data)
        output.write(csv_line + '\n')

print("抽出が完了しました。")

# 入力ファイル名と出力ファイル名
input_file = "outputlog.csv"
output_file = "outputlog_cleaned.csv"

# "Fail:"の後の数字を残し、"Fail:"を削除してCSVファイルに変換
with open(input_file, 'r') as f, open(output_file, 'w', newline='') as output:
    for line in f:
        cleaned_line = re.sub(r'Fail:(\d+),', r'\1,', line)
        output.write(cleaned_line)

print("クリーニングが完了しました.")

# 入力ファイル名と出力ファイル名
input_file = "outputlog_cleaned.csv"
output_file = "outputlog_final.csv"

# 1列目から"[32m"を削除
with open(input_file, 'r') as f, open(output_file, 'w', newline='') as output:
    for line in f:
        cleaned_line = line.replace('[32m', '')
        output.write(cleaned_line)

# 入力ファイル名と出力ファイル名
input_file = "outputlog_final.csv"
output_file = "outputlog_final_cleaned.csv"

# 1列目から 'a' を削除してCSVファイルに変換
with open(input_file, 'r') as f, open(output_file, 'w', newline='') as output:
    for line in f:
        cleaned_line = line.replace('', '')
        output.write(cleaned_line)

print("最終処理が完了しました。")

