import pandas as pd

# スペース区切りのテキストファイルのファイル名
input_filename = 'txt/result_trans_180tps_mac_resouse.txt'
print("input_filename: ", input_filename)

# CSVファイルのファイル名
output_filename = 'output1.csv'

# テキストファイルを読み取り
with open(input_filename, 'r') as input_file:
    lines = input_file.readlines()

# 特定の文字列を含む行のみを抽出します############################################
lines = [line.strip() for line in lines if '561ddd7c39ae' in line]
# ###########################################################################


# データをDataFrameに変換します
df = pd.DataFrame([line.split() for line in lines])

# 特定の列（例：3番目の列）のデータに対してGiBからMiBに変換します
column_index = 3  # 3番目の列（0から始まるインデックス）
for i, value in enumerate(df[column_index]):
    if 'GiB' in value:
        df.at[i, column_index] = f'{float(value.split("GiB")[0]) * 1024:.2f} MiB'
    
    # "MiB" の文字列を削除します
    df.at[i, column_index] = df.at[i, column_index].replace("MiB", "").strip()
# 特定の列の各行の値から '%' を削除して数字だけに変換
df[column_index-1] = df[column_index-1].str.replace('%', '').str.extract('(\d+\.*\d*)')[0]
df[column_index+3] = df[column_index+3].str.replace('%', '').str.extract('(\d+\.*\d*)')[0]

#tyousei = 698587##後ろ削りたいならここを小さくする############################################
#if len(df) > tyousei:
#    df = df.iloc[:tyousei]

#if len(df) > 691200:
 #   excess_rows = len(df) - 691200
  #  if excess_rows % 2 == 0:
   #     df = df.iloc[excess_rows // 2:-(excess_rows // 2)]
    #else:
     #   df = df.iloc[(excess_rows // 2) + 1:-(excess_rows // 2)]


# DataFrameをCSVファイルに書き込みます（カンマ区切り）
df.to_csv(output_filename, index=False, header=False, sep=',')

print(f'{input_filename} を {output_filename} に変換し、GiBからMiBに変換し、カンマ区切りにし、前後左右120個の行を削除しました。')
