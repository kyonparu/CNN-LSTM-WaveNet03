# Description: 100Hzの音素ラベルデータを250Hzに補完するスクリプト
import numpy as np
import os

# 話者コードのリスト
speakers = ['M0101', 'M0102', 'M0201', 'M0301', 'M00302', 'W0101', 'W0201', 'W0301', 'W0401', 'W0501', 'W0601', 'W0701', 'W0801']

# 発話番号の範囲
utterance_range = range(1, 504)

# サンプリング周波数
sampling_rate = 100  # 100Hz
time_step = 1 / sampling_rate  # 0.01秒 (10ms)

processed_files = []

for speaker in speakers:
    for utterance in utterance_range:
        lab_file_path = f'C:/Users/erima/py_code/DNN_exp/data_set/label_data/{speaker}/ATR503{speaker}_{utterance:03d}.lab'
        csv_file_path = f'C:/Users/erima/py_code/DNN_exp/data_set/articulatory_data/{speaker}/ATR503{speaker}_{utterance:03d}_UL.csv'

        # ファイルが存在しない場合はスキップ
        if not os.path.exists(lab_file_path) or not os.path.exists(csv_file_path):
            continue

        with open(lab_file_path, 'r') as file:
            lines = file.readlines()

        # 各サンプルの音素を格納するリスト
        phonemes = []

        # ファイルの各行を処理
        for line in lines:
            start_time, end_time, phoneme = line.strip().split()
            start_time = float(start_time)
            end_time = float(end_time)
            
            # 開始時間から終了時間までの各サンプルの音素を追加
            start_sample = int(start_time * sampling_rate)
            end_sample = int(end_time * sampling_rate)
            
            for i in range(start_sample, end_sample):
                phonemes.append(phoneme)

        # 最後にsilEの音素を追加
        phonemes.append('silE')

        # 100Hzのデータを250Hzに補完する
        phonemes_250Hz = []
        for i in range(len(phonemes)):
            if i % 2 == 0:
                # 偶数個目の音素は2回繰り返す
                phonemes_250Hz.append(phonemes[i])
                phonemes_250Hz.append(phonemes[i])
            else:
                # 奇数個目の音素は3回繰り返す
                phonemes_250Hz.append(phonemes[i])
                phonemes_250Hz.append(phonemes[i])
                phonemes_250Hz.append(phonemes[i])

        # CSVファイルの行数を確認
        with open(csv_file_path, 'r') as csv_file:
            csv_lines = csv_file.readlines()

        # 合計サンプル数が5で割り切れない場合はsilEを最後に2つ追加
        if len(csv_lines) % 5 != 0:
            phonemes_250Hz.append('silE')
            phonemes_250Hz.append('silE')

        # ディレクトリが存在しない場合は作成
        os.makedirs(f'C:/Users/erima/py_code/DNN_exp/data_set/label_data_100/{speaker}', exist_ok=True)
        os.makedirs(f'C:/Users/erima/py_code/DNN_exp/data_set/label_data_250/{speaker}', exist_ok=True)

        # 結果をファイルに保存
        with open(f'C:/Users/erima/py_code/DNN_exp/data_set/label_data_100/{speaker}/ATR503{speaker}_{utterance:03d}.csv', 'w') as f:
            for phoneme in phonemes:
                f.write(f"{phoneme}\n")

        with open(f'C:/Users/erima/py_code/DNN_exp/data_set/label_data_250/{speaker}/ATR503{speaker}_{utterance:03d}.csv', 'w') as f:
            for phoneme in phonemes_250Hz:
                f.write(f"{phoneme}\n")

        # 処理したファイルを記録
        processed_files.append((speaker, utterance))

# 処理が完了した話者コードと発話番号を表示
for speaker, utterance in processed_files:
    print(f"処理が完了しました: 話者コード {speaker}, 発話番号 {utterance:03d}")

print("全ての処理が完了しました。")