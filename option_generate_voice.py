import os
import torch
import numpy as np
import soundfile as sf
from datetime import datetime
from espnet2.bin.tts_inference import Text2Speech
from main import load_config  # 設定ファイルの読み込み関数をインポート

# HiFi-GANモデルのロード
model = Text2Speech.from_pretrained("kan-bayashi/jsut_hifigan.v1", device="cpu")
vocoder = model.vocoder

# サンプルレート（HiFi-GANの仕様に基づく）
SAMPLERATE = 22050

# testlistのパス（configから取得する場合）
config = load_config(config_path="config.yaml")
testlist_path = config['testlist']

# testlistを読み込む
if not os.path.exists(testlist_path):
    raise FileNotFoundError(f"Test list file not found: {testlist_path}")

with open(testlist_path, "r") as f:
    test_items = f.readlines()

# 保存先ディレクトリを作成
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = f"save_speech_{current_time}"
os.makedirs(output_dir, exist_ok=True)

# 音声合成を実行
for item in test_items:
    item = item.strip()
    if not item:
        continue

    # speaker_idとsentence_idを抽出
    parts = item.split("_")
    if len(parts) < 2:
        print(f"Skipping invalid test item: {item}")
        continue

    speaker_id = parts[0][5:]  # 例: "ATR503M0102" -> "M0102"
    sentence_id = parts[1]     # 例: "051"

    # 推定されたメルスペクトログラムのパス
    mel_path = os.path.join("dump", "test", f"{speaker_id}_{sentence_id}_predicted.npy")

    # メルスペクトログラムをロード
    if not os.path.exists(mel_path):
        print(f"Mel-spectrogram file not found: {mel_path}")
        continue

    mel = np.load(mel_path)  # 推定されたメルスペクトログラムをロード
    mel = torch.from_numpy(mel).unsqueeze(0)  # [B, T, 80]

    # 音声合成（vocoderだけ）
    with torch.no_grad():
        wav = vocoder(mel.transpose(1, 2)).view(-1).cpu().numpy()

    # 保存
    output_path = os.path.join(output_dir, f"{speaker_id}_{sentence_id}_out.wav")
    sf.write(output_path, wav, samplerate=SAMPLERATE)
    print(f"Generated audio saved to: {output_path}")