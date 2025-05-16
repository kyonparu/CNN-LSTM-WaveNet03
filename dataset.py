import os
import numpy as np
import pandas as pd
import pyworld as pw
import pysptk
import soundfile as sf
from tqdm import tqdm
import torch
import torch.nn as nn
import librosa
import logging
from utils import extract_speaker_and_sentence_id

# ログの設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 音素埋め込み用の辞書（例）
PHONEME_LIST = ["a", "a:", "i", "i:", "u", "u:", "e", "e:", "o", "o:", "k", "s",
                "t", "n", "f", "h", "m", "y", "r", "w", "g", "j", "z", "d", "b",
                "p", "q", "N", "ky", "gy", "sh", "ty", "ch", "ny", "hy", "my", "ry", "py",
                "ts", "silB", "silE", "sp"]
PHONEME_TO_ID = {p: i for i, p in enumerate(PHONEME_LIST)}

class PhonemeEmbedding(nn.Module):
    def __init__(self, num_phonemes, embedding_dim):
        super(PhonemeEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_phonemes, embedding_dim)
    
    def forward(self, phoneme_ids):
        return self.embedding(phoneme_ids)

class SpeechDataset:
    def __init__(self, data_dir, trainlist_path):
        """
        コンストラクタ
        :param data_dir: データディレクトリのパス
        :param trainlist_path: trainlistファイルのパス
        """
        self.data_dir = data_dir
        self.trainlist_path = trainlist_path
        self.phoneme_embedding = PhonemeEmbedding(len(PHONEME_LIST), embedding_dim=16)

    def load_articulation_data(self, speaker_id, sentence_id):
        """
        調音位置データの読み込みと一次微分値の計算
        :param speaker_id: 話者コード（例: M0101）
        :param sentence_id: 発話番号（例: 001）
        :return: 特徴量 (N, 6, 6) の NumPy 配列
        """
        features = []
        positions = ["UL", "LL", "LJ", "T1", "T2", "T3"]
        for pos in positions:
            file_name = f"ATR503{speaker_id}_{sentence_id}_{pos}.csv"
            file_path = os.path.join(self.data_dir, 'articulatory_data', speaker_id, file_name)
            data = pd.read_csv(file_path, header=None).values
            coords = data[:, :3]
            diff_coords = np.diff(coords, axis=0, prepend=coords[:1])
            combined = np.concatenate([coords, diff_coords], axis=1)
            features.append(combined)
        
        return np.stack(features, axis=1)

    def load_audio_features(self, speaker_id, sentence_id, target_length):
        """
        音声特徴量の抽出（80次元のlog-Melフィルタバンク）
        """
        file_name = f"ATR503{speaker_id}_{sentence_id}.wav"
        file_path = os.path.join(self.data_dir, 'audio_data16k', speaker_id, file_name)
        
        # 音声データを読み込み（サンプリングレートを16kHzに固定）
        waveform, sr = librosa.load(file_path, sr=16000)
        
        # Melスペクトログラムを計算
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=sr,
            n_fft=1024,
            hop_length=64,
            win_length=1024,
            n_mels=80,
            fmin=0.0,
            fmax=8000.0,
            center=False  # HiFi-GANと合わせるならFalse
        )
        
        # logスケールに変換
        log_mel = np.log(mel_spec + 1e-6)
        
        logging.info(f"log_mel.shape[1]: {log_mel.shape[1]}, target_length: {target_length}")
        # 長さを調整
        if log_mel.shape[1] < target_length:
            pad_width = target_length - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='edge')
        elif log_mel.shape[1] > target_length:
            log_mel = log_mel[:, :target_length]
        
        # 転置して [time_steps, 80] の形状にする
        log_mel = log_mel.T
        
        return torch.tensor(log_mel, dtype=torch.float32)

    def load_lab_file(self, speaker_id, sentence_id):
        """
        ラベルファイルの読み込み
        :param speaker_id: 話者コード（例: M0101）
        :param sentence_id: 発話番号（例: 001）
        :return: 音素ラベルのリスト
        """
        file_name = f"ATR503{speaker_id}_{sentence_id}.csv"
        file_path = os.path.join(self.data_dir, 'label_data_250', speaker_id, file_name)
        labels = []
        with open(file_path, "r") as f:
            for line in f:
                phoneme = line.strip()
                phoneme_id = PHONEME_TO_ID.get(phoneme, 0)
                labels.append(phoneme_id)

        return torch.tensor(labels, dtype=torch.long)

    def __call__(self, speaker_id, sentence_id):
        """
        データローダーを呼び出して音声と調音位置の特徴量を取得
        :param speaker_id: 話者コード（例: M0101）
        :param sentence_id: 発話番号（例: 001）
        :return: 音声特徴量, 調音位置特徴量, 言語特徴量
        """
        # 調音特徴量の読み込み
        articulation_features = self.load_articulation_data(speaker_id, sentence_id)

        # 音声特徴量の読み込み
        audio_features = self.load_audio_features(speaker_id, sentence_id, target_length=articulation_features.shape[0])

        # 言語特徴量の読み込み
        phoneme_ids = self.load_lab_file(speaker_id, sentence_id)
        phoneme_embeddings = self.phoneme_embedding(phoneme_ids)
        phoneme_embeddings = phoneme_embeddings.clone().detach().float()

        # NumPy配列をテンソルに変換
        audio_features = audio_features.clone().detach().float()
        articulation_features = torch.tensor(articulation_features, dtype=torch.float32).clone().detach()
        phoneme_embeddings = phoneme_embeddings.clone().detach().float()

        # 特徴量の型を表示
        print("stage-chage feature")
        print(f"audio_features type: {type(audio_features)}, shape: {audio_features.shape}")
        print(f"articulation_features type: {type(articulation_features)}, shape: {articulation_features.shape}")
        print(f"phoneme_embeddings type: {type(phoneme_embeddings)}, shape: {phoneme_embeddings.shape}")

        return audio_features, articulation_features, phoneme_embeddings

    def get_feature_dimensions(self, output_dir, list_name):
        """
        保存されたテンソルを使用して特徴量の次元数を取得
        """
        # trainlistの最初のデータを取得
        with open(self.trainlist_path, 'r') as f:
            first_line = f.readline().strip()
        logging.info(f"First line in trainlist: {first_line}")

        # 話者IDと発話IDを抽出
        speaker_id, sentence_id = extract_speaker_and_sentence_id(first_line)
        if speaker_id is None or sentence_id is None:
            raise ValueError("Invalid data format in trainlist.")
        logging.info(f"Extracted speaker_id: {speaker_id}, sentence_id: {sentence_id}")

        # 保存されたテンソルをロード
        audio_features, articulation_features, phoneme_embeddings = load_saved_features(
            speaker_id, sentence_id, output_dir, list_name
        )
        logging.info(f"audio_features type: {type(audio_features)}, shape: {audio_features.shape}")
        logging.info(f"articulatory_features type: {type(articulation_features)}, shape: {articulation_features.shape}")
        logging.info(f"phoneme_embeddings type: {type(phoneme_embeddings)}, shape: {phoneme_embeddings.shape}")

        # 次元数を計算
        audio_dim = audio_features.shape[1]
        articulatory_dim = articulation_features.shape[2]
        linguistic_dim = phoneme_embeddings.shape[1]

        logging.info(f"Feature dimensions: audio_dim={audio_dim}, articulatory_dim={articulatory_dim}, linguistic_dim={linguistic_dim}")

        return audio_dim, articulatory_dim, linguistic_dim


def extract_features(config):
    """
    特徴量抽出を行い、テンソルを保存する関数
    :param config: 設定ファイルの情報を含む辞書
    """
    dataset = SpeechDataset(data_dir=config['data_dir'], trainlist_path=config['trainlist'])
    
    # 出力ディレクトリの設定
    output_dirs = {
        'train': config['train_output_dir'],
        'dev': config['dev_output_dir'],
        'test': config['test_output_dir']
    }

    norm_dirs = {key: os.path.join(output_dirs[key], 'norm', key) for key in output_dirs}
    orig_dirs = {key: os.path.join(output_dirs[key], 'orig', key) for key in output_dirs}

    # ディレクトリ構造を作成
    for dirs in [norm_dirs, orig_dirs]:
        for key in dirs:
            os.makedirs(os.path.join(dirs[key], 'in'), exist_ok=True)
            os.makedirs(os.path.join(dirs[key], 'out'), exist_ok=True)
    
    # 特徴量抽出と保存
    for list_name, list_file in zip(['train', 'dev', 'test'], [config['trainlist'], config['devlist'], config['testlist']]):
        with open(list_file, 'r') as f:
            lines = f.readlines()
        # tqdmを使って進捗バーを表示
        for line in tqdm(lines, desc=f"Extracting features for {list_name}"):
            item = line.strip()
            speaker_id, sentence_id = extract_speaker_and_sentence_id(item)
            if speaker_id is None or sentence_id is None:
                continue

            # 特徴量を取得
            audio_features, articulation_features, phoneme_embeddings = dataset(speaker_id, sentence_id)

            # 正規化データの保存
            norm_output_dir = norm_dirs[list_name]
            torch.save(audio_features, os.path.join(norm_output_dir, 'in', f"audio_features_{speaker_id}_{sentence_id}.pt"))
            torch.save(articulation_features, os.path.join(norm_output_dir, 'out', f"articulation_features_{speaker_id}_{sentence_id}.pt"))  # [time_steps, 6, 6]
            torch.save(phoneme_embeddings, os.path.join(norm_output_dir, 'out', f"phoneme_embeddings_{speaker_id}_{sentence_id}.pt"))

            # 元データの保存
            orig_output_dir = orig_dirs[list_name]
            torch.save(audio_features, os.path.join(orig_output_dir, 'in', f"audio_features_{speaker_id}_{sentence_id}.pt"))
            torch.save(articulation_features, os.path.join(orig_output_dir, 'out', f"articulation_features_{speaker_id}_{sentence_id}.pt"))  # [time_steps, 6, 6]
            torch.save(phoneme_embeddings, os.path.join(orig_output_dir, 'out', f"phoneme_embeddings_{speaker_id}_{sentence_id}.pt"))

def load_saved_features(speaker_id, sentence_id, output_dir, list_name):
    """
    保存されたテンソルをロードする関数
    :param speaker_id: 話者コード
    :param sentence_id: 発話番号
    :param output_dir: 特徴量が保存されているベースディレクトリ (例: dump/)
    :param list_name: データセットの種類（train, dev, test）
    :return: ロードされたテンソル
    """
    norm_output_dir = os.path.join(output_dir, 'norm', list_name)
    audio_features = torch.load(os.path.join(norm_output_dir, 'in', f"audio_features_{speaker_id}_{sentence_id}.pt"), weights_only=True)
    articulation_features = torch.load(os.path.join(norm_output_dir, 'out', f"articulation_features_{speaker_id}_{sentence_id}.pt"), weights_only=True)
    phoneme_embeddings = torch.load(os.path.join(norm_output_dir, 'out', f"phoneme_embeddings_{speaker_id}_{sentence_id}.pt"), weights_only=True)

    return audio_features, articulation_features, phoneme_embeddings