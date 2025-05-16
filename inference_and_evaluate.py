import torch
import soundfile as sf
import pyworld as pw
import numpy as np
from dataset import SpeechDataset
from model import CNN_LSTM_WaveNet
from scipy.fftpack import dct
import pysptk
import os
import yaml
from utils import extract_speaker_and_sentence_id  # 追加
import logging
import matplotlib.pyplot as plt
from datetime import datetime

# ログの設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# モデルとデータセットの準備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(checkpoint_path, device, config):
    """
    モデルをロードする関数
    :param checkpoint_path: チェックポイントファイルのパス
    :param device: 使用するデバイス (CPU/GPU)
    :param config: 設定ファイルの辞書
    :return: ロードされたモデル
    """
    # データセットの初期化
    dataset = SpeechDataset(data_dir=config['data_dir'], trainlist_path=config['trainlist'])

    # 特徴量の次元数を取得
    audio_dim, articulatory_dim, linguistic_dim = dataset.get_feature_dimensions(
        output_dir=config['train_output_dir'], list_name='train'
    )

    # モデルの初期化
    model = CNN_LSTM_WaveNet(
        in_channels=articulatory_dim,
        cnn_channels=64,
        lstm_hidden=128,
        output_dim=audio_dim,
        wavenet_channels=128,
        embed_dim=linguistic_dim
    ).to(device)

    # チェックポイントのロード
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model

def generate_audio(model, dataset, speaker_id, sentence_id, device):
    # データセットから特徴量を取得
    _, articulation_features, phoneme_embeddings = dataset(speaker_id, sentence_id)

    # CNN用の調音特徴量
    articulation_features = torch.stack([articulation_features]).float()  # [1, time_steps, 6, 6]
    articulation_features = articulation_features.permute(0, 3, 1, 2).to(device)  # [1, 6, 6, time_steps]

    # 言語特徴量
    phoneme_embeddings = torch.stack([phoneme_embeddings]).float().to(device)  # [1, time_steps, embed_dim]

    with torch.no_grad():
        # モデルの推論
        predicted_mgc = model(articulation_features, phoneme_embeddings)  # [1, time_steps, 80]

        # メルスペクトログラムを波形に変換
        predicted_mgc = predicted_mgc.squeeze(0).cpu().numpy().astype(np.float64)  # [time_steps, 80]
        fft_size = 1024
        alpha = 0.42  # サンプリングレートに応じたalpha値
        spectral_envelope = pysptk.mc2sp(predicted_mgc, alpha, fft_size).astype(np.float64)

        # C連続配列に変換
        spectral_envelope = np.ascontiguousarray(spectral_envelope)
        f0 = np.zeros(predicted_mgc.shape[0], dtype=np.float64)  # F0はゼロ（無音）
        aperiodicity = np.ones_like(spectral_envelope, dtype=np.float64)  # 非周期成分は1（完全周期）
        aperiodicity = np.ascontiguousarray(aperiodicity)

        # WORLDで音声を合成
        output_waveform = pw.synthesize(f0, spectral_envelope, aperiodicity, 16000)

    return output_waveform, predicted_mgc

def load_testlist(testlist_path):
    with open(testlist_path, 'r') as f:
        lines = f.readlines()
    test_items = [line.strip() for line in lines]
    return test_items

def calculate_mcd(target, generated):
    """Calculate Mel-Cepstral Distortion (MCD)"""
    mcd = 0
    for t, g in zip(target, generated):
        mcd += np.sqrt(np.sum((t - g)**2))
    return (10.0 / np.log(10)) * (mcd / len(target))

def calculate_gpe(target_f0, generated_f0, threshold=0.2):
    """Calculate Gross Pitch Error (GPE)"""
    target_voiced = target_f0 > 0
    generated_voiced = generated_f0 > 0
    errors = np.abs(target_f0 - generated_f0) / (target_f0 + 1e-10)
    gpe = np.sum((errors > threshold) & target_voiced & generated_voiced) / np.sum(target_voiced)
    return gpe

def calculate_vde(target_voiced, generated_voiced):
    """Calculate Voicing Decision Error (VDE)"""
    return np.sum(target_voiced != generated_voiced) / len(target_voiced)

def calculate_ffe(target_f0, generated_f0):
    """Calculate F0 Frame Error (FFE)"""
    return np.sum((target_f0 == 0) != (generated_f0 == 0)) / len(target_f0)

def evaluate_generated_audio(config, device):
    logging.info("Starting evaluation process...")
    
    # モデルのロード
    logging.info("Loading model...")
    model = load_model(config['checkpoint'], device, config)
    logging.info("Model loaded successfully.")

    # データセットの初期化
    logging.info("Initializing dataset...")
    dataset = SpeechDataset(data_dir=config['data_dir'], trainlist_path=config['testlist'])
    logging.info("Dataset initialized successfully.")

    # テストリストの読み込み
    logging.info("Loading test list...")
    test_items = load_testlist(config['testlist'])
    logging.info(f"Test list loaded successfully. {len(test_items)} items found.")

    # 推定結果の保存先ディレクトリ
    dump_dir = os.path.join("dump", "test")
    os.makedirs(dump_dir, exist_ok=True)

    # スペクトログラムの保存先ディレクトリ（dumpの外）
    date_str = datetime.now().strftime("%Y-%m-%d")
    spectrogram_dir = f"save_spectrogram_{date_str}"
    os.makedirs(spectrogram_dir, exist_ok=True)

    # 評価指標の初期化
    mcd_scores = []

    for idx, item in enumerate(test_items):
        logging.info(f"Processing item {idx + 1}/{len(test_items)}: {item}")

        # ファイル名からスピーカーIDと発話番号を抽出
        speaker_id, sentence_id = extract_speaker_and_sentence_id(item)
        if speaker_id is None or sentence_id is None:
            logging.warning(f"Skipping item {item}: Unable to extract speaker ID or sentence ID.")
            continue

        # ターゲットMGCの読み込み
        target_mgc_path = os.path.join(
            config['test_output_dir'], 'norm', 'test', 'in', f"audio_features_{speaker_id}_{sentence_id}.pt"
        )
        if not os.path.exists(target_mgc_path):
            logging.warning(f"Target MGC file not found: {target_mgc_path}")
            continue

        target_mgc = torch.load(target_mgc_path).numpy()  # ターゲットのメルスペクトログラムをロード

        # 推論結果を生成
        logging.info(f"Generating MGC for speaker {speaker_id}, sentence {sentence_id}...")
        _, articulation_features, phoneme_embeddings = dataset(speaker_id, sentence_id)
        articulation_features = torch.stack([articulation_features]).float().permute(0, 3, 1, 2).to(device)
        phoneme_embeddings = torch.stack([phoneme_embeddings]).float().to(device)

        with torch.no_grad():
            predicted_mgc = model(articulation_features, phoneme_embeddings).squeeze(0).cpu().numpy()

        # 推定されたメルスペクトログラムを保存
        predicted_mgc_path = os.path.join(dump_dir, f"{speaker_id}_{sentence_id}_predicted.npy")
        np.save(predicted_mgc_path, predicted_mgc)
        logging.info(f"Predicted MGC saved: {predicted_mgc_path}")

        # MCDを計算
        mcd = calculate_mcd(target_mgc, predicted_mgc)
        mcd_scores.append(mcd)
        logging.info(f"MCD calculated for speaker {speaker_id}, sentence {sentence_id}: {mcd:.4f}")

    # MCDの平均値を計算
    if mcd_scores:
        average_mcd = np.mean(mcd_scores)
        logging.info(f"Average MCD across all sentences: {average_mcd:.4f}")
    else:
        logging.warning("No MCD scores were calculated.")

    # メルスペクトログラムをプロットして保存
        plot_path = os.path.join(spectrogram_dir, f"{speaker_id}_{sentence_id}_mel_plot.png")
        plot_mel_spectrogram(target_mgc, predicted_mgc, plot_path, speaker_id, sentence_id)
        logging.info(f"Mel-spectrogram plot saved: {plot_path}")

    # 評価結果を返す
    logging.info("Evaluation process completed.")
    return mcd_scores

def plot_mel_spectrogram(target_mgc, predicted_mgc, output_path, speaker_id, sentence_id):
    """
    ターゲットと推定されたメルスペクトログラムをプロットして保存する
    """
    plt.figure(figsize=(12, 6))

    # ターゲットのメルスペクトログラム
    plt.subplot(1, 2, 1)
    plt.imshow(target_mgc.T, aspect='auto', origin='lower', interpolation='none')
    plt.title(f"Target Mel-Spectrogram\nSpeaker: {speaker_id}, Sentence: {sentence_id}")
    plt.colorbar()
    plt.xlabel("Time")
    plt.ylabel("Mel-Cepstrum Coefficients")

    # 推定されたメルスペクトログラム
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_mgc.T, aspect='auto', origin='lower', interpolation='none')
    plt.title(f"Predicted Mel-Spectrogram\nSpeaker: {speaker_id}, Sentence: {sentence_id}")
    plt.colorbar()
    plt.xlabel("Time")
    plt.ylabel("Mel-Cepstrum Coefficients")

    # 保存
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    config_path = 'config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    logging.info("Starting evaluation script...")
    metrics = evaluate_generated_audio(config, device)

    # numpy.float64をfloatに変換
    metrics = tuple([float(score) for score in metric_list] for metric_list in metrics)

    # ログに出力
    logging.info(f"MCD scores: [{', '.join([f'{score:.4f}' for score in metrics[0]])}]")
    logging.info(f"GPE scores: [{', '.join([f'{score:.4f}' for score in metrics[1]])}]")
    logging.info(f"VDE scores: [{', '.join([f'{score:.4f}' for score in metrics[2]])}]")
    logging.info(f"FFE scores: [{', '.join([f'{score:.4f}' for score in metrics[3]])}]")
    logging.info("Evaluation script completed.")
