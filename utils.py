import torch
import os
import matplotlib.pyplot as plt

def save_checkpoint(model, optimizer, epoch, path='checkpoint.pth'):
    """
    モデルのチェックポイントを保存する関数
    :param model: 学習済みモデル
    :param optimizer: オプティマイザ
    :param epoch: 現在のエポック数
    :param path: 保存先のパス
    """
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(state, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(model, optimizer, path='checkpoint.pth'):
    """
    モデルのチェックポイントを読み込む関数
    :param model: 学習済みモデル
    :param optimizer: オプティマイザ
    :param path: チェックポイントファイルのパス
    :return: 再開するエポック数
    """
    if os.path.isfile(path):
        state = torch.load(path)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        epoch = state['epoch']
        print(f"Checkpoint loaded from {path} at epoch {epoch}")
        return epoch
    else:
        print(f"No checkpoint found at {path}")
        return 0

def plot_loss(training_losses, validation_losses=None, save_path=None):
    """
    損失のプロットを行う関数
    :param training_losses: トレーニング損失のリスト
    :param validation_losses: バリデーション損失のリスト（省略可能）
    :param save_path: プロット画像の保存先パス（省略可能）
    """
    plt.figure(figsize=(10, 6))
    plt.plot(training_losses, label='Training Loss')
    if validation_losses:
        plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"Loss plot saved to {save_path}")
    else:
        plt.show()


def print_lr(optimizer):
    """
    現在の学習率を表示する関数
    :param optimizer: 使用中のオプティマイザ
    """
    for param_group in optimizer.param_groups:
        print(f"Current Learning Rate: {param_group['lr']}")

def print_metrics(mcd, gpe, vde, ffe):
    """
    評価メトリクスを表示する関数
    :param mcd: メルケプストラム歪
    :param gpe: グロスピッチ誤差
    :param vde: ボイス決定エラー
    :param ffe: F0フレームエラー
    """
    print(f"Mel-Cepstral Distortion (MCD): {mcd:.4f}")
    print(f"Gross Pitch Error (GPE): {gpe:.4f}")
    print(f"Voicing Decision Error (VDE): {vde:.4f}")
    print(f"F0 Frame Error (FFE): {ffe:.4f}")

def extract_speaker_and_sentence_id(filename):
    """
    ファイル名からスピーカーIDと発話番号を抽出する関数
    :param filename: ファイル名（例: 'ATR503M0102_051'）
    :return: スピーカーID（例: 'M0102'）と発話番号（例: '051'）
    """
    try:
        speaker_id = filename.split('_')[0][6:]  # 例: 'M0102'
        sentence_id = filename.split('_')[1]  # 例: '051'
        return speaker_id, sentence_id
    except IndexError:
        print(f"Error processing filename: {filename}")
        return None, None
