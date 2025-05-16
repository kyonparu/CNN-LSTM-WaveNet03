import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SpeechDataset, load_saved_features  # データセットのロード
from utils import save_checkpoint, plot_loss, print_lr
import logging
import os
import numpy 
import time
from datetime import datetime

# ロギングの設定
logging.basicConfig(level=logging.INFO)

def extract_speaker_and_sentence_id(item):
    parts = item.split('_')
    if len(parts) != 2:
        raise ValueError(f"Invalid item format: {item}")
    speaker_id = parts[0][-5:]
    sentence_id = parts[1]
    return speaker_id, sentence_id

def load_list(list_path):
    with open(list_path, 'r') as f:
        lines = f.readlines()
    items = [line.strip().split() for line in lines]
    #print(f"List items: {items[:5]}")  # 最初の5個のアイテムを表示
    return items

def create_dataloader(list_path, output_dir, batch_size, num_workers=0, list_name=None):
    logging.info(f"Entering create_dataloader with list_path={list_path}, output_dir={output_dir}, list_name={list_name}")
    if list_name is None:
        raise ValueError("list_name must be provided.")

    with open(list_path, 'r') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        item = line.strip()
        speaker_id, sentence_id = extract_speaker_and_sentence_id(item)
        combined_audio_features, articulation_features, phoneme_embeddings = load_saved_features(
            speaker_id, sentence_id, output_dir, list_name
        )
        data.append((combined_audio_features, articulation_features, phoneme_embeddings))

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    logging.info("Exiting create_dataloader")
    return dataloader


def collate_fn(batch):
    logging.info("Entering collate_fn")
    
    combined_audio_features, articulation_features, linguistic_features = zip(*batch)

    # 音響特徴量
    combined_audio_features = torch.stack(combined_audio_features).float()

    # CNN用の調音特徴量
    articulation_features = torch.stack(articulation_features).float()  # [batch_size, time_steps, 6, 6]
    articulation_features = articulation_features.permute(0, 3, 1, 2)

    # 言語特徴量
    linguistic_features = torch.stack(linguistic_features).float()  # [batch_size, time_steps, embed_dim]

    logging.info(f"Exiting collate_fn with shapes: combined_audio_features={combined_audio_features.shape}, articulation_features={articulation_features.shape}, linguistic_features={linguistic_features.shape}")
    return combined_audio_features, articulation_features, linguistic_features


def train_model(model, optimizer, start_epoch, epochs, batch_size, trainlist_path, devlist_path, train_output_dir, dev_output_dir, device, config):
    logging.info("Entering train_model")
    
    use_scheduler = config.get('use_scheduler', True)
    if use_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        logging.info("Learning rate scheduler enabled.")
    else:
        scheduler = None
        logging.info("Learning rate scheduler disabled.")

    logging.info("Creating training dataloader...")
    train_dataloader = create_dataloader(trainlist_path, train_output_dir, batch_size, num_workers=0, list_name='train')
    logging.info("Training dataloader created.")

    logging.info("Creating validation dataloader...")
    dev_dataloader = create_dataloader(devlist_path, dev_output_dir, batch_size, num_workers=0, list_name='dev')
    logging.info("Validation dataloader created.")

    training_losses = []
    validation_losses = []

    for epoch in range(start_epoch, epochs):
        logging.info(f"Starting epoch {epoch+1}/{epochs}...")
        model.train()
        total_loss = 0.0

        # Training loop
        for batch_idx, batch in enumerate(train_dataloader):
            logging.info(f"Processing training batch {batch_idx+1}/{len(train_dataloader)}...")
            loss = process_batch(model, batch, device, is_training=True, optimizer=optimizer)
            total_loss += loss
            logging.info(f"Training loss updated: {total_loss}")

        logging.info(f"Epoch [{epoch+1}/{epochs}] completed. Average Training Loss: {total_loss/len(train_dataloader):.4f}")
        training_losses.append(total_loss / len(train_dataloader))

        # Save checkpoint
        checkpoint_path = config['checkpoint']
        save_checkpoint(model, optimizer, epoch, path=checkpoint_path)
        logging.info(f"Checkpoint saved for epoch {epoch+1} at {checkpoint_path}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(dev_dataloader):
                logging.info(f"Processing validation batch {batch_idx+1}/{len(dev_dataloader)}...")
                loss = process_batch(model, batch, device, is_training=False)
                val_loss += loss
                logging.info(f"Validation loss updated: {val_loss}")

        logging.info(f"Validation Loss for epoch {epoch+1}/{epochs}: {val_loss/len(dev_dataloader):.4f}")
        validation_losses.append(val_loss / len(dev_dataloader))

        # Scheduler step
        scheduler.step()
        logging.info(f"Scheduler step completed for epoch {epoch+1}/{epochs}.")

    # 損失プロットの保存
    plot_dir = config.get('plot_output_dir', 'plots')  # configから保存先を取得（デフォルトは'plots'）
    os.makedirs(plot_dir, exist_ok=True)

    # ファイル名に日付とハイパーパラメータを含める
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = os.path.join(plot_dir, f"loss_plot_lr{config['learning_rate']}_e{epochs}_bs{batch_size}_{current_time}.png")

    plot_loss(training_losses, validation_losses, save_path=plot_file)
    logging.info(f"Loss plot saved to {plot_file}")

    logging.info("Training and validation losses plotted.")
    logging.info("Exiting train_model")


def process_batch(model, batch, device, is_training=True, optimizer=None):
    """
    1つのバッチを処理する関数
    :param model: モデル
    :param batch: バッチデータ
    :param device: 使用するデバイス
    :param is_training: 学習モードかどうか
    :param optimizer: オプティマイザ（学習時のみ必要）
    :return: バッチの損失
    """
    combined_audio_features, articulation_features, linguistic_features = batch
    combined_audio_features, articulation_features, linguistic_features = (
        combined_audio_features.to(device),
        articulation_features.to(device),
        linguistic_features.to(device),
    )
    logging.info("Batch data moved to device.")

    if is_training:
        optimizer.zero_grad()
        logging.info("Optimizer gradients zeroed.")

    # モデルのフォワードパス
    outputs = model(articulation_features, linguistic_features)
    logging.info("Model forward pass completed.")

    # 損失計算
    loss = nn.MSELoss()(outputs, combined_audio_features)
    logging.info(f"Loss computed: {loss.item()}")

    if is_training:
        loss.backward()
        logging.info("Loss backward pass completed.")
        optimizer.step()
        logging.info("Optimizer step completed.")

    return loss.item()
