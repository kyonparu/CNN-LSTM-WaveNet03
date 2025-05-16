import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import logging

# ログの重複設定を防止（Jupyterなどを考慮）
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0
        )

    def forward(self, x):
        pad = (self.kernel_size - 1) * self.dilation
        x = F.pad(x, (pad, 0))  # 右側にパディング
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.filter_conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.gate_conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.res_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        tanh_out = torch.tanh(self.filter_conv(x))
        sigm_out = torch.sigmoid(self.gate_conv(x))
        z = tanh_out * sigm_out
        res = self.res_conv(z)
        skip = self.skip_conv(z)

        return (x + res), skip


class WaveNet(nn.Module):
    def __init__(self, in_channels=128, res_channels=64, out_channels=80, 
                 kernel_size=2, dilation_cycles=3, layers_per_cycle=10):
        super().__init__()
        self.front_conv = CausalConv1d(in_channels, res_channels, kernel_size=1, dilation=1)
        
        dilations = [2 ** i for i in range(layers_per_cycle)]  # dilation_cyclesを削除
        self.receptive_field = (kernel_size - 1) * sum(dilations) + 1
        logging.info(f"Receptive field: {self.receptive_field}")

        self.res_blocks = nn.ModuleList([ResidualBlock(res_channels, kernel_size, d) for d in dilations])
        self.relu = nn.ReLU()
        self.output_conv1 = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.output_conv2 = nn.Conv1d(res_channels, out_channels, kernel_size=1)

    def forward(self, x, target=None):
        logging.info(f"WaveNet input shape: {x.shape}")
        x = self.front_conv(x)
        logging.info(f"After front_conv: {x.shape}")
        skip_connections = []
        for block in self.res_blocks:
            x, skip = block(x)
            #logging.info(f"After ResidualBlock: x={x.shape}, skip={skip.shape}")
            skip_connections.append(skip)
        out = sum(skip_connections)
        logging.info(f"After skip connection sum: {out.shape}")
        out = self.relu(out)
        out = self.output_conv1(out)
        logging.info(f"After output_conv1: {out.shape}")
        out = self.relu(out)
        out = self.output_conv2(out)
        logging.info(f"After output_conv2: {out.shape}")

        # 出力の時間次元をターゲットに揃える
        if target is not None:
            target_time_steps = target.size(2)
            logging.info(f"Target time steps: {target_time_steps}")
            logging.info(f"Output time steps before trimming: {out.size(2)}")
            if out.size(2) > target_time_steps:
                out = out[:, :, :target_time_steps]
                logging.info(f"Trimmed output shape: {out.shape}")

        return out


class CNN_LSTM_WaveNet(nn.Module):
    def __init__(self, in_channels, cnn_channels, lstm_hidden, output_dim, wavenet_channels, embed_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, cnn_channels, kernel_size=(3, 3), padding=(1, 1)), nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=(3, 3), padding=(1, 1)), nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=(3, 3), padding=(1, 1)), nn.BatchNorm2d(cnn_channels),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(cnn_channels + embed_dim, lstm_hidden, batch_first=True, bidirectional=True)

        self.output_layer = nn.Linear(lstm_hidden * 2, output_dim)  # メルスペクトログラム次元に対応
        self.wavenet = WaveNet(in_channels=output_dim, res_channels=wavenet_channels, out_channels=output_dim)

    def cnn_forward(self, articulation_features):
        cnn_outputs = self.cnn(articulation_features)  # [B, C, T, 6]
        return cnn_outputs.mean(dim=-1)  # -> [B, C, T]

    def lstm_forward(self, lstm_inputs):
        lstm_outputs, _ = self.lstm(lstm_inputs)
        return lstm_outputs

    def forward(self, articulation_features, linguistic_features, target=None):
        cnn_outputs = self.cnn_forward(articulation_features)
        linguistic_features = linguistic_features.permute(0, 2, 1)  # [B, D, T]

        if cnn_outputs.size(2) != linguistic_features.size(2):
            raise ValueError(f"Time dimension mismatch: CNN {cnn_outputs.size(2)} vs Ling {linguistic_features.size(2)}")

        combined_features = torch.cat((cnn_outputs, linguistic_features), dim=1)  # [B, C+D, T]
        combined_features = combined_features.permute(0, 2, 1)  # [B, T, C+D]

        lstm_outputs = self.lstm_forward(combined_features)  # [B, T, H*2]
        predicted = self.output_layer(lstm_outputs)  # [B, T, output_dim]

        predicted = predicted.permute(0, 2, 1)  # [B, output_dim, T]
        wavenet_outputs = self.wavenet(predicted, target=target)
        return wavenet_outputs.permute(0, 2, 1)  # [B, T, output_dim]
