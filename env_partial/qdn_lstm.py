import torch
import torch.nn as nn


class QDNLSTM(nn.Module):
    def __init__(self, obs_shape, n_actions, seq_len=7):
        super().__init__()
        c, h, w = obs_shape
        self.seq_len = seq_len

        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        conv_out_dim = 64 * h * w

        self.lstm = nn.LSTM(input_size=conv_out_dim,
                            hidden_size=256, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x_seq, hidden=None):
        """
        x_seq: [batch, seq_len, c, h, w]
        """
        b, s, c, h, w = x_seq.size()
        x_seq = x_seq.view(b * s, c, h, w)
        conv_out = self.cnn(x_seq)  # [b * s, feat]
        conv_out = conv_out.view(b, s, -1)  # [b, seq_len, feat]
        # lstm_out: [b, seq_len, 256]
        lstm_out, hidden = self.lstm(conv_out, hidden)
        return self.fc(lstm_out[:, -1]), hidden  # 取最後一個時間步的輸出
