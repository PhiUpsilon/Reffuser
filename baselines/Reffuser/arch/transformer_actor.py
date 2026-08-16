import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerActor(nn.Module):
    def __init__(self, seq_len, num_nodes, num_features,
                 e_layers=3, n_heads=16, d_model=128, d_ff=256,
                 dropout=0.2, fc_dropout=0.2, head_dropout=0.0, **kwargs):
        super(TransformerActor, self).__init__()
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.d_model = d_model

        # Input embedding
        self.input_embedding = nn.Linear(num_nodes * num_features, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len=seq_len)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        # Output layer
        self.output_layer = nn.Linear(d_model, num_nodes * num_features)

        # Dropout layers
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.head_dropout = nn.Dropout(head_dropout)

    def forward(self, x, future_data, batch_seen, epoch, train):
        # x shape: [B, L, N, C]
        B, L, N, C = x.shape
        x = x.view(B, L, N * C)  # Reshape to [B, L, N*C]

        # Input embedding
        x = self.input_embedding(x)  # [B, L, d_model]

        # Positional encoding
        x = self.positional_encoding(x)

        # Transformer encoder
        x = x.permute(1, 0, 2)  # [L, B, d_model]
        x = self.transformer_encoder(x)  # [L, B, d_model]

        # Output layer
        x = x.permute(1, 0, 2)  # [B, L, d_model]
        x = self.fc_dropout(x)
        x = self.output_layer(x)  # [B, L, N*C]

        # Reshape back to [B, L, N, C]
        x = x.view(B, L, N, C)
        return x