import torch.nn as nn
import torch
import numpy as np
class _TransformerEncode(nn.Module):
    def __init__(self, input_dim, embd_dim, ff_dim, num_head, num_layer) -> None:
        super(_TransformerEncode,self).__init__()
        self.peak_mlp = nn.Sequential(
            nn.Linear(1, embd_dim),
            nn.ReLU(),
            nn.Linear(embd_dim,embd_dim)
        )
        self.pos_embedding = nn.Linear(input_dim, embd_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model= embd_dim, nhead= num_head, dim_feedforward= ff_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers = num_layer)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(embd_dim, 128)

    def forward(self,peak_input):
        if isinstance(peak_input, np.ndarray):
            peak_input = torch.tensor(peak_input, dtype=torch.float32)
        peak_pos = peak_input[:,:, 0:1]
        intensity = peak_input[:,:, 1:2]
        intensity_embd = self.peak_mlp(intensity)
        pos_embd = self.pos_embedding(peak_pos)
        x = intensity_embd + pos_embd
        x = x.permute(1,0,2)
        x = self.transformer_encoder(x)
        x = x.permute(1,2,0)
        x = self.pooling(x).squeeze(-1)
        x = self.fc(x)

        return x
