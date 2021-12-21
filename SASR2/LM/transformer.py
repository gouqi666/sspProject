from torch import nn
import torch

if __name__ == '__main__':
    layer = nn.Embedding(1200, 512)
    encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=512, nhead=8), num_layers=8)
    x = layer(torch.tensor([[1, 2, 3]], dtype=torch.int))
    print(x.shape)
    print(encoder(x).shape)
