import math

from torch import nn
import torch
from torch.autograd import Variable

from utils import TextFeaturizer


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class ConvTransformerLayer(nn.Module):
    def __init__(self, d_model, drop_rate, dff, num_heads):
        super().__init__()
        self.enc_layer = nn.TransformerEncoderLayer(d_model, dim_feedforward=dff, dropout=drop_rate, nhead=num_heads)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=(3,), padding='same')
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        enc_output = self.enc_layer(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        enc_output = enc_output.transpose(-1, -2)
        conv_output = self.conv(enc_output).transpose(-1, -2)
        return self.dropout(self.activation(conv_output))


class Transformer(nn.Module):
    def __init__(self,
                 am_featurizer: TextFeaturizer,
                 lm_featurizer: TextFeaturizer,
                 bert_dim=768,
                 d_model=512,
                 num_heads=4,
                 dff=1024,
                 drop_rate=0.1):
        super().__init__()
        self.lm_featurizer = lm_featurizer
        self.am_featurizer = am_featurizer
        self.embedding = nn.Embedding(am_featurizer.vocab_size, d_model)
        self.pos_embedding = PositionalEncoding(d_model=d_model, dropout=drop_rate, max_len=64)
        self.encoder = nn.TransformerEncoder(
            ConvTransformerLayer(d_model, drop_rate, dff, num_heads),
            num_layers=5
        )
        self.bert_feature_layer = nn.Linear(d_model, bert_dim)
        self.hidden_state = nn.Linear(bert_dim, d_model)
        self.feed_forward = nn.Linear(d_model, lm_featurizer.vocab_size)

    def forward(self, x, mask=None):
        if mask is not None:
            enc_output = self.encoder(self.pos_embedding(self.embedding(x)),
                                      src_key_padding_mask=mask.transpose(-1, 0))
        else:
            enc_output = self.encoder(self.pos_embedding(self.embedding(x)))
        bert_out = self.bert_feature_layer(enc_output)
        hidden_out = self.hidden_state(bert_out)
        final_out = self.feed_forward(hidden_out)
        return final_out, bert_out
