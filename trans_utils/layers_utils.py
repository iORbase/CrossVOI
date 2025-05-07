import torch
import torch.nn as nn

class PosWiseFF(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate, **kwargs):
        super(PosWiseFF, self).__init__(**kwargs)
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        self.gelu = torch.nn.GELU()
        self.dense_1 = torch.nn.Linear(in_features=d_model, out_features=self.d_ff)
        self.dense_2 = torch.nn.Linear(in_features=self.d_ff, out_features=d_model)

        self.dropout_layer_1 = nn.Dropout(self.dropout_rate)
        self.dropout_layer_2 = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        x = self.gelu(self.dense_1(x))
        x = self.dropout_layer_1(x)
        x = self.gelu(self.dense_2(x))
        x = self.dropout_layer_2(x)

        return x

class attn_pad_mask(nn.Module):
    def __init__(self, **kwargs):
        super(attn_pad_mask, self).__init__(**kwargs)

    def forward(self, x):
        return (x == 0)[:, None, None,:].int()

def add_reg_token(x, voc_size):
    reg_token = torch.IntTensor([voc_size + 1]).repeat(x.shape[0], 1)

    return torch.cat([reg_token, x], dim=1)