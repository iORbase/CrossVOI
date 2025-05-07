# -*- coding: utf-8 -*-
import sys
sys.path.append("../..")

from Human.trans_utils.layers_utils import *
from Human.trans_utils.mha_layer import *

import torch
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 num_heads,
                 d_ff,
                 dropout_rate,
                 **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        self.mha_layer = MultiHeadAttention(self.d_model,
                                            self.num_heads,
                                            self.dropout_rate)

        self.poswiseff_layer = PosWiseFF(self.d_model,
                                         self.d_ff,
                                         self.dropout_rate)

        self.layernorm1 = torch.nn.LayerNorm(normalized_shape=self.d_model, eps=1e-5)
        self.layernorm2 = torch.nn.LayerNorm(normalized_shape=self.d_model, eps=1e-5)

    def forward(self, inputs, mask=None, DistMatrix=None):
        x = inputs

        attn_out, attn_w = self.mha_layer([x, x, x], mask=mask, DistMatrix=DistMatrix)
        sublayer1_out = self.layernorm1(x + attn_out)  # [batch_size, input_seq_len, d_model]

        poswiseff_out = self.poswiseff_layer(sublayer1_out)
        sublayer2_out = self.layernorm2(sublayer1_out + poswiseff_out)  # [batch_size, input_seq_len, d_model]

        return sublayer2_out, attn_w

class Encoder(nn.Module):
    def __init__(self,
                 d_model,
                 num_layers,
                 num_heads,
                 d_ff,
                 dropout_rate,
                 device,
                 **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.device = device

        self.enc_layers = torch.nn.ModuleList([EncoderLayer(self.d_model, self.num_heads, self.d_ff,
                                        self.dropout_rate).to(device)
                           for i in range(self.num_layers)])

    def forward(self, inputs, mask=None, DistMatrix=None):
        x = inputs
        attention_weights = {}

        for layer_index in range(self.num_layers):
            layer = self.enc_layers[layer_index]
            if layer_index == 0:
                x, attn_enc_w = layer(x, mask, DistMatrix)
            else:
                x, attn_enc_w = layer(x, mask)
            attention_weights['encoder_layer{}'.format(layer_index + 1)] = attn_enc_w

        return x, attention_weights