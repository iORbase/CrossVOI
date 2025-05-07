# -*- coding: utf-8 -*-
import sys
sys.path.append("../..")

from Human.trans_utils.layers_utils import *
from Human.trans_utils.mha_layer import *

import torch
import torch.nn as nn

class CrossAttnLayer(nn.Module):
    def __init__(self,
                 d_model,
                 cross_num_heads,
                 x1_num_heads, x2_num_heads,
                 x1_d_ff, x2_d_ff,
                 dropout_rate,
                 device,
                 **kwargs):
        super(CrossAttnLayer, self).__init__(**kwargs)

        self.d_model = d_model
        self.cross_num_heads = cross_num_heads
        self.x1_num_heads = x1_num_heads
        self.x2_num_heads = x2_num_heads
        self.x1_d_ff = x1_d_ff
        self.x2_d_ff = x2_d_ff
        self.dropout_rate = dropout_rate
        self.device = device

        self.mha_layer_1 = MultiHeadAttention(self.d_model, self.cross_num_heads, self.dropout_rate)
        self.mha_layer_2 = MultiHeadAttention(self.d_model, self.cross_num_heads, self.dropout_rate)

        self.ln_1 = nn.LayerNorm(normalized_shape=self.d_model, eps=1e-5)
        self.ln_2 = nn.LayerNorm(normalized_shape=self.d_model, eps=1e-5)
        self.ln_3 = nn.LayerNorm(normalized_shape=self.d_model, eps=1e-5)
        self.ln_4 = nn.LayerNorm(normalized_shape=self.d_model, eps=1e-5)
        self.ln_5 = nn.LayerNorm(normalized_shape=self.d_model, eps=1e-5)
        self.ln_6 = nn.LayerNorm(normalized_shape=self.d_model, eps=1e-5)

        self.mha_layer_3 = MultiHeadAttention(self.d_model, self.x1_num_heads, self.dropout_rate)
        self.mha_layer_4 = MultiHeadAttention(self.d_model, self.x2_num_heads, self.dropout_rate)

        self.poswiseff_layer_1 = PosWiseFF(self.d_model, self.x1_d_ff, self.dropout_rate)
        self.poswiseff_layer_2 = PosWiseFF(self.d_model, self.x2_d_ff, self.dropout_rate)

    def rearrange_qkv(self, input1, input2):
        input1_pred_token = input1[:, 0, None,:]
        input1_tokens = input1[:, 1:, :]
        input2_pred_token = input2[:, 0, None, :]
        input2_tokens = input2[:, 1:, :]

        return input1_pred_token, input1_tokens, input2_pred_token, input2_tokens

    def forward(self, inputs, mask_x12=None, mask_x21=None):
        x1_p_t, x1_t, x2_p_t, x2_t = self.rearrange_qkv(inputs[0], inputs[1])

        x12_qkv = torch.cat([x1_p_t, x2_t], dim=1)
        x21_qkv = torch.cat([x2_p_t, x1_t], dim=1)

        attn_x12_out, attn_x12_w = self.mha_layer_1([x12_qkv[:, 0, None, :],
                                                     x12_qkv,
                                                     x12_qkv],
                                                    mask=mask_x12)

        attn_x21_out, attn_x21_w = self.mha_layer_2([x21_qkv[:, 0, None, :],
                                                     x21_qkv,
                                                     x21_qkv],
                                                    mask=mask_x21)


        #print('attn_x12_w:' + str(attn_x12_w.shape))
        #print('attn_x21_w:' + str(attn_x21_w.shape))
        #print('mask_x12:' + str(mask_x12.shape))
        #print('mask_x21:' + str(mask_x21))

        #for i in mask_x12:
        #    for j in i:
        #        print(np.count_nonzero(j[0].cpu()))
        '''
        result = []

        for i in attn_x12_w:
            arr = np.zeros(401)
            for j in i:
                #print(np.count_nonzero(j[0].cpu()))
                arr = np.add(arr, j[0].cpu())
            print(arr)
        '''
        x1_p_t_cross = self.ln_1(x1_p_t + attn_x12_out)
        x2_p_t_cross = self.ln_2(x2_p_t + attn_x21_out)

        x1_cross = torch.cat([x1_p_t_cross, x1_t], dim=1)
        x2_cross = torch.cat([x2_p_t_cross, x2_t], dim=1)

        attn_x1_out, attn_x1_w = self.mha_layer_3([x1_cross, x1_cross, x1_cross], mask=mask_x21)
        attn_x2_out, attn_x2_w = self.mha_layer_4([x2_cross, x2_cross, x2_cross], mask=mask_x12)

        x1_cross = self.ln_3(x1_cross + attn_x1_out)
        x2_cross = self.ln_4(x2_cross + attn_x2_out)

        x1_cross_posff_out = self.poswiseff_layer_1(x1_cross)
        x2_cross_posff_out = self.poswiseff_layer_2(x2_cross)

        x1_cross = self.ln_5(x1_cross + x1_cross_posff_out)
        x2_cross = self.ln_6(x2_cross + x2_cross_posff_out)

        return [x1_cross, x2_cross], attn_x12_w, attn_x21_w, attn_x1_w, attn_x2_w

class CrossAttnBlock(nn.Module):
    def __init__(self,
                 d_model,
                 num_layers,
                 cross_num_heads,
                 x1_num_heads, x2_num_heads,
                 x1_d_ff, x2_d_ff,
                 dropout_rate,
                 device,
                 **kwargs):

        super(CrossAttnBlock, self).__init__(**kwargs)

        self.device = device
        self.d_model = d_model
        self.num_layers = num_layers
        self.cross_num_heads = cross_num_heads
        self.x1_num_heads = x1_num_heads
        self.x2_num_heads = x2_num_heads
        self.x1_d_ff = x1_d_ff
        self.x2_d_ff = x2_d_ff
        self.dropout_rate = dropout_rate

        self.cross_attn_layers = torch.nn.ModuleList([CrossAttnLayer(self.d_model,
                                                 self.cross_num_heads,
                                                 self.x1_num_heads, self.x2_num_heads,
                                                 self.x1_d_ff, self.x2_d_ff,
                                                 self.dropout_rate, self.device).to(self.device)
                                  for i in range(self.num_layers)])

    def forward(self, inputs, mask_12=None, mask_21=None):
        x = inputs
        attention_weights = {}

        for layer_index in range(self.num_layers):
            layer = self.cross_attn_layers[layer_index]
            x, x12_attn_w, x21_attn_w, x1_cross_attn_w, x2_cross_attn_w = layer(x, mask_12, mask_21)

            attention_weights['attn_weights_layer{}'.format(layer_index + 1)] = \
                [x12_attn_w, x21_attn_w, x1_cross_attn_w, x2_cross_attn_w]

        return x, attention_weights
