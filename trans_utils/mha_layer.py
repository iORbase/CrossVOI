# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout_rate, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, inputs, mask=None, DistMatrix=None):
        query, key, value = inputs

        dim_k = key.shape[-1]
        scale = 1 / np.sqrt(dim_k)

        matmul_q_transp_k = torch.matmul(query, key.transpose(2,3))
        scaled_attention_scores = matmul_q_transp_k * scale

        if mask is not None:
            scaled_attention_scores += (mask * -1e9)

        if DistMatrix is not None:
            scaled_attention_scores = DistMatrix.unsqueeze(1) * scaled_attention_scores

        attention_weights = self.softmax(scaled_attention_scores)
        attention_weights = self.dropout_layer(attention_weights)

        attention_output = torch.matmul(attention_weights, value)
        return attention_output, attention_weights

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads, dropout_rate, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)

        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        assert d_model % num_heads == 0

        self.attention = ScaledDotProductAttention(self.dropout_rate)

        self.query_dense = torch.nn.Linear(in_features=d_model, out_features=d_model)
        self.key_dense = torch.nn.Linear(in_features=d_model, out_features=d_model)
        self.value_dense = torch.nn.Linear(in_features=d_model, out_features=d_model)

        self.out = torch.nn.Linear(in_features=d_model, out_features=d_model)
        self.dropout_layer = torch.nn.Dropout(self.dropout_rate)

    def forward(self, inputs, mask=None, DistMatrix=None):
        query = inputs[0]
        key = inputs[1]
        value = inputs[2]

        batch_size = query.shape[0]

        query = self.query_dense(query)  # [batch_size, seq_len, d_model]
        key = self.key_dense(key)  # [batch_size, seq_len, d_model]
        value = self.value_dense(value)  # [batch_size, seq_len, d_model]

        query = query.view(batch_size, -1, self.num_heads, self.d_model // self.num_heads).transpose(2, 1).contiguous()
        key = key.view(batch_size, -1, self.num_heads, self.d_model // self.num_heads).transpose(2, 1).contiguous()
        value = value.view(batch_size, -1, self.num_heads, self.d_model // self.num_heads).transpose(2, 1).contiguous()

        attention_output, attention_weights = self.attention([query, key, value], mask=mask, DistMatrix=DistMatrix)
        # # [batch_size, num_heads, seq_len_q, head_dim], # [batch_size, num_heads, seq_len_q, seq_len_k]

        attention_output = attention_output.transpose(2, 1).contiguous() # [batch_size, seq_len_q, num_heads, head_dim]
        attention_output = attention_output.view(batch_size, -1, self.d_model)  # [batch_size, seq_len_q, d_model]

        mh_attention_output = self.dropout_layer(self.out(attention_output))  # [batch_size, seq_len_q, d_model]

        return mh_attention_output, attention_weights