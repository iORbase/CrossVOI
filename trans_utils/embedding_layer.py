import numpy as np
import torch
import torch.nn as nn

class EmbeddingLayer(nn.Module):
    def __init__(self, d_model, dropout_rate, device, num_embeddings, **kwargs):
        super(EmbeddingLayer, self).__init__(**kwargs)

        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.device = device

        self.emb_layer = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=d_model).to(self.device)
        self.dropout_layer = torch.nn.Dropout(self.dropout_rate).to(self.device)

    def position_embedding(self, max_len):
        angle = torch.arange(self.d_model, dtype=torch.float32).to(self.device)
        angle = 10000 ** (2 * (angle / self.d_model))
        angle = torch.arange(max_len, dtype=torch.float32).unsqueeze(1).to(self.device) / angle
        values = torch.stack([torch.sin(angle[:, 0::2]), torch.cos(angle[:, 1::2])], dim=2)
        pos_enc = torch.reshape(values, shape=[values.shape[0], -1])
        return pos_enc.float()

    def forward(self, sequences):
        max_len = sequences.shape[1]

        output = self.emb_layer(sequences) * np.sqrt(self.d_model)

        output = output + self.position_embedding(max_len)
        output = self.dropout_layer(output)

        return output

