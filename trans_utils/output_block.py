# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class OutputMLP(nn.Module):
    def __init__(self,
                 d_model,
                 mlp_depth,
                 mlp_units,
                 dropout_rate,
                 **kwargs):
        super(OutputMLP, self).__init__(**kwargs)

        mlp_units.insert(0, 2 * d_model)
        self.mlp_depth = mlp_depth
        self.mlp_units = mlp_units
        self.dropout_rate = dropout_rate
        self.mlp_head = nn.Sequential()

        for i in range(mlp_depth):
            self.mlp_head.add_module("linear %d " % i, nn.Linear(self.mlp_units[i], self.mlp_units[i+1]))
            self.mlp_head.add_module("gelu %d " % i, nn.GELU())
            self.mlp_head.add_module("dropout %d " % i, nn.Dropout(dropout_rate))

        self.mlp_head.add_module("linear %d " % mlp_depth, nn.Linear(self.mlp_units[-1], 1))
        self.mlp_head.add_module("sigmoid", nn.Sigmoid())

    def forward(self, inputs):
        prot_input = inputs[0][:,0,:]
        smiles_input = inputs[1][:,0,:]

        concat_input = torch.cat([prot_input, smiles_input], dim=1)
        output = self.mlp_head(concat_input)

        return output