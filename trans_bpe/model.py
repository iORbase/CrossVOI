import torch
import torch.nn as nn
import sys
sys.path.append("../../..")
from Human.trans_utils.embedding_layer import EmbeddingLayer
from Human.trans_utils.transformer_encoder import Encoder
from Human.trans_utils.cross_attention_transformer_encoder import CrossAttnBlock
from Human.trans_utils.output_block import OutputMLP
from Human.trans_utils.layers_utils import attn_pad_mask
from torch.optim.lr_scheduler import StepLR

class Trans(nn.Module):
    def __init__(self,
                 d_model,
                 prot_atten_layers, prot_atten_heads,
                 smile_atten_layers, smile_atten_heads,
                 cross_atten_layers, cross_atten_heads,
                 prot_d_ff, smiles_d_ff,
                 x1_num_heads, x2_num_heads,
                 x1_d_ff, x2_d_ff,
                 mlp_depth, mlp_units,
                 dropout_rate,
                 lr, betas, eps, weight_decay,
                 atomemb, resemb,
                 is_grad_dec, step, gamma,
                 device):

        super(Trans, self).__init__()
        self.device = torch.device(device)
        self.is_grad_dec = is_grad_dec
        self.Get_mask = attn_pad_mask().to(self.device)

        self.AtomEmb = EmbeddingLayer(
                              d_model=d_model,
                              dropout_rate=dropout_rate,
                              device=device,
                              num_embeddings=atomemb + 2)

        self.ResidueEmb = EmbeddingLayer(
                              d_model=d_model,
                              dropout_rate=dropout_rate,
                              device=device,
                              num_embeddings=resemb + 2)

        self.Atom_atten = Encoder(d_model=d_model,
                               num_layers=smile_atten_layers,
                               num_heads=smile_atten_heads,
                               d_ff=smiles_d_ff,
                               dropout_rate=dropout_rate,
                               device=device
                               )

        self.Residue_atten = Encoder(d_model=d_model,
                               num_layers=prot_atten_layers,
                               num_heads=prot_atten_heads,
                               d_ff=prot_d_ff,
                               dropout_rate=dropout_rate,
                               device=device
                               )

        self.Cross_atten = CrossAttnBlock(d_model=d_model,
                                     num_layers=cross_atten_layers,
                                     cross_num_heads=cross_atten_heads,
                                     x1_num_heads=x1_num_heads,
                                     x2_num_heads=x2_num_heads,
                                     x1_d_ff=x1_d_ff,
                                     x2_d_ff=x2_d_ff,
                                     dropout_rate=dropout_rate,
                                     device=device)

        self.Out = OutputMLP(d_model=d_model,
                             mlp_depth=mlp_depth,
                             mlp_units=mlp_units,
                             dropout_rate=dropout_rate).to(self.device)

        self.opt = torch.optim.RAdam(params=self.parameters(), betas=betas, eps=eps, lr=lr, weight_decay=weight_decay)
        # self.scheduler = ExponentialLR(self.opt, gamma=0.99)
        if is_grad_dec:
            self.scheduler = StepLR(self.opt, step_size=step, gamma=gamma)
        self.BCEloss = nn.BCELoss()

    def forward(self, atomF, residueF, atomDistMatrix, resDistMatrix):
        atom_mask = self.Get_mask(atomF)
        residue_mask = self.Get_mask(residueF)

        #print(residueF.shape)
        #for i in residueF:
        #    print(i.nonzero().shape[0])

        embed_atom = self.AtomEmb(atomF)
        embed_resd = self.ResidueEmb(residueF)

        embed_atom, _ = self.Atom_atten(inputs=embed_atom, mask=atom_mask, DistMatrix=atomDistMatrix)
        embed_resd, res_attn_w = self.Residue_atten(inputs=embed_resd, mask=residue_mask, DistMatrix=resDistMatrix)

        #print(res_attn_w.shape)

        f, _ = self.Cross_atten([embed_atom, embed_resd], residue_mask, atom_mask)
        out = self.Out(f)

        return out

    def train_model(self, atomF, residueF, EC50, atomDistMatrix, resDistMatrix):
        out = self(atomF, residueF, atomDistMatrix, resDistMatrix)
        loss = self.BCEloss(out, EC50)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        if self.is_grad_dec:
            self.scheduler.step()

        return loss.item()
