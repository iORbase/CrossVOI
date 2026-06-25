
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from datetime import datetime

class SelfAttentionLayer(nn.Module):
    """自注意力层，用于处理蛋白质或小分子的特征"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # 使输入形状为 (batch, seq_len, embed_dim)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attn_mask=None):
        # x形状: (batch_size, seq_len, embed_dim)
        attn_output, attn_weights = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(attn_output)  # 残差连接
        x = self.norm(x)  # 层归一化
        return x, attn_weights

class CrossAttentionLayer(nn.Module):
    """交叉注意力层，用于融合蛋白质和小分子的特征"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, attn_mask=None):
        # query: 通常是蛋白质特征 (batch_size, L, embed_dim)
        # key/value: 通常是小分子特征 (batch_size, S, embed_dim)
        attn_output, attn_weights = self.cross_attn(query, key, value, attn_mask=attn_mask)
        output = query + self.dropout(attn_output)  # 残差连接
        output = self.norm(output)  # 层归一化
        return output, attn_weights

class ProteinLigandInteractionModel(nn.Module):
    """蛋白质和小分子相互作用预测模型"""
    def __init__(self, 
                 protein_embed_dim=1024,  # 蛋白质特征向量维度
                 ligand_embed_dim=768,    # 小分子特征向量维度
                 hidden_dim=256,          # 隐藏层维度
                 protein_self_attn_layers=2,  # 蛋白质自注意力层数
                 ligand_self_attn_layers=2,   # 小分子自注意力层数
                 cross_attn_layers=2,         # 交叉注意力层数
                 num_heads=4,                 # 注意力头数
                 fc_hidden_dims=[512, 256],   # 全连接层隐藏维度
                 dropout_rate=0.1,
                 max_protein_len=400,         # 蛋白质最大长度
                 max_ligand_len=200):         # 小分子最大长度
        super().__init__()
        
        # 保存超参数
        self.hparams = {
            'protein_embed_dim': protein_embed_dim,
            'ligand_embed_dim': ligand_embed_dim,
            'hidden_dim': hidden_dim,
            'protein_self_attn_layers': protein_self_attn_layers,
            'ligand_self_attn_layers': ligand_self_attn_layers,
            'cross_attn_layers': cross_attn_layers,
            'num_heads': num_heads,
            'fc_hidden_dims': fc_hidden_dims,
            'dropout_rate': dropout_rate,
            'max_protein_len': max_protein_len,
            'max_ligand_len': max_ligand_len
        }
        
        # 特征映射层 - 将不同维度的特征映射到相同的隐藏维度
        self.protein_projection = nn.Linear(protein_embed_dim, hidden_dim)
        self.ligand_projection = nn.Linear(ligand_embed_dim, hidden_dim)
        
        # 蛋白质自注意力层
        self.protein_self_attn = nn.ModuleList([
            SelfAttentionLayer(hidden_dim, num_heads, dropout_rate)
            for _ in range(protein_self_attn_layers)
        ])
        
        # 小分子自注意力层
        self.ligand_self_attn = nn.ModuleList([
            SelfAttentionLayer(hidden_dim, num_heads, dropout_rate)
            for _ in range(ligand_self_attn_layers)
        ])
        
        # 交叉注意力层 - 蛋白质关注小分子
        self.protein_cross_attn = nn.ModuleList([
            CrossAttentionLayer(hidden_dim, num_heads, dropout_rate)
            for _ in range(cross_attn_layers)
        ])
        
        # 交叉注意力层 - 小分子关注蛋白质
        self.ligand_cross_attn = nn.ModuleList([
            CrossAttentionLayer(hidden_dim, num_heads, dropout_rate)
            for _ in range(cross_attn_layers)
        ])
        
        # 全连接网络
        fc_layers = []
        input_dim = 2 * hidden_dim  # 蛋白质和小分子特征拼接
        for curr_hidden_dim in fc_hidden_dims:
            fc_layers.extend([
                nn.Linear(input_dim, curr_hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(curr_hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            input_dim = curr_hidden_dim
        
        # 输出层（二分类）
        fc_layers.append(nn.Linear(input_dim, 1))
        fc_layers.append(nn.Sigmoid())
        
        self.classifier = nn.Sequential(*fc_layers)
        
        # 用于存储注意力权重（可视化用）
        self.attention_weights = {}
        
    def forward(self, protein_features, ligand_features, protein_mask=None, ligand_mask=None, 
                save_attention=False, save_path=None):
        """
        前向传播
        
        参数:
            protein_features: 蛋白质特征 (batch_size, L, protein_embed_dim)
            ligand_features: 小分子特征 (batch_size, S, ligand_embed_dim)
            protein_mask: 蛋白质序列掩码 (batch_size, L)，1表示有效位置，0表示填充
            ligand_mask: 小分子序列掩码 (batch_size, S)，1表示有效位置，0表示填充
            save_attention: 是否保存注意力权重
            save_path: 注意力权重保存路径
        """
        # 重置注意力权重存储
        if save_attention:
            self.attention_weights = {
                'protein_self': [],
                'ligand_self': [],
                'protein_cross': [],
                'ligand_cross': []
            }
        
        # 将不同维度的特征映射到相同的隐藏维度
        protein_x = self.protein_projection(protein_features)
        ligand_x = self.ligand_projection(ligand_features)
        
        # 处理蛋白质自注意力
        for i, attn_layer in enumerate(self.protein_self_attn):
            # 创建注意力掩码（忽略填充部分）
            attn_mask = None
            if protein_mask is not None:
                # 转换为多头注意力所需的掩码格式 (batch_size * num_heads, L, L)
                attn_mask = self._create_attn_mask(protein_mask, protein_mask)
            
            protein_x, attn_weights = attn_layer(protein_x, attn_mask)
            
            # 保存注意力权重
            if save_attention:
                self.attention_weights['protein_self'].append(attn_weights.detach())
        
        # 处理小分子自注意力
        for i, attn_layer in enumerate(self.ligand_self_attn):
            # 创建注意力掩码
            attn_mask = None
            if ligand_mask is not None:
                attn_mask = self._create_attn_mask(ligand_mask, ligand_mask)
            
            ligand_x, attn_weights = attn_layer(ligand_x, attn_mask)
            
            # 保存注意力权重
            if save_attention:
                self.attention_weights['ligand_self'].append(attn_weights.detach())
        
        # 交叉注意力 - 蛋白质关注小分子
        cross_protein_x = protein_x
        for i, attn_layer in enumerate(self.protein_cross_attn):
            # 创建注意力掩码
            attn_mask = None
            if protein_mask is not None and ligand_mask is not None:
                attn_mask = self._create_attn_mask(protein_mask, ligand_mask)
            
            cross_protein_x, attn_weights = attn_layer(
                query=cross_protein_x, 
                key=ligand_x, 
                value=ligand_x,
                attn_mask=attn_mask
            )
            
            # 保存注意力权重
            if save_attention:
                self.attention_weights['protein_cross'].append(attn_weights.detach())
        
        # 交叉注意力 - 小分子关注蛋白质
        cross_ligand_x = ligand_x
        for i, attn_layer in enumerate(self.ligand_cross_attn):
            # 创建注意力掩码
            attn_mask = None
            if ligand_mask is not None and protein_mask is not None:
                attn_mask = self._create_attn_mask(ligand_mask, protein_mask)
            
            cross_ligand_x, attn_weights = attn_layer(
                query=cross_ligand_x, 
                key=protein_x, 
                value=protein_x,
                attn_mask=attn_mask
            )
            
            # 保存注意力权重
            if save_attention:
                self.attention_weights['ligand_cross'].append(attn_weights.detach())
        
        # 池化操作 - 使用平均池化
        protein_pooled = torch.mean(cross_protein_x, dim=1)  # (batch_size, hidden_dim)
        ligand_pooled = torch.mean(cross_ligand_x, dim=1)    # (batch_size, hidden_dim)
        
        # 特征拼接
        combined = torch.cat([protein_pooled, ligand_pooled], dim=1)  # (batch_size, 2*hidden_dim)
        
        # 分类预测
        output = self.classifier(combined)  # (batch_size, 1)
        
        # 如果需要，保存注意力权重到文件
        if save_attention and save_path is not None:
            self.save_attention_weights(save_path)
        
        return output
    
    def _create_attn_mask(self, query_mask, key_mask):
        """
        创建注意力掩码，防止模型关注填充部分
        
        参数:
            query_mask: 查询序列掩码 (batch_size, query_len)
            key_mask: 键序列掩码 (batch_size, key_len)
        
        返回:
            attn_mask: 注意力掩码 (batch_size * num_heads, query_len, key_len)
        """
        # 计算掩码矩阵 (batch_size, query_len, key_len)
        attn_mask = torch.bmm(
            query_mask.unsqueeze(2).float(),  # (batch_size, query_len, 1)
            key_mask.unsqueeze(1).float()     # (batch_size, 1, key_len)
        )
        
        # 将0转换为-1e9，1转换为0，符合PyTorch注意力掩码格式
        attn_mask = (1 - attn_mask) * -1e9
        
        # 扩展掩码以适应多头注意力：(batch_size, query_len, key_len) -> (batch_size * num_heads, query_len, key_len)
        batch_size, query_len, key_len = attn_mask.size()
        num_heads = self.hparams['num_heads']
        
        # 为每个注意力头复制一份掩码
        attn_mask = attn_mask.repeat_interleave(num_heads, dim=0)
        
        return attn_mask
    
    def save_attention_weights(self, save_path):
        """
        保存注意力权重到文件，方便可视化
        
        参数:
            save_path: 保存路径
        """
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)
        
        # 获取当前时间作为文件名的一部分
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存每种注意力权重
        for attn_type, weights_list in self.attention_weights.items():
            for layer_idx, weights in enumerate(weights_list):
                # weights形状: (batch_size, num_heads, seq_len_q, seq_len_k)
                file_name = f"{attn_type}_layer_{layer_idx}_{timestamp}.npz"
                file_path = os.path.join(save_path, file_name)
                
                # 转换为numpy数组并保存
                np.savez(
                    file_path,
                    weights=weights.cpu().numpy(),
                    type=attn_type,
                    layer=layer_idx
                )
        
        print(f"注意力权重已保存到 {save_path}")