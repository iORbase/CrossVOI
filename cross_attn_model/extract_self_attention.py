# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import csv
import pickle

def extract_self_attention_scores(attention_file, save_path, attention_type='protein_self', aggregation_method='max'):
    """
    从注意力分数文件中提取自注意力分数并保存到CSV文件
    
    Args:
        attention_file: 注意力分数文件路径
        save_path: 结果保存路径
        attention_type: 注意力类型，'protein_self' 或 'ligand_self'
        aggregation_method: 聚合方法，'max' (最大值) 或 'mean' (平均值)
    """
    print(f"正在处理文件: {attention_file}")
    
    # 加载注意力分数文件
    try:
        with open(attention_file, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"文件加载失败: {e}")
        return {}
    
    # 提取数据
    attention_scores_dict = data.get('attention_scores_dict', {})
    attention_weights = data.get('attention_weights', {})
    
    # 检查数据结构
    print(f"attention_weights 包含的键: {list(attention_weights.keys())}")
    
    # 提取指定类型的自注意力权重
    self_attention_weights = attention_weights.get(attention_type, [])
    print(f"{attention_type} 层数: {len(self_attention_weights)}")
    
    if len(self_attention_weights) == 0:
        print(f"警告: {attention_type} 为空列表")
        return {}
    
    # 检查第一个元素的形状
    first_layer = self_attention_weights[0]
    if hasattr(first_layer, 'shape'):
        print(f"第一层形状: {first_layer.shape}")
    
    # 创建一个字典来存储每个OR的自注意力分数
    or_attention_sums = {}
    
    # 处理每个样本
    total_samples = len(attention_scores_dict)
    print(f"总样本数: {total_samples}")
    
    # 检查是否有PyTorch库可用
    has_torch = False
    try:
        import torch
        has_torch = True
    except ImportError:
        pass
    
    # 处理每个样本
    success_count = 0
    fail_count = 0
    for i, (key, sample_info) in enumerate(attention_scores_dict.items()):
        or_name = sample_info.get('or_name', 'unknown')
        voc_name = sample_info.get('voc_name', 'unknown')
        
        try:
            # 从 attention_weights 中获取自注意力矩阵
            attention_matrix = None
            
            # self_attention_weights 的结构：
            # self_attention_weights[层索引] -> 张量 (batch_size, seq_len, seq_len)
            if len(self_attention_weights) > 0:
                # 收集所有层的注意力权重
                layer_attention_matrices = []
                
                for layer_idx in range(len(self_attention_weights)):
                    layer_tensor = self_attention_weights[layer_idx]
                    
                    if i < layer_tensor.size(0):
                        sample_tensor = layer_tensor[i]
                        sample_matrix = sample_tensor.cpu().detach().numpy().astype(np.float32)
                        layer_attention_matrices.append(sample_matrix)
                
                # 如果有多层，取平均
                if len(layer_attention_matrices) > 0:
                    attention_matrix = np.mean(layer_attention_matrices, axis=0)
            
            # 确保矩阵是有效的
            if attention_matrix is None:
                attention_sums = np.array([], dtype=np.float32)
            elif len(attention_matrix.shape) == 3:
                # 对于多头注意力 (num_heads, seq_len, seq_len)，取平均
                attention_matrix = np.mean(attention_matrix, axis=0)
                if len(attention_matrix.shape) == 2 and attention_matrix.shape[0] > 0:
                    if aggregation_method == 'max':
                        attention_sums = np.max(attention_matrix, axis=1)
                    elif aggregation_method == 'mean':
                        attention_sums = np.mean(attention_matrix, axis=1)
                    else:
                        attention_sums = np.array([], dtype=np.float32)
                else:
                    attention_sums = np.array([], dtype=np.float32)
            elif len(attention_matrix.shape) == 2 and attention_matrix.shape[0] > 0:
                # 已经是二维矩阵 (seq_len, seq_len)
                if aggregation_method == 'max':
                    attention_sums = np.max(attention_matrix, axis=1)
                elif aggregation_method == 'mean':
                    attention_sums = np.mean(attention_matrix, axis=1)
                else:
                    attention_sums = np.array([], dtype=np.float32)
            else:
                attention_sums = np.array([], dtype=np.float32)
            
            # 将结果添加到字典中
            or_attention_sums[or_name] = attention_sums
            success_count += 1
            
        except Exception as e:
            or_attention_sums[or_name] = np.array([], dtype=np.float32)
            fail_count += 1
    
    print(f"处理完成: 成功 {success_count}, 失败 {fail_count}")
    
    # 检查是否有非空的注意力分数
    non_empty_count = sum(1 for sums in or_attention_sums.values() if len(sums) > 0)
    print(f"非空注意力分数的OR数: {non_empty_count}")
    
    if non_empty_count > 0:
        for or_name, sums in or_attention_sums.items():
            if len(sums) > 0:
                print(f"示例 - OR: {or_name}, 序列长度: {len(sums)}")
                break
    
    # 保存到CSV文件
    if save_path:
        print(f"正在保存到 {save_path}")
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        # 找出最长的OR序列长度
        max_length = 0
        if or_attention_sums:
            max_length = max(len(sums) for sums in or_attention_sums.values())
        
        # 创建CSV文件并写入数据
        with open(save_path, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            
            # 写入表头
            header = ['OR_name'] + [f'position_{i+1}' for i in range(max_length)]
            csv_writer.writerow(header)
            
            # 写入每个OR的数据
            for or_name, sums in or_attention_sums.items():
                if not isinstance(sums, np.ndarray):
                    sums = np.array(sums, dtype=np.float32)
                
                if len(sums) > 0:
                    padded_sums = np.pad(sums, (0, max_length - len(sums)), 'constant', constant_values=0)
                else:
                    padded_sums = np.zeros(max_length, dtype=np.float32)
                
                row = [or_name] + padded_sums.tolist()
                csv_writer.writerow(row)
        
        print(f"自注意力分数已保存到 {save_path}")
    
    return or_attention_sums

# ==================== 配置参数 ====================
ATTENTION_FILE = 'attention_weights/full100_hOR_attention_scores.pkl'
OUTPUT_PATH = 'protein_self_attention_scores.csv'
ATTENTION_TYPE = 'protein_self'  # 可选: 'protein_self' 或 'ligand_self'
AGGREGATION_METHOD = 'max'  # 可选: 'max' (最大值) 或 'mean' (平均值)
# ==================== 配置参数 ====================

def main():
    # 提取自注意力分数并保存
    or_attention_sums = extract_self_attention_scores(
        ATTENTION_FILE,
        OUTPUT_PATH,
        ATTENTION_TYPE,
        AGGREGATION_METHOD
    )
    
    print(f"\n处理完成，共处理 {len(or_attention_sums)} 个OR")

if __name__ == '__main__':
    main()