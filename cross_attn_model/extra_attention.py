import numpy as np
import os
import argparse
import csv
import pickle

def extract_attention_scores(attention_file, save_path, batch_size=100):
    """
    从注意力分数文件中提取蛋白质-配体交叉注意力分数并保存到CSV文件
    
    Args:
        attention_file: 注意力分数文件路径
        save_path: 结果保存路径
        batch_size: 批处理大小，用于内存优化
    """
    # 加载注意力分数文件
    with open(attention_file, 'rb') as f:
        data = pickle.load(f)
    
    # 提取数据
    attention_scores_dict = data.get('attention_scores_dict', {})
    attention_weights = data.get('attention_weights', {})
    
    # 检查数据结构
    print("数据结构检查:")
    print(f"attention_scores_dict 键: {list(attention_scores_dict.keys())}")
    print(f"attention_weights 键: {list(attention_weights.keys())}")
    
    # 提取 protein_cross 注意力权重
    protein_cross_weights = attention_weights.get('protein_cross', [])
    print(f"protein_cross 长度: {len(protein_cross_weights)}")
    
    if len(protein_cross_weights) > 0:
        print(f"第一个元素形状: {protein_cross_weights[0].size() if hasattr(protein_cross_weights[0], 'size') else np.array(protein_cross_weights[0]).shape}")
    
    # 创建一个字典来存储每个OR的注意力分数
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
    for i, (key, sample_info) in enumerate(attention_scores_dict.items()):
        or_name = sample_info.get('or_name', 'unknown')
        print(f"处理样本 {i+1}/{total_samples}: {or_name}")
        
        try:
            # 获取注意力矩阵
            attention_matrix = None
            
            # 优先从 attention_weights 中获取
            if len(protein_cross_weights) > i:
                tensor_data = protein_cross_weights[i]
                
                # 处理PyTorch张量
                if has_torch and hasattr(tensor_data, 'size'):
                    # 如果是CUDA张量，先转移到CPU
                    if hasattr(tensor_data, 'is_cuda') and tensor_data.is_cuda:
                        attention_matrix = tensor_data.cpu().detach().numpy().astype(np.float32)
                    else:
                        attention_matrix = tensor_data.detach().numpy().astype(np.float32)
                else:
                    attention_matrix = np.array(tensor_data, dtype=np.float32)
            
            # 如果没有获取到，从 attention_scores_dict 中获取
            if attention_matrix is None:
                attention_matrix = np.array(sample_info.get('attention_matrix', []), dtype=np.float32)
            
            # 确保矩阵是有效的
            if attention_matrix is not None and len(attention_matrix.shape) == 3:
                # 对于多头注意力，取平均值
                attention_matrix = np.mean(attention_matrix, axis=0)
            
            # 确保矩阵是有效的
            if attention_matrix is None or len(attention_matrix.shape) != 2:
                attention_matrix = np.array([], dtype=np.float32)
            
            # 对每一位上的注意力分数求最大值（axis=1 表示对行求最大值，得到400*1的矩阵）
            if len(attention_matrix.shape) == 2 and attention_matrix.shape[0] > 0:
                # 计算所有位置的注意力最大值
                attention_sums = np.max(attention_matrix, axis=1)
            else:
                attention_sums = np.array([], dtype=np.float32)
            
            # 将结果添加到字典中
            or_attention_sums[or_name] = attention_sums
        except Exception as e:
            print(f"处理样本 {or_name} 时出错: {e}")
            # 添加空结果，确保所有OR都有对应的条目
            or_attention_sums[or_name] = np.array([], dtype=np.float32)
    
    # 保存到CSV文件
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        # 找出最长的OR序列长度
        max_length = 0
        if or_attention_sums:
            max_length = max(len(sums) for sums in or_attention_sums.values())
        print(f"最大序列长度: {max_length}")
        
        # 创建CSV文件并写入数据
        with open(save_path, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            
            # 写入表头
            header = ['OR_name'] + [f'position_{i+1}' for i in range(max_length)]
            csv_writer.writerow(header)
            
            # 写入每个OR的数据
            for or_name, sums in or_attention_sums.items():
                # 确保sums是numpy数组
                if not isinstance(sums, np.ndarray):
                    sums = np.array(sums, dtype=np.float32)
                
                # 确保长度一致，不足的地方用0填充
                if len(sums) > 0:
                    padded_sums = np.pad(sums, (0, max_length - len(sums)), 'constant', constant_values=0)
                else:
                    padded_sums = np.zeros(max_length, dtype=np.float32)
                
                # 构建一行数据并写入
                row = [or_name] + padded_sums.tolist()
                csv_writer.writerow(row)
        
        print(f"注意力分数已保存到 {save_path}")
    
    return or_attention_sums

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='提取注意力分数并保存到CSV文件')
    parser.add_argument('--attention_file', type=str, required=True,
                      help='注意力分数文件路径')
    parser.add_argument('--save_path', type=str, required=True,
                      help='结果保存路径')
    parser.add_argument('--batch_size', type=int, default=100,
                      help='批处理大小（用于内存优化）')
    args = parser.parse_args()
    
    # 提取注意力分数并保存
    or_attention_sums = extract_attention_scores(
        args.attention_file,
        args.save_path,
        args.batch_size
    )
    
    print(f"处理完成，共处理 {len(or_attention_sums)} 个OR")

if __name__ == '__main__':
    main()