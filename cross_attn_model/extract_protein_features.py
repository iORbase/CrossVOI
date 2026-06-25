import numpy as np
import pandas as pd
import h5py
import os

# ==================== 配置参数 ====================
PROTEIN_EMBEDDING_PATH = 'data/per_residue_embeddings_hOR.h5'
OUTPUT_PATH = 'protein_features.csv'
MAX_LENGTH = 400
COMPRESSION_METHOD = 'mean'  # 可选: 'mean', 'max', 'sum', 'norm'
# ==================== 配置参数 ====================

def load_protein_features(protein_embedding_path):
    """
    从HDF5文件加载蛋白质特征

    Args:
        protein_embedding_path: 蛋白质嵌入文件路径

    Returns:
        protein_features: 字典，键为蛋白质名称，值为特征矩阵 (L, D)
    """
    protein_features = {}

    with h5py.File(protein_embedding_path, 'r') as f:
        for key in f.keys():
            protein_features[str(key)] = np.array(f[key])

    print(f"共加载 {len(protein_features)} 个蛋白质特征")

    return protein_features

def compress_features(features, compression_method='mean'):
    """
    将蛋白质特征从 (L, D) 压缩为 (L,)

    Args:
        features: 特征矩阵 (L, D)
        compression_method: 压缩方法，可选 'mean', 'max', 'sum', 'norm'

    Returns:
        compressed: 压缩后的特征向量 (L,)
    """
    if features.ndim == 1:
        return features

    if features.ndim == 2:
        L, D = features.shape

        if compression_method == 'mean':
            compressed = np.mean(features, axis=1)
        elif compression_method == 'max':
            compressed = np.max(features, axis=1)
        elif compression_method == 'sum':
            compressed = np.sum(features, axis=1)
        elif compression_method == 'norm':
            compressed = np.linalg.norm(features, axis=1)
        else:
            raise ValueError(f"不支持的压缩方法: {compression_method}")

        return compressed
    else:
        raise ValueError(f"特征维度不正确: {features.ndim}")

def save_to_csv(protein_features, output_path, max_length=400, compression_method='mean'):
    """
    将蛋白质特征保存到CSV文件

    Args:
        protein_features: 蛋白质特征字典
        output_path: 输出CSV文件路径
        max_length: 最大序列长度，不足填充0，超过截断
        compression_method: 压缩方法
    """
    protein_names = sorted(protein_features.keys())

    max_seq_length = min(max_length, max(len(features) for features in protein_features.values()))
    print(f"最大序列长度: {max_seq_length}")

    df = pd.DataFrame()
    df['protein_name'] = protein_names

    for pos in range(max_seq_length):
        df[f'position_{pos+1}'] = 0.0

    for protein_name in protein_names:
        features = protein_features[protein_name]

        compressed = compress_features(features, compression_method)

        if len(compressed) > max_seq_length:
            compressed = compressed[:max_seq_length]
        else:
            compressed = np.pad(compressed, (0, max_seq_length - len(compressed)), 'constant')

        idx = df[df['protein_name'] == protein_name].index[0]
        for pos in range(max_seq_length):
            df.at[idx, f'position_{pos+1}'] = compressed[pos]

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"蛋白质特征已保存到 {output_path}")

    return df

def main():
    protein_features = load_protein_features(PROTEIN_EMBEDDING_PATH)
    save_to_csv(protein_features, OUTPUT_PATH, MAX_LENGTH, COMPRESSION_METHOD)

if __name__ == '__main__':
    main()