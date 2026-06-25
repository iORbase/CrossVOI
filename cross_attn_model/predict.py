
import torch
import numpy as np
import pandas as pd
import h5py
import os
import json
import pickle
import csv
from models import ProteinLigandInteractionModel

# Set random seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Data loading function
def load_data(csv_path, protein_embedding_path, ligand_embedding_path, n_rows=None):
    # Read CSV file with optional row limit
    data = pd.read_csv(csv_path, nrows=n_rows)
    
    # Ensure CSV file has at least 4 columns (idx, voc_name, or_name, label)
    if len(data.columns) < 4:
        raise ValueError(f"CSV file must contain at least 4 columns, but found {len(data.columns)} columns")
    
    # Set column names to match train.py
    data.columns = ['idx','voc_name', 'or_name', 'label'] + list(data.columns[4:])
    
    # Read protein and ligand embeddings
    protein_embeddings = {}
    ligand_embeddings = {}
    
    # Read protein embeddings
    with h5py.File(protein_embedding_path, 'r') as f:
        for key in f.keys():
            protein_embeddings[str(key)] = np.array(f[key])
    
    # Read ligand embeddings
    with h5py.File(ligand_embedding_path, 'r') as f:
        for key in f.keys():
            ligand_embeddings[str(key)] = np.array(f[key])
    
    # Check data matching
    if len(data) > 0:
        matched = 0
        total = min(100, len(data))
        for i in range(total):
            voc_name = str(data.iloc[i]['voc_name'])
            or_name = str(data.iloc[i]['or_name'])
            if voc_name in ligand_embeddings and or_name in protein_embeddings:
                matched += 1
        

    
    return data, protein_embeddings, ligand_embeddings

# Feature preparation function
def prepare_features(samples, protein_embeddings, ligand_embeddings, max_protein_len=400, max_ligand_len=200):
    protein_features = []
    ligand_features = []
    protein_masks = []  # 新增：存储蛋白质特征掩码
    ligand_masks = []  # 新增：存储配体特征掩码
    sample_info = []  # Store sample information (names, etc.)
    valid_indices = []
    
    for idx, sample in enumerate(samples):
        # Support 4-column format: idx, voc_name, or_name, label (matching train.py)
        if len(sample) >= 4:
            _, voc_name, or_name, _ = sample
        else:
            continue
        
        # Ensure string format for names
        voc_name = str(voc_name)
        or_name = str(or_name)
        
        # Skip if features not found
        if voc_name not in ligand_embeddings or or_name not in protein_embeddings:
            continue
        
        protein_feat = protein_embeddings[or_name]
        ligand_feat = ligand_embeddings[voc_name]
        
        # 创建蛋白质掩码：判断每个特征向量是否全为0
        original_protein_len = len(protein_feat)
        protein_mask = np.array([not np.all(vec == 0) for vec in protein_feat])
        
        # 创建配体掩码：判断每个特征向量是否全为0
        original_ligand_len = len(ligand_feat)
        ligand_mask = np.array([not np.all(vec == 0) for vec in ligand_feat])
        
        # Truncate or pad to max length
        if len(protein_feat) > max_protein_len:
            protein_feat = protein_feat[:max_protein_len]
            protein_mask = protein_mask[:max_protein_len]  # 截断掩码
        else:
            pad_length = max_protein_len - len(protein_feat)
            protein_feat = np.pad(protein_feat, ((0, pad_length), (0, 0)), 'constant')
            protein_mask = np.pad(protein_mask, (0, pad_length), 'constant')  # 填充掩码
        
        if len(ligand_feat) > max_ligand_len:
            ligand_feat = ligand_feat[:max_ligand_len]
            ligand_mask = ligand_mask[:max_ligand_len]  # 截断掩码
        else:
            pad_length = max_ligand_len - len(ligand_feat)
            ligand_feat = np.pad(ligand_feat, ((0, pad_length), (0, 0)), 'constant')
            ligand_mask = np.pad(ligand_mask, (0, pad_length), 'constant')  # 填充掩码
        
        protein_features.append(protein_feat)
        ligand_features.append(ligand_feat)
        protein_masks.append(protein_mask)
        ligand_masks.append(ligand_mask)
        sample_info.append((voc_name, or_name))
        valid_indices.append(idx)
    
    # Return None if no valid samples
    if len(valid_indices) == 0:
        return None, None, None, None, None, None
    
    # Convert to PyTorch tensors
    protein_features = torch.tensor(np.array(protein_features), dtype=torch.float32)
    ligand_features = torch.tensor(np.array(ligand_features), dtype=torch.float32)
    protein_masks = torch.tensor(np.array(protein_masks), dtype=torch.bool)  # 转换为布尔张量
    ligand_masks = torch.tensor(np.array(ligand_masks), dtype=torch.bool)  # 转换为布尔张量
    
    return protein_features, ligand_features, protein_masks, ligand_masks, sample_info, valid_indices

# Batch generator
def get_batches(data, protein_embeddings, ligand_embeddings, batch_size=32, indices=None, max_protein_len=400, max_ligand_len=200):
    # Use specified indices or all data
    if indices is not None:
        data_subset = data.iloc[indices].values.tolist()
    else:
        data_subset = data.values.tolist()
    
    # Generate batches
    for i in range(0, len(data_subset), batch_size):
        batch_samples = data_subset[i:i+batch_size]
        batch = prepare_features(batch_samples, protein_embeddings, ligand_embeddings, max_protein_len, max_ligand_len)
        
        # Skip empty batches
        if batch[0] is None:
            continue
        
        yield batch

# Prediction function with enhanced attention analysis
def predict(model, data, protein_embeddings, ligand_embeddings, 
           device, batch_size=32, max_protein_len=400, max_ligand_len=200, 
           threshold=0.5, save_attention=False, attention_dir=None):
    model.eval()
    all_predictions = []
    all_sample_info = []
    attention_scores_dict = {}
    missing_count = 0
    
    # Ensure attention directory exists
    if save_attention and attention_dir:
        os.makedirs(attention_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(get_batches(data, protein_embeddings, ligand_embeddings, 
                                                    batch_size, None, max_protein_len, max_ligand_len)):
            # 解包包含掩码的批次数据
            protein_features, ligand_features, protein_masks, ligand_masks, sample_info, valid_indices = batch
            
            # Move to device
            protein_features = protein_features.to(device)
            ligand_features = ligand_features.to(device)
            protein_masks = protein_masks.to(device)  # 掩码也移至设备
            ligand_masks = ligand_masks.to(device)  # 掩码也移至设备
            
            # Forward pass with attention extraction
            if save_attention:
                # Extract attention scores during forward pass
                outputs = model(protein_features, ligand_features, protein_masks, ligand_masks, save_attention=True)
                
                # Process and save attention scores for each sample in the batch
                for sample_idx, (voc_name, or_name) in enumerate(sample_info):
                    sample_key = f"{voc_name}_{or_name}"
                    
                    # Get cross attention from the model's attention_weights attribute
                    # We'll use protein_cross attention which represents protein's attention to ligand
                    if 'protein_cross' in model.attention_weights and model.attention_weights['protein_cross']:
                        # Average attention scores across heads and layers
                        cross_attn_layers = model.attention_weights['protein_cross']
                        avg_attention = torch.mean(torch.stack(cross_attn_layers), dim=0)
                        # For each sample, get the attention matrix
                        sample_attention = avg_attention[sample_idx].cpu().numpy()
                        
                        # Store attention scores with corresponding protein and ligand names
                        attention_scores_dict[sample_key] = {
                            'voc_name': voc_name,
                            'or_name': or_name,
                            'attention_matrix': sample_attention.tolist()
                        }
                
                # 注意力权重将在处理完所有批次后统一保存到单个文件中
            else:
                outputs = model(protein_features, ligand_features, protein_masks, ligand_masks)
            
            # Save predictions and sample info
            all_predictions.extend(outputs.cpu().numpy().flatten().tolist())
            all_sample_info.extend(sample_info)
            
            
    
    # Compute binary predictions
    binary_predictions = (np.array(all_predictions) >= threshold).astype(int)
    
    # Count samples with missing features
    total_samples = len(data)
    missing_count = total_samples - len(all_predictions)
    
    if missing_count > 0:
        print(f"Warning: {missing_count} samples were skipped due to missing features")
    
    # Create results DataFrame
    results = pd.DataFrame({
        'voc_name': [info[0] for info in all_sample_info],
        'or_name': [info[1] for info in all_sample_info],
        'prediction_score': all_predictions,
        'prediction_label': binary_predictions
    })
    
    # Save attention scores as a single pickle file if requested
    if save_attention and attention_dir:
        # Collect all attention data in one dictionary
        all_attention_data = {
            'attention_scores_dict': attention_scores_dict,
            'attention_weights': model.attention_weights
        }
        
        # Save in pickle format
        pickle_save_path = os.path.join(attention_dir, "best_fold3_hOR_attention_scores.pkl")
        with open(pickle_save_path, 'wb') as f:
            pickle.dump(all_attention_data, f)
            
        print(f"Attention data saved to {pickle_save_path}")

    
    return results

def predict_with_self_attention(model, data, protein_embeddings, ligand_embeddings,
                                device, batch_size=32, max_protein_len=400, max_ligand_len=200,
                                save_path='protein_self_attention_scores.csv', aggregation_method='max'):
    """
    预测函数：提取并保存自注意力分数到CSV文件

    Args:
        model: 训练好的模型
        data: DataFrame，包含 voc_name, or_name 等列
        protein_embeddings: 蛋白质嵌入字典
        ligand_embeddings: 配体嵌入字典
        device: 计算设备
        batch_size: 批处理大小
        max_protein_len: 蛋白质最大长度
        max_ligand_len: 配体最大长度
        save_path: 自注意力分数保存路径
        aggregation_method: 聚合方法，'max' 或 'mean'
    """
    model.eval()

    all_sample_info = []
    all_attention_tensors = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(get_batches(data, protein_embeddings, ligand_embeddings,
                                                      batch_size, None, max_protein_len, max_ligand_len)):
            protein_features, ligand_features, protein_masks, ligand_masks, sample_info, valid_indices = batch

            all_sample_info.extend(sample_info)

            protein_features = protein_features.to(device)
            ligand_features = ligand_features.to(device)
            protein_masks = protein_masks.to(device)
            ligand_masks = ligand_masks.to(device)

            model(protein_features, ligand_features, protein_masks, ligand_masks, save_attention=True)

            protein_self_weights = model.attention_weights.get('protein_self', [])
            if protein_self_weights:
                stacked = torch.stack([w.detach().cpu() for w in protein_self_weights], dim=0)
                avg_across_layers = torch.mean(stacked, dim=0)
                all_attention_tensors.append(avg_across_layers)

            print(f"批次 {batch_idx+1}: 处理 {len(sample_info)} 个样本")

    print(f"\n累计了 {len(all_attention_tensors)} 个批次的蛋白质自注意力")
    print(f"总样本数: {len(all_sample_info)}")

    protein_self_tensor = torch.cat(all_attention_tensors, dim=0)
    print(f"蛋白质自注意力总形状: {protein_self_tensor.shape}")

    num_samples = protein_self_tensor.shape[0]
    sample_attention_sums = []

    for i in range(num_samples):
        sample_matrix = protein_self_tensor[i].cpu().numpy()

        if aggregation_method == 'max':
            attn_sums = np.max(sample_matrix, axis=1)
        else:
            attn_sums = np.mean(sample_matrix, axis=1)

        sample_attention_sums.append(attn_sums)

    or_attention_dict = {}
    for i, (voc_name, or_name) in enumerate(all_sample_info):
        if or_name not in or_attention_dict:
            or_attention_dict[or_name] = sample_attention_sums[i]

    print(f"不同OR数量: {len(or_attention_dict)}")

    if save_path:
        max_length = max(len(sums) for sums in or_attention_dict.values())

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

        with open(save_path, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            header = ['OR'] + [f'position_{j+1}' for j in range(max_length)]
            csv_writer.writerow(header)

            for or_name, sums in or_attention_dict.items():
                padded_sums = np.pad(sums, (0, max_length - len(sums)), 'constant', constant_values=0)
                row = [or_name] + padded_sums.tolist()
                csv_writer.writerow(row)

        print(f"自注意力分数已保存到 {save_path}")

    return or_attention_dict


# Load model configuration
def load_model_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

# Main function
def main():
    # Set random seed
    set_seed(42)
    
    # Configuration parameters - use the same file paths as train.py
    csv_path = 'data/hOR_inter.csv'
    protein_embedding_path = 'data/per_residue_embeddings_hOR.h5'
    ligand_embedding_path = 'data/per_atom_embeddings_hOR.h5'
    model_path = './models/best_model_fold_3.pt'
    config_path = None  # Use default parameters if no config file
    output_path = 'best_fold3_hOR_prediction_results.csv'
    batch_size = 32
    threshold = 0.5
    save_attention = True  # Enable attention analysis by default
    attention_dir = 'attention_weights'
    extract_self_attention = False  # 新增：是否提取自注意力并保存到CSV
    self_attention_output = 'best_protein_self_attention_scores.csv'  # 自注意力CSV输出路径
    
    # Check if files exist
    for file_path in [csv_path, protein_embedding_path, ligand_embedding_path, model_path]:
        if file_path and not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data - read only the first 100 rows for testing purposes
    data, protein_embeddings, ligand_embeddings = load_data(csv_path, protein_embedding_path, ligand_embedding_path)
    
    # Load model configuration (if available)
    if config_path:
        config = load_model_config(config_path)
        model = ProteinLigandInteractionModel(**config).to(device)
    else:
        # Create model with default parameters matching train.py
        model = ProteinLigandInteractionModel(
            protein_embed_dim=1024,
            ligand_embed_dim=768,
            hidden_dim=256,
            protein_self_attn_layers=2,
            ligand_self_attn_layers=2,
            cross_attn_layers=2,
            num_heads=4,
            fc_hidden_dims=[512, 256],
            dropout_rate=0.1,
            max_protein_len=400,
            max_ligand_len=200
        ).to(device)
    

    model.load_state_dict(torch.load(model_path, map_location=device))

    if extract_self_attention:
        predict_with_self_attention(model, data, protein_embeddings, ligand_embeddings,
                                   device, batch_size,
                                   model.hparams['max_protein_len'],
                                   model.hparams['max_ligand_len'],
                                   save_path=self_attention_output)
    else:
        results = predict(model, data, protein_embeddings, ligand_embeddings,
                         device, batch_size,
                         model.hparams['max_protein_len'],
                         model.hparams['max_ligand_len'],
                         threshold, save_attention, attention_dir)
        results.to_csv(output_path, index=False)

    
if __name__ == '__main__':
    main()
