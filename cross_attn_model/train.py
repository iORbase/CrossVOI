import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import h5py
import os
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from models import ProteinLigandInteractionModel
from torch.utils.tensorboard import SummaryWriter

# Set random seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Data loading function
def load_data(csv_path, protein_embedding_path, ligand_embedding_path):
    # Read CSV file
    df = pd.read_csv(csv_path)
    df.columns = ['idx','voc_name', 'or_name', 'label']
    
    # Read protein embeddings
    protein_embeddings = {}
    with h5py.File(protein_embedding_path, 'r') as f:
        for key in f.keys():
            protein_embeddings[key] = np.array(f[key])
    
    # Read ligand embeddings
    ligand_embeddings = {}
    with h5py.File(ligand_embedding_path, 'r') as f:
        for key in f.keys():
            ligand_embeddings[key] = np.array(f[key])
    
    # Check data matching
    if len(df) > 0:
        matched = 0
        total = min(100, len(df))
        for i in range(total):
            voc_name = df.iloc[i]['voc_name']
            or_name = df.iloc[i]['or_name']
            if str(voc_name) in ligand_embeddings and str(or_name) in protein_embeddings:
                matched += 1
        
        print(f"Feature matching rate in first {total} samples: {matched/total:.2%}")
    
    return df, protein_embeddings, ligand_embeddings

# Feature preparation function
def prepare_features(samples, protein_embeddings, ligand_embeddings, max_protein_len=400, max_ligand_len=200):
    protein_features = []
    ligand_features = []
    protein_masks = []
    ligand_masks = []
    labels = []
    valid_indices = []
    
    for idx, sample in enumerate(samples):
        # Support 4-column format: idx, voc_name, or_name, label
        if len(sample) == 4:
            _, voc_name, or_name, label = sample
        else:
            voc_name, or_name, label = sample
        
        # Ensure string format for names
        voc_name = str(voc_name)
        or_name = str(or_name)
        
        # Skip if features not found
        if voc_name not in ligand_embeddings or or_name not in protein_embeddings:
            continue
        
        protein_feat = protein_embeddings[or_name]
        ligand_feat = ligand_embeddings[voc_name]
        
        # Truncate or pad to max length
        original_protein_len = len(protein_feat)
        if len(protein_feat) > max_protein_len:
            protein_feat = protein_feat[:max_protein_len]
        else:
            pad_length = max_protein_len - len(protein_feat)
            protein_feat = np.pad(protein_feat, ((0, pad_length), (0, 0)), 'constant')
        
        original_ligand_len = len(ligand_feat)
        if len(ligand_feat) > max_ligand_len:
            ligand_feat = ligand_feat[:max_ligand_len]
        else:
            pad_length = max_ligand_len - len(ligand_feat)
            ligand_feat = np.pad(ligand_feat, ((0, pad_length), (0, 0)), 'constant')
        
        # Create masks: True for valid positions (non-zero vectors), False for padding (all-zero vectors)
        # For protein mask
        protein_mask = []
        for i in range(len(protein_feat)):
            # Check if the feature vector is all zeros
            is_all_zero = np.all(protein_feat[i] == 0)
            # True means valid (not masked), False means masked
            protein_mask.append(not is_all_zero)
        
        # For ligand mask
        ligand_mask = []
        for i in range(len(ligand_feat)):
            # Check if the feature vector is all zeros
            is_all_zero = np.all(ligand_feat[i] == 0)
            # True means valid (not masked), False means masked
            ligand_mask.append(not is_all_zero)
        
        protein_features.append(protein_feat)
        ligand_features.append(ligand_feat)
        protein_masks.append(protein_mask)
        ligand_masks.append(ligand_mask)
        labels.append(label)
        valid_indices.append(idx)
    
    # Return None if no valid samples
    if len(valid_indices) == 0:
        return None, None, None, None, None, None
    
    # Convert to PyTorch tensors
    protein_features = torch.tensor(np.array(protein_features), dtype=torch.float32)
    ligand_features = torch.tensor(np.array(ligand_features), dtype=torch.float32)
    protein_masks = torch.tensor(np.array(protein_masks), dtype=torch.bool)
    ligand_masks = torch.tensor(np.array(ligand_masks), dtype=torch.bool)
    labels = torch.tensor(np.array(labels), dtype=torch.float32).unsqueeze(1)
    
    return protein_features, ligand_features, protein_masks, ligand_masks, labels, valid_indices

# Batch generator
def get_batches(data, protein_embeddings, ligand_embeddings, batch_size=32, indices=None, max_protein_len=400, max_ligand_len=200):
    # Use specified indices or all data
    if indices is not None:
        data_subset = data.iloc[indices].values.tolist()
    else:
        data_subset = data.values.tolist()
    
    # Shuffle data
    np.random.shuffle(data_subset)
    
    # Generate batches
    for i in range(0, len(data_subset), batch_size):
        batch_samples = data_subset[i:i+batch_size]
        batch = prepare_features(batch_samples, protein_embeddings, ligand_embeddings, max_protein_len, max_ligand_len)
        
        # Skip empty batches
        if batch[0] is None:
            continue
        
        yield batch

# Model training function
def train_model(model, data, protein_embeddings, ligand_embeddings, 
                optimizer, criterion, device, batch_size=32, 
                max_protein_len=400, max_ligand_len=200, 
                train_indices=None, verbose=True):
    model.train()
    running_loss = 0.0
    total_samples = 0
    batch_count = 0
    
    # Prepare data subset and calculate total batches for progress display
    if train_indices is not None:
        data_subset = data.iloc[train_indices].values.tolist()
    else:
        data_subset = data.values.tolist()
    
    # Shuffle data (to match get_batches behavior)
    np.random.shuffle(data_subset)
    
    # Calculate approximate total batches (without considering skipped empty batches)
    total_batches_approx = max(1, len(data_subset) // batch_size)
    
    # Process batches with progress display
    for batch_idx, batch in enumerate(get_batches(data, protein_embeddings, ligand_embeddings, 
                                               batch_size, train_indices, 
                                               max_protein_len, max_ligand_len)):
        protein_features, ligand_features, protein_masks, ligand_masks, labels, valid_indices = batch
        
        # Move to device
        protein_features = protein_features.to(device)
        ligand_features = ligand_features.to(device)
        protein_masks = protein_masks.to(device)
        ligand_masks = ligand_masks.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(protein_features, ligand_features, protein_masks, ligand_masks)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update statistics
        running_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
        batch_count += 1
        
        # Display progress
        if verbose and batch_idx % 5 == 0:  # Update every 5 batches
            progress = min(100, (batch_idx + 1) / total_batches_approx * 100)
            avg_loss_so_far = running_loss / total_samples if total_samples > 0 else 0.0
            # Use \r to overwrite the current line
            print(f"Training progress: {progress:.1f}% | Avg Loss: {avg_loss_so_far:.4f}", end='\r')
    
    # Calculate average loss
    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
    
    if verbose:
        # Print a new line after progress updates
        print(f"Training completed. Avg Loss: {avg_loss:.4f}")
    
    return avg_loss

# Model evaluation function
def evaluate_model(model, data, protein_embeddings, ligand_embeddings, 
                   device, batch_size=32, max_protein_len=400, 
                   max_ligand_len=200, indices=None, threshold=0.5):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in get_batches(data, protein_embeddings, ligand_embeddings, 
                                batch_size, indices, 
                                max_protein_len, max_ligand_len):
            protein_features, ligand_features, protein_masks, ligand_masks, labels, valid_indices = batch
            
            # Move to device
            protein_features = protein_features.to(device)
            ligand_features = ligand_features.to(device)
            protein_masks = protein_masks.to(device)
            ligand_masks = ligand_masks.to(device)
            
            # Forward pass
            outputs = model(protein_features, ligand_features, protein_masks, ligand_masks)
            
            # Save predictions and labels
            all_predictions.extend(outputs.cpu().numpy().flatten().tolist())
            all_labels.extend(labels.cpu().numpy().flatten().tolist())
    
    # Check if there are predictions
    if len(all_predictions) == 0:
        print("Warning: No valid samples found during evaluation")
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'roc_auc': 0.0,
            'pr_auc': 0.0
        }
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Compute binary predictions
    binary_predictions = (all_predictions >= threshold).astype(int)
    
    # Calculate evaluation metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, binary_predictions),
        'precision': precision_score(all_labels, binary_predictions, zero_division=0),
        'recall': recall_score(all_labels, binary_predictions, zero_division=0),
        'f1': f1_score(all_labels, binary_predictions, zero_division=0),
        'roc_auc': roc_auc_score(all_labels, all_predictions) if len(np.unique(all_labels)) > 1 else 0.0,
        'pr_auc': average_precision_score(all_labels, all_predictions) if len(np.unique(all_labels)) > 1 else 0.0
    }
    
    return metrics

def _generate_protein_based_splits(data, n_splits, random_state=42):
    """根据蛋白质名称生成交叉验证划分，确保训练集和验证集中没有重复的蛋白质，同时保持标签比例
    
    Args:
        data: 数据DataFrame，包含 'or_name' 列表示蛋白质名称和 'label' 列表示标签
        n_splits: 折数
        random_state: 随机种子
    
    Yields:
        (train_indices, val_indices): 训练集和验证集的索引数组
    """
    # 设置随机种子
    np.random.seed(random_state)
    
    # 获取所有唯一的蛋白质名称
    unique_proteins = data['or_name'].unique()
    print(f"Total unique proteins: {len(unique_proteins)}")
    
    # 计算总体标签比例
    total_positive = len(data[data['label'] == 1])
    total_negative = len(data[data['label'] == 0])
    overall_pos_ratio = total_positive / (total_positive + total_negative)
    print(f"Overall positive ratio: {overall_pos_ratio:.4f}")
    
    # 统计每个蛋白质的正负样本数量
    protein_stats = {}
    for protein in unique_proteins:
        protein_data = data[data['or_name'] == protein]
        pos_count = len(protein_data[protein_data['label'] == 1])
        neg_count = len(protein_data[protein_data['label'] == 0])
        protein_stats[protein] = {'pos': pos_count, 'neg': neg_count, 'total': pos_count + neg_count}
    
    # 按蛋白质的正样本比例排序，用于分层抽样
    proteins_sorted = sorted(unique_proteins, key=lambda p: (protein_stats[p]['pos'] / protein_stats[p]['total']) if protein_stats[p]['total'] > 0 else 0)
    
    # 采用分层策略：将蛋白质均匀分配到各个fold，保持整体标签比例
    # 方法：将排序后的蛋白质依次分配到各个fold
    fold_proteins = [set() for _ in range(n_splits)]
    
    # 按顺序循环分配蛋白质到各个fold
    for i, protein in enumerate(proteins_sorted):
        fold_idx = i % n_splits
        fold_proteins[fold_idx].add(protein)
    
    # 生成每个fold的划分
    for fold in range(n_splits):
        # 当前fold的蛋白质作为验证集
        val_proteins = fold_proteins[fold]
        # 其他fold的蛋白质作为训练集
        train_proteins = set()
        for i in range(n_splits):
            if i != fold:
                train_proteins.update(fold_proteins[i])
        
        # 获取索引
        train_indices = data[data['or_name'].isin(train_proteins)].index.values
        val_indices = data[data['or_name'].isin(val_proteins)].index.values
        
        # 打乱索引顺序
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        
        # 计算并打印标签分布
        train_data = data.loc[train_indices]
        val_data = data.loc[val_indices]
        train_pos = len(train_data[train_data['label'] == 1])
        train_neg = len(train_data[train_data['label'] == 0])
        val_pos = len(val_data[val_data['label'] == 1])
        val_neg = len(val_data[val_data['label'] == 0])
        
        print(f"Fold {fold+1}: Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
        print(f"  Train proteins: {len(train_proteins)}, Val proteins: {len(val_proteins)}")
        print(f"  Train pos/neg: {train_pos}/{train_neg} ({train_pos/(train_pos+train_neg):.4f}), Val pos/neg: {val_pos}/{val_neg} ({val_pos/(val_pos+val_neg):.4f})")
        
        yield train_indices, val_indices

def _generate_molecular_based_splits(data, n_splits, random_state=42):
    """根据小分子名称生成交叉验证划分，确保训练集和验证集中没有重复的小分子，同时保持标签比例
    
    Args:
        data: 数据DataFrame，包含 'voc_name' 列表示小分子名称和 'label' 列表示标签
        n_splits: 折数
        random_state: 随机种子
    
    Yields:
        (train_indices, val_indices): 训练集和验证集的索引数组
    """
    # 设置随机种子
    np.random.seed(random_state)
    
    # 获取所有唯一的小分子名称
    unique_molecules = data['voc_name'].unique()
    print(f"Total unique molecules: {len(unique_molecules)}")
    
    # 计算总体标签比例
    total_positive = len(data[data['label'] == 1])
    total_negative = len(data[data['label'] == 0])
    overall_pos_ratio = total_positive / (total_positive + total_negative)
    print(f"Overall positive ratio: {overall_pos_ratio:.4f}")
    
    # 统计每个小分子的正负样本数量
    molecule_stats = {}
    for molecule in unique_molecules:
        molecule_data = data[data['voc_name'] == molecule]
        pos_count = len(molecule_data[molecule_data['label'] == 1])
        neg_count = len(molecule_data[molecule_data['label'] == 0])
        molecule_stats[molecule] = {'pos': pos_count, 'neg': neg_count, 'total': pos_count + neg_count}
    
    # 按小分子的正样本比例排序，用于分层抽样
    molecules_sorted = sorted(unique_molecules, key=lambda m: (molecule_stats[m]['pos'] / molecule_stats[m]['total']) if molecule_stats[m]['total'] > 0 else 0)
    
    # 采用分层策略：将小分子均匀分配到各个fold，保持整体标签比例
    # 方法：将排序后的小分子依次分配到各个fold
    fold_molecules = [set() for _ in range(n_splits)]
    
    # 按顺序循环分配小分子到各个fold
    for i, molecule in enumerate(molecules_sorted):
        fold_idx = i % n_splits
        fold_molecules[fold_idx].add(molecule)
    
    # 生成每个fold的划分
    for fold in range(n_splits):
        # 当前fold的小分子作为验证集
        val_molecules = fold_molecules[fold]
        # 其他fold的小分子作为训练集
        train_molecules = set()
        for i in range(n_splits):
            if i != fold:
                train_molecules.update(fold_molecules[i])
        
        # 获取索引
        train_indices = data[data['voc_name'].isin(train_molecules)].index.values
        val_indices = data[data['voc_name'].isin(val_molecules)].index.values
        
        # 打乱索引顺序
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        
        # 计算并打印标签分布
        train_data = data.loc[train_indices]
        val_data = data.loc[val_indices]
        train_pos = len(train_data[train_data['label'] == 1])
        train_neg = len(train_data[train_data['label'] == 0])
        val_pos = len(val_data[val_data['label'] == 1])
        val_neg = len(val_data[val_data['label'] == 0])
        
        print(f"Fold {fold+1}: Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
        print(f"  Train molecules: {len(train_molecules)}, Val molecules: {len(val_molecules)}")
        print(f"  Train pos/neg: {train_pos}/{train_neg} ({train_pos/(train_pos+train_neg):.4f}), Val pos/neg: {val_pos}/{val_neg} ({val_pos/(val_pos+val_neg):.4f})")
        
        yield train_indices, val_indices

# Cross-validation function
def cross_validate(model, data, protein_embeddings, ligand_embeddings, 
                  device, n_splits=5, batch_size=32, 
                  max_protein_len=400, max_ligand_len=200, 
                  learning_rate=0.0001, num_epochs=50,
                  split_strategy='stratified'):
    """交叉验证函数
    
    Args:
        model: 模型实例
        data: 数据DataFrame
        protein_embeddings: 蛋白质嵌入字典
        ligand_embeddings: 配体嵌入字典
        device: 计算设备
        n_splits: 折数
        batch_size: 批次大小
        max_protein_len: 蛋白质最大长度
        max_ligand_len: 配体最大长度
        learning_rate: 学习率
        num_epochs: 训练轮数
        split_strategy: 数据划分策略，'stratified'（按标签分层）、'protein'（按蛋白质划分）或 'molecular'（按小分子划分）
    """
    fold_metrics = []
    
    # Initialize TensorBoard writer for cross-validation
    log_dir = os.path.join('tensorboard_logs', f'cross_val_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}')
    writer = SummaryWriter(log_dir=log_dir)
    print(f"Cross-validation TensorBoard logs will be saved to: {log_dir}")
    print(f"Using split strategy: {split_strategy}")
    
    # 根据策略生成划分
    if split_strategy == 'protein':
        splits = _generate_protein_based_splits(data, n_splits)
    elif split_strategy == 'molecular':
        splits = _generate_molecular_based_splits(data, n_splits)
    else:
        # Use StratifiedKFold to ensure balanced labels in each fold
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = skf.split(data, data['label'])
    
    for fold, (train_indices, val_indices) in enumerate(splits):
        # Initialize model
        fold_model = ProteinLigandInteractionModel(**model.hparams).to(device)
        optimizer = optim.Adam(fold_model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        
        # Add learning rate scheduler for cross-validation
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=False
        )
        
        best_val_f1 = 0.0
        best_model_path = f"./models/best_model_orsplit_fold_{fold+1}.pt"
        
        # Training loop
        for epoch in range(num_epochs):
            # Train model
            train_loss = train_model(
                fold_model,
                data,
                protein_embeddings,
                ligand_embeddings,
                optimizer,
                criterion,
                device,
                batch_size,
                max_protein_len,
                max_ligand_len,
                train_indices,
                verbose=False
            )
            
            # Evaluate model
            val_metrics = evaluate_model(
                fold_model,
                data,
                protein_embeddings,
                ligand_embeddings,
                device,
                batch_size,
                max_protein_len,
                max_ligand_len,
                val_indices
            )
            
            # Step the scheduler
            scheduler.step(val_metrics['f1'])
            
            # Save best model
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                torch.save(fold_model.state_dict(), best_model_path)
            
            # Log metrics to TensorBoard
            writer.add_scalar(f'Loss/fold_{fold+1}', train_loss, epoch)
            writer.add_scalar(f'Accuracy/fold_{fold+1}', val_metrics['accuracy'], epoch)
            writer.add_scalar(f'Precision/fold_{fold+1}', val_metrics['precision'], epoch)
            writer.add_scalar(f'Recall/fold_{fold+1}', val_metrics['recall'], epoch)
            writer.add_scalar(f'F1/fold_{fold+1}', val_metrics['f1'], epoch)
            writer.add_scalar(f'ROC-AUC/fold_{fold+1}', val_metrics['roc_auc'], epoch)
            writer.add_scalar(f'PR-AUC/fold_{fold+1}', val_metrics['pr_auc'], epoch)
        
        # Load best model and compute final metrics
        fold_model.load_state_dict(torch.load(best_model_path))
        final_metrics = evaluate_model(
            fold_model,
            data,
            protein_embeddings,
            ligand_embeddings,
            device,
            batch_size,
            max_protein_len,
            max_ligand_len,
            val_indices
        )
        
        fold_metrics.append(final_metrics)
        print(f"Fold {fold+1} Final Metrics: Acc: {final_metrics['accuracy']:.4f}, Prec: {final_metrics['precision']:.4f}, Rec: {final_metrics['recall']:.4f}, F1: {final_metrics['f1']:.4f}, ROC-AUC: {final_metrics['roc_auc']:.4f}, PR-AUC: {final_metrics['pr_auc']:.4f}")
    
    # Calculate average metrics
    avg_metrics = {
        'accuracy': np.mean([m['accuracy'] for m in fold_metrics]),
        'precision': np.mean([m['precision'] for m in fold_metrics]),
        'recall': np.mean([m['recall'] for m in fold_metrics]),
        'f1': np.mean([m['f1'] for m in fold_metrics]),
        'roc_auc': np.mean([m['roc_auc'] for m in fold_metrics]),
        'pr_auc': np.mean([m['pr_auc'] for m in fold_metrics])
    }
    
    print(f"\nCross-Validation Summary:")
    print(f"Avg Accuracy: {avg_metrics['accuracy']:.4f} ± {np.std([m['accuracy'] for m in fold_metrics]):.4f}")
    print(f"Avg Precision: {avg_metrics['precision']:.4f} ± {np.std([m['precision'] for m in fold_metrics]):.4f}")
    print(f"Avg Recall: {avg_metrics['recall']:.4f} ± {np.std([m['recall'] for m in fold_metrics]):.4f}")
    print(f"Avg F1 Score: {avg_metrics['f1']:.4f} ± {np.std([m['f1'] for m in fold_metrics]):.4f}")
    print(f"Avg AUROC: {avg_metrics['roc_auc']:.4f} ± {np.std([m['roc_auc'] for m in fold_metrics]):.4f}")
    print(f"Avg AUPRC: {avg_metrics['pr_auc']:.4f} ± {np.std([m['pr_auc'] for m in fold_metrics]):.4f}")
    
    # Close TensorBoard writer
    writer.close()
    print("Cross-validation TensorBoard writer closed.")
    
    return avg_metrics

# Main function
def main():
    # Set random seed
    set_seed(42)
    
    # File paths - keep existing path settings
    csv_path = 'data/hOR_inter.csv'
    protein_embedding_path = 'data/per_residue_embeddings_hOR.h5'
    ligand_embedding_path = 'data/per_atom_embeddings_hOR.h5'
    
    # Model and training parameters
    batch_size = 128
    learning_rate = 0.0001
    num_epochs = 50
    n_splits = 5
    cross_validation = True
    use_full_data = False
    save_model_path = './models/best_model_orsplit_data.pt'
    
    # Check if files exist
    for file_path in [csv_path, protein_embedding_path, ligand_embedding_path]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
    
    # Ensure model save directory exists
    os.makedirs(os.path.dirname(save_model_path) if os.path.dirname(save_model_path) else '.', exist_ok=True)
    
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    data, protein_embeddings, ligand_embeddings = load_data(csv_path, protein_embedding_path, ligand_embedding_path)
    
    # Create model
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
    

    if cross_validation:
        # Perform cross-validation
        cross_validate(model, data, protein_embeddings, ligand_embeddings, 
                      device, n_splits, batch_size, 
                      model.hparams['max_protein_len'], 
                      model.hparams['max_ligand_len'], 
                      learning_rate, num_epochs, split_strategy='protein')
    elif use_full_data:
        # Use full data for training and evaluation
        train_indices = np.arange(len(data))
        test_indices = train_indices
        # Train the model
        train_and_evaluate(model, data, protein_embeddings, ligand_embeddings, device, 
                          train_indices, test_indices, batch_size, learning_rate, 
                          num_epochs, save_model_path)
    else:
        # Split into train and test sets with stratified sampling to ensure balanced labels
        # Get indices of positive and negative samples
        positive_indices = data[data['label'] == 1].index
        negative_indices = data[data['label'] == 0].index
        
        # Determine the size of the smaller class
        min_class_size = min(len(positive_indices), len(negative_indices))
        
        # Use stratified sampling with train_test_split to ensure balanced distribution
        # This ensures that both train and test sets have similar proportions of each class
        train_indices_pos, test_indices_pos = train_test_split(
            positive_indices, test_size=0.2, random_state=42
        )
        train_indices_neg, test_indices_neg = train_test_split(
            negative_indices, test_size=0.1, random_state=42
        )
        
        # Combine indices
        train_indices = np.concatenate([train_indices_pos, train_indices_neg])
        test_indices = np.concatenate([test_indices_pos, test_indices_neg])
        
        # Shuffle the indices
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)
        
        # Train the model
        train_and_evaluate(model, data, protein_embeddings, ligand_embeddings, device, 
                          train_indices, test_indices, batch_size, learning_rate, 
                          num_epochs, save_model_path)

# Train and evaluate model function
def train_and_evaluate(model, data, protein_embeddings, ligand_embeddings, device, 
                      train_indices, test_indices, batch_size=32, learning_rate=0.0001, 
                      num_epochs=50, save_model_path='./models/best_model.pt'):
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    
    # Initialize TensorBoard writer
    log_dir = os.path.join('tensorboard_logs', f'runs_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}')
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    
    # Add learning rate scheduler
    # ReduceLROnPlateau: reduce learning rate when a metric has stopped improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    best_val_f1 = 0.0
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.8f}")
        
        # Train model
        train_loss = train_model(
            model,
            data,
            protein_embeddings,
            ligand_embeddings,
            optimizer,
            criterion,
            device,
            batch_size,
            model.hparams['max_protein_len'],
            model.hparams['max_ligand_len'],
            train_indices=train_indices,
            verbose=True
        )
        
        # Evaluate model
        val_metrics = evaluate_model(
            model,
            data,
            protein_embeddings,
            ligand_embeddings,
            device,
            batch_size,
            model.hparams['max_protein_len'],
            model.hparams['max_ligand_len'],
            indices=test_indices
        )
        
        # Step the scheduler with the F1 score
        scheduler.step(val_metrics['f1'])
        
        # Print evaluation results
        print(f"Test Metrics: Acc: {val_metrics['accuracy']:.4f}, Prec: {val_metrics['precision']:.4f}, Rec: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}, ROC-AUC: {val_metrics['roc_auc']:.4f}, PR-AUC: {val_metrics['pr_auc']:.4f}")
        
        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/test', val_metrics['accuracy'], epoch)
        writer.add_scalar('Precision/test', val_metrics['precision'], epoch)
        writer.add_scalar('Recall/test', val_metrics['recall'], epoch)
        writer.add_scalar('F1/test', val_metrics['f1'], epoch)
        writer.add_scalar('ROC-AUC/test', val_metrics['roc_auc'], epoch)
        writer.add_scalar('PR-AUC/test', val_metrics['pr_auc'], epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save(model.state_dict(), save_model_path)
            print(f"Best model saved with F1 score: {best_val_f1:.4f}")
    
    # Load best model and compute final metrics
    model.load_state_dict(torch.load(save_model_path))
    final_metrics = evaluate_model(
        model,
        data,
        protein_embeddings,
        ligand_embeddings,
        device,
        batch_size,
        model.hparams['max_protein_len'],
        model.hparams['max_ligand_len'],
        indices=test_indices
    )
    
    print(f"\nFinal Test Metrics:")
    print(f"Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall: {final_metrics['recall']:.4f}")
    print(f"F1 Score: {final_metrics['f1']:.4f}")
    print(f"AUROC: {final_metrics['roc_auc']:.4f}")
    print(f"AUPRC: {final_metrics['pr_auc']:.4f}")
    
    # Close TensorBoard writer
    writer.close()
    print("TensorBoard writer closed.")
    
    return final_metrics
    
if __name__ == '__main__':
    main()
