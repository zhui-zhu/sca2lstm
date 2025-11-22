import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import json
from datetime import datetime
import importlib.util
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import multiprocessing as mp
from hydrologyDataset import HydrologyDataset, preprocess_batch_data, parallel_preprocess_batches
from utils import get_discharge_scaler_params, denormalize_discharge, plot_training_curves, plot_prediction_comparison, plot_loss_distribution


def load_config():
    """加载配置"""
    class Config:
        """SCA2LSTM配置类"""
        def __init__(self, config_dict=None):
            if config_dict is None:
                config_dict = self._load_config_from_file()
            
            # 设置所有配置属性
            self.LSTM1_DERIVED_FEATS = config_dict.get('LSTM1_DERIVED_FEATS', [])
            self.LSTM2_FEATURES = config_dict.get('LSTM2_FEATURES', [])
            self.TARGET_COL = config_dict.get('TARGET_COL', 'discharge_vol')
            self.SEQ_LEN = config_dict.get('SEQ_LEN', 7)
            self.PRED_LEN = config_dict.get('PRED_LEN', 1)
            self.BATCH_SIZE = config_dict.get('BATCH_SIZE', 32)
            self.N_EPOCHS = config_dict.get('N_EPOCHS', 60)
            self.PATIENCE = config_dict.get('PATIENCE', 10)
            self.LR = config_dict.get('LR', 1e-5)
            self.LSTM_HIDDEN_DIM = config_dict.get('LSTM_HIDDEN_DIM', 64)
            self.LSTM_LAYERS = config_dict.get('LSTM_LAYERS', 2)
            self.DROPOUT = config_dict.get('DROPOUT', 0.3)
            self.EMBEDDING_DIM = config_dict.get('EMBEDDING_DIM', 16)
            self.N_FEATURES = len(self.LSTM2_FEATURES)
            self.LSTM1_INPUT_DIM = config_dict.get('LSTM1_INPUT_DIM', 29)
            self.SEED = config_dict.get('SEED', 42)
            self.MODEL_SAVE_PATH = config_dict.get('MODEL_SAVE_PATH', './model_output/sca2lstm.pth')
            self.DATA_INPUT_DIR = config_dict.get('DATA_INPUT_DIR', './model_input_data/')
            self.TRAIN_BASIN_IDS = config_dict.get('TRAIN_BASIN_IDS', [])
            self.VAL_BASIN_IDS = config_dict.get('VAL_BASIN_IDS', [])
            self.MIN_VALID_LABEL_RATIO = config_dict.get('MIN_VALID_LABEL_RATIO', 0.3)
            self.MIN_VALID_ROWS = config_dict.get('MIN_VALID_ROWS', 10000)
            self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        def _load_config_from_file(self):
            """从配置文件加载配置参数"""
            # 获取当前文件目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, 'run.config')
            
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
            # 创建配置命名空间
            config_namespace = {}
            
            try:
                # 读取并执行配置文件
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_code = f.read()
                
                # 在安全命名空间中执行配置代码
                exec(config_code, config_namespace)
                
            except Exception as e:
                raise ImportError(f"加载配置文件失败: {str(e)}")
            
            return config_namespace
        
        def __getstate__(self):
            """支持pickle序列化"""
            return self.__dict__.copy()
        
        def __setstate__(self, state):
            """支持pickle反序列化"""
            self.__dict__.update(state)
    return Config()

# ======================== SCA2LSTM模型（修复维度+优化初始化）=======================
class SCA2LSTM(nn.Module):
    def __init__(self, config):
        super(SCA2LSTM, self).__init__()
        self.config = config
        self.n_features = config.N_FEATURES
        self.lstm1_input_dim = config.LSTM1_INPUT_DIM
        self.lstm_hidden_dim = config.LSTM_HIDDEN_DIM 
        self.lstm_layers = config.LSTM_LAYERS
        self.dropout = config.DROPOUT
        self.embedding_dim = config.EMBEDDING_DIM
        self.basin_ids = config.TRAIN_BASIN_IDS + config.VAL_BASIN_IDS
        self.basin_num = len(set(self.basin_ids))
        
        # 流域嵌入层（更小的初始化方差）
        self.basin_embedding = nn.Embedding(
            num_embeddings=self.basin_num,
            embedding_dim=self.embedding_dim,
            padding_idx=-1
        )
        nn.init.normal_(self.basin_embedding.weight, mean=0.0, std=0.001)  # 降低方差
        
        # LSTM1（优化初始化）
        self.lstm1_cell = nn.LSTMCell(
            input_size=self.lstm1_input_dim,
            hidden_size=self.lstm_hidden_dim
        )
        
        # 权重输出头（更稳定的激活）
        self.weight_head = nn.Sequential(
            nn.Linear(self.lstm_hidden_dim, 32),
            nn.LeakyReLU(0.01),  # 避免ReLU死亡问题
            nn.Dropout(self.dropout),
            nn.Linear(32, self.n_features),
            nn.Softmax(dim=-1)
        )
        
        # LSTM2（优化初始化）
        self.lstm2_input_dim = self.n_features + self.embedding_dim
        self.lstm2_cell = nn.LSTMCell(
            input_size=self.lstm2_input_dim,
            hidden_size=self.lstm_hidden_dim
        )
        
        # 预测头（更稳定的激活）
        self.predict_head = nn.Sequential(
            nn.Linear(self.lstm_hidden_dim, 16),
            nn.LeakyReLU(0.01),
            nn.Dropout(self.dropout),
            nn.Linear(16, 1)
        )
        
        # 残差归一化层（更小的初始化）
        self.residual_norm = nn.LayerNorm(1)
        self._init_weights()

    def _init_weights(self):
        # LSTM权重初始化（降低方差）
        for name, param in self.lstm1_cell.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data, gain=0.1)  # 降低增益
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data, gain=0.1)
            elif "bias" in name:
                param.data.fill_(0.01)  # 降低bias初始化值
        
        for name, param in self.lstm2_cell.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data, gain=0.1)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data, gain=0.1)
            elif "bias" in name:
                param.data.fill_(0.01)
        
        # 全连接层初始化
        for module in self.weight_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    module.bias.data.fill_(0.01)
        
        for module in self.predict_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    module.bias.data.fill_(0.01)
        
        # 残差归一化层初始化
        for name, param in self.residual_norm.named_parameters():
            if "weight" in name:
                nn.init.ones_(param.data) * 0.1  # 降低权重
            elif "bias" in name:
                param.data.fill_(0.0)

    def forward(self, seq_features, lstm1_input, missing_bool, basin_ids, residual, return_weights=False):
        batch_size = seq_features.shape[0]
        seq_len = seq_features.shape[1]
        device = self.config.DEVICE
        
        # 确保残差是2维（batch, 1）
        if residual.dim() == 1:
            residual = residual.unsqueeze(-1)  # (batch,) → (batch, 1)
        elif residual.dim() != 2:
            residual = residual.view(-1, 1)  # 动态reshape为2维
        
        # 流域嵌入
        basin_id_to_idx = {bid: idx for idx, bid in enumerate(set(self.basin_ids))}
        basin_indices = torch.tensor([basin_id_to_idx[bid.item()] for bid in basin_ids], dtype=torch.long).to(device)
        basin_embed = self.basin_embedding(basin_indices)  # (batch, 16)
        
        # 残差处理（增加裁剪，避免异常值）
        residual_norm = self.residual_norm(residual)  # (batch, 16)
        residual_norm = torch.clamp(residual_norm, min=0.0, max=2.0)  # 限制残差范围
        residual_broadcast = residual_norm.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, 7, 16)
        
        lstm2_outputs = []
        
        # 如果return_weights=True，收集所有时间步的权重
        all_feature_weights = torch.zeros(batch_size, seq_len, self.n_features).to(device)
        
        for t in range(seq_len):
            # LSTM1输入处理（裁剪极端值）
            lstm1_input_t = lstm1_input[:, t, :]  # (batch, 29)
            lstm1_input_t = torch.clamp(lstm1_input_t, min=-3.0, max=3.0)  # 严格裁剪
            
            # LSTM1前向（初始隐态更平缓）
            h1 = torch.zeros(batch_size, self.lstm_hidden_dim, device=device) * 0.01
            c1 = torch.zeros(batch_size, self.lstm_hidden_dim, device=device) * 0.01
            h1, c1 = self.lstm1_cell(lstm1_input_t, (h1, c1))
            h1 = torch.clamp(h1, min=-5.0, max=5.0)
            
            # 生成权重（更稳定的归一化）
            feature_weights = self.weight_head(h1)  # (batch, 19)
            feature_weights = feature_weights * missing_bool[:, t, :]
            weight_sums = feature_weights.sum(dim=-1, keepdim=True) + 1e-8
            feature_weights = feature_weights / weight_sums
            feature_weights = torch.clamp(feature_weights, min=1e-6, max=0.5)  # 放宽权重范围
            
            # 保存权重（如果需要返回）
            if return_weights:
                all_feature_weights[:, t, :] = feature_weights
            
            # LSTM2输入处理（裁剪极端值）
            seq_features_t = seq_features[:, t, :]  # (batch, 19)
            seq_features_t = torch.clamp(seq_features_t, min=-3.0, max=3.0)
            weighted_features = seq_features_t * feature_weights
            weighted_features = torch.clamp(weighted_features, min=-3.0, max=3.0)
            
            lstm2_input_final = torch.cat([weighted_features, basin_embed], dim=-1)
            lstm2_input_final = torch.clamp(lstm2_input_final, min=-3.0, max=3.0)
            
            # LSTM2前向（初始隐态更平缓）
            h2 = torch.zeros(batch_size, self.lstm_hidden_dim, device=device) * 0.01
            c2 = torch.zeros(batch_size, self.lstm_hidden_dim, device=device) * 0.01
            h2, c2 = self.lstm2_cell(lstm2_input_final, (h2, c2))
            h2 = torch.clamp(h2, min=-5.0, max=5.0)
            
            lstm2_outputs.append(h2)
        
        # 预测输出（裁剪非负）
        lstm2_last = lstm2_outputs[-1]
        pred = self.predict_head(lstm2_last)
        pred = torch.clamp(pred, min=0.0, max=100.0)  # 限制预测值范围
        
        if return_weights:
            return pred, all_feature_weights
        else:
            return pred

# ======================== 训练/验证工具函数（修复梯度检查+优化流程）=======================
def train_one_epoch(model, dataloader, criterion, optimizer, config):
    model.train()
    total_loss = 0.0
    skipped_batches = 0
    prev_residual = torch.zeros(config.BATCH_SIZE, 1).to(config.DEVICE)  # 确保初始残差是2维
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="训练epoch")):
        # 预处理批次数据
        processed = preprocess_batch_data(batch, config)
        if processed is None:
            skipped_batches += 1
            continue
        
        # 获取处理后的数据
        seq_features = processed['seq_features']
        lstm1_input = processed['lstm1_input']
        missing_bool = processed['missing_bool']
        basin_ids = processed['basin_ids']
        target = processed['target']
        
        # 调整残差batch大小
        current_batch_size = seq_features.shape[0]
        if current_batch_size != prev_residual.shape[0]:
            prev_residual = torch.zeros(current_batch_size, 1).to(config.DEVICE)
        
        # 前向传播
        try:
            pred = model(
                seq_features=seq_features,
                lstm1_input=lstm1_input,
                missing_bool=missing_bool,
                basin_ids=basin_ids,
                residual=prev_residual,
                return_weights=False  # 训练时不需要返回权重
            )
        except Exception as e:
            print(f"⚠️  前向传播失败：{str(e)}，跳过此批次")
            skipped_batches += 1
            continue
        
        # 检查预测值
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            skipped_batches += 1
            continue
        
        # 反归一化预测值和目标值，计算真实损失
        try:
            # 获取当前批次中所有流域的缩放参数
            pred_denorm_list = []
            target_denorm_list = []
            
            for i in range(current_batch_size):
                basin_id = basin_ids[i].item()
                # 读取该流域的缩放参数
                discharge_min, discharge_max = get_discharge_scaler_params(str(basin_id), config.DATA_INPUT_DIR)
                
                # 反归一化预测值和目标值
                pred_denorm = denormalize_discharge(pred[i].squeeze(), discharge_min, discharge_max)
                target_denorm = denormalize_discharge(target[i].squeeze(), discharge_min, discharge_max)
                
                pred_denorm_list.append(pred_denorm)
                target_denorm_list.append(target_denorm)
            
            # 将反归一化后的值组合成张量
            pred_denorm_tensor = torch.stack(pred_denorm_list)
            target_denorm_tensor = torch.stack(target_denorm_list)
            
            # 计算真实损失（使用反归一化后的值）
            loss = criterion(pred_denorm_tensor, target_denorm_tensor)
            
        except Exception as e:
            # 如果反归一化失败，回退到使用归一化值计算损失
            print(f"⚠️  反归一化失败，使用归一化值计算损失：{str(e)}")
            loss = criterion(pred, target)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度检查（放宽阈值，只过滤NaN/Inf）
        nan_grad_found = False
        for p in model.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    nan_grad_found = True
                    break
        
        if nan_grad_found:
            optimizer.zero_grad()
            skipped_batches += 1
            continue
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        # 优化器更新
        optimizer.step()
        
        total_loss += loss.item() * current_batch_size
        
        # 更新残差（使用相对误差，更稳定）
        current_residual = torch.abs(pred - target) / (target + 1e-8)  # 相对误差
        # 确保残差是2维（batch_size, 1）
        if current_residual.dim() == 1:
            prev_residual = current_residual.unsqueeze(-1).detach()  # (batch_size,) → (batch_size, 1)
        else:
            prev_residual = current_residual.view(-1, 1).detach()  # 保持(batch_size, 1)形状
    
    # 打印跳过批次统计
    if skipped_batches > 0:
        skip_ratio = skipped_batches / len(dataloader) * 100
        print(f"⚠️  本epoch跳过了{skipped_batches}个批次（{skip_ratio:.1f}%）")
    
    avg_loss = total_loss / len(dataloader.dataset) if len(dataloader.dataset) > 0 else float('inf')
    return avg_loss

def validate_one_epoch(model, dataloader, criterion, config, epoch=None):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_basin_ids = []  # 新增：收集流域ID
    
    # 新增：收集特征权重数据
    feature_weights_history = {}  # {basin_id: []}
    
    prev_residual = torch.zeros(config.BATCH_SIZE, 1).to(config.DEVICE)  # 确保初始残差是2维
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="验证epoch")):
            # 预处理批次数据
            processed = preprocess_batch_data(batch, config)
            if processed is None:
                continue
            
            # 获取处理后的数据
            seq_features = processed['seq_features']
            lstm1_input = processed['lstm1_input']
            missing_bool = processed['missing_bool']
            basin_ids = processed['basin_ids']
            target = processed['target']
            
            # 调整残差batch大小
            current_batch_size = seq_features.shape[0]
            if current_batch_size != prev_residual.shape[0]:
                prev_residual = torch.zeros(current_batch_size, 1).to(config.DEVICE)
            
            # 前向传播并收集权重 - 使用return_weights=True获取权重
            pred, batch_feature_weights = model(
                seq_features=seq_features,
                lstm1_input=lstm1_input,
                missing_bool=missing_bool,
                basin_ids=basin_ids,
                residual=prev_residual,
                return_weights=True
            )
            
            # 计算平均权重（跨时间步）
            avg_feature_weights = batch_feature_weights.mean(dim=1)  # (batch, n_features)
            
            # 保存权重数据（按流域分组）
            for i in range(current_batch_size):
                basin_id = basin_ids[i].item()
                if basin_id not in feature_weights_history:
                    feature_weights_history[basin_id] = []
                feature_weights_history[basin_id].append(avg_feature_weights[i].cpu().numpy())
            
            if torch.isnan(pred).any() or torch.isinf(pred).any():
                continue
            
            # 反归一化预测值和目标值，计算真实损失
            try:
                # 获取当前批次中所有流域的缩放参数
                pred_denorm_list = []
                target_denorm_list = []
                
                for i in range(current_batch_size):
                    basin_id = basin_ids[i].item()
                    # 读取该流域的缩放参数
                    discharge_min, discharge_max = get_discharge_scaler_params(str(basin_id), config.DATA_INPUT_DIR)
                    
                    # 反归一化预测值和目标值
                    pred_denorm = denormalize_discharge(pred[i].squeeze(), discharge_min, discharge_max)
                    target_denorm = denormalize_discharge(target[i].squeeze(), discharge_min, discharge_max)
                    
                    pred_denorm_list.append(pred_denorm)
                    target_denorm_list.append(target_denorm)
                
                # 将反归一化后的值组合成张量
                pred_denorm_tensor = torch.stack(pred_denorm_list)
                target_denorm_tensor = torch.stack(target_denorm_list)
                
                # 计算真实损失（使用反归一化后的值）
                loss = criterion(pred_denorm_tensor, target_denorm_tensor)
                
            except Exception as e:
                # 如果反归一化失败，回退到使用归一化值计算损失
                print(f"⚠️  验证阶段反归一化失败，使用归一化值计算损失：{str(e)}")
                loss = criterion(pred, target)
            
            total_loss += loss.item() * current_batch_size
            
            # 更新残差
            current_residual = torch.abs(pred - target) / (target + 1e-8)
            # 确保残差是2维（batch_size, 1）
            if current_residual.dim() == 1:
                prev_residual = current_residual.unsqueeze(-1).detach()  # (batch_size,) → (batch_size, 1)
            else:
                prev_residual = current_residual.view(-1, 1).detach()  # 保持(batch_size, 1)形状
            
            # 收集结果
            all_preds.extend(pred.squeeze().cpu().numpy())
            all_targets.extend(target.squeeze().cpu().numpy())
            all_basin_ids.extend(basin_ids.cpu().numpy())  # 新增：收集流域ID
    
    avg_loss = total_loss / len(dataloader.dataset) if len(dataloader.dataset) > 0 else float('inf')
    
    # 处理权重数据 - 计算每个流域的平均权重
    final_weights_data = {}
    for basin_id, weights_list in feature_weights_history.items():
        if weights_list:
            # 计算该流域在所有batch中的平均权重
            basin_weights = np.array(weights_list)
            final_weights_data[basin_id] = basin_weights.mean(axis=0)  # (n_features,)
    
    # 将权重数据存储到config中，用于后续可视化
    if not hasattr(config, 'feature_weights_history'):
        config.feature_weights_history = {}
    if epoch is not None:
        config.feature_weights_history[epoch] = final_weights_data
    
    # 使用utils.py中的可视化函数
    if epoch is not None and len(all_preds) > 0:
        try:
            # 使用与训练过程相同的时间戳目录
            timestamp = getattr(config, 'VIZ_TIMESTAMP', None)
            if timestamp is None:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            viz_dir = os.path.join('model_output', 'visualizations', timestamp)
            os.makedirs(viz_dir, exist_ok=True)
            
            # 绘制预测对比图
            plot_prediction_comparison(
                pred_values=all_preds,
                target_values=all_targets,
                basin_ids=all_basin_ids if 'all_basin_ids' in locals() else None,
                epoch=epoch,
                save_dir=viz_dir,
                sample_size=50
            )
            
            # 每10个epoch绘制一次损失分布
            if (epoch + 1) % 10 == 0:
                individual_losses = []
                # 重新计算每个样本的损失用于分布图
                for i in range(min(len(all_preds), 100)):  # 限制样本数量
                    pred_val = torch.tensor(all_preds[i])
                    target_val = torch.tensor(all_targets[i])
                    individual_loss = nn.MSELoss()(pred_val, target_val).item()
                    individual_losses.append(individual_loss)
                
                if individual_losses:
                    plot_loss_distribution(individual_losses, epoch=epoch, save_dir=viz_dir)
            
            # 每5个epoch绘制一次特征权重热力图
            if (epoch + 1) % 5 == 0 and final_weights_data:
                from utils import plot_feature_weights_heatmap
                plot_feature_weights_heatmap(
                    feature_weights_history=config.feature_weights_history,
                    feature_names=config.LSTM2_FEATURES if hasattr(config, 'LSTM2_FEATURES') else None,
                    save_dir=viz_dir
                )
            
        except Exception as e:
            print(f"⚠️  可视化失败：{str(e)}")
    
    return avg_loss

# ======================== 主训练流程（优化配置）=======================
def train_sca2lstm(config, use_parallel=True, use_multithreading=True):
    # 使用工厂函数创建DataLoader（简化代码结构）
    from hydrologyDataset import create_hydrology_dataloaders
    train_dataset, train_loader, val_dataset, val_loader = create_hydrology_dataloaders(config, use_parallel, use_multithreading)
    
    # 模型初始化
    model = SCA2LSTM(config).to(config.DEVICE)
    criterion = nn.MSELoss() 
    # 优化器（更稳定的参数）
    optimizer = optim.AdamW(model.parameters(), 
                           lr=config.LR, 
                           weight_decay=1e-5,  # 降低权重衰减
                           betas=(0.9, 0.999),
                           eps=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                    patience=15,  # 延长耐心值
                                                    factor=0.5,
                                                    threshold=0.001,
                                                    min_lr=1e-7)
    
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = config.MODEL_SAVE_PATH
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    
    # 初始化损失历史记录
    train_losses_history = []
    val_losses_history = []
    
    # 创建带时间戳的可视化目录
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_dir = os.path.join('model_output', 'visualizations', timestamp)
    os.makedirs(viz_dir, exist_ok=True)
    print(f"📁 可视化目录已创建: {viz_dir}")
    
    # 将时间戳存储到config中，供验证函数使用
    config.VIZ_TIMESTAMP = timestamp
    
    print(f"\n{'='*30} 开始训练SCA2LSTM {'='*30}")
    print(f"模型配置：{ {k: v for k, v in config.__dict__.items() if not k.startswith('__') and not callable(v)} }")
    print(f"设备：{config.DEVICE}")
    print(f"训练样本数：{len(train_dataset)}，训练批次：{len(train_loader)}")
    print(f"验证样本数：{len(val_dataset)}，验证批次：{len(val_loader)}")
    print(f"学习率：{config.LR}，Dropout：{config.DROPOUT}")
    
    # 训练循环
    for epoch in range(config.N_EPOCHS):
        print(f"\n📌 Epoch {epoch+1}/{config.N_EPOCHS}")
        
        # 训练
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, config)
        # 验证
        val_loss = validate_one_epoch(
            model, val_loader, criterion, config, 
            epoch=epoch
        )
        
        # 记录损失历史
        train_losses_history.append(train_loss)
        val_losses_history.append(val_loss)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 打印日志
        print(f"训练损失：{train_loss:.6f}，验证损失：{val_loss:.6f}")
        print(f"当前学习率：{optimizer.param_groups[0]['lr']:.8f}")
        
        # 每5个epoch绘制一次损失曲线
        if (epoch + 1) % 5 == 0 or epoch == config.N_EPOCHS - 1:
            try:
                plot_training_curves(train_losses_history, val_losses_history, save_dir=viz_dir)
                print(f"📊 已更新损失曲线图")
            except Exception as e:
                print(f"⚠️  损失曲线绘制失败: {str(e)}")
        
        # 早停逻辑
        if val_loss < best_val_loss - 1e-6:  # 增加微小阈值，避免浮点误差
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "config": {k: v for k, v in config.__dict__.items() if not k.startswith('__') and not callable(v)}
            }, best_model_path)
            print(f"✅ 保存最优模型（验证损失：{best_val_loss:.6f}）")
        else:
            patience_counter += 1
            print(f"⚠️  早停计数器：{patience_counter}/{config.PATIENCE}")
            if patience_counter >= config.PATIENCE:
                print(f"❌ 早停触发，训练结束")
                break
    
    # 权重分析
    print(f"\n{'='*30} 模型权重分析 {'='*30}")
    try:
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print("📊 LSTM1权重统计:")
        for name, param in model.named_parameters():
            if 'lstm1_cell' in name:
                print(f"   {name}: 均值={param.mean().item():.6f}, 标准差={param.std().item():.6f}")
        
        print("\n📊 LSTM2权重统计:")
        for name, param in model.named_parameters():
            if 'lstm2_cell' in name:
                print(f"   {name}: 均值={param.mean().item():.6f}, 标准差={param.std().item():.6f}")
    
    except Exception as e:
        print(f"⚠️  权重分析失败：{str(e)}")
    
    print(f"\n{'='*30} 训练完成 {'='*30}")
    print(f"最优验证损失：{best_val_loss:.6f}")
    print(f"模型保存路径：{best_model_path}")
    
    return best_model_path

# ======================== 运行训练 ========================
if __name__ == "__main__":
    import argparse
    
    # 添加命令行参数
    parser = argparse.ArgumentParser(description='SCA2LSTM 训练脚本')
    parser.add_argument('--parallel', action='store_true', help='使用并行数据集（多线程/多进程）')
    parser.add_argument('--serial', action='store_true', help='使用串行数据集（默认）')
    parser.add_argument('--epochs', type=int, default=60, help='训练轮数')
    args = parser.parse_args()
    
    # 确定是否使用并行
    use_parallel = args.parallel and not args.serial
    if args.parallel and args.serial:
        print("⚠️  同时指定了--parallel和--serial，默认使用并行模式")
        use_parallel = True
    elif not args.parallel and not args.serial:
        # 默认使用串行模式（Windows系统下更安全）
        import platform
        is_windows = platform.system() == 'Windows'
        use_parallel = not is_windows  # Windows下默认串行，其他系统默认并行
        print(f"🎯 默认模式：{'并行' if use_parallel else '串行'}数据集")
    
    # 加载配置
    config = load_config()
    
    # 固定随机种子
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.SEED)
    
    try:
        # 更新配置中的epochs参数
        config.N_EPOCHS = args.epochs
        best_model_path = train_sca2lstm(
            config, 
            use_parallel=use_parallel,
            use_multithreading=True
        )
        print(f"\n🎉 训练成功完成！最优模型已保存至: {best_model_path}")
        print(f"📈 使用{'并行' if use_parallel else '串行'}数据集完成训练")
    except Exception as e:
        print(f"\n❌ 训练失败：{str(e)}")
        import traceback
        traceback.print_exc()