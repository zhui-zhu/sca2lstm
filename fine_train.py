#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCA2LSTM 流域微调脚本
在预训练模型基础上对特定流域进行微调
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import time
import argparse
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 导入项目模块
from sca2lstm import SCA2LSTM, load_config
from hydrologyDataset import HydrologyDataset
from utils import plot_training_curves, plot_prediction_comparison, plot_loss_distribution, plot_feature_weights_heatmap

def load_pretrained_model(model_path, config, device):
    """
    加载预训练模型
    
    Args:
        model_path: 预训练模型路径
        config: 配置对象
        device: 计算设备
    
    Returns:
        model: 加载的模型
        checkpoint: 检查点信息
    """
    print(f"📂 加载预训练模型: {model_path}")
    
    # 初始化模型
    model = SCA2LSTM(config).to(device)
    
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 加载模型状态
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ 预训练模型加载成功")
    print(f"📊 原始训练信息:")
    print(f"   - 训练轮次: {checkpoint.get('epoch', '未知')}")
    best_val_loss = checkpoint.get('best_val_loss', '未知')
    if isinstance(best_val_loss, (int, float)):
        print(f"   - 验证损失: {best_val_loss:.6f}")
    else:
        print(f"   - 验证损失: {best_val_loss}")
    
    return model, checkpoint

def freeze_layers(model, freeze_embedding=True, freeze_lstm1=True, freeze_weight_head=False):
    """
    冻结模型层
    
    Args:
        model: SCA2LSTM模型
        freeze_embedding: 是否冻结嵌入层
        freeze_lstm1: 是否冻结LSTM1层
        freeze_weight_head: 是否冻结权重头
    """
    print("🔒 设置层冻结策略:")
    
    # 冻结嵌入层
    if freeze_embedding:
        for param in model.basin_embedding.parameters():
            param.requires_grad = False
        print("   - 流域嵌入层: 冻结")
    else:
        print("   - 流域嵌入层: 可训练")
    
    # 冻结LSTM1层
    if freeze_lstm1:
        for param in model.lstm1_cell.parameters():
            param.requires_grad = False
        print("   - LSTM1层: 冻结")
    else:
        print("   - LSTM1层: 可训练")
    
    # 冻结权重头
    if freeze_weight_head:
        for param in model.weight_head.parameters():
            param.requires_grad = False
        print("   - 权重头: 冻结")
    else:
        print("   - 权重头: 可训练")
    
    # LSTM2层和预测头保持可训练（微调重点）
    for param in model.lstm2_cell.parameters():
        param.requires_grad = True
    for param in model.predict_head.parameters():
        param.requires_grad = True
    
    print("   - LSTM2层: 可训练")
    print("   - 预测头: 可训练")

def prepare_basin_specific_data(target_basin_id, config, fine_tune_ratio=0.8):
    """
    准备特定流域的微调和验证数据
    
    Args:
        target_basin_id: 目标流域ID
        config: 配置对象
        fine_tune_ratio: 微调数据比例
    
    Returns:
        fine_tune_dataset: 微调数据集
        val_dataset: 验证数据集
    """
    print(f"📊 准备流域 {target_basin_id} 的微调数据")
    
    # 检查目标流域是否存在数据
    data_path = os.path.join(config.DATA_INPUT_DIR,str(target_basin_id), f"model_input_{target_basin_id}.csv")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"流域 {target_basin_id} 的数据文件不存在: {data_path}")
    
    # 读取数据
    df = pd.read_csv(data_path)
    total_samples = len(df)
    
    print(f"📈 流域 {target_basin_id} 数据概况:")
    print(f"   - 总样本数: {total_samples}")
    print(f"   - 时间范围: {df['date'].min()} 到 {df['date'].max()}")
    
    # 创建完整的数据集（用于数据分割）
    full_dataset = HydrologyDataset(
        basin_ids=[target_basin_id], 
        config=config, 
        mode="fine_tune",
        use_parallel=False  # 微调时禁用并行
    )
    
    # 分割数据集
    total_size = len(full_dataset)
    fine_tune_size = int(total_size * fine_tune_ratio)
    val_size = total_size - fine_tune_size
    
    fine_tune_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [fine_tune_size, val_size],
        generator=torch.Generator().manual_seed(config.SEED)
    )
    
    print(f"   - 微调样本数: {len(fine_tune_dataset)}")
    print(f"   - 验证样本数: {len(val_dataset)}")
    
    return fine_tune_dataset, val_dataset

def fine_tune_one_epoch(model, dataloader, criterion, optimizer, config, epoch):
    """
    微调单个epoch
    
    Args:
        model: SCA2LSTM模型
        dataloader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        config: 配置对象
        epoch: 当前轮次
    
    Returns:
        avg_loss: 平均损失
    """
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f"微调 Epoch {epoch+1}")
    
    for batch_idx, batch_data in enumerate(progress_bar):
        # 数据转移到设备
        seq_features = batch_data["seq_features"].to(config.DEVICE)
        lstm1_input = batch_data["lstm1_input"].to(config.DEVICE)
        missing_bool = batch_data["missing_bool"].to(config.DEVICE)
        basin_ids = batch_data["basin_id"].to(config.DEVICE)
        target = batch_data["target"].to(config.DEVICE)
        
        # 初始化residual为0（演示数据中没有residual）
        residual = torch.zeros((target.size(0), 1), device=config.DEVICE)
        
        # 前向传播
        pred = model(
            seq_features=seq_features,
            lstm1_input=lstm1_input,
            missing_bool=missing_bool,
            basin_ids=basin_ids,
            residual=residual,
            return_weights=False
        )
        
        # 检查预测值的有效性
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            print(f"⚠️  批次{batch_idx}包含NaN/Inf预测值，跳过")
            continue
            
        # 计算损失
        loss = criterion(pred.squeeze(), target)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 统计
        total_loss += loss.item() * target.size(0)
        
        # 更新进度条
        progress_bar.set_postfix({
            '损失': f'{loss.item():.6f}',
            '平均损失': f'{total_loss / ((batch_idx + 1) * target.size(0)):.6f}'
        })
    
    return total_loss / len(dataloader.dataset)

def validate_fine_tune(model, dataloader, criterion, config, epoch):
    """
    微调验证
    
    Args:
        model: SCA2LSTM模型
        dataloader: 验证数据加载器
        criterion: 损失函数
        config: 配置对象
        epoch: 当前轮次
    
    Returns:
        avg_loss: 平均验证损失
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc=f"验证 Epoch {epoch+1}"):
            # 数据转移到设备
            seq_features = batch_data["seq_features"].to(config.DEVICE)
            lstm1_input = batch_data["lstm1_input"].to(config.DEVICE)
            missing_bool = batch_data["missing_bool"].to(config.DEVICE)
            basin_ids = batch_data["basin_id"].to(config.DEVICE)
            target = batch_data["target"].to(config.DEVICE)
            
            # 初始化residual为0（演示数据中没有residual）
            residual = torch.zeros((target.size(0), 1), device=config.DEVICE)
            
            # 前向传播
            pred = model(
                seq_features=seq_features,
                lstm1_input=lstm1_input,
                missing_bool=missing_bool,
                basin_ids=basin_ids,
                residual=residual,
                return_weights=False
            )
            
            # 检查预测值的有效性
            if torch.isnan(pred).any() or torch.isinf(pred).any():
                print(f"⚠️  验证批次包含NaN/Inf预测值，跳过")
                continue
                
            # 计算损失
            loss = criterion(pred.squeeze(), target)
            total_loss += loss.item() * target.size(0)
            
            # 收集预测结果（过滤NaN值）
            pred_numpy = pred.squeeze().cpu().numpy()
            target_numpy = target.cpu().numpy()
            
            # 确保数组是一维的
            if pred_numpy.ndim > 1:
                pred_numpy = pred_numpy.flatten()
            if target_numpy.ndim > 1:
                target_numpy = target_numpy.flatten()
            
            # 只收集有效的预测值
            valid_mask = ~(np.isnan(pred_numpy) | np.isinf(pred_numpy) | np.isnan(target_numpy) | np.isinf(target_numpy))
            if valid_mask.any():
                all_preds.extend(pred_numpy[valid_mask])
                all_targets.extend(target_numpy[valid_mask])
    
    avg_loss = total_loss / len(dataloader.dataset)
    
    # 检查是否有足够的有效数据
    if len(all_preds) == 0 or len(all_targets) == 0:
        print("⚠️  没有有效的预测数据用于评估")
        return float('inf')
    
    # 计算评估指标
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    
    print(f"📊 验证结果:")
    print(f"   - 平均损失: {avg_loss:.6f}")
    print(f"   - RMSE: {rmse:.6f}")
    print(f"   - MAE: {mae:.6f}")
    print(f"   - R²: {r2:.6f}")
    
    return avg_loss

def fine_tune_basin(target_basin_id, config, args):
    """
    对特定流域进行微调
    
    Args:
        target_basin_id: 目标流域ID
        config: 配置对象
        args: 命令行参数
    """
    print(f"\n{'='*50}")
    print(f"🎯 开始微调流域: {target_basin_id}")
    print(f"{'='*50}")
    
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    config.DEVICE = device
    print(f"💻 使用设备: {device}")
    
    # 加载预训练模型
    model, checkpoint = load_pretrained_model(args.pretrained_model, config, device)
    
    # 设置层冻结策略
    freeze_layers(
        model, 
        freeze_embedding=not args.unfreeze_embedding,
        freeze_lstm1=not args.unfreeze_lstm1,
        freeze_weight_head=not args.unfreeze_weight_head
    )
    
    # 准备微调数据
    fine_tune_dataset, val_dataset = prepare_basin_specific_data(
        target_basin_id, config, args.fine_tune_ratio
    )
    
    # 创建数据加载器
    fine_tune_loader = DataLoader(
        fine_tune_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0,  # 微调时禁用多进程
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 设置优化器（使用较小的学习率进行微调）
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        patience=args.patience // 2,
        factor=0.5,
        threshold=0.001,
        min_lr=1e-7
    )
    
    # 损失函数
    criterion = nn.MSELoss()
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join('model_output', 'fine_tune', f'basin_{target_basin_id}', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 输出目录: {output_dir}")
    
    # 训练历史
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    # 微调循环
    print(f"\n🚀 开始微调 (最大轮次: {args.epochs})")
    
    for epoch in range(args.epochs):
        print(f"\n📌 微调轮次 {epoch+1}/{args.epochs}")
        
        # 微调
        train_loss = fine_tune_one_epoch(
            model, fine_tune_loader, criterion, optimizer, config, epoch
        )
        train_losses.append(train_loss)
        
        # 验证
        val_loss = validate_fine_tune(
            model, val_loader, criterion, config, epoch
        )
        val_losses.append(val_loss)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 早停逻辑
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_counter = 0
            
            # 保存最佳模型
            best_model_path = os.path.join(output_dir, f'best_model_basin_{target_basin_id}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'config': {k: v for k, v in config.__dict__.items() if not k.startswith('__') and not callable(v)},
                'target_basin_id': target_basin_id,
                'fine_tune_config': {
                    'freeze_embedding': not args.unfreeze_embedding,
                    'freeze_lstm1': not args.unfreeze_lstm1,
                    'freeze_weight_head': not args.unfreeze_weight_head,
                    'learning_rate': args.learning_rate,
                    'batch_size': args.batch_size,
                    'fine_tune_ratio': args.fine_tune_ratio
                }
            }, best_model_path)
            print(f"💾 保存最佳模型: {best_model_path}")
        else:
            patience_counter += 1
            print(f"⚠️  早停计数器: {patience_counter}/{args.patience}")
            
            if patience_counter >= args.patience:
                print(f"❌ 早停触发，微调结束")
                break
    
    # 绘制训练曲线
    try:
        plot_training_curves(train_losses, val_losses, save_dir=output_dir)
        print(f"📊 训练曲线已保存")
    except Exception as e:
        print(f"⚠️  训练曲线绘制失败: {str(e)}")
    
    # 最终评估
    print(f"\n{'='*30}")
    print(f"🏁 微调完成总结")
    print(f"{'='*30}")
    print(f"📊 最佳验证损失: {best_val_loss:.6f}")
    print(f"📊 最终训练损失: {train_losses[-1]:.6f}")
    print(f"📊 最终验证损失: {val_losses[-1]:.6f}")
    print(f"📁 模型保存路径: {output_dir}")
    
    return best_model_path

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='SCA2LSTM 流域微调脚本')
    
    # 必需参数
    parser.add_argument('--target_basin', type=int, required=True, 
                       help='目标流域ID')
    parser.add_argument('--pretrained_model', type=str, required=True,
                       help='预训练模型路径')
    
    # 微调参数
    parser.add_argument('--epochs', type=int, default=20,
                       help='微调轮次 (默认: 20)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小 (默认: 16)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='学习率 (默认: 0.0001)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='权重衰减 (默认: 1e-5)')
    parser.add_argument('--patience', type=int, default=8,
                       help='早停耐心值 (默认: 8)')
    parser.add_argument('--fine_tune_ratio', type=float, default=0.8,
                       help='微调数据比例 (默认: 0.8)')
    
    # 层解冻参数
    parser.add_argument('--unfreeze_embedding', action='store_true',
                       help='解冻嵌入层')
    parser.add_argument('--unfreeze_lstm1', action='store_true',
                       help='解冻LSTM1层')
    parser.add_argument('--unfreeze_weight_head', action='store_true',
                       help='解冻权重头')
    
    # 其他参数
    parser.add_argument('--no_cuda', action='store_true',
                       help='禁用CUDA')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认: 42)')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available() and not args.no_cuda:
        torch.cuda.manual_seed(args.seed)
    
    # 加载配置
    config = load_config()
    
    try:
        # 执行微调
        best_model_path = fine_tune_basin(args.target_basin, config, args)
        print(f"\n🎉 流域 {args.target_basin} 微调成功完成！")
        print(f"📂 最佳模型路径: {best_model_path}")
        
    except Exception as e:
        print(f"\n❌ 微调失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())