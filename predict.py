#!/usr/bin/env python3
"""
SCA2LSTM 预测脚本
使用微调后的模型进行流量预测
"""

import torch
import numpy as np
import pandas as pd
import os
import sys
import argparse
from datetime import datetime, timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt
import ast

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sca2lstm import SCA2LSTM
from hydrologyDataset import HydrologyDataset

def load_config_from_file(config_path="run.config"):
    """
    从配置文件加载配置参数
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置对象
    """
    config_dict = {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 逐行解析配置文件
        for line in content.split('\n'):
            line = line.strip()
            # 跳过空行和注释行
            if not line or line.startswith('#'):
                continue
                
            # 处理列表类型的配置
            if line.startswith('LSTM1_DERIVED_FEATS') or line.startswith('LSTM2_FEATURES'):
                key = line.split('=')[0].strip()
                # 提取列表内容（在方括号之间的内容）
                start_idx = line.find('[')
                end_idx = line.rfind(']')
                if start_idx != -1 and end_idx != -1:
                    list_content = line[start_idx:end_idx+1]
                    try:
                        config_dict[key] = ast.literal_eval(list_content)
                    except:
                        # 如果解析失败，手动解析
                        items = []
                        item_content = line[start_idx+1:end_idx]
                        for item in item_content.split(','):
                            item = item.strip().strip('"\'')
                            if item:
                                items.append(item)
                        config_dict[key] = items
            
            # 处理简单键值对
            elif '=' in line and not line.startswith('['):
                parts = line.split('=', 1)
                key = parts[0].strip()
                value = parts[1].strip()
                
                # 移除注释
                if '#' in value:
                    value = value.split('#')[0].strip()
                
                # 转换值类型
                if value.startswith('[') and value.endswith(']'):
                    # 列表类型
                    try:
                        config_dict[key] = ast.literal_eval(value)
                    except:
                        config_dict[key] = []
                elif value in ['True', 'False']:
                    # 布尔类型
                    config_dict[key] = value == 'True'
                elif value.isdigit():
                    # 整数类型
                    config_dict[key] = int(value)
                elif value.replace('.', '').isdigit():
                    # 浮点数类型
                    config_dict[key] = float(value)
                else:
                    # 字符串类型，移除引号
                    config_dict[key] = value.strip('"\'')
    
    except FileNotFoundError:
        print(f"⚠️  配置文件 {config_path} 不存在，使用默认配置")
        return None
    except Exception as e:
        print(f"⚠️  配置文件解析失败: {str(e)}，使用默认配置")
        return None
    
    return config_dict

class Config:
    """预测配置类 - 优先从配置文件加载参数"""
    
    def __init__(self, config_path="run.config"):
        """初始化配置，优先从配置文件加载"""
        # 首先设置默认值
        self._set_default_values()
        
        # 然后从配置文件加载（如果存在）
        config_dict = load_config_from_file(config_path)
        if config_dict:
            self._load_from_dict(config_dict)
            print(f"✅ 配置已从 {config_path} 文件加载")
        else:
            print("⚠️  使用默认配置")
    
    def _set_default_values(self):
        """设置默认配置值"""
        # 模型特征配置
        self.LSTM1_DERIVED_FEATS = [
            "ssi", "high_prec_running_days", "low_prec_running_days", "prec_7day_sum", "prec_30day_sum"
        ]
        self.LSTM2_FEATURES = [
            "precipitation", "peti", "temperature", "discharge_vol",
            "area", "dpsbar", "elev_mean", "aridity", "p_seasonality",
            "tawc", "porosity_cosby", "baseflow_index", "dwood_perc", "ewood_perc",
            "grass_perc", "urban_perc", "inwater_perc", "benchmark_catch", "reservoir_cap"
        ]
        
        # 模型参数
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.SEQ_LEN = 3
        self.PRED_LEN = 1
        self.LSTM_HIDDEN_DIM = 64
        self.LSTM_LAYERS = 2
        self.DROPOUT = 0.3
        self.EMBEDDING_DIM = 16
        self.N_FEATURES = len(self.LSTM2_FEATURES)
        self.LSTM1_INPUT_DIM = len(self.LSTM1_DERIVED_FEATS) + 1  # 5个衍生特征 + 1个LSTM2反馈残差 = 6
        
        # 数据配置
        self.TARGET_COL = "discharge_vol"
        self.DATA_INPUT_DIR = "./model_input_data/"
        self.TRAIN_BASIN_IDS = [32006, 42003, 51001, 75003, 79005]
        self.VAL_BASIN_IDS = [75003]
        self.MIN_VALID_LABEL_RATIO = 0.3  # 训练集流域必须包含30%以上有效标签
        self.MIN_VALID_ROWS = 10000  # 每个流域必须包含10000行有效数据
    
    def _load_from_dict(self, config_dict):
        """从字典加载配置"""
        # 列表类型配置
        if 'LSTM1_DERIVED_FEATS' in config_dict:
            self.LSTM1_DERIVED_FEATS = config_dict['LSTM1_DERIVED_FEATS']
        if 'LSTM2_FEATURES' in config_dict:
            self.LSTM2_FEATURES = config_dict['LSTM2_FEATURES']
        if 'TRAIN_BASIN_IDS' in config_dict:
            self.TRAIN_BASIN_IDS = config_dict['TRAIN_BASIN_IDS']
        if 'VAL_BASIN_IDS' in config_dict:
            self.VAL_BASIN_IDS = config_dict['VAL_BASIN_IDS']
        
        # 字符串类型配置
        if 'TARGET_COL' in config_dict:
            self.TARGET_COL = config_dict['TARGET_COL']
        if 'DATA_INPUT_DIR' in config_dict:
            self.DATA_INPUT_DIR = config_dict['DATA_INPUT_DIR']
        if 'MODEL_SAVE_PATH' in config_dict:
            self.MODEL_SAVE_PATH = config_dict['MODEL_SAVE_PATH']
        
        # 整数类型配置
        if 'SEQ_LEN' in config_dict:
            self.SEQ_LEN = config_dict['SEQ_LEN']
        if 'PRED_LEN' in config_dict:
            self.PRED_LEN = config_dict['PRED_LEN']
        if 'LSTM_HIDDEN_DIM' in config_dict:
            self.LSTM_HIDDEN_DIM = config_dict['LSTM_HIDDEN_DIM']
        if 'LSTM_LAYERS' in config_dict:
            self.LSTM_LAYERS = config_dict['LSTM_LAYERS']
        if 'EMBEDDING_DIM' in config_dict:
            self.EMBEDDING_DIM = config_dict['EMBEDDING_DIM']
        if 'BATCH_SIZE' in config_dict:
            self.BATCH_SIZE = config_dict['BATCH_SIZE']
        if 'N_EPOCHS' in config_dict:
            self.N_EPOCHS = config_dict['N_EPOCHS']
        if 'PATIENCE' in config_dict:
            self.PATIENCE = config_dict['PATIENCE']
        if 'MIN_VALID_ROWS' in config_dict:
            self.MIN_VALID_ROWS = config_dict['MIN_VALID_ROWS']
        
        # 浮点数类型配置
        if 'DROPOUT' in config_dict:
            self.DROPOUT = config_dict['DROPOUT']
        if 'LR' in config_dict:
            self.LR = config_dict['LR']
        if 'MIN_VALID_LABEL_RATIO' in config_dict:
            self.MIN_VALID_LABEL_RATIO = config_dict['MIN_VALID_LABEL_RATIO']
        
        # 重新计算依赖字段
        self.N_FEATURES = len(self.LSTM2_FEATURES)
        self.LSTM1_INPUT_DIM = len(self.LSTM1_DERIVED_FEATS) + 1

def load_model(model_path, basin_id):
    """
    加载微调后的模型
    
    Args:
        model_path: 模型文件路径
        basin_id: 目标流域ID
    
    Returns:
        加载好的模型
    """
    print(f"📂 加载微调模型: {model_path}")
    
    # 创建配置
    config = Config()
    
    # 创建模型
    model = SCA2LSTM(config)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ 模型权重加载成功")
        print(f"📊 微调轮次: {checkpoint.get('epoch', '未知')}")
        print(f"📊 训练损失: {checkpoint.get('train_loss', '未知')}")
        print(f"📊 验证损失: {checkpoint.get('val_loss', '未知')}")
    else:
        model.load_state_dict(checkpoint)
        print(f"✅ 模型权重加载成功")
    
    model.eval()
    model.to(config.DEVICE)
    
    return model, config

def predict_basin_flow(model, config, basin_id, num_samples=50):
    """
    预测指定流域的流量
    
    Args:
        model: 训练好的模型
        config: 配置对象
        basin_id: 流域ID
        num_samples: 预测样本数量
    
    Returns:
        预测结果DataFrame
    """
    print(f"🧪 开始预测流域 {basin_id} 的流量")
    
    # 创建数据集
    dataset = HydrologyDataset([basin_id], config, mode="test", use_parallel=False)
    
    if len(dataset) == 0:
        print(f"❌ 流域 {basin_id} 没有可用数据")
        return None
    
    print(f"📊 可用样本数: {len(dataset)}")
    
    # 获取数据
    all_predictions = []
    all_dates = []
    all_actuals = []
    
    # 随机选择一些样本进行预测展示
    sample_indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    # 获取该流域的缩放参数
    discharge_min, discharge_max = get_discharge_scaler_params(str(basin_id), config.DATA_INPUT_DIR)
    
    with torch.no_grad():
        for idx in tqdm(sample_indices, desc="预测进度"):
            # 获取数据
            sample_data = dataset[idx]
            seq_features = sample_data["seq_features"]
            lstm1_input = sample_data["lstm1_input"]
            missing_bool = sample_data["missing_bool"]
            basin_ids = sample_data["basin_id"]
            target = sample_data["target"]
            
            # 转换为batch格式
            seq_features = seq_features.unsqueeze(0).to(config.DEVICE)
            lstm1_input = lstm1_input.unsqueeze(0).to(config.DEVICE)
            missing_bool = missing_bool.unsqueeze(0).to(config.DEVICE)
            basin_ids = torch.tensor([basin_id], dtype=torch.long).to(config.DEVICE)
            residual = torch.zeros(1, 1).to(config.DEVICE)
            
            # 预测
            prediction = model(seq_features, lstm1_input, missing_bool, basin_ids, residual, return_weights=False)
            
            # 保存结果
            pred_value = prediction.cpu().numpy().flatten()[0]
            target_value = target.cpu().numpy().flatten()[0] if isinstance(target, torch.Tensor) else target
            
            # 对预测值和真实值进行反归一化处理
            pred_value_denorm = denormalize_discharge(pred_value, discharge_min, discharge_max)
            target_value_denorm = denormalize_discharge(target_value, discharge_min, discharge_max)
            
            # 获取日期（从数据集样本中提取）
            sample_data = dataset.samples[idx]
            date = sample_data.get('date', dataset.data.iloc[idx]['date'] if hasattr(dataset, 'data') else None)
            
            all_predictions.append(pred_value_denorm)
            all_actuals.append(target_value_denorm)
            all_dates.append(date)
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'date': all_dates,
        'basin_id': basin_id,
        'predicted_flow': all_predictions,
        'actual_flow': all_actuals,
        'abs_error': np.abs(np.array(all_predictions) - np.array(all_actuals))
    })
    
    return results_df

def get_output_directory():
    """创建并返回时间戳输出目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"model_output/predict/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def plot_water_level_comparison(results_df, basin_id, output_dir=None):
    """Plot water level comparison chart"""
    # Sort by date
    results_df = results_df.sort_values('date')
    
    # Use provided output directory or create new one
    if output_dir is None:
        output_dir = get_output_directory()
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    fig.suptitle(f'Water Level Prediction vs Actual Comparison - Basin {basin_id}', fontsize=16, fontweight='bold')
    
    # Plot 1: Time series comparison
    ax1 = axes[0]
    ax1.plot(results_df['date'], results_df['actual_flow'], 'b-', linewidth=2, 
            label='Actual Flow', alpha=0.8)
    ax1.plot(results_df['date'], results_df['predicted_flow'], 'r--', linewidth=2, 
            label='Predicted Flow', alpha=0.8)
    ax1.fill_between(results_df['date'], results_df['actual_flow'], 
                    results_df['predicted_flow'], alpha=0.3, color='gray', 
                    label='Error Area')
    
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Discharge (m³/s)', fontsize=12)
    ax1.set_title('Time Series Comparison', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 2: Scatter plot (Predicted vs Actual)
    ax2 = axes[1]
    ax2.scatter(results_df['actual_flow'], results_df['predicted_flow'], 
               alpha=0.6, s=30, c='blue')
    
    # Add perfect prediction line
    min_val = min(results_df['actual_flow'].min(), results_df['predicted_flow'].min())
    max_val = max(results_df['actual_flow'].max(), results_df['predicted_flow'].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
            label='Perfect Prediction Line')
    
    ax2.set_xlabel('Actual Flow (m³/s)', fontsize=12)
    ax2.set_ylabel('Predicted Flow (m³/s)', fontsize=12)
    ax2.set_title('Predicted vs Actual Scatter Plot', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Calculate statistics
    mae = results_df['abs_error'].mean()
    rmse = np.sqrt(np.mean((results_df['actual_flow'] - results_df['predicted_flow']) ** 2))
    correlation = results_df['actual_flow'].corr(results_df['predicted_flow'])
    
    # Add statistics info on scatter plot
    stats_text = f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nCorrelation: {correlation:.4f}'
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_file = f"{output_dir}/water_level_comparison_basin_{basin_id}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Water level comparison chart saved to: {output_file}")
    
    # Close figure to release memory
    plt.close(fig)

def fill_missing_data(data_df, start_date, end_date):
    """Fill missing data in the specified date range"""
    # Create complete date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Reindex to include all dates
    data_df = data_df.set_index('date').reindex(date_range)
    data_df.index.name = 'date'
    
    # Count missing values
    missing_count = data_df.isnull().sum().sum()
    total_count = len(data_df)
    
    if missing_count > 0:
        print(f"⚠️  Found {missing_count} missing values out of {total_count} total data points")
        
        # For each column with missing data
        for col in data_df.columns:
            if data_df[col].isnull().sum() > 0:
                missing_dates = data_df[data_df[col].isnull()].index
                print(f"📅 Column '{col}' missing data on dates: {missing_dates[:5].tolist()}...")
                
                # Use different strategies based on missing data amount
                if len(missing_dates) <= 2:
                    # For few missing values, use interpolation with neighboring days
                    data_df[col] = data_df[col].interpolate(method='linear', limit_direction='both')
                    print(f"✅ Filled {len(missing_dates)} missing values using linear interpolation")
                else:
                    # For many missing values, use 7-day rolling mean
                    data_df[col] = data_df[col].fillna(data_df[col].rolling(window=7, min_periods=1, center=True).mean())
                    print(f"✅ Filled {len(missing_dates)} missing values using 7-day rolling mean")
    
    return data_df.reset_index()

def predict_continuous_flow(model, config, basin_id, start_date, end_date):
    """
    Continuous prediction for specified date range
    
    Args:
        model: Trained model
        config: Configuration object
        basin_id: Basin ID
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        Prediction results DataFrame
    """
    print(f"🧪 Starting continuous prediction for basin {basin_id} from {start_date} to {end_date}")
    
    # Convert dates to datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Create dataset for the basin
    dataset = HydrologyDataset([basin_id], config, mode="test", use_parallel=False)
    
    if len(dataset) == 0:
        print(f"❌ No available data for basin {basin_id}")
        return None
    
    # Get all data for the basin
    basin_data = dataset.data[dataset.data['basin_id'] == basin_id].copy()
    basin_data['date'] = pd.to_datetime(basin_data['date'])
    
    # Filter data within the specified date range
    mask = (basin_data['date'] >= start_date) & (basin_data['date'] <= end_date)
    target_data = basin_data[mask].copy()
    
    if len(target_data) == 0:
        print(f"❌ No data available in the specified date range: {start_date} to {end_date}")
        return None
    
    # Fill missing data
    target_data = fill_missing_data(target_data, start_date, end_date)
    
    print(f"📊 Processing {len(target_data)} days of data from {start_date.date()} to {end_date.date()}")
    
    # 获取该流域的缩放参数
    discharge_min, discharge_max = get_discharge_scaler_params(str(basin_id), config.DATA_INPUT_DIR)
    
    # Prepare results storage
    all_predictions = []
    all_actuals = []
    all_dates = []
    
    # Create a working copy of data for continuous prediction
    working_data = target_data.copy()
    
    with torch.no_grad():
        for i in tqdm(range(len(target_data)), desc="Continuous prediction progress"):
            current_date = target_data.iloc[i]['date']
            
            # For each day, we need 3 days of historical data
            if i < 3:
                # For the first few days, use actual historical data
                historical_end_idx = i
                historical_start_idx = max(0, i - 3)
            else:
                # For subsequent days, use a mix of actual and predicted data
                historical_end_idx = i
                historical_start_idx = i - 3
            
            # Prepare sequence data
            historical_data = working_data.iloc[historical_start_idx:historical_end_idx + 1].copy()
            
            if len(historical_data) < 3 and i > 0:
                # If we don't have enough historical data, skip or use available data
                print(f"⚠️  Insufficient historical data for {current_date.date()}, using available data")
                continue
            
            # Create a temporary dataset for this prediction
            temp_data = historical_data.tail(3)  # Use last 3 days
            
            if len(temp_data) < 3:
                continue
            
            # Convert to model input format
            # This is a simplified version - you may need to adapt based on your exact data structure
            try:
                # Extract features (this part needs to be adapted to your exact data structure)
                seq_features = []  # You'll need to implement this based on your data format
                lstm1_input = []   # You'll need to implement this based on your data format
                
                # For now, let's use a simpler approach by finding matching samples in the dataset
                matching_samples = []
                for j in range(len(dataset.samples)):
                    sample_date = dataset.samples[j].get('date', dataset.data.iloc[j]['date'])
                    if pd.to_datetime(sample_date) == current_date:
                        matching_samples.append(j)
                        break
                
                if matching_samples:
                    # Use the matching sample from dataset
                    sample_data = dataset[matching_samples[0]]
                    seq_features = sample_data["seq_features"].unsqueeze(0).to(config.DEVICE)
                    lstm1_input = sample_data["lstm1_input"].unsqueeze(0).to(config.DEVICE)
                    missing_bool = sample_data["missing_bool"].unsqueeze(0).to(config.DEVICE)
                    basin_ids = torch.tensor([basin_id], dtype=torch.long).to(config.DEVICE)
                    residual = torch.zeros(1, 1).to(config.DEVICE)
                    
                    # Get actual value
                    actual_value = target_data.iloc[i]['discharge_vol']
                    
                    # Make prediction
                    prediction = model(seq_features, lstm1_input, missing_bool, basin_ids, residual, return_weights=False)
                    pred_value = prediction.cpu().numpy().flatten()[0]
                    
                    # 对预测值和真实值进行反归一化处理
                    pred_value_denorm = denormalize_discharge(pred_value, discharge_min, discharge_max)
                    actual_value_denorm = denormalize_discharge(actual_value, discharge_min, discharge_max)
                    
                    # Store results
                    all_predictions.append(pred_value_denorm)
                    all_actuals.append(actual_value_denorm)
                    all_dates.append(current_date)
                    
                    # Update working data with prediction for next iterations
                    working_data.loc[working_data['date'] == current_date, 'discharge_vol'] = pred_value
                    
                else:
                    print(f"⚠️  No matching sample found for {current_date.date()}")
                    continue
                    
            except Exception as e:
                print(f"⚠️  Error processing {current_date.date()}: {str(e)}")
                continue
    
    # Create results DataFrame
    if all_predictions:
        results_df = pd.DataFrame({
            'date': all_dates,
            'basin_id': basin_id,
            'predicted_flow': all_predictions,
            'actual_flow': all_actuals,
            'abs_error': np.abs(np.array(all_predictions) - np.array(all_actuals))
        })
        
        print(f"✅ Continuous prediction completed: {len(results_df)} days predicted")
        return results_df
    else:
        print("❌ No predictions were made")
        return None

def plot_error_analysis(results_df, basin_id, output_dir=None):
    """Plot comprehensive error analysis chart"""
    # Sort by date
    results_df = results_df.sort_values('date')
    
    # Use provided output directory or create new one
    if output_dir is None:
        output_dir = get_output_directory()
    
    # Create error analysis figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Error Analysis - Basin {basin_id}', fontsize=16, fontweight='bold')
    
    # Plot 1: Error time series
    ax1 = axes[0, 0]
    ax1.plot(results_df['date'], results_df['abs_error'], 'r-', linewidth=1, alpha=0.7)
    ax1.axhline(y=results_df['abs_error'].mean(), color='blue', linestyle='--', 
                label=f'Mean Error: {results_df["abs_error"].mean():.4f}')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Absolute Error', fontsize=12)
    ax1.set_title('Error Time Series', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 2: Error distribution histogram
    ax2 = axes[0, 1]
    ax2.hist(results_df['abs_error'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(x=results_df['abs_error'].mean(), color='red', linestyle='--', 
                label=f'Mean: {results_df["abs_error"].mean():.4f}')
    ax2.axvline(x=results_df['abs_error'].median(), color='green', linestyle='--', 
                label=f'Median: {results_df["abs_error"].median():.4f}')
    ax2.set_xlabel('Absolute Error', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Error Distribution Histogram', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Relative error
    results_df['relative_error'] = np.abs((results_df['actual_flow'] - results_df['predicted_flow']) / results_df['actual_flow']) * 100
    ax3 = axes[1, 0]
    ax3.plot(results_df['date'], results_df['relative_error'], 'orange', linewidth=1, alpha=0.7)
    ax3.axhline(y=results_df['relative_error'].mean(), color='red', linestyle='--', 
                label=f'Mean Relative Error: {results_df["relative_error"].mean():.2f}%')
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_ylabel('Relative Error (%)', fontsize=12)
    ax3.set_title('Relative Error Time Series', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 4: Error by flow ranges
    ax4 = axes[1, 1]
    # Divide flow into several ranges
    flow_bins = pd.qcut(results_df['actual_flow'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    error_by_flow = results_df.groupby(flow_bins, observed=False)['abs_error'].mean()
    
    bars = ax4.bar(error_by_flow.index, error_by_flow.values, 
                   color=['red', 'orange', 'yellow', 'lightgreen', 'green'], alpha=0.7)
    ax4.set_xlabel('Flow Range', fontsize=12)
    ax4.set_ylabel('Mean Absolute Error', fontsize=12)
    ax4.set_title('Error by Flow Ranges', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, error_by_flow.values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    output_file = f"{output_dir}/error_analysis_basin_{basin_id}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Error analysis chart saved to: {output_file}")
    
    # Close figure to release memory
    plt.close(fig)

def main():
    """Main function with command line arguments"""
    print("🚀 SCA2LSTM Flow Prediction System")
    print("=" * 50)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SCA2LSTM Continuous Flow Prediction')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)', default='2000-01-01')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)', default='2000-12-31')
    parser.add_argument('--basin', type=int, help='Basin ID', default=32006)
    parser.add_argument('--model', type=str, help='Model path', default=None)
    
    args = parser.parse_args()
    
    # Create unified timestamped output directory for this prediction run
    output_dir = get_output_directory()
    print(f"📁 输出目录: {output_dir}")
    
    # Set default model path if not provided
    if args.model is None:
        model_path = "model_output/fine_tune/basin_32006/20251123_163859/best_model_basin_32006.pth"
    else:
        model_path = args.model
    
    if not os.path.exists(model_path):
        print(f"❌ Model file does not exist: {model_path}")
        print("Please confirm the fine-tuned model path is correct")
        return
    
    # Load model
    try:
        model, config = load_model(model_path, args.basin)
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        return
    
    # Determine prediction mode
    if args.start and args.end:
        # Continuous prediction mode
        print(f"\n🧪 Starting continuous prediction for basin {args.basin}...")
        print(f"📅 Date range: {args.start} to {args.end}")
        
        results = predict_continuous_flow(model, config, args.basin, args.start, args.end)
        
        if results is not None and len(results) > 0:
            print(f"\n📊 Continuous Prediction Summary:")
            print(f"Prediction period: {args.start} to {args.end}")
            print(f"Total days predicted: {len(results)}")
            print(f"Average predicted flow: {results['predicted_flow'].mean():.4f}")
            print(f"Average actual flow: {results['actual_flow'].mean():.4f}")
            print(f"Mean absolute error: {results['abs_error'].mean():.4f}")
            print(f"Maximum predicted flow: {results['predicted_flow'].max():.4f}")
            print(f"Minimum predicted flow: {results['predicted_flow'].min():.4f}")
            print(f"Prediction standard deviation: {results['predicted_flow'].std():.4f}")
            
            # Save results
            output_file = f"{output_dir}/prediction_results_basin_{args.basin}_{args.start}_to_{args.end}.csv"
            results.to_csv(output_file, index=False)
            print(f"\n✅ Prediction results saved to: {output_file}")
            
            # Generate water level comparison chart
            print("\n📊 Generating water level comparison chart...")
            plot_water_level_comparison(results, args.basin, output_dir)
            
            # Generate error analysis chart
            print("\n📊 Generating error analysis chart...")
            plot_error_analysis(results, args.basin, output_dir)
            
            # Show first 10 prediction results
            print(f"\n📈 First 10 prediction results:")
            print(results.head(10).to_string())
            
        else:
            print("❌ Continuous prediction failed")
    
    else:
        # Original random sampling mode
        print(f"\n🧪 Starting random sampling prediction for basin {args.basin}...")
        results = predict_basin_flow(model, config, args.basin, num_samples=100)
        
        if results is not None and len(results) > 0:
            print(f"\n📊 Prediction Summary:")
            print(f"Sample count: {len(results)}")
            print(f"Average predicted flow: {results['predicted_flow'].mean():.4f}")
            print(f"Average actual flow: {results['actual_flow'].mean():.4f}")
            print(f"Mean absolute error: {results['abs_error'].mean():.4f}")
            print(f"Maximum predicted flow: {results['predicted_flow'].max():.4f}")
            print(f"Minimum predicted flow: {results['predicted_flow'].min():.4f}")
            print(f"Prediction standard deviation: {results['predicted_flow'].std():.4f}")
            
            # Save results
            output_file = f"{output_dir}/prediction_results_basin_{args.basin}.csv"
            results.to_csv(output_file, index=False)
            print(f"\n✅ Prediction results saved to: {output_file}")
            
            # Generate water level comparison chart
            print("\n📊 Generating water level comparison chart...")
            plot_water_level_comparison(results, args.basin, output_dir)
            
            # Generate error analysis chart
            print("\n📊 Generating error analysis chart...")
            plot_error_analysis(results, args.basin, output_dir)
            
            # Show first 10 prediction results
            print(f"\n📈 First 10 prediction results:")
            print(results.head(10).to_string())
            
        else:
            print("❌ Prediction failed")

if __name__ == "__main__":
    main()