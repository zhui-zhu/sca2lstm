import numpy as np
import pandas as pd
from typing import Union, Tuple, List
import json
import matplotlib.pyplot as plt
import torch
import warnings
import os
warnings.filterwarnings('ignore')


def get_discharge_scaler_params(basin_id: str, scaler_dir: str = "model_input_data"):
    """
    获取指定流域的discharge_vol缩放参数（Min-Max归一化的min和max值）
    
    参数:
    ---------
    basin_id : str
        流域ID
    scaler_dir : str, default="model_input_data"
        缩放参数文件所在目录
        
    返回:
    ---------
    discharge_min : float
        discharge_vol的最小值
    discharge_max : float
        discharge_vol的最大值
        
    异常:
    ---------
    FileNotFoundError: 如果缩放参数文件不存在
    KeyError: 如果文件中缺少discharge_vol的缩放参数
    """
    scaler_file = f"{scaler_dir}/{basin_id}/ts_scaler_{basin_id}.json"
    
    try:
        with open(scaler_file, 'r', encoding='utf-8') as f:
            scaler_data = json.load(f)
        
        # 获取discharge_vol的缩放参数
        if "discharge_vol" not in scaler_data:
            raise KeyError(f"流域{basin_id}的缩放参数文件中缺少'discharge_vol'字段")
        
        discharge_params = scaler_data["discharge_vol"]
        discharge_min = discharge_params["min"]
        discharge_max = discharge_params["max"]
        
        return discharge_min, discharge_max
        
    except FileNotFoundError:
        raise FileNotFoundError(f"流域{basin_id}的缩放参数文件不存在：{scaler_file}")
    except json.JSONDecodeError as e:
        raise ValueError(f"流域{basin_id}的缩放参数文件格式错误：{e}")


def denormalize_discharge(normalized_values: Union[float, np.ndarray, torch.Tensor], 
                         discharge_min: float, discharge_max: float):
    """
    反归一化discharge_vol值（将归一化值转换为真实值）
    
    参数:
    ---------
    normalized_values : float, np.ndarray, or torch.Tensor
        归一化后的discharge_vol值（范围通常在0-1之间）
    discharge_min : float
        原始数据的最小值
    discharge_max : float
        原始数据的最大值
        
    返回:
    ---------
    denormalized_values : 与输入类型相同
        反归一化后的真实discharge_vol值
        
    公式:
    ---------
    真实值 = 归一化值 × (max - min) + min
    """
    if isinstance(normalized_values, torch.Tensor):
        return normalized_values * (discharge_max - discharge_min) + discharge_min
    elif isinstance(normalized_values, np.ndarray):
        return normalized_values * (discharge_max - discharge_min) + discharge_min
    else:  # float or int
        return normalized_values * (discharge_max - discharge_min) + discharge_min


def identify_flood_events(
    discharge_series: Union[np.ndarray, pd.Series, List],
    threshold_method: str = "percentile",
    threshold_value: float = 90.0,
    min_duration: int = 3,
    min_interval: int = 5,
    smoothing_window: int = 3
) -> np.ndarray:
    """
    洪水识别函数 - 基于流量时序数据识别洪水事件
    
    参数:
    -----------
    discharge_series : array-like
        流量时序数据 (单位: m³/s 或 mm/day)
    threshold_method : str, default="percentile"
        阈值确定方法: "percentile" (百分位数), "mean_plus_std" (均值+标准差), "fixed" (固定值)
    threshold_value : float, default=95.0
        阈值参数: 百分位数(如95表示95%分位数) 或 固定阈值 或 标准差倍数
    min_duration : int, default=3
        最小洪水持续时间 (天)
    min_interval : int, default=5
        相邻洪水事件的最小间隔 (天)
    smoothing_window : int, default=3
        平滑窗口大小 (天), 用于消除噪声
    
    返回:
    -----------
    flood_mask : np.ndarray
        布尔数组, True表示洪水期, False表示非洪水期
    
    算法说明:
    -----------
    1. 数据平滑处理 (移动平均)
    2. 根据指定方法计算洪水阈值
    3. 识别超过阈值的连续时段
    4. 应用最小持续时间过滤
    5. 合并相邻的洪水事件 (间隔小于min_interval)
    """
    
    # 输入验证和转换
    if isinstance(discharge_series, (list, pd.Series)):
        discharge_series = np.array(discharge_series, dtype=float)
    elif isinstance(discharge_series, np.ndarray):
        discharge_series = discharge_series.astype(float)
    else:
        raise TypeError("discharge_series 必须是 np.ndarray, pd.Series 或 list")
    
    if len(discharge_series) == 0:
        return np.array([], dtype=bool)
    
    if np.all(np.isnan(discharge_series)):
        return np.full(len(discharge_series), False, dtype=bool)
    
    # 1. 数据平滑处理
    if smoothing_window > 1 and len(discharge_series) >= smoothing_window:
        # 使用移动平均进行平滑
        kernel = np.ones(smoothing_window) / smoothing_window
        smoothed_discharge = np.convolve(discharge_series, kernel, mode='same')
    else:
        smoothed_discharge = discharge_series.copy()
    
    # 2. 计算洪水阈值
    valid_data = smoothed_discharge[~np.isnan(smoothed_discharge)]
    
    if len(valid_data) == 0:
        return np.full(len(discharge_series), False, dtype=bool)
    
    if threshold_method == "percentile":
        # 百分位数方法 (默认95%分位数)
        flood_threshold = np.percentile(valid_data, threshold_value)
    elif threshold_method == "mean_plus_std":
        # 均值 + n倍标准差方法
        mean_flow = np.mean(valid_data)
        std_flow = np.std(valid_data)
        flood_threshold = mean_flow + threshold_value * std_flow
    elif threshold_method == "fixed":
        # 固定阈值方法
        flood_threshold = threshold_value
    else:
        raise ValueError("threshold_method 必须是 'percentile', 'mean_plus_std' 或 'fixed'")
    
    # 确保阈值不低于数据的最小值
    flood_threshold = max(flood_threshold, np.min(valid_data))
    
    # 3. 识别超过阈值的时段
    above_threshold = smoothed_discharge >= flood_threshold
    
    # 4. 寻找连续的高流量时段
    flood_mask = np.full(len(discharge_series), False, dtype=bool)
    
    # 找到所有连续的高流量段
    high_flow_periods = []
    start_idx = None
    
    for i, is_high in enumerate(above_threshold):
        if is_high and start_idx is None:
            start_idx = i
        elif not is_high and start_idx is not None:
            high_flow_periods.append((start_idx, i - 1))
            start_idx = None
    
    # 处理最后一个段
    if start_idx is not None:
        high_flow_periods.append((start_idx, len(above_threshold) - 1))
    
    # 5. 应用最小持续时间过滤
    valid_periods = []
    for start, end in high_flow_periods:
        duration = end - start + 1
        if duration >= min_duration:
            valid_periods.append((start, end))
    
    # 6. 合并相邻的洪水事件
    if len(valid_periods) > 0:
        merged_periods = [valid_periods[0]]
        
        for current_start, current_end in valid_periods[1:]:
            last_start, last_end = merged_periods[-1]
            
            # 如果当前段与上一段间隔小于min_interval，则合并
            if current_start - last_end <= min_interval:
                merged_periods[-1] = (last_start, current_end)
            else:
                merged_periods.append((current_start, current_end))
        
        # 标记最终的洪水时段
        for start, end in merged_periods:
            flood_mask[start:end + 1] = True
    
    return flood_mask


def get_flood_statistics(
    discharge_series: Union[np.ndarray, pd.Series, List],
    flood_mask: np.ndarray
) -> dict:
    """
    计算洪水事件的统计特征
    
    参数:
    -----------
    discharge_series : array-like
        原始流量时序数据
    flood_mask : np.ndarray
        洪水识别结果 (来自 identify_flood_events)
    
    返回:
    -----------
    stats : dict
        洪水统计信息，包括:
        - n_floods: 洪水事件数量
        - total_flood_days: 总洪水天数
        - avg_flood_duration: 平均洪水持续时间
        - max_flood_duration: 最长洪水持续时间
        - avg_flood_intensity: 平均洪水强度
        - max_flood_intensity: 最大洪水强度
        - flood_frequency: 洪水频率 (%)
    """
    
    if isinstance(discharge_series, (list, pd.Series)):
        discharge_series = np.array(discharge_series, dtype=float)
    
    if len(discharge_series) != len(flood_mask):
        raise ValueError("discharge_series 和 flood_mask 长度必须一致")
    
    # 基本统计
    total_days = len(flood_mask)
    flood_days = np.sum(flood_mask)
    flood_frequency = (flood_days / total_days) * 100 if total_days > 0 else 0
    
    # 洪水事件统计
    flood_events = []
    start_idx = None
    
    for i, is_flood in enumerate(flood_mask):
        if is_flood and start_idx is None:
            start_idx = i
        elif not is_flood and start_idx is not None:
            flood_events.append((start_idx, i - 1))
            start_idx = None
    
    # 处理最后一个事件
    if start_idx is not None:
        flood_events.append((start_idx, len(flood_mask) - 1))
    
    n_floods = len(flood_events)
    
    if n_floods > 0:
        durations = [end - start + 1 for start, end in flood_events]
        avg_duration = np.mean(durations)
        max_duration = np.max(durations)
        
        # 计算洪水强度 (使用洪水期间的平均流量)
        intensities = []
        for start, end in flood_events:
            flood_flow = discharge_series[start:end + 1]
            if len(flood_flow) > 0:
                intensities.append(np.mean(flood_flow))
        
        avg_intensity = np.mean(intensities) if intensities else 0
        max_intensity = np.max(intensities) if intensities else 0
    else:
        avg_duration = max_duration = avg_intensity = max_intensity = 0
    
    return {
        "n_floods": n_floods,
        "total_flood_days": flood_days,
        "flood_frequency": flood_frequency,
        "avg_flood_duration": avg_duration,
        "max_flood_duration": max_duration,
        "avg_flood_intensity": avg_intensity,
        "max_flood_intensity": max_intensity
    }


def evaluate_flood_prediction(
    true_discharge: Union[np.ndarray, pd.Series, List],
    pred_discharge: Union[np.ndarray, pd.Series, List],
    threshold_method: str = "percentile",
    threshold_value: float = 95.0
) -> dict:
    """
    评估洪水预测效果 - 对比真实流量和预测流量的洪水识别结果
    
    参数:
    -----------
    true_discharge : array-like
        真实流量数据
    pred_discharge : array-like
        预测流量数据
    threshold_method : str
        洪水识别阈值方法
    threshold_value : float
        洪水识别阈值参数
    
    返回:
    -----------
    evaluation : dict
        洪水预测评估结果，包括:
        - true_floods: 真实洪水事件数量
        - pred_floods: 预测洪水事件数量
        - true_flood_days: 真实洪水天数
        - pred_flood_days: 预测洪水天数
        - flood_detection_rate: 洪水检测率 (%)
        - false_alarm_rate: 误报率 (%)
        - flood_day_accuracy: 洪水日准确率 (%)
    """
    
    # 转换为numpy数组
    if isinstance(true_discharge, (list, pd.Series)):
        true_discharge = np.array(true_discharge, dtype=float)
    if isinstance(pred_discharge, (list, pd.Series)):
        pred_discharge = np.array(pred_discharge, dtype=float)
    
    if len(true_discharge) != len(pred_discharge):
        raise ValueError("真实流量和预测流量长度必须一致")
    
    # 识别洪水事件
    true_flood_mask = identify_flood_events(
        true_discharge, 
        threshold_method=threshold_method,
        threshold_value=threshold_value
    )
    
    pred_flood_mask = identify_flood_events(
        pred_discharge,
        threshold_method=threshold_method, 
        threshold_value=threshold_value
    )
    
    # 计算评估指标
    total_days = len(true_discharge)
    
    # 洪水日统计
    true_flood_days = np.sum(true_flood_mask)
    pred_flood_days = np.sum(pred_flood_mask)
    
    # 混淆矩阵计算
    true_positives = np.sum(true_flood_mask & pred_flood_mask)  # 正确识别的洪水日
    false_positives = np.sum(~true_flood_mask & pred_flood_mask)  # 误报
    false_negatives = np.sum(true_flood_mask & ~pred_flood_mask)  # 漏报
    true_negatives = np.sum(~true_flood_mask & ~pred_flood_mask)  # 正确识别的非洪水日
    
    # 计算评估指标
    flood_detection_rate = (true_positives / true_flood_days * 100) if true_flood_days > 0 else 0
    false_alarm_rate = (false_positives / (false_positives + true_negatives) * 100) if (false_positives + true_negatives) > 0 else 0
    flood_day_accuracy = ((true_positives + true_negatives) / total_days * 100) if total_days > 0 else 0
    
    # 获取洪水事件统计
    true_stats = get_flood_statistics(true_discharge, true_flood_mask)
    pred_stats = get_flood_statistics(pred_discharge, pred_flood_mask)
    
    return {
        "true_floods": true_stats["n_floods"],
        "pred_floods": pred_stats["n_floods"],
        "true_flood_days": true_flood_days,
        "pred_flood_days": pred_flood_days,
        "flood_detection_rate": flood_detection_rate,
        "false_alarm_rate": false_alarm_rate,
        "flood_day_accuracy": flood_day_accuracy,
        "true_flood_statistics": true_stats,
        "pred_flood_statistics": pred_stats,
        "confusion_matrix": {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_negatives": true_negatives
        }
    }


def test_flood_identification():
    """
    测试洪水识别函数
    """
    print("=== 洪水识别函数测试 ===")
    
    # 生成模拟流量数据 (包含明显的洪水事件)
    np.random.seed(42)
    n_days = 365
    base_flow = 10.0  # 基础流量
    
    # 生成季节性流量模式
    seasonal_pattern = 5.0 * np.sin(2 * np.pi * np.arange(n_days) / 365)
    
    # 添加几个明显的洪水事件
    flood_events = [
        (50, 60),   # 第50-60天
        (120, 130), # 第120-130天
        (200, 210), # 第200-210天
        (280, 290), # 第280-290天
    ]
    
    discharge_data = base_flow + seasonal_pattern + np.random.normal(0, 1, n_days)
    
    # 在洪水事件期间增加流量
    for start, end in flood_events:
        flood_magnitude = np.random.uniform(15, 25)  # 洪水强度
        discharge_data[start:end+1] += flood_magnitude
    
    # 确保没有负值
    discharge_data = np.maximum(discharge_data, 0.1)
    
    print(f"生成 {n_days} 天的模拟流量数据")
    print(f"流量范围: {np.min(discharge_data):.2f} - {np.max(discharge_data):.2f} m³/s")
    
    # 测试不同的洪水识别方法
    methods = [
        ("percentile", 90.0),
        ("percentile", 95.0),
        ("mean_plus_std", 2.0),
    ]
    
    for method, value in methods:
        print(f"\n--- 方法: {method} (阈值: {value}) ---")
        
        # 识别洪水
        flood_mask = identify_flood_events(
            discharge_data,
            threshold_method=method,
            threshold_value=value,
            min_duration=3,
            min_interval=5
        )
        
        # 计算统计
        stats = get_flood_statistics(discharge_data, flood_mask)
        
        print(f"洪水事件数量: {stats['n_floods']}")
        print(f"总洪水天数: {stats['total_flood_days']}")
        print(f"洪水频率: {stats['flood_frequency']:.1f}%")
        print(f"平均持续时间: {stats['avg_flood_duration']:.1f} 天")
        print(f"平均洪水强度: {stats['avg_flood_intensity']:.2f} m³/s")
    
    print("\n=== 测试完成 ===")


def visualize_lstm2_weights(model, sample_input, save_dir=None):
    """
    LSTM2 Weight Visualization - Display feature weights over time
    
    Parameters:
    -----------
    model : SCA2LSTM model
        Trained SCA2LSTM model
    sample_input : dict
        Dictionary containing model inputs
    save_dir : str, optional
        Directory to save images, if None only display
    
    Returns:
    -----------
    figs : list of matplotlib.figure.Figure
        List of generated figure objects
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    model.eval()
    with torch.no_grad():
        # 准备输入数据
        device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cpu')
        seq_features = sample_input["seq_features"].unsqueeze(0).to(device)
        lstm1_input = sample_input["lstm1_input"].unsqueeze(0).to(device)
        missing_bool = sample_input["missing_bool"].unsqueeze(0).to(device)
        basin_ids = sample_input["basin_id"].unsqueeze(0).to(device)
        
        # 使用新的双向反馈架构获取LSTM1输出
        # 初始化隐状态
        batch_size = lstm1_input.size(0)
        h1 = torch.zeros(batch_size, model.config.LSTM_HIDDEN_DIM).to(device)
        c1 = torch.zeros(batch_size, model.config.LSTM_HIDDEN_DIM).to(device)
        h2 = torch.zeros(batch_size, model.config.LSTM_HIDDEN_DIM).to(device)
        c2 = torch.zeros(batch_size, model.config.LSTM_HIDDEN_DIM).to(device)
        
        # 通过时间步传递获取最终的隐状态
        seq_len = lstm1_input.size(1)
        for t in range(seq_len):
            lstm1_input_t = lstm1_input[:, t, :]
            # LSTM1前向传播（使用零初始化的h2，因为我们只关心权重生成）
            lstm1_input_with_feedback = torch.cat([lstm1_input_t, h1, h2], dim=-1)
            h1, c1 = model.lstm1_cell(lstm1_input_with_feedback, (h1, c1))
        
        # 使用最终的隐状态生成特征权重
        lstm1_out = h1.unsqueeze(1)  # 添加序列维度
        feature_weights = model.weight_head(lstm1_out)
        
        # Apply missing mask and normalization
        feature_weights = feature_weights * missing_bool
        feature_weights = feature_weights / (feature_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Convert to numpy
        weights_np = feature_weights.squeeze(0).cpu().numpy()  # (seq_len, n_features)
        
        # Get feature names
        feature_names = model.config.LSTM2_FEATURES
        
        # Set large font sizes
        plt.rcParams.update({'font.size': 14})
        figs = []
        
        # Figure 1: Heatmap of weight changes
        fig1, ax1 = plt.subplots(figsize=(16, 10))
        sns.heatmap(weights_np.T, 
                   xticklabels=range(1, weights_np.shape[0] + 1),
                   yticklabels=feature_names,
                   cmap='YlOrRd', 
                   cbar_kws={'label': 'Feature Weights', 'shrink': 0.8},
                   ax=ax1)
        ax1.set_title('LSTM2 Feature Weights Heatmap Over Time', fontsize=18, fontweight='bold', pad=20)
        ax1.set_xlabel('Time Step (Days)', fontsize=14)
        ax1.set_ylabel('Features', fontsize=14)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        figs.append(fig1)
        
        if save_dir:
            fig1.savefig(f"{save_dir}/lstm2_weights_heatmap.png", dpi=300, bbox_inches='tight')
        
        # Figure 2: Average weights bar chart
        fig2, ax2 = plt.subplots(figsize=(14, 10))
        avg_weights = np.mean(weights_np, axis=0)
        bars = ax2.bar(range(len(feature_names)), avg_weights, color='steelblue', alpha=0.7, linewidth=1.5)
        ax2.set_title('LSTM2 Average Feature Weights Distribution', fontsize=18, fontweight='bold', pad=20)
        ax2.set_xlabel('Features', fontsize=14)
        ax2.set_ylabel('Average Weights', fontsize=14)
        ax2.set_xticks(range(len(feature_names)))
        ax2.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=12)
        ax2.tick_params(axis='y', which='major', labelsize=12)
        ax2.grid(axis='y', alpha=0.3)
        
        # Display values on bars
        for bar, weight in zip(bars, avg_weights):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{weight:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        figs.append(fig2)
        if save_dir:
            fig2.savefig(f"{save_dir}/lstm2_average_weights.png", dpi=300, bbox_inches='tight')
        
        # Figure 3: Weight time series
        fig3, ax3 = plt.subplots(figsize=(16, 10))
        time_steps = range(weights_np.shape[0])
        colors = plt.cm.Set3(np.linspace(0, 1, len(feature_names)))
        for i, (feature_name, color) in enumerate(zip(feature_names, colors)):
            ax3.plot(time_steps, weights_np[:, i], label=feature_name, alpha=0.8, linewidth=3, color=color)
        
        ax3.set_title('LSTM2 Feature Weights Time Series', fontsize=18, fontweight='bold', pad=20)
        ax3.set_xlabel('Time Step (Days)', fontsize=14)
        ax3.set_ylabel('Feature Weights', fontsize=14)
        ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12, framealpha=0.8)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='both', which='major', labelsize=12)
        figs.append(fig3)
        
        if save_dir:
            fig3.savefig(f"{save_dir}/lstm2_weights_timeseries.png", dpi=300, bbox_inches='tight')
        
        # Figure 4: Weight statistics
        fig4, ax4 = plt.subplots(figsize=(12, 10))
        ax4.axis('off')
        stats_text = f"""Weight Statistics Summary:

Mean Weight: {np.mean(avg_weights):.4f}
Weight Std Dev: {np.std(avg_weights):.4f}
Max Weight: {np.max(avg_weights):.4f} ({feature_names[np.argmax(avg_weights)]})
Min Weight: {np.min(avg_weights):.4f} ({feature_names[np.argmin(avg_weights)]})

Weight Distribution:
> 0.1: {np.sum(avg_weights > 0.1)} features
0.05-0.1: {np.sum((avg_weights > 0.05) & (avg_weights <= 0.1))} features
< 0.05: {np.sum(avg_weights <= 0.05)} features

Temporal Stability:
Mean Weight Variance: {np.mean(np.var(weights_np, axis=0)):.4f}
Weight Range: {np.max(weights_np) - np.min(weights_np):.4f}"""
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                verticalalignment='top', fontsize=14, fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1.0', facecolor='lightblue', alpha=0.8, edgecolor='navy', linewidth=2))
        
        figs.append(fig4)
        if save_dir:
            fig4.savefig(f"{save_dir}/lstm2_weight_statistics.png", dpi=300, bbox_inches='tight')
        
        return figs


def visualize_feature_weights(model, sample_input, feature_names=None, save_dir=None):
    """
    Feature Weight Visualization - Display attention weights for each feature
    
    Parameters:
    -----------
    model : SCA2LSTM model
        Trained SCA2LSTM model
    sample_input : dict
        Dictionary containing model inputs
    feature_names : list, optional
        List of feature names, if None use model config
    save_dir : str, optional
        Directory to save images, if None only display
    
    Returns:
    -----------
    figs : list of matplotlib.figure.Figure
        List of generated figure objects
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    model.eval()
    with torch.no_grad():
        # 准备输入数据
        device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cpu')
        seq_features = sample_input["seq_features"].unsqueeze(0).to(device)
        lstm1_input = sample_input["lstm1_input"].unsqueeze(0).to(device)
        missing_bool = sample_input["missing_bool"].unsqueeze(0).to(device)
        basin_ids = sample_input["basin_id"].unsqueeze(0).to(device)
        
        # 获取特征权重 - 使用模型内部逻辑提取权重而不是预测值
        batch_size = seq_features.shape[0]
        seq_len = seq_features.shape[1]
        
        # 初始化隐状态
        h1 = torch.zeros(batch_size, model.lstm_hidden_dim, device=device)
        c1 = torch.zeros(batch_size, model.lstm_hidden_dim, device=device)
        h2 = torch.zeros(batch_size, model.lstm_hidden_dim, device=device)
        c2 = torch.zeros(batch_size, model.lstm_hidden_dim, device=device)
        
        # 流域嵌入
        basin_id_to_idx = {bid: idx for idx, bid in enumerate(set(model.basin_ids))}
        basin_indices = torch.tensor([basin_id_to_idx[bid.item()] for bid in basin_ids], dtype=torch.long).to(device)
        basin_embed = model.basin_embedding(basin_indices)
        
        # 收集每个时间步的特征权重
        feature_weights_list = []
        
        for t in range(seq_len):
            # LSTM1：生成特征权重
            lstm1_input_t = lstm1_input[:, t, :]
            lstm1_input_with_feedback = torch.cat([lstm1_input_t, h1, h2], dim=-1)
            h1, c1 = model.lstm1_cell(lstm1_input_with_feedback, (h1, c1))
            
            # 生成特征权重
            feature_weights = model.weight_head(h1)
            feature_weights = feature_weights * missing_bool[:, t, :]
            # 防止除零：如果所有特征都缺失，使用均匀分布
            weight_sums = feature_weights.sum(dim=-1, keepdim=True)
            uniform_weights = torch.ones_like(feature_weights) / feature_weights.shape[-1]
            feature_weights = torch.where(
                weight_sums < 1e-8,
                uniform_weights * missing_bool[:, t, :],
                feature_weights / (weight_sums + 1e-8)
            )
            feature_weights_list.append(feature_weights)
            
            # LSTM2：更新h2状态（用于下一个时间步的反馈）
            seq_features_t = seq_features[:, t, :]
            weighted_features = seq_features_t * feature_weights
            basin_embed_t = basin_embed
            lstm2_input_with_feedback = torch.cat([weighted_features, basin_embed_t, h1], dim=-1)
            h2, c2 = model.lstm2_cell(lstm2_input_with_feedback, (h2, c2))
        
        # 堆叠所有时间步的权重
        feature_weights = torch.stack(feature_weights_list, dim=1)  # (batch, seq_len, n_features)
        
        # 检查是否有NaN或inf
        if torch.isnan(feature_weights).any() or torch.isinf(feature_weights).any():
            print("⚠️  警告：特征权重包含NaN或inf，使用备用权重")
            # 使用均匀权重作为备用
            feature_weights = torch.ones_like(feature_weights) / feature_weights.shape[-1]
        
        # Convert to numpy
        weights_np = feature_weights.squeeze(0).cpu().numpy()  # (seq_len, n_features)
        
        # 检查numpy数组是否有效
        if np.isnan(weights_np).any() or np.isinf(weights_np).any():
            print("⚠️  警告：转换后的权重包含NaN或inf，使用均匀权重")
            weights_np = np.ones_like(weights_np) / weights_np.shape[-1]
        
        # Get feature names
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(weights_np.shape[1])]
        
        # Set large font sizes
        plt.rcParams.update({'font.size': 14})
        figs = []
        
        # Figure 1: Feature weights heatmap
        fig1, ax1 = plt.subplots(figsize=(16, 10))
        sns.heatmap(weights_np.T, 
                   xticklabels=range(1, weights_np.shape[0] + 1),
                   yticklabels=feature_names,
                   cmap='RdYlBu_r', 
                   cbar_kws={'label': 'Feature Weight', 'shrink': 0.8},
                   ax=ax1,
                   annot=False,
                   fmt='.3f',
                   linewidths=0.5)
        
        ax1.set_title('Feature Attention Weights Heatmap', fontsize=20, fontweight='bold', pad=20)
        ax1.set_xlabel('Time Step (Days)', fontsize=16)
        ax1.set_ylabel('Features', fontsize=16)
        ax1.tick_params(axis='both', which='major', labelsize=14)
        figs.append(fig1)
        
        if save_dir:
            fig1.savefig(f"{save_dir}/feature_weights_heatmap.png", dpi=300, bbox_inches='tight')
        
        # Figure 2: Average feature weights
        fig2, ax2 = plt.subplots(figsize=(14, 10))
        avg_weights = np.mean(weights_np, axis=0)
        
        # Create color map based on weight values
        colors = plt.cm.RdYlBu_r(avg_weights / np.max(avg_weights))
        bars = ax2.bar(range(len(feature_names)), avg_weights, color=colors, alpha=0.8, linewidth=1.5)
        
        ax2.set_title('Average Feature Attention Weights', fontsize=20, fontweight='bold', pad=20)
        ax2.set_xlabel('Features', fontsize=16)
        ax2.set_ylabel('Average Weight', fontsize=16)
        ax2.set_xticks(range(len(feature_names)))
        ax2.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=14)
        ax2.tick_params(axis='y', which='major', labelsize=14)
        ax2.grid(axis='y', alpha=0.3)
        
        # Display values on bars
        for bar, weight in zip(bars, avg_weights):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{weight:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        figs.append(fig2)
        if save_dir:
            fig2.savefig(f"{save_dir}/feature_weights_average.png", dpi=300, bbox_inches='tight')
        
        # Figure 3: Feature weight time series
        fig3, ax3 = plt.subplots(figsize=(16, 10))
        time_steps = range(weights_np.shape[0])
        
        # Use distinct colors for different features
        colors = plt.cm.Set3(np.linspace(0, 1, len(feature_names)))
        for i, (feature_name, color) in enumerate(zip(feature_names, colors)):
            ax3.plot(time_steps, weights_np[:, i], label=feature_name, 
                    alpha=0.8, linewidth=3, color=color, marker='', markersize=0, antialiased=True)
        
        ax3.set_title('Feature Attention Weights Over Time', fontsize=20, fontweight='bold', pad=20)
        ax3.set_xlabel('Time Step (Days)', fontsize=16)
        ax3.set_ylabel('Feature Weight', fontsize=16)
        ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12, framealpha=0.8)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='both', which='major', labelsize=14)
        
        figs.append(fig3)
        if save_dir:
            fig3.savefig(f"{save_dir}/feature_weights_timeseries.png", dpi=300, bbox_inches='tight')
        
        # Figure 4: Feature weight distribution
        fig4, ax4 = plt.subplots(figsize=(12, 10))
        
        # Create violin plot for weight distribution
        data_for_violin = [weights_np[:, i] for i in range(len(feature_names))]
        parts = ax4.violinplot(data_for_violin, positions=range(len(feature_names)), 
                              showmeans=True, showmedians=True)
        
        # Customize violin plot colors
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(plt.cm.RdYlBu_r(i / len(feature_names)))
            pc.set_alpha(0.7)
        
        ax4.set_title('Feature Weight Distribution', fontsize=20, fontweight='bold', pad=20)
        ax4.set_xlabel('Features', fontsize=16)
        ax4.set_ylabel('Weight Value', fontsize=16)
        ax4.set_xticks(range(len(feature_names)))
        ax4.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=14)
        ax4.tick_params(axis='y', which='major', labelsize=14)
        ax4.grid(axis='y', alpha=0.3)
        
        figs.append(fig4)
        if save_dir:
            fig4.savefig(f"{save_dir}/feature_weights_distribution.png", dpi=300, bbox_inches='tight')
        
        return figs


def plot_water_level_comparison(true_discharge, pred_discharge, dates=None, 
                               flood_mask=None, save_dir=None, title="Water Level Comparison"):
    """
    Water Level Comparison - Visualize the difference between observed and predicted values
    
    Parameters:
    -----------
    true_discharge : array-like
        Observed discharge data
    pred_discharge : array-like
        Predicted discharge data
    dates : array-like, optional
        Date sequence for x-axis
    flood_mask : array-like, optional
        Flood event mask for highlighting flood periods
    save_dir : str, optional
        Directory to save images
    title : str
        Figure title
    
    Returns:
    -----------
    figs : list of matplotlib.figure.Figure
        List of generated figure objects
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime, timedelta
    
    # Data preparation
    true_discharge = np.array(true_discharge, dtype=float)
    pred_discharge = np.array(pred_discharge, dtype=float)
    
    if len(true_discharge) != len(pred_discharge):
        raise ValueError("Observed and predicted values must have the same length")
    
    # Generate default dates (if not provided)
    if dates is None:
        start_date = datetime(2020, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(len(true_discharge))]
    else:
        dates = pd.to_datetime(dates)
    
    # Set large font sizes
    plt.rcParams.update({'font.size': 14})
    figs = []
    
    # Figure 1: Main comparison plot
    fig1, ax1 = plt.subplots(figsize=(16, 10))
    # Smooth curve plotting with enhanced visual appeal
    ax1.plot(dates, true_discharge, color='#1f77b4', linewidth=3.5, label='Observed', 
             alpha=0.9, linestyle='-', marker='', markersize=0, antialiased=True)
    ax1.plot(dates, pred_discharge, color='#d62728', linewidth=3.5, label='Predicted', 
             alpha=0.9, linestyle='-', marker='', markersize=0, antialiased=True)
    
    # Highlight flood periods
    if flood_mask is not None:
        flood_dates = np.array(dates)[flood_mask]
        flood_true = true_discharge[flood_mask]
        if len(flood_dates) > 0:
            ax1.scatter(flood_dates, flood_true, c='orange', s=100, 
                       label='Flood Period Observations', alpha=0.8, zorder=5, edgecolors='black', linewidth=1)
    
    ax1.set_title(title, fontsize=20, fontweight='bold', pad=20)
    ax1.set_ylabel('Discharge (m³/s)', fontsize=16)
    ax1.legend(loc='upper right', fontsize=14, framealpha=0.8)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    
    # Set x-axis date format
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    figs.append(fig1)
    if save_dir:
        fig1.savefig(f"{save_dir}/water_level_comparison_main.png", dpi=300, bbox_inches='tight')
    
    # Figure 2: Residual analysis
    fig2, ax2 = plt.subplots(figsize=(16, 10))
    residuals = true_discharge - pred_discharge
    # Smooth residual curve with enhanced styling
    ax2.plot(dates, residuals, color='#2ca02c', linewidth=3, alpha=0.85, 
             linestyle='-', antialiased=True)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.9, linewidth=2.5)
    ax2.axhline(y=np.mean(residuals), color='#d62728', linestyle='--', linewidth=3,
                label=f'Mean Residual: {np.mean(residuals):.2f}')
    ax2.fill_between(dates, residuals, 0, alpha=0.3, color='green')
    
    ax2.set_title('Residual Analysis (Observed - Predicted)', fontsize=20, fontweight='bold', pad=20)
    ax2.set_ylabel('Residuals (m³/s)', fontsize=16)
    ax2.legend(fontsize=14, framealpha=0.8)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    
    # Set x-axis date format
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    figs.append(fig2)
    if save_dir:
        fig2.savefig(f"{save_dir}/water_level_residuals.png", dpi=300, bbox_inches='tight')
    
    # Figure 3: Scatter plot and correlation analysis
    fig3, ax3 = plt.subplots(figsize=(14, 12))
    ax3.scatter(true_discharge, pred_discharge, alpha=0.7, s=60, color='purple', edgecolors='black', linewidth=0.5)
    
    # Add 1:1 line
    min_val = min(np.min(true_discharge), np.min(pred_discharge))
    max_val = max(np.max(true_discharge), np.max(pred_discharge))
    ax3.plot([min_val, max_val], [min_val, max_val], color='black', linewidth=3.5, 
             linestyle='--', label='1:1 Line', alpha=0.9, antialiased=True)
    
    # Calculate statistics
    from scipy import stats
    correlation, p_value = stats.pearsonr(true_discharge, pred_discharge)
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    nse = 1 - np.sum(residuals**2) / np.sum((true_discharge - np.mean(true_discharge))**2)
    
    # Add statistical information text
    stats_text = f'Correlation Coefficient: {correlation:.3f}\nRMSE: {rmse:.2f} m³/s\nMAE: {mae:.2f} m³/s\nNSE: {nse:.3f}'
    ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.8', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=2),
             fontsize=16, fontweight='bold')
    
    ax3.set_xlabel('Observed Discharge (m³/s)', fontsize=16)
    ax3.set_ylabel('Predicted Discharge (m³/s)', fontsize=16)
    ax3.set_title('Observed vs Predicted Scatter Plot', fontsize=20, fontweight='bold', pad=20)
    ax3.legend(fontsize=14, framealpha=0.8)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    
    figs.append(fig3)
    if save_dir:
        fig3.savefig(f"{save_dir}/water_level_scatter.png", dpi=300, bbox_inches='tight')
    
    print(f"Water level comparison plots saved to: {save_dir}")
    return figs


def create_evaluation_report(true_discharge, pred_discharge, dates=None, basin_id=None):
    """
    生成完整的模型评估报告
    
    参数:
    -----------
    true_discharge : array-like
        观测流量数据
    pred_discharge : array-like
        预测流量数据
    dates : array-like, optional
        日期序列
    basin_id : str/int, optional
        流域ID
    
    返回:
    -----------
    report : dict
        包含各种评估指标和可视化结果的字典
    """
    from scipy import stats
    
    # 基础统计
    true_discharge = np.array(true_discharge, dtype=float)
    pred_discharge = np.array(pred_discharge, dtype=float)
    residuals = true_discharge - pred_discharge
    
    # 计算评估指标
    # 检查预测值是否有变化（标准差是否为0）
    pred_std = np.std(pred_discharge)
    if pred_std == 0:
        # 如果预测值恒定，相关系数为0，p值为1
        correlation = 0.0
        p_value = 1.0
    else:
        correlation, p_value = stats.pearsonr(true_discharge, pred_discharge)
    
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    bias = np.mean(residuals)
    
    # Nash-Sutcliffe效率系数
    nse = 1 - np.sum(residuals**2) / np.sum((true_discharge - np.mean(true_discharge))**2)
    
    # 相对误差
    relative_rmse = rmse / np.mean(true_discharge) * 100
    relative_mae = mae / np.mean(true_discharge) * 100
    
    # 洪水期评估（如果有洪水识别结果）
    flood_evaluation = None
    try:
        flood_mask = identify_flood_events(true_discharge)
        flood_evaluation = evaluate_flood_prediction(true_discharge, pred_discharge)
        
        # 转换洪水评估中的numpy类型为Python原生类型
        if flood_evaluation:
            def convert_flood_types(obj):
                """递归转换洪水评估中的numpy类型"""
                if isinstance(obj, dict):
                    return {k: convert_flood_types(v) for k, v in obj.items()}
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj
            
            flood_evaluation = convert_flood_types(flood_evaluation)
            
    except:
        pass
    
    report = {
        'basin_id': basin_id,
        'sample_size': len(true_discharge),
        'correlation': {'value': float(correlation), 'p_value': float(p_value)},
        'rmse': float(rmse),
        'mae': float(mae),
        'bias': float(bias),
        'nse': float(nse),
        'relative_rmse': float(relative_rmse),
        'relative_mae': float(relative_mae),
        'true_stats': {
            'mean': float(np.mean(true_discharge)),
            'std': float(np.std(true_discharge)),
            'min': float(np.min(true_discharge)),
            'max': float(np.max(true_discharge))
        },
        'pred_stats': {
            'mean': float(np.mean(pred_discharge)),
            'std': float(np.std(pred_discharge)),
            'min': float(np.min(pred_discharge)),
            'max': float(np.max(pred_discharge))
        },
        'flood_evaluation': flood_evaluation
    }
    
    return report


def test_visualization_functions():
    """
    测试可视化函数
    """
    print("🧪 开始测试可视化函数...")
    
    # 生成模拟数据
    np.random.seed(42)
    n_days = 365
    
    # 模拟流量数据（含季节性变化和洪水事件）
    t = np.linspace(0, 4*np.pi, n_days)
    base_flow = 50 + 30 * np.sin(t)  # 季节性基流
    noise = np.random.normal(0, 5, n_days)
    
    # 添加洪水事件
    flood_events = [
        (50, 70, 150),   # 开始, 结束, 峰值
        (150, 180, 200),
        (250, 280, 180)
    ]
    
    true_discharge = base_flow + noise
    for start, end, peak in flood_events:
        flood_shape = np.exp(-((np.arange(end-start) - (end-start)//2)**2) / (2*5**2))
        true_discharge[start:end] += peak * flood_shape
    
    # 生成预测数据（添加一些误差）
    pred_discharge = true_discharge + np.random.normal(0, 8, n_days)
    
    # 生成日期序列
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    
    # 测试水位对比图
    print("📊 测试水位对比图...")
    try:
        figs = plot_water_level_comparison(
            true_discharge=true_discharge,
            pred_discharge=pred_discharge,
            dates=dates,
            save_dir='.'
        )
        for fig in figs:
            plt.close(fig)
        print("✅ 水位对比图测试成功")
    except Exception as e:
        print(f"❌ 水位对比图测试失败: {str(e)}")
    
    # 测试评估报告
    print("📋 测试评估报告...")
    try:
        report = create_evaluation_report(
            true_discharge=true_discharge,
            pred_discharge=pred_discharge,
            dates=dates,
            basin_id="TEST_BASIN"
        )
        
        # 保存报告（处理numpy数据类型）
        def convert_numpy_types(obj):
            """转换numpy数据类型为Python原生类型"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        with open('test_evaluation_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=convert_numpy_types)
        
        print("✅ 评估报告测试成功")
        print(f"   - 样本数量: {report['sample_size']}")
        print(f"   - 相关系数: {report['correlation']['value']:.3f}")
        print(f"   - RMSE: {report['rmse']:.2f}")
        print(f"   - NSE: {report['nse']:.3f}")
        
    except Exception as e:
        print(f"❌ 评估报告测试失败: {str(e)}")
    
    print("🎉 可视化函数测试完成！")


def plot_training_curves(train_losses, val_losses, save_dir=None, show_plot=False):
    """
    绘制训练和验证损失曲线
    
    参数:
    -----------
    train_losses : list
        训练损失历史
    val_losses : list
        验证损失历史
    save_dir : str, optional
        保存图像的目录
    show_plot : bool, default=False
        是否显示图像
    """
    try:
        plt.figure(figsize=(12, 5))
        
        # 绘制损失曲线
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        # 标记最佳验证损失
        if val_losses:
            best_epoch = np.argmin(val_losses)
            best_loss = val_losses[best_epoch]
            plt.scatter(best_epoch + 1, best_loss, c='red', s=100, marker='*', 
                       label=f'Best Validation Loss: {best_loss:.4f}')
        
        plt.title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'training_curves.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 损失曲线已保存至: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
            
    except Exception as e:
        print(f"❌ 绘制损失曲线失败: {str(e)}")


def plot_prediction_comparison(pred_values, target_values, basin_ids=None, epoch=None, save_dir=None, sample_size=50):
    """
    绘制预测值与真实值的对比图
    
    参数:
    -----------
    pred_values : array-like
        预测值列表
    target_values : array-like
        真实值列表
    basin_ids : list, optional
        流域ID列表
    epoch : int, optional
        当前轮次
    save_dir : str, optional
        保存图像的目录
    sample_size : int, default=50
        显示的样本数量
    """
    try:
        if len(pred_values) == 0 or len(target_values) == 0:
            print("⚠️  没有预测数据可供可视化")
            return
        
        # 转换为numpy数组并取前sample_size个样本
        pred_array = np.array(pred_values)[:sample_size]
        target_array = np.array(target_values)[:sample_size]
        
        plt.figure(figsize=(12, 5))
        
        # 绘制对比图
        sample_indices = range(len(pred_array))
        plt.plot(sample_indices, target_array, 'b-', label='True Values', linewidth=2, marker='o', markersize=4)
        plt.plot(sample_indices, pred_array, 'r--', label='Predictions', linewidth=2, marker='s', markersize=4)
        
        # 添加标题信息
        title = 'Prediction vs True Values Comparison'
        if epoch is not None:
            title += f' (Epoch {epoch+1})'
        if basin_ids and len(set(basin_ids)) <= 3:  # 只显示少量流域ID
            unique_basins = list(set(basin_ids))
            title += f'\nBasins: {", ".join(map(str, unique_basins))}'
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Sample Index', fontsize=12)
        plt.ylabel('Discharge Value', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 计算并显示误差指标
        if len(pred_array) == len(target_array):
            mse = np.mean((pred_array - target_array) ** 2)
            mae = np.mean(np.abs(pred_array - target_array))
            plt.text(0.02, 0.98, f'MSE: {mse:.4f}\nMAE: {mae:.4f}', 
                    transform=plt.gca().transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            epoch_str = f'_epoch_{epoch+1}' if epoch is not None else ''
            save_path = os.path.join(save_dir, f'prediction_comparison{epoch_str}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 预测对比图已保存至: {save_path}")
        
        plt.close()
        
    except Exception as e:
        print(f"❌ 绘制预测对比图失败: {str(e)}")


def plot_loss_distribution(losses, epoch=None, save_dir=None, bins=30):
    """
    绘制损失分布直方图
    
    参数:
    -----------
    losses : array-like
        损失值列表
    epoch : int, optional
        当前轮次
    save_dir : str, optional
        保存图像的目录
    bins : int, default=30
        直方图的分箱数量
    """
    try:
        if len(losses) == 0:
            print("⚠️  没有损失数据可供可视化")
            return
        
        plt.figure(figsize=(10, 6))
        
        # 绘制直方图
        plt.hist(losses, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        
        # 添加统计信息
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        
        plt.axvline(mean_loss, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_loss:.4f}')
        plt.axvline(mean_loss + std_loss, color='orange', linestyle='--', linewidth=2, label=f'Mean+Std: {mean_loss + std_loss:.4f}')
        plt.axvline(mean_loss - std_loss, color='orange', linestyle='--', linewidth=2, label=f'Mean-Std: {mean_loss - std_loss:.4f}')
        
        title = 'Loss Distribution'
        if epoch is not None:
            title += f' (Epoch {epoch+1})'
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Loss Value', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息文本
        plt.text(0.02, 0.98, f'Samples: {len(losses)}\nMean: {mean_loss:.4f}\nStd: {std_loss:.4f}', 
                transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            epoch_str = f'_epoch_{epoch+1}' if epoch is not None else ''
            save_path = os.path.join(save_dir, f'loss_distribution{epoch_str}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 损失分布图已保存至: {save_path}")
        
        plt.close()
        
    except Exception as e:
        print(f"❌ 绘制损失分布图失败: {str(e)}")


def plot_feature_weights_heatmap(feature_weights_history, feature_names=None, save_dir=None, show_plot=False):
    """
    绘制特征权重热力图，展示不同特征权重随训练轮数的变化
    
    参数:
    -----------
    feature_weights_history : dict
        特征权重历史数据，格式：{epoch: {basin_id: weights_array}}
        其中weights_array形状为 (seq_len, n_features) 或 (n_features,)
    feature_names : list, optional
        特征名称列表，用于Y轴标签
    save_dir : str, optional
        保存图像的目录
    show_plot : bool, default=False
        是否显示图像
    """
    try:
        if not feature_weights_history:
            print("⚠️  没有特征权重数据可供可视化")
            return
        
        # 获取所有epoch和流域ID
        all_epochs = sorted(feature_weights_history.keys())
        all_basins = set()
        for epoch_data in feature_weights_history.values():
            all_basins.update(epoch_data.keys())
        all_basins = sorted(list(all_basins))
        
        if not all_basins:
            print("⚠️  没有找到流域数据")
            return
        
        # 为每个流域创建单独的热力图
        for basin_id in all_basins:
            # 收集该流域的权重数据
            basin_weights = []
            valid_epochs = []
            
            for epoch in all_epochs:
                if basin_id in feature_weights_history[epoch]:
                    weights = feature_weights_history[epoch][basin_id]
                    if weights is not None and len(weights) > 0:
                        # 处理不同形状的权重数据
                        if weights.ndim == 2:  # (seq_len, n_features)
                            # 对时间维度取平均，得到 (n_features,)
                            avg_weights = np.mean(weights, axis=0)
                            basin_weights.append(avg_weights)
                        elif weights.ndim == 1:  # (n_features,)
                            basin_weights.append(weights)
                        else:
                            print(f"⚠️  权重数据维度不支持: {weights.ndim}")
                            continue
                        valid_epochs.append(epoch)
            
            if not basin_weights:
                print(f"⚠️  流域 {basin_id} 没有有效的权重数据")
                continue
            
            # 转换为numpy数组
            weights_matrix = np.array(basin_weights)  # shape: (n_epochs, n_features)
            
            # 创建大字体、大图像的热力图
            plt.figure(figsize=(16, 10))
            
            # 根据epoch数量调整X轴密度
            n_epochs = len(valid_epochs)
            if n_epochs <= 20:
                x_tick_interval = 1  # 每轮都显示
            elif n_epochs <= 50:
                x_tick_interval = 5  # 每5轮显示一次
            else:
                x_tick_interval = 10  # 每10轮显示一次
            
            # 创建热力图
            im = plt.imshow(weights_matrix.T, 
                           aspect='auto', 
                           cmap='YlOrRd',  # 黄-橙-红渐变
                           interpolation='nearest')
            
            # 设置坐标轴
            plt.title(f'Feature Weights Heatmap - Basin {basin_id}', 
                     fontsize=20, fontweight='bold', pad=20)
            plt.xlabel('Training Epoch', fontsize=16, fontweight='bold')
            plt.ylabel('Features', fontsize=16, fontweight='bold')
            
            # 设置Y轴标签
            if feature_names and len(feature_names) == weights_matrix.shape[1]:
                plt.yticks(range(len(feature_names)), feature_names, fontsize=12)
            else:
                plt.yticks(range(weights_matrix.shape[1]), 
                        [f'Feature {i}' for i in range(weights_matrix.shape[1])], 
                        fontsize=12)
            
            # 设置X轴标签
            x_ticks = range(0, n_epochs, x_tick_interval)
            x_labels = [valid_epochs[i] + 1 for i in x_ticks]  # +1转换为人类可读的epoch编号
            plt.xticks(x_ticks, x_labels, fontsize=12)
            
            # 添加颜色条
            cbar = plt.colorbar(im, shrink=0.8)
            cbar.set_label('Weight Value', fontsize=14, fontweight='bold')
            cbar.ax.tick_params(labelsize=12)
            
            # 在热力图上显示具体数值
            for i in range(0, n_epochs, max(1, n_epochs // 10)):  # 限制显示的数值数量
                for j in range(weights_matrix.shape[1]):
                    text = plt.text(i, j, f'{weights_matrix[i, j]:.3f}',
                                   ha="center", va="center", 
                                   color="white" if weights_matrix[i, j] < 0.3 else "black",
                                   fontsize=10, fontweight='bold')
            
            # 添加网格线
            plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            plt.tight_layout()
            
            # 保存图像
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'feature_weights_basin_{basin_id}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"🔥 特征权重热力图已保存至: {save_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()
        
        print(f"✅ 已完成 {len(all_basins)} 个流域的特征权重热力图绘制")
        
    except Exception as e:
        print(f"❌ 绘制特征权重热力图失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_flood_identification()
    print("\n" + "="*50)
    test_visualization_functions()