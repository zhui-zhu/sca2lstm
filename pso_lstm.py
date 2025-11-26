# PSO-LSTM适配版本 - 单流域22001，仅使用discharge_vol
# 直接读取CSV文件，无需归一化，优化LSTM超参数
# Usage: python pso_lstm.py --lead_time 1

import os
import sys
import argparse
import json
import datetime as dt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from typing import Tuple
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------- Utilities -------------------------

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def nse(obs, sim):
    obs = np.asarray(obs).flatten()
    sim = np.asarray(sim).flatten()
    denom = np.sum((obs - np.mean(obs)) ** 2)
    if denom == 0:
        return -np.inf
    return 1 - np.sum((obs - sim) ** 2) / denom

def rmse(obs, sim):
    return float(np.sqrt(mean_squared_error(np.asarray(obs).flatten(),
                                            np.asarray(sim).flatten())))

def bias_pct(obs, sim):
    """
    计算百分比偏差（Bias%）
    Args:
        obs: 观测值数组
        sim: 模拟值数组
    Returns:
        bias_pct: 百分比偏差值
    """
    obs = np.asarray(obs).flatten()
    sim = np.asarray(sim).flatten()
    s_obs = np.sum(obs)
    if s_obs == 0:
        return np.nan
    return float((np.sum(sim - obs) / s_obs) * 100.0)

def log(level: str, message: str):
    """简单的日志函数"""
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")

# ------------------------- 数据加载函数 -------------------------

def load_discharge_data(basin_id: str = "22001") -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """
    直接从CSV文件加载指定流域的discharge_vol数据和日期信息
    Args:
        basin_id: 流域ID, 默认为22001
    Returns:
        discharge_series: discharge_vol时间序列数据
        dates: 对应的日期索引
    """
    csv_path = f"./datasets/CAMELS_GB/CAMELS_GB_timeseries/CAMELS_GB_hydromet_timeseries_{basin_id}_19701001-20150930.csv"
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"数据文件不存在: {csv_path}")
    
    # 读取CSV文件
    df = pd.read_csv(csv_path, parse_dates=['date'])
    
    # 提取discharge_vol列和日期
    discharge_series = df['discharge_vol'].values
    dates = pd.DatetimeIndex(df['date'])
    
    # 检查缺失值
    if np.isnan(discharge_series).any():
        print(f"警告：发现{np.isnan(discharge_series).sum()}个缺失值，使用前向填充处理")
        # 简单的前向填充处理缺失值
        discharge_series = pd.Series(discharge_series).ffill().values
    
    return discharge_series, dates

def create_sequences(discharge_series: np.ndarray, time_steps: int, lead_time: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建LSTM训练序列
    Args:
        discharge_series: discharge_vol时间序列
        time_steps: 输入序列长度（历史时间步）
        lead_time: 预测步长（未来时间步）
    Returns:
        X: 输入序列，shape=(N_samples, time_steps, 1)
        y: 输出序列，shape=(N_samples, lead_time)
    """
    X, y = [], []
    
    # 总序列长度需求
    total_length = time_steps + lead_time
    
    for i in range(len(discharge_series) - total_length + 1):
        # 输入序列：过去time_steps个时间步
        x_seq = discharge_series[i:i + time_steps]
        # 输出序列：未来lead_time个时间步
        y_seq = discharge_series[i + time_steps:i + time_steps + lead_time]
        
        X.append(x_seq.reshape(-1, 1))  # 添加特征维度
        y.append(y_seq)
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def split_train_val(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    划分训练集和验证集
    Args:
        X: 输入序列
        y: 输出序列  
        train_ratio: 训练集比例
    Returns:
        X_train, X_val, y_train, y_val
    """
    split_idx = int(len(X) * train_ratio)
    
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    return X_train, X_val, y_train, y_val

# ------------------------- Model -------------------------

class RunoffLSTM(nn.Module):
    """简化版LSTM - 仅使用discharge_vol历史数据预测未来"""
    def __init__(self, time_steps: int, hidden_size: int, lead_time: int = 1, num_layers: int = 1):
        super().__init__()
        self.lead_time = lead_time
        # 输入维度为1（只有discharge_vol一个特征）
        self.lstm = nn.LSTM(input_size=1,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True) # 输入形状为 [B, T, 1]
        self.fc = nn.Linear(hidden_size, lead_time)  # 输出维度为lead_time

    def forward(self, x):
        # x: [B, T, 1]
        out, _ = self.lstm(x)      # [B, T, H]
        out = out[:, -1, :]        # last time step, [B, H]
        out = self.fc(out)         # [B, lead_time]
        return out

def train_eval_lstm(X_train, y_train, X_val, y_val,
                    batch_size: int = 64,
                    hidden_size: int = 64,
                    max_epochs: int = 25,
                    verbose: bool = False) -> float:
    """Return validation NSE (higher is better). (用于 PSO 阶段，不画图)"""
    if len(X_train) < 5 or len(X_val) < 3:
        return -9999

    # 确保y是2D数组 (N_samples, 1)
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
    if y_val.ndim == 1:
        y_val = y_val.reshape(-1, 1)

    # 创建训练集和验证集
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.float32))
    val_ds   = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                             torch.tensor(y_val, dtype=torch.float32))
    # 动态调整批量大小
    bs_train = max(4, min(batch_size, len(train_ds)))
    bs_val   = max(4, min(batch_size, len(val_ds)))

    # 创建数据加载器
    train_loader = DataLoader(train_ds, batch_size=bs_train, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=bs_val,   shuffle=False, drop_last=False)

    model = RunoffLSTM(time_steps=X_train.shape[1], hidden_size=hidden_size, lead_time=y_train.shape[1]).to(DEVICE)
    crit = nn.MSELoss()
    opt  = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_nse = -np.inf
    train_losses, val_losses = [], []
    
    for epoch in range(max_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                val_loss += crit(pred, yb).item()
                y_true.append(yb.cpu().numpy())
                y_pred.append(pred.cpu().numpy())
        
        # 计算指标
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        val_nse = nse(y_true, y_pred)
        
        if verbose:
            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == max_epochs - 1:
                print(f"  Epoch {epoch+1:02d}/{max_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val NSE: {val_nse:.4f}")
        
        best_val_nse = max(best_val_nse, val_nse)

    return best_val_nse

def train_evaluate(X_train, y_train, X_val, y_val,
                   time_steps, batch_size, hidden_size,
                   epochs, plot=True, save_path=None, verbose=True):
    """
    训练并评估LSTM模型，支持多lead_time输出
    """
    # 确保y是2D数组 (N_samples, lead_time)
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
    if y_val.ndim == 1:
        y_val = y_val.reshape(-1, 1)

    # 创建数据加载器
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                           torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # 初始化模型 - 输入维度为1（只有discharge_vol）
    model = RunoffLSTM(time_steps=time_steps, hidden_size=hidden_size, lead_time=y_train.shape[1]).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练循环
    train_losses, val_losses = [], []
    best_val_rmse = np.inf

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                val_loss += criterion(pred, yb).item()
                y_true.append(yb.cpu().numpy())
                y_pred.append(pred.cpu().numpy())

        # 计算指标
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        val_rmse = rmse(y_true, y_pred)
        val_nse = nse(y_true, y_pred)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            if save_path:
                torch.save(model.state_dict(), save_path)

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch+1:03d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val RMSE: {val_rmse:.3f} | Val NSE: {val_nse:.3f}")

    # 绘制训练曲线
    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f'Training Curve - Lead Time: {y_train.shape[1]}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path.replace('.pth', '_training_curve.png'))
        plt.show()

    return best_val_rmse, val_nse

def plot_prediction_comparison(dates, actual, predicted, basin_id, lead_time, start_date=None, end_date=None, save_path=None):
    """
    绘制预测值与实际值的对比图
    Args:
        dates: 日期索引
        actual: 实际值数组
        predicted: 预测值数组  
        basin_id: 流域ID
        lead_time: 预测步长
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        save_path: 保存路径
    """
    # 创建图形
    plt.figure(figsize=(15, 10))
    
    # 时间范围筛选
    mask = np.ones(len(dates), dtype=bool)
    if start_date:
        start_dt = pd.to_datetime(start_date)
        mask = mask & (dates >= start_dt)
    if end_date:
        end_dt = pd.to_datetime(end_date)
        mask = mask & (dates <= end_dt)
    
    filtered_dates = dates[mask]
    filtered_actual = actual[mask]
    filtered_predicted = predicted[mask]
    
    if len(filtered_dates) == 0:
        print("⚠️  指定时间范围内无数据，使用全部数据")
        filtered_dates = dates
        filtered_actual = actual
        filtered_predicted = predicted
    
    # 计算评估指标
    nse_score = nse(filtered_actual, filtered_predicted)
    rmse_score = rmse(filtered_actual, filtered_predicted)
    bias_score = bias_pct(filtered_actual, filtered_predicted)
    
    # 绘制对比图
    plt.subplot(2, 1, 1)
    plt.plot(filtered_dates, filtered_actual, label='Actual Discharge', color='blue', alpha=0.8, linewidth=1.5)
    plt.plot(filtered_dates, filtered_predicted, label='Predicted Discharge', color='red', alpha=0.8, linewidth=1.5)
    plt.title(f'Basin {basin_id} Discharge Prediction Comparison (Lead Time: {lead_time})', fontsize=14, fontweight='bold')
    plt.ylabel('Discharge (m³/s)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 添加指标信息
    textstr = f'NSE: {nse_score:.4f} | RMSE: {rmse_score:.2f} | Bias: {bias_score:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=props)
    
    # 绘制散点图
    plt.subplot(2, 1, 2)
    plt.scatter(filtered_actual, filtered_predicted, alpha=0.6, s=20)
    
    # 添加完美预测线
    min_val = min(np.min(filtered_actual), np.min(filtered_predicted))
    max_val = max(np.max(filtered_actual), np.max(filtered_predicted))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2, label='Perfect Prediction Line')
    
    plt.xlabel('Actual Discharge (m³/s)', fontsize=12)
    plt.ylabel('Predicted Discharge (m³/s)', fontsize=12)
    plt.title(f'Predicted vs Actual Discharge Scatter Plot (NSE: {nse_score:.4f})', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 时间范围信息
    time_range_info = f"Time Range: {filtered_dates[0].strftime('%Y-%m-%d')} to {filtered_dates[-1].strftime('%Y-%m-%d')}"
    if start_date or end_date:
        time_range_info += f" (Filtered: {len(filtered_dates)} data points)"
    else:
        time_range_info += f" (All Data: {len(filtered_dates)} data points)"
    
    plt.figtext(0.5, 0.01, time_range_info, ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])  # 调整布局，为底部文本留出空间
    
    # 保存图片（在show之前保存）
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Prediction comparison plot saved to: {save_path}")
    
    plt.show()
    
    # 打印统计信息
    print(f"📊 Prediction Statistics:")
    print(f"   Data Points: {len(filtered_actual)}")
    print(f"   Actual Range: [{np.min(filtered_actual):.2f}, {np.max(filtered_actual):.2f}]")
    print(f"   Predicted Range: [{np.min(filtered_predicted):.2f}, {np.max(filtered_predicted):.2f}]")
    print(f"   NSE: {nse_score:.4f}")
    print(f"   RMSE: {rmse_score:.2f}")
    print(f"   Bias: {bias_score:.2f}%")
    
    return nse_score, rmse_score, bias_score

# ------------------------- PSO -------------------------

class Particle:
    def __init__(self, dim, lb, ub):
        self.dim = dim
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.position = np.random.uniform(self.lb, self.ub)
        self.velocity = np.zeros(dim)
        self.best_position = self.position.copy()
        self.best_score = -np.inf
        self.score = -np.inf

def pso_optimize(objective_func, dim, lb, ub, n_particles=15, max_iter=20, verbose=True):
    """最大化目标函数"""
    particles = [Particle(dim, lb, ub) for _ in range(n_particles)]
    global_best_position = None
    global_best_score = -np.inf

    for iter_idx in range(max_iter):
        # 评估所有粒子
        iter_scores = []
        for p_idx, p in enumerate(particles):
            p.score = objective_func(p.position)
            iter_scores.append(p.score)
            
            # 更新个体最佳
            if p.score > p.best_score:
                p.best_score = p.score
                p.best_position = p.position.copy()
            
            # 更新全局最佳
            if p.score > global_best_score:
                global_best_score = p.score
                global_best_position = p.position.copy()

        # 计算本轮统计信息
        iter_scores = np.array(iter_scores)
        mean_score = np.mean(iter_scores)
        std_score = np.std(iter_scores)
        min_score = np.min(iter_scores)
        max_score = np.max(iter_scores)

        # 更新速度和位置
        w, c1, c2 = 0.5, 1.5, 1.5
        for p in particles:
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            p.velocity = (w * p.velocity +
                          c1 * r1 * (p.best_position - p.position) +
                          c2 * r2 * (global_best_position - p.position))
            p.position = p.position + p.velocity
            p.position = np.clip(p.position, lb, ub)

        if verbose:
            # 获取当前最佳参数
            best_params_str = ""
            if global_best_position is not None:
                ts, bs, hs = int(global_best_position[0]), int(global_best_position[1]), int(global_best_position[2])
                best_params_str = f"| Best: ts={ts}, bs={bs}, hs={hs}"
            
            print(f"🔄 PSO 第 {iter_idx+1:02d}/{max_iter} 轮 | "
                  f"平均: {mean_score:.4f} ± {std_score:.4f} | "
                  f"范围: [{min_score:.4f}, {max_score:.4f}] | "
                  f"全局最佳: {global_best_score:.4f} {best_params_str}")

    return global_best_position, global_best_score

# ------------------------- Main -------------------------
# python .\pso_lstm.py --basin_id 32006 --pso_particles 10 --pso_iters 5 --epochs 30 --final_epochs 50 --plot --plot_prediction --start 2000-01-01 --end 2000-12-31
def main():
    parser = argparse.ArgumentParser(description="PSO-LSTM for discharge_vol prediction (basin 22001)")
    parser.add_argument("--basin_id", type=str, default="22001", help="Basin ID (default: 22001)")
    parser.add_argument("--time_steps", type=int, default=None, help="Input sequence length (use PSO if None)")
    parser.add_argument("--lead_time", type=int, default=1, help="Lead time (forecast horizon)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (use PSO if None)")
    parser.add_argument("--hidden_size", type=int, default=None, help="Hidden size (use PSO if None)")
    parser.add_argument("--pso_particles", type=int, default=10, help="PSO particles")
    parser.add_argument("--pso_iters", type=int, default=5, help="PSO iterations")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs (PSO phase)")
    parser.add_argument("--final_epochs", type=int, default=50, help="Final training epochs")
    parser.add_argument("--plot", action="store_true", help="Plot results")
    parser.add_argument("--plot_prediction", action="store_true", help="绘制预测对比图")
    parser.add_argument("--start", type=str, help="预测开始时间 (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="预测结束时间 (YYYY-MM-DD)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    set_seed(args.seed)

    # 设置算法名称和输出目录
    algo_name = "PSO_LSTM"
    
    # 创建输出目录 - 使用不带冒号的时间格式
    current_time = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_output", algo_name, f"{current_time}")
    os.makedirs(output_dir, exist_ok=True)
    
    log("INFO", f"开始运行 {algo_name} 算法")
    log("INFO", f"输出目录: {output_dir}")

    # 1. 加载数据
    log("INFO", f"加载流域 {args.basin_id} 的 discharge_vol 数据...")
    discharge_series, dates = load_discharge_data(basin_id=args.basin_id)
    log("INFO", f"数据加载完成，共 {len(discharge_series)} 个时间步")

    # 2. 参数优化（PSO）
    if args.time_steps is None or args.batch_size is None or args.hidden_size is None:
        log("INFO", "开始 PSO 超参数优化...")
        log("INFO", f"PSO 配置: 粒子数={args.pso_particles}, 迭代次数={args.pso_iters}")
        log("INFO", "搜索空间: time_steps∈[2,48], batch_size∈[4,256], hidden_size∈[8,128]")

        def objective(params):
            ts, bs, hs = int(params[0]), int(params[1]), int(params[2])
            
            # 参数有效性检查
            if ts < 2 or bs < 4 or hs < 8:
                print(f"⚠️  参数无效: ts={ts}, bs={bs}, hs={hs} (小于最小值)")
                return -9999
            
            print(f"🧪 测试参数组合: time_steps={ts:2d}, batch_size={bs:3d}, hidden_size={hs:3d}", end="")
            
            # 创建序列数据
            X, y = create_sequences(discharge_series, ts, args.lead_time)
            if len(X) < 100:
                print(f" → 数据不足: {len(X)} < 100")
                return -9999
            
            # 划分训练验证集
            X_train, X_val, y_train, y_val = split_train_val(X, y)
            
            # 训练并评估
            val_nse = train_eval_lstm(X_train, y_train, X_val, y_val,
                                      batch_size=bs, hidden_size=hs,
                                      max_epochs=args.epochs, verbose=True)
            
            print(f" → NSE: {val_nse:.4f}")
            return val_nse

        # PSO参数范围
        dim = 3
        lb = [2, 4, 8]     # time_steps, batch_size, hidden_size
        ub = [48, 256, 128]
        best_params, best_score = pso_optimize(objective, dim, lb, ub,
                                               n_particles=args.pso_particles,
                                               max_iter=args.pso_iters, verbose=True)
        best_time_steps, best_batch_size, best_hidden_size = map(int, best_params)
        log("INFO", f"🎯 PSO 优化完成!")
        log("INFO", f"   最佳参数: time_steps={best_time_steps}, batch_size={best_batch_size}, hidden_size={best_hidden_size}")
        log("INFO", f"   最佳验证NSE: {best_score:.4f}")
        log("INFO", f"   总评估次数: {args.pso_particles * args.pso_iters}")
    else:
        best_time_steps = args.time_steps
        best_batch_size = args.batch_size
        best_hidden_size = args.hidden_size
        log("INFO", f"使用指定参数: time_steps={best_time_steps}, batch_size={best_batch_size}, hidden_size={best_hidden_size}")

    # 3. 最终训练
    log("INFO", "开始最终模型训练...")
    X, y = create_sequences(discharge_series, best_time_steps, args.lead_time)
    X_train, X_val, y_train, y_val = split_train_val(X, y)
    
    log("INFO", f"训练集: {X_train.shape}, 验证集: {X_val.shape}")

    # 最终模型训练（带绘图）
    model_path = os.path.join(output_dir, f"lstm_basin{args.basin_id}_lead{args.lead_time}.pth")
    print(f"\n📊 最终训练详细损失输出:")
    best_rmse, best_nse = train_evaluate(X_train, y_train, X_val, y_val,
                                         best_time_steps, best_batch_size, best_hidden_size,
                                         epochs=args.final_epochs, plot=args.plot, save_path=model_path, verbose=True)

    # 4. 保存结果
    results = {
        "basin_id": args.basin_id,
        "lead_time": args.lead_time,
        "best_time_steps": best_time_steps,
        "best_batch_size": best_batch_size,
        "best_hidden_size": best_hidden_size,
        "best_val_rmse": float(best_rmse),
        "best_val_nse": float(best_nse),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "timestamp": current_time
    }

    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log("INFO", f"运行完成！最佳验证RMSE: {best_rmse:.3f}, NSE: {best_nse:.3f}")
    log("INFO", f"结果已保存至: {output_dir}")

    # 5. 预测对比图（如果指定）
    if args.plot_prediction:
        log("INFO", "开始生成预测对比图...")
        
        # 加载训练好的模型进行预测
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建最终模型
        final_model = RunoffLSTM(time_steps=best_time_steps, hidden_size=best_hidden_size, lead_time=args.lead_time)
        final_model.load_state_dict(torch.load(model_path, map_location=device))
        final_model.to(device)
        final_model.eval()
        
        # 创建完整序列数据用于预测
        X_full, y_full = create_sequences(discharge_series, best_time_steps, args.lead_time)
        
        # 转换为tensor
        X_tensor = torch.FloatTensor(X_full).to(device)
        y_tensor = torch.FloatTensor(y_full).to(device)
        
        # 生成预测
        with torch.no_grad():
            y_pred_tensor = final_model(X_tensor)
            y_pred = y_pred_tensor.cpu().numpy().flatten()
            y_actual = y_tensor.cpu().numpy().flatten()
        
        # 获取对应的日期（需要调整索引以匹配预测结果）
        pred_dates = dates[best_time_steps + args.lead_time - 1:]
        
        # 生成预测对比图
        plot_save_path = os.path.join(output_dir, f"prediction_comparison_basin{args.basin_id}_lead{args.lead_time}.png")
        plot_prediction_comparison(
            dates=pred_dates,
            actual=y_actual,
            predicted=y_pred,
            basin_id=args.basin_id,
            lead_time=args.lead_time,
            start_date=args.start,
            end_date=args.end,
            save_path=plot_save_path
        )
        
        log("INFO", "预测对比图生成完成！")

if __name__ == "__main__":
    main()