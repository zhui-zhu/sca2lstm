import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import time
import psutil
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple
import threading
import functools
import platform

class HydrologyDataset(Dataset):
    def __init__(self, basin_ids: list, config, mode: str = "train", 
                 use_parallel: bool = False, max_load_threads: int = 20, 
                 max_sample_processes: int = 10, enable_monitoring: bool = True):
        self.config = config
        self.mode = mode
        self.basin_ids = basin_ids
        self.lstm2_features = config.LSTM2_FEATURES
        self.lstm1_derived_feats = config.LSTM1_DERIVED_FEATS
        self.target_col = config.TARGET_COL
        self.seq_len = config.SEQ_LEN
        self.pred_len = config.PRED_LEN
        
        # 并行化配置
        self.use_parallel = use_parallel
        self.max_load_threads = max_load_threads
        self.max_sample_processes = max_sample_processes
        self.enable_monitoring = enable_monitoring
        
        # Windows系统检测
        self.is_windows = platform.system() == 'Windows'
        if self.is_windows and self.use_parallel:
            print("⚠️  Windows系统下建议禁用多进程，自动切换到串行模式")
            self.use_parallel = False
        
        # 性能监控
        self.performance_stats = {
            'load_time': 0,
            'sample_time': 0,
            'total_time': 0,
            'memory_peak': 0,
            'errors': []
        }
        
        print(f"🚀 初始化水文数据集 ({mode}模式)")
        print(f"📊 配置参数:")
        print(f"   - 流域数量: {len(basin_ids)}")
        print(f"   - 并行化模式: {'✅ 启用' if self.use_parallel else '❌ 禁用'}")
        if self.use_parallel:
            print(f"   - 数据加载线程: {max_load_threads}")
            print(f"   - 样本生成进程: {max_sample_processes}")
        print(f"   - 序列长度: {config.SEQ_LEN}")
        print(f"   - 预测长度: {config.PRED_LEN}")
        print()
        
        start_time = time.time()
        
        # 根据模式选择数据加载方式
        if self.use_parallel:
            self.data = self._load_all_basins_data_parallel()
            self.samples = self._generate_samples_parallel()
        else:
            self.data = self._load_all_basins_data()
            self.samples = self._generate_samples()
        
        end_time = time.time()
        self.performance_stats['total_time'] = end_time - start_time
        
        # 显示性能统计
        if self.enable_monitoring:
            self._print_performance_stats()

    def _load_single_basin_data(self, basin_id: int) -> pd.DataFrame or None:
        basin_dir = os.path.join(self.config.DATA_INPUT_DIR, str(basin_id))
        data_path = os.path.join(basin_dir, f"model_input_{basin_id}.csv")
        
        if not os.path.exists(data_path):
            print(f"⚠️  流域{basin_id}的数据文件不存在：{data_path}，跳过")
            return None
        
        df = pd.read_csv(data_path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        original_rows = len(df)
        
        # 过滤目标值NaN
        df = df.dropna(subset=["discharge_vol", self.target_col]).reset_index(drop=True)
        valid_rows = len(df)
        valid_ratio = valid_rows / original_rows if original_rows > 0 else 0
        
        if valid_ratio < self.config.MIN_VALID_LABEL_RATIO:
            print(f"⚠️  流域{basin_id}有效标签比例{valid_ratio:.2f}（<{self.config.MIN_VALID_LABEL_RATIO}），跳过")
            return None
        if valid_rows < self.config.MIN_VALID_ROWS:
            print(f"⚠️  流域{basin_id}有效标签行数{valid_rows}（<{self.config.MIN_VALID_ROWS}），跳过")
            return None
        
        required_cols = self.lstm2_features + self.lstm1_derived_feats + [self.target_col, "date", "catchment_id"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"⚠️  流域{basin_id}缺少必需列：{missing_cols}，跳过")
            return None
        
        df.rename(columns={"catchment_id": "basin_id"}, inplace=True)
        return df

    def _load_all_basins_data(self) -> pd.DataFrame:
        all_data = []
        for basin_id in tqdm(self.basin_ids, desc=f"加载{self.mode}流域数据"):
            try:
                df = self._load_single_basin_data(basin_id)
                if df is not None:
                    all_data.append(df)
            except Exception as e:
                print(f"❌ 加载流域{basin_id}失败：{str(e)}，跳过该流域")
                continue
        
        if not all_data:
            raise ValueError(f"⚠️  无有效流域数据（所有流域标签全为NaN或有效标签不足）！")
        
        return pd.concat(all_data, ignore_index=True) 

    def _load_all_basins_data_parallel(self) -> pd.DataFrame:
        """并行加载所有流域数据"""
        print(f"🔄 开始并行数据加载...")
        start_time = time.time()
        
        # 计算最优线程数
        n_threads = min(self.max_load_threads, multiprocessing.cpu_count() * 2, len(self.basin_ids))
        
        all_data = []
        completed_count = 0
        failed_count = 0
        
        # 使用线程池并行加载
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            # 提交所有任务
            future_to_basin = {
                executor.submit(self._load_single_basin_data_with_retry, basin_id): basin_id
                for basin_id in self.basin_ids
            }
            
            # 收集结果并显示进度
            with tqdm(total=len(self.basin_ids), desc="并行加载流域数据") as pbar:
                for future in as_completed(future_to_basin):
                    basin_id = future_to_basin[future]
                    try:
                        df = future.result()
                        if df is not None:
                            all_data.append(df)
                            completed_count += 1
                        else:
                            failed_count += 1
                    except Exception as e:
                        print(f"❌ 流域{basin_id}加载失败: {str(e)}")
                        failed_count += 1
                        self.performance_stats['errors'].append(f"流域{basin_id}: {str(e)}")
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        "成功": completed_count,
                        "失败": failed_count
                    })
        
        if not all_data:
            raise ValueError(f"⚠️ 无有效流域数据！成功: {completed_count}, 失败: {failed_count}")
        
        # 合并数据
        result_df = pd.concat(all_data, ignore_index=True)
        
        end_time = time.time()
        self.performance_stats['load_time'] = end_time - start_time
        
        print(f"✅ 并行数据加载完成！")
        print(f"   - 成功加载: {completed_count} 个流域")
        print(f"   - 失败: {failed_count} 个流域") 
        print(f"   - 总数据行数: {len(result_df)}")
        print(f"   - 耗时: {self.performance_stats['load_time']:.2f} 秒")
        print()
        
        return result_df
    
    def _load_single_basin_data_with_retry(self, basin_id: int, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """带重试机制的单流域数据加载"""
        for attempt in range(max_retries):
            try:
                return self._load_single_basin_data(basin_id)
            except Exception as e:
                if attempt < max_retries - 1:
                    # 指数退避重试
                    time.sleep(0.1 * (attempt + 1))
                else:
                    # 最后一次重试也失败
                    return None
        return None

    def _get_time_encoding(self, date_series: pd.Series) -> np.ndarray:
        month = date_series.dt.month
        day = date_series.dt.day
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        day_sin = np.sin(2 * np.pi * day / 31)
        day_cos = np.cos(2 * np.pi * day / 31)
        return np.stack([month_sin, month_cos, day_sin, day_cos], axis=1)

    def _get_lstm1_input(self, df_seq: pd.DataFrame) -> np.ndarray:
        lstm1_inputs = []
        # time_encoding = self._get_time_encoding(df_seq["date"])
        # lstm1_inputs.append(time_encoding)
        
        derived_feats = df_seq[self.lstm1_derived_feats].values.astype(np.float32)
        lstm1_inputs.append(derived_feats)
        
        # lstm2_features_data = df_seq[self.lstm2_features]
        # missing_bool = (~np.isnan(lstm2_features_data.values)).astype(np.float32)
        # lstm1_inputs.append(missing_bool)
        
        # 添加LSTM2反馈残差（初始化为0）
        seq_len = len(df_seq)
        feedback_residual = np.zeros((seq_len, 1), dtype=np.float32)
        lstm1_inputs.append(feedback_residual)

        result = np.concatenate(lstm1_inputs, axis=1).astype(np.float32)
        return result

    def _generate_samples(self) -> list:
        print(f"开始生成{self.mode}集序列样本（序列长度={self.seq_len}天）...")
        samples = []
        for basin_id, df_basin in self.data.groupby("basin_id"):
            df_basin = df_basin.reset_index(drop=True)
            n_potential_samples = len(df_basin) - self.seq_len - self.pred_len + 1
            if n_potential_samples <= 0:
                print(f"⚠️  流域{basin_id}数据不足，无法生成样本（需{self.seq_len+self.pred_len}天，实际{len(df_basin)}天）")
                continue
            valid_sample_count = 0
            for i in range(n_potential_samples):
                seq_start = i
                seq_end = i + self.seq_len
                df_seq = df_basin.iloc[seq_start:seq_end].copy()
                
                target_start = seq_end
                target_end = seq_end + self.pred_len
                if target_end > len(df_basin):
                    continue
                
                target = df_basin.iloc[target_start:target_end][self.target_col].values
                if np.isnan(target).any():
                    continue
                
                # 填充特征NaN（用序列均值）
                seq_features = df_seq[self.lstm2_features].copy()
                for feat in self.lstm2_features:
                    if seq_features[feat].isna().any():
                        seq_features[feat].fillna(seq_features[feat].mean(), inplace=True)
                seq_features = seq_features.values.astype(np.float32)
                
                lstm1_input = self._get_lstm1_input(df_seq)
                missing_bool = (~np.isnan(df_seq[self.lstm2_features].values)).astype(np.float32)
                
                samples.append({
                    "seq_features": seq_features,
                    "lstm1_input": lstm1_input,
                    "missing_bool": missing_bool,
                    "basin_id": basin_id,
                    "target": target.astype(np.float32)
                })
                valid_sample_count += 1
        
        print(f"{self.mode}集生成完成：共{len(samples)}个有效样本（标签均非NaN）")
        return samples
    
    def _generate_samples_parallel(self) -> List[Dict]:
        """并行生成样本"""
        print(f"🔄 开始并行样本生成...")
        start_time = time.time()
        
        # 按流域分组数据
        basin_groups = list(self.data.groupby("basin_id"))
        all_samples = []
        completed_basins = 0
        failed_basins = 0
        
        # 准备配置参数（转换为基本类型，避免pickle问题）
        config_params = {
            'seq_len': self.config.SEQ_LEN,
            'pred_len': self.config.PRED_LEN,
            'target_col': self.config.TARGET_COL,
            'lstm2_features': list(self.config.LSTM2_FEATURES),
            'lstm1_derived_feats': list(self.config.LSTM1_DERIVED_FEATS)
        }
        
        # 计算最优进程数
        n_processes = min(self.max_sample_processes, multiprocessing.cpu_count(), len(basin_groups))
        
        # 使用进程池并行处理
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            # 准备任务参数
            tasks = [
                (basin_id, df_basin.copy(), config_params, self.mode)
                for basin_id, df_basin in basin_groups
            ]
            
            # 提交任务
            futures = [
                executor.submit(self._generate_single_basin_samples_parallel, *task)
                for task in tasks
            ]
            
            # 收集结果
            with tqdm(total=len(basin_groups), desc="并行生成样本") as pbar:
                for future in as_completed(futures):
                    try:
                        basin_samples = future.result()
                        if basin_samples:
                            all_samples.extend(basin_samples)
                            completed_basins += 1
                        else:
                            failed_basins += 1
                    except Exception as e:
                        failed_basins += 1
                        print(f"❌ 样本生成失败: {str(e)}")
                        self.performance_stats['errors'].append(f"样本生成: {str(e)}")
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        "完成流域": completed_basins,
                        "失败流域": failed_basins,
                        "样本数": len(all_samples)
                    })
        
        end_time = time.time()
        self.performance_stats['sample_time'] = end_time - start_time
        
        print(f"✅ 并行样本生成完成！")
        print(f"   - 成功处理: {completed_basins} 个流域")
        print(f"   - 失败: {failed_basins} 个流域")
        print(f"   - 生成样本: {len(all_samples)} 个")
        print(f"   - 耗时: {self.performance_stats['sample_time']:.2f} 秒")
        print(f"   - 平均速度: {len(all_samples)/self.performance_stats['sample_time']:.1f} 样本/秒")
        print()
        
        return all_samples
    
    @staticmethod
    def _generate_single_basin_samples_parallel(basin_id: int, df_basin: pd.DataFrame, 
                                               config_params: dict, mode: str) -> List[Dict]:
        """并行处理单个流域的样本生成（静态方法，用于多进程）"""
        try:
            samples = []
            seq_len = config_params['seq_len']
            pred_len = config_params['pred_len']
            target_col = config_params['target_col']
            lstm2_features = config_params['lstm2_features']
            lstm1_derived_feats = config_params['lstm1_derived_feats']
            
            # 重置索引
            df_basin = df_basin.reset_index(drop=True)
            n_potential_samples = len(df_basin) - seq_len - pred_len + 1
            
            if n_potential_samples <= 0:
                return samples
            
            # 预计算时间编码（优化性能）
            month = df_basin["date"].dt.month.values
            day = df_basin["date"].dt.day.values
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)
            day_sin = np.sin(2 * np.pi * day / 31)
            day_cos = np.cos(2 * np.pi * day / 31)
            
            for i in range(n_potential_samples):
                seq_start = i
                seq_end = i + seq_len
                
                # 数据切片
                df_seq = df_basin.iloc[seq_start:seq_end]
                
                # 目标值检查
                target_start = seq_end
                target_end = seq_end + pred_len
                if target_end > len(df_basin):
                    continue
                
                target = df_basin.iloc[target_start:target_end][target_col].values
                if np.isnan(target).any():
                    continue
                
                # 特征处理（向量化优化）
                seq_features_matrix = df_seq[lstm2_features].values.astype(np.float32)
                
                # NaN填充（向量化）
                col_means = np.nanmean(seq_features_matrix, axis=0)
                nan_mask = np.isnan(seq_features_matrix)
                if np.any(nan_mask):
                    # 使用向量化填充而不是循环
                    row_indices, col_indices = np.where(nan_mask)
                    seq_features_matrix[nan_mask] = col_means[col_indices]
                
                # LSTM1输入生成
                time_encoding = np.stack([
                    month_sin[seq_start:seq_end],
                    month_cos[seq_start:seq_end],
                    day_sin[seq_start:seq_end],
                    day_cos[seq_start:seq_end]
                ], axis=1)
                
                derived_feats = df_seq[lstm1_derived_feats].values.astype(np.float32)
                missing_bool = (~np.isnan(df_seq[lstm2_features].values)).astype(np.float32)
                
                # 添加LSTM2反馈残差（初始化为0）
                feedback_residual = np.zeros((seq_len, 1), dtype=np.float32)
                
                lstm1_input = np.concatenate([
                    time_encoding,
                    derived_feats,
                    missing_bool,
                    feedback_residual
                ], axis=1).astype(np.float32)
                
                # 添加到样本列表
                samples.append({
                    "seq_features": seq_features_matrix,
                    "lstm1_input": lstm1_input,
                    "missing_bool": missing_bool,
                    "basin_id": basin_id,
                    "target": target.astype(np.float32)
                })
            
            return samples
            
        except Exception as e:
            print(f"❌ 流域{basin_id}样本生成失败: {str(e)}")
            return []

    def __len__(self) -> int:
        return len(self.samples)
    
    def _print_performance_stats(self):
        """打印性能统计信息"""
        if not self.enable_monitoring:
            return
        
        stats = self.performance_stats
        total_time = stats['total_time']
        load_time = stats['load_time']
        sample_time = stats['sample_time']
        
        print("=" * 60)
        print("📊 数据集性能统计")
        print("=" * 60)
        print(f"⏱️  总耗时: {total_time:.2f} 秒")
        if self.use_parallel:
            print(f"📂 数据加载: {load_time:.2f} 秒 ({load_time/total_time*100:.1f}%)")
            print(f"🎯 样本生成: {sample_time:.2f} 秒 ({sample_time/total_time*100:.1f}%)")
            print(f"⚡ 并行加速比: {total_time/(load_time + sample_time):.2f}x")
        
        if stats['errors']:
            print(f"❌ 错误数量: {len(stats['errors'])}")
            for error in stats['errors'][:3]:  # 只显示前3个错误
                print(f"   - {error}")
        
        print("=" * 60)
        print()
    
    def get_performance_comparison(self, serial_time: float = None) -> Dict:
        """获取性能对比信息"""
        if not self.use_parallel:
            return {'mode': 'serial', 'total_time': self.performance_stats['total_time']}
        
        parallel_time = self.performance_stats['total_time']
        if serial_time is None:
            # 估算串行时间（基于经验法则）
            serial_time = parallel_time * 2.5  # 保守估计
        
        speedup = serial_time / parallel_time if parallel_time > 0 else 0
        
        return {
            'mode': 'parallel',
            'serial_time': serial_time,
            'parallel_time': parallel_time,
            'speedup': speedup,
            'dataset_size': len(self.samples),
            'efficiency': min(speedup / 4, 1.0) * 100  # 假设4核并行
        }

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        return {
            "seq_features": torch.from_numpy(sample["seq_features"]),
            "lstm1_input": torch.from_numpy(sample["lstm1_input"]),
            "missing_bool": torch.from_numpy(sample["missing_bool"]),
            "basin_id": torch.tensor(sample["basin_id"], dtype=torch.long),
            "target": torch.from_numpy(sample["target"])
        }

# ======================== 多线程数据预处理 =======================
def preprocess_batch_data(batch_data, config):
    """多线程数据预处理函数"""
    device = config.DEVICE
    # 数据移到设备
    seq_features = batch_data["seq_features"].to(device)
    lstm1_input = batch_data["lstm1_input"].to(device)
    missing_bool = batch_data["missing_bool"].to(device)
    basin_ids = batch_data["basin_id"].to(device)
    target = batch_data["target"].to(device).unsqueeze(-1)  # (batch, 1)
    return {
        'seq_features': seq_features,
        'lstm1_input': lstm1_input,
        'missing_bool': missing_bool,
        'basin_ids': basin_ids,
        'target': target
    }

def parallel_preprocess_batches(batches, config, max_workers=4):
    """并行预处理多个批次"""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(preprocess_batch_data, batch, config) for batch in batches]
        results = [future.result() for future in futures]
    return [result for result in results if result is not None]

# ======================== DataLoader工厂函数 =======================
def create_hydrology_dataloaders(config, use_parallel=False, use_multithreading=True):
    """
    创建水文数据集的DataLoader
    
    参数:
        config: 配置对象
        use_parallel: 是否使用并行数据集
        use_multithreading: 是否使用多线程数据加载
    
    返回:
        train_dataset, train_loader, val_dataset, val_loader: 训练和验证数据集及DataLoader
    """
    print(f"\n{'='*30} 加载数据 {'='*30}")
    
    # 选择数据集类型
    dataset_type = "并行" if use_parallel else "串行"
    print(f"📊 使用{dataset_type}数据集")
    
    # 创建数据集（现在统一使用HydrologyDataset类）
    train_dataset = HydrologyDataset(config.TRAIN_BASIN_IDS, config, mode="train", use_parallel=use_parallel)
    val_dataset = HydrologyDataset(config.VAL_BASIN_IDS, config, mode="val", use_parallel=use_parallel)
    
    if len(train_dataset) == 0:
        raise ValueError("⚠️  训练集无有效样本（所有样本标签均为NaN）！")
    if len(val_dataset) == 0:
        raise ValueError("⚠️  验证集无有效样本（所有样本标签均为NaN）！")
    
    # 数据加载器配置（Windows系统下避免多进程pickle问题）
    import platform
    is_windows = platform.system() == 'Windows'
    
    if is_windows:
        # Windows系统：使用主进程数据加载，避免pickle序列化问题
        num_workers = 0
        print(f"🪟 Windows系统：使用主进程数据加载（num_workers=0）")
    else:
        # Linux/Mac系统：可以使用多进程
        num_workers = min(4, multiprocessing.cpu_count()) if use_multithreading else 0
        print(f"🐧 非Windows系统：使用多进程数据加载（num_workers={num_workers}）")
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False if is_windows else (num_workers > 0)  # Windows下禁用
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False if is_windows else (num_workers > 0)  # Windows下禁用
    )
    
    return train_dataset, train_loader, val_dataset, val_loader