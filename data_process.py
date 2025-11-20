import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import json

# ======================== 核心配置 ========================
TS_DATA_DIR = "./datasets/CAMELS_GB/CAMELS_GB_timeseries/"
STATIC_DATA_DIR = "./datasets/CAMELS_GB/CAMELS_GB_static/"
OUTPUT_DIR = "./model_input_data/"
CATCHMENT_IDS = [10002, 10003, 22001, 22006, 32006, 39025, 42003, 45003, 51001, 54017, 75003, 79005]

# 静态数据文件名称
STATIC_FILES = {
    "topo": "CAMELS_GB_topographic_attributes.csv",
    "clim": "CAMELS_GB_climatic_attributes.csv",
    "land": "CAMELS_GB_landcover_attributes.csv",
    "soil": "CAMELS_GB_soil_attributes.csv",
    "hydro": "CAMELS_GB_hydrologic_attributes.csv",
    "human": "CAMELS_GB_humaninfluence_attributes.csv"
}

# 静态指标物理参考范围
STATIC_REF_RANGE = {
    "area": (0, 10000),           # km²
    "dpsbar": (0, 500),           # m/km
    "elev_mean": (0, 700),        # m
    "aridity": (0, 1),            # 干旱指数
    "p_seasonality": (-1, 1),     # 降水季节性
    "tawc": (0, 250),             # mm
    "porosity_cosby": (0, 100),   # 孔隙度 
    "baseflow_index": (0, 1),     # 基流指数
    "dwood_perc": (0, 100),       # 落叶林占比 （%）
    "ewood_perc": (0, 100),       # 常绿林占比 （%）
    "grass_perc": (0, 100),       # 草地占比 （%）
    "urban_perc": (0, 100),       # 城镇占比 （%）
    "inwater_perc": (0, 100),     # 水域占比 （%）
    "benchmark_catch": (0, 1),    # 基准流域（0/1）
    "reservoir_cap": (0, 1e8)     # 水库库容（m³）
}

# 时序指标物理硬约束(统计数据适用 CAMELS_GB 数据集，英国地区)
TS_PHYSICAL_CONSTRAINTS = {
    "precipitation": (0, 200),   # 日降水0-200mm（沿海可放宽至800）
    "peti": (0, 15),             # 日蒸散发0-15mm   
    "temperature": (-10, 40),    # 日温-10~40℃
    "discharge_vol": (0, 1000)     # 流量0-1000m³/s
}

# ======================== 核心工具函数 ========================
def fill_timeseries_nan(series: pd.Series, is_extreme_context: pd.Series = None) -> pd.Series:
    """时序指标缺失值填充"""
    series_filled = series.copy()
    nan_mask = series_filled.isna()
    if nan_mask.sum() == 0:
        return series_filled

    # 极端事件前后缺失：用极端事件上下文的有效均值×0.8填充
    if is_extreme_context is not None:
        extreme_nan_idx = nan_mask & is_extreme_context
        if extreme_nan_idx.sum() > 0:
            extreme_valid_val = series_filled[is_extreme_context & (~nan_mask)].mean() # 极端事件上下文的有效均值
            if not pd.isna(extreme_valid_val):
                series_filled.loc[extreme_nan_idx] = extreme_valid_val * 0.8 # 极端事件上下文的有效均值×0.8
                nan_mask = series_filled.isna()
                if nan_mask.sum() == 0:
                    return series_filled

    # 单个缺失值：前后一天均值
    single_nan_idx = []
    for idx in series_filled[nan_mask].index:
        prev_valid = (idx > 0) and (~pd.isna(series_filled.iloc[idx-1]))
        next_valid = (idx < len(series_filled)-1) and (~pd.isna(series_filled.iloc[idx+1]))
        if prev_valid and next_valid:
            single_nan_idx.append(idx)
    
    if single_nan_idx:
        series_filled.loc[single_nan_idx] = (
            series_filled.iloc[[i-1 for i in single_nan_idx]].values +
            series_filled.iloc[[i+1 for i in single_nan_idx]].values
        ) / 2
        nan_mask = series_filled.isna()
        if nan_mask.sum() == 0:
            return series_filled

    # 剩余连续缺失：7天滚动均值
    rolling_mean = series_filled.rolling(window=7, center=True, min_periods=3).mean()
    series_filled.loc[nan_mask] = rolling_mean.loc[nan_mask]

    # 极端情况：全局均值兜底
    final_nan_mask = series_filled.isna()
    if final_nan_mask.sum() > 0:
        series_filled.loc[final_nan_mask] = series_filled.dropna().mean()

    return series_filled

def handle_timeseries_outliers(series: pd.Series, indicator: str, catchment_id: int) -> pd.Series:
    """时序指标异常值处理（保留真实极端值）"""
    series_clean = series.copy()
    valid_data = series_clean.dropna()
    if len(valid_data) < 10: # 数据不足10天，不处理
        return series_clean

    # 物理硬约束
    min_phys, max_phys = TS_PHYSICAL_CONSTRAINTS[indicator]
    series_clean = np.clip(series_clean, min_phys, max_phys)

    # 流域自适应阈值（历史最大×1.2）
    historical_max = valid_data.max()
    adaptive_thresh = historical_max * 1.2
    series_clean = np.where(
        (series_clean > adaptive_thresh) & (series_clean <= max_phys),
        adaptive_thresh,
        series_clean
    )

    print(f"流域{catchment_id}-{indicator}：历史最大={historical_max:.2f}，自适应阈值={adaptive_thresh:.2f}")
    return series_clean

def calculate_dynamic_features(ts_data: pd.DataFrame) -> pd.DataFrame:
    """计算时序衍生指标"""
    ts_data = ts_data.copy()
    # 避免重复计算衍生列（如果已存在则跳过）
    dynamic_cols = ["high_prec_running_days", "low_prec_running_days", "prec_7day_sum", "prec_30day_sum"]

    # 高降水持续天数
    daily_mean_prec = ts_data["precipitation"].dropna().mean()
    high_prec_thresh = daily_mean_prec * 3 # 高降水阈值（3倍日降水均值）
    ts_data["is_high_prec"] = (ts_data["precipitation"] >= high_prec_thresh).astype(int)
    ts_data["high_prec_running_days"] = ts_data["is_high_prec"].groupby(
        (ts_data["is_high_prec"] != ts_data["is_high_prec"].shift()).cumsum()
    ).cumsum() * ts_data["is_high_prec"]

    # 干旱持续天数
    ts_data["is_low_prec"] = (ts_data["precipitation"] < 1).astype(int) # 干旱阈值（1mm）
    ts_data["low_prec_running_days"] = ts_data["is_low_prec"].groupby(
        (ts_data["is_low_prec"] != ts_data["is_low_prec"].shift()).cumsum()
    ).cumsum() * ts_data["is_low_prec"]

    # 累计降水
    ts_data["prec_7day_sum"] = ts_data["precipitation"].rolling(window=7, min_periods=1).sum()
    ts_data["prec_30day_sum"] = ts_data["precipitation"].rolling(window=30, min_periods=7).sum()

    # 删除中间列
    ts_data = ts_data.drop(columns=["is_high_prec", "is_low_prec"], errors="ignore")
    return ts_data

def normalize_static_feature(value: float, feature_name: str) -> float:
    """静态指标归一化"""
    if pd.isna(value):
        return np.nan
    min_ref, max_ref = STATIC_REF_RANGE[feature_name]
    if max_ref - min_ref < 1e-8:
        return 0.5
    value_clipped = np.clip(value, min_ref, max_ref)
    value_norm = (value_clipped - min_ref) / (max_ref - min_ref)
    return round(value_norm, 6)

def load_static_data_complete(catchment_id: int) -> pd.DataFrame:
    """完整加载15个静态指标"""
    static_data = pd.DataFrame({"gauge_id": [catchment_id]})

    # 1. 地形指标（area/dpsbar/elev_mean）
    topo_path = os.path.join(STATIC_DATA_DIR, STATIC_FILES["topo"])
    if os.path.exists(topo_path):
        topo_df = pd.read_csv(topo_path)
        topo_df["gauge_id"] = pd.to_numeric(topo_df["gauge_id"], errors="coerce").fillna(-1).astype(int)
        if catchment_id in topo_df["gauge_id"].values:
            row = topo_df[topo_df["gauge_id"] == catchment_id].iloc[0]
            static_data["area"] = row.get("area", np.nan)
            static_data["dpsbar"] = row.get("dpsbar", np.nan)
            static_data["elev_mean"] = row.get("elev_mean", np.nan)
        else:
            static_data["area"] = static_data["dpsbar"] = static_data["elev_mean"] = np.nan
    else:
        static_data["area"] = static_data["dpsbar"] = static_data["elev_mean"] = np.nan

    # 2. 气候指标（aridity/p_seasonality）
    clim_path = os.path.join(STATIC_DATA_DIR, STATIC_FILES["clim"])
    if os.path.exists(clim_path):
        clim_df = pd.read_csv(clim_path)
        clim_df["gauge_id"] = pd.to_numeric(clim_df["gauge_id"], errors="coerce").fillna(-1).astype(int)
        if catchment_id in clim_df["gauge_id"].values:
            row = clim_df[clim_df["gauge_id"] == catchment_id].iloc[0]
            static_data["aridity"] = row.get("aridity", np.nan)
            static_data["p_seasonality"] = row.get("p_seasonality", np.nan)
        else:
            static_data["aridity"] = static_data["p_seasonality"] = np.nan
    else:
        static_data["aridity"] = static_data["p_seasonality"] = np.nan

    # 3. 土壤指标（tawc/porosity_cosby）
    soil_path = os.path.join(STATIC_DATA_DIR, STATIC_FILES["soil"])
    if os.path.exists(soil_path):
        soil_df = pd.read_csv(soil_path)
        soil_df["gauge_id"] = pd.to_numeric(soil_df["gauge_id"], errors="coerce").fillna(-1).astype(int)
        if catchment_id in soil_df["gauge_id"].values:
            row = soil_df[soil_df["gauge_id"] == catchment_id].iloc[0]
            static_data["tawc"] = row.get("tawc", np.nan)
            static_data["porosity_cosby"] = row.get("porosity_cosby", np.nan)
        else:
            static_data["tawc"] = static_data["porosity_cosby"] = np.nan
    else:
        static_data["tawc"] = static_data["porosity_cosby"] = np.nan

    # 4. 水文指标（baseflow_index）
    hydro_path = os.path.join(STATIC_DATA_DIR, STATIC_FILES["hydro"])
    if os.path.exists(hydro_path):
        hydro_df = pd.read_csv(hydro_path)
        hydro_df["gauge_id"] = pd.to_numeric(hydro_df["gauge_id"], errors="coerce").fillna(-1).astype(int)
        if catchment_id in hydro_df["gauge_id"].values:
            row = hydro_df[hydro_df["gauge_id"] == catchment_id].iloc[0]
            static_data["baseflow_index"] = row.get("baseflow_index", np.nan)
        else:
            static_data["baseflow_index"] = np.nan
    else:
        static_data["baseflow_index"] = np.nan

    # 5. 土地覆盖指标（各占比）
    land_path = os.path.join(STATIC_DATA_DIR, STATIC_FILES["land"])
    if os.path.exists(land_path):
        land_df = pd.read_csv(land_path)
        land_df["gauge_id"] = pd.to_numeric(land_df["gauge_id"], errors="coerce").fillna(-1).astype(int)
        if catchment_id in land_df["gauge_id"].values:
            row = land_df[land_df["gauge_id"] == catchment_id].iloc[0]
            static_data["dwood_perc"] = row.get("dwood_perc", np.nan)
            static_data["ewood_perc"] = row.get("ewood_perc", np.nan)
            static_data["grass_perc"] = row.get("grass_perc", np.nan)
            static_data["urban_perc"] = row.get("urban_perc", np.nan)
            static_data["inwater_perc"] = row.get("inwater_perc", np.nan)
        else:
            land_cols = ["dwood_perc", "ewood_perc", "grass_perc", "urban_perc", "inwater_perc"]
            static_data[land_cols] = np.nan
    else:
        land_cols = ["dwood_perc", "ewood_perc", "grass_perc", "urban_perc", "inwater_perc"]
        static_data[land_cols] = np.nan

    # 6. 人类影响指标（benchmark_catch/reservoir_cap）
    human_path = os.path.join(STATIC_DATA_DIR, STATIC_FILES["human"])
    if os.path.exists(human_path):
        human_df = pd.read_csv(human_path)
        human_df["gauge_id"] = pd.to_numeric(human_df["gauge_id"], errors="coerce").fillna(-1).astype(int)
        if catchment_id in human_df["gauge_id"].values:
            row = human_df[human_df["gauge_id"] == catchment_id].iloc[0]
            bench_val = row.get("benchmark_catch", np.nan)
            static_data["benchmark_catch"] = 1 if bench_val == "Y" else 0 if bench_val == "N" else np.nan
            static_data["reservoir_cap"] = row.get("reservoir_cap", np.nan)
        else:
            static_data["benchmark_catch"] = static_data["reservoir_cap"] = np.nan
    else:
        static_data["benchmark_catch"] = static_data["reservoir_cap"] = np.nan

    # 确保无重复列名
    static_data = static_data.loc[:, ~static_data.columns.duplicated()]
    return static_data

def fill_discharge_nan(ts_data: pd.DataFrame) -> pd.DataFrame:
    """流量缺失值分场景填充"""
    ts_data = ts_data.copy()
    discharge = ts_data["discharge_vol"].copy()
    flood_condition = ts_data["precipitation"] > ts_data["precipitation"].quantile(0.9) # 90%分位数为洪水期

    # 非洪水期缺失
    non_flood_nan_idx = discharge[(discharge.isna()) & (~flood_condition)].index
    if len(non_flood_nan_idx) > 0:
        rolling_mean = discharge.rolling(window=7, center=True, min_periods=1).mean() # 非洪水期7天滚动均值
        discharge.loc[non_flood_nan_idx] = rolling_mean.loc[non_flood_nan_idx]

    # 洪水期缺失
    flood_nan_idx = discharge[(discharge.isna()) & (flood_condition)].index
    for idx in flood_nan_idx: 
        start_idx = max(0, idx - 3)
        end_idx = min(len(ts_data), idx + 4)
        window_mask = (ts_data.index >= start_idx) & (ts_data.index < end_idx)
        window_flood_mask = window_mask & flood_condition
        
        flood_window_data = discharge[window_flood_mask]
        if not flood_window_data.empty:
            discharge.loc[idx] = flood_window_data.mean()
        else:
            discharge.loc[idx] = discharge[window_mask].mean()

    ts_data["discharge_vol"] = discharge
    return ts_data

# ======================== 单流域处理主函数 ========================
def preprocess_single_catchment(catchment_id: int):
    catchment_output_dir = os.path.join(OUTPUT_DIR, str(catchment_id))
    os.makedirs(catchment_output_dir, exist_ok=True)
    print(f"\n{'='*50} 开始处理流域 {catchment_id} {'='*50}")

    # ---------------------- 步骤1：读取时序数据 ----------------------
    ts_filename = f"CAMELS_GB_hydromet_timeseries_{catchment_id}_19701001-20150930.csv"
    ts_file_path = os.path.join(TS_DATA_DIR, ts_filename)
    if not os.path.exists(ts_file_path):
        print(f"❌ 流域{catchment_id}时序文件不存在：{ts_file_path}，跳过")
        return

    ts_df = pd.read_csv(ts_file_path)
    required_ts_cols = ["date", "precipitation", "peti", "temperature", "discharge_vol"]
    missing_cols = [col for col in required_ts_cols[1:] if col not in ts_df.columns]
    if missing_cols:
        print(f"⚠️  时序文件缺少指标：{missing_cols}，仅用现有指标")
    used_ts_cols = [col for col in required_ts_cols if col in ts_df.columns]
    ts_data = ts_df[used_ts_cols].copy()

    # 日期格式化+
    ts_data["date"] = pd.to_datetime(ts_data["date"], errors="coerce")
    ts_data = ts_data.dropna(subset=["date"]).drop_duplicates("date").reset_index(drop=True)
    print(f"✅ 时序数据读取完成：{len(ts_data)} 条记录")

    # ---------------------- 步骤2：时序数据预处理 ----------------------
    # 定义极端事件上下文（90%以上降水为极端事件）
    extreme_prec_thresh = ts_data["precipitation"].dropna().quantile(0.9) if "precipitation" in ts_data.columns else 0
    is_extreme_context = ts_data["precipitation"] >= extreme_prec_thresh if "precipitation" in ts_data.columns else pd.Series(False, index=ts_data.index)

    # 缺失值填充
    for col in ["precipitation", "peti", "temperature"]:
        if col in ts_data.columns:
            ts_data[col] = fill_timeseries_nan(ts_data[col], is_extreme_context)
            print(f"✅ {col}缺失值填充完成")

    # 流量缺失值填充
    if "discharge_vol" in ts_data.columns:
        ts_data = fill_discharge_nan(ts_data)
        print(f"✅ 流量缺失值填充完成")

    # 异常值处理
    ts_num_cols = [col for col in used_ts_cols if col != "date"]
    for col in ts_num_cols:
        if col in TS_PHYSICAL_CONSTRAINTS:
            ts_data[col] = handle_timeseries_outliers(ts_data[col], col, catchment_id)
    print(f"✅ 时序异常值处理完成")

    # ---------------------- 步骤3：计算时序衍生指标 ----------------------
    dynamic_cols = []
    if "precipitation" in ts_data.columns:
        ts_data = calculate_dynamic_features(ts_data)
        dynamic_cols = ["high_prec_running_days", "low_prec_running_days", "prec_7day_sum", "prec_30day_sum"]
        dynamic_cols = [col for col in dynamic_cols if col in ts_data.columns]  # 过滤实际存在的衍生列
        print(f"✅ 衍生指标计算完成：{dynamic_cols}")
    else:
        print(f"⚠️  无降水数据，未计算衍生指标")

    # ---------------------- 步骤4：时序指标归一化 ----------------------
    all_ts_cols = list(set(ts_num_cols + dynamic_cols))  # 去重
    all_ts_cols = [col for col in all_ts_cols if col in ts_data.columns]  # 确保列存在
    ts_scaler_params = {}
    ts_normalized = pd.DataFrame(index=ts_data.index)  # 明确索引

    for col in all_ts_cols:
        min_val = ts_data[col].min()
        max_val = ts_data[col].max()
        ts_scaler_params[col] = {"min": float(min_val), "max": float(max_val)}
        # 归一化
        if max_val - min_val < 1e-8:
            ts_normalized[col] = 0.5
        else:
            ts_normalized[col] = (ts_data[col] - min_val) / (max_val - min_val)

    # 保存时序缩放器
    ts_scaler_path = os.path.join(catchment_output_dir, f"ts_scaler_{catchment_id}.json")
    with open(ts_scaler_path, "w") as f:
        json.dump(ts_scaler_params, f, indent=2)
    print(f"✅ 时序归一化完成，缩放器保存至：{ts_scaler_path}")

    # ---------------------- 步骤5：静态数据预处理 ----------------------
    static_raw = load_static_data_complete(catchment_id)
    static_cols = list(STATIC_REF_RANGE.keys())
    static_processed = pd.DataFrame(index=[0])  # 明确索引

    # 静态指标归一化（确保无重复列）
    for col in static_cols:
        if col not in static_processed.columns:  # 避免重复添加
            if col in static_raw.columns:
                raw_val = static_raw[col].iloc[0]
                norm_val = normalize_static_feature(raw_val, col)
                static_processed[col] = [norm_val]
            else:
                static_processed[col] = [np.nan]

    # 保存静态参考范围
    static_scaler_path = os.path.join(catchment_output_dir, f"static_scaler_{catchment_id}.json")
    with open(static_scaler_path, "w") as f:
        json.dump(STATIC_REF_RANGE, f, indent=2)
    print(f"✅ 静态数据归一化完成，参考范围保存至：{static_scaler_path}")

    # ---------------------- 步骤6：数据融合（关键修复：避免列名重复） ----------------------
    # 1. 收集所有要拼接的DataFrame
    date_df = ts_data[["date"]].copy()
    catchment_df = pd.DataFrame({"catchment_id": [catchment_id]*len(ts_data)})

    # 2. 检查所有列名是否重复
    all_cols = (
        date_df.columns.tolist() +
        ts_normalized.columns.tolist() +
        static_processed.columns.tolist() +
        catchment_df.columns.tolist()
    )
    duplicate_cols = [col for col in set(all_cols) if all_cols.count(col) > 1]
    if duplicate_cols:
        print(f"⚠️  发现重复列名：{duplicate_cols}，自动重命名")
        # 重命名重复列（给静态列添加前缀"static_"）
        static_processed = static_processed.rename(columns={col: f"static_{col}" for col in duplicate_cols if col in static_processed.columns})
        # 给时序列添加前缀"ts_"（如果仍有重复）
        remaining_dups = [col for col in set(
            date_df.columns.tolist() + ts_normalized.columns.tolist() + static_processed.columns.tolist() + catchment_df.columns.tolist()
        ) if all_cols.count(col) > 1]
        if remaining_dups:
            ts_normalized = ts_normalized.rename(columns={col: f"ts_{col}" for col in remaining_dups if col in ts_normalized.columns})

    # 3. 静态数据重复到时序长度
    static_repeated = pd.DataFrame(
        np.tile(static_processed.values, (len(ts_data), 1)),
        columns=static_processed.columns,
        index=ts_data.index  # 确保索引一致
    )

    # 4. 拼接所有数据（添加verify_integrity=True便于调试）
    try:
        final_data = pd.concat([
            date_df,
            ts_normalized,
            static_repeated,
            catchment_df
        ], axis=1, verify_integrity=True)
    except ValueError as e:
        print(f"❌ 拼接失败：{str(e)}")
        print(f"当前各部分列名：")
        print(f"- 日期列：{date_df.columns.tolist()}")
        print(f"- 时序列：{ts_normalized.columns.tolist()}")
        print(f"- 静态列：{static_repeated.columns.tolist()}")
        print(f"- 流域ID列：{catchment_df.columns.tolist()}")
        raise e

    # ---------------------- 步骤7：保存最终结果 ----------------------
    final_output_path = os.path.join(catchment_output_dir, f"model_input_{catchment_id}.csv")
    final_data.to_csv(final_output_path, index=False, na_rep="NaN")
    print(f"✅ 最终数据保存至：{final_output_path}")
    print(f"📊 数据维度：{final_data.shape}（时间步×指标数）")
    print(f"📋 包含指标：{len(ts_normalized.columns)}个时序指标 + {len(static_repeated.columns)}个静态指标")

    print(f"{'='*50} 流域 {catchment_id} 处理完成 {'='*50}\n")

# ======================== 主流程（批量处理）========================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"开始处理 {len(CATCHMENT_IDS)} 个流域...\n")

    for catchment_id in tqdm(CATCHMENT_IDS, desc="整体处理进度", unit="流域"):
        try:
            preprocess_single_catchment(catchment_id)
        except Exception as e:
            print(f"❌ 流域{catchment_id}处理异常：{str(e)}，跳过")
            continue

    print(f"\n所有流域处理完成！最终数据存放于：{OUTPUT_DIR}")