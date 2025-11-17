import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from tqdm import tqdm  # 进度条（可选，需安装：pip install tqdm）

# ======================== 1. 配置参数（用户需根据自身文件修改）========================
# 文件路径配置
TS_DATA_DIR = "./CAMELS_GB_timeseries/"  # 时序数据文件夹路径（存放所有流域的时序CSV）
STATIC_DATA_DIR = "./CAMELS_GB_static/"  # 静态属性文件文件夹路径
OUTPUT_DIR = "./model_input_data/"  # 最终输入数据输出路径

# 流域ID列表（替换为你要处理的流域ID，多个用逗号分隔）
CATCHMENT_IDS = ["1001", "1002", "1003"]  # 示例：3个流域，实际替换为你的NRFA测站ID

# 静态属性文件名称（无需修改，按CAMELS_GB标准命名）
STATIC_FILES = {
    "topo": "CAMELS_GB_topographic_attributes.txt",
    "clim": "CAMELS_GB_climatic_attributes.txt",
    "land": "CAMELS_GB_landcover_attributes.txt",
    "soil": "CAMELS_GB_soil_attributes.txt",
    "hydro": "CAMELS_GB_hydrologic_attributes.txt",
    "human": "CAMELS_GB_humaninfluence_attributes.txt"
}

# 数据分隔符（CAMELS_GB静态文件默认制表符，时序文件默认逗号，无需修改）
STATIC_SEP = "\t"
TS_SEP = ","

# ======================== 2. 工具函数定义（无需修改）========================
def handle_outliers(series):
    """异常值处理：3σ法则，仅替换低于下限的异常值（保留极端洪水）"""
    mean = series.mean()
    std = series.std()
    lower_bound = mean - 3 * std
    series = series.where(series >= lower_bound, lower_bound)
    return series

def fill_discharge_nan(ts_data):
    """填充discharge_vol的缺失值（区分洪水期/非洪水期）"""
    flood_condition = ts_data["precipitation"] > 50  # 洪水期判断：日降雨>50mm
    discharge = ts_data["discharge_vol"].copy()
    
    # 非洪水期：7天滚动均值填充（中心窗口）
    non_flood_nan_idx = discharge[(discharge.isna()) & (~flood_condition)].index
    if len(non_flood_nan_idx) > 0:
        rolling_mean = discharge.rolling(window=7, center=True, min_periods=1).mean()
        discharge.loc[non_flood_nan_idx] = rolling_mean.loc[non_flood_nan_idx]
    
    # 洪水期：前后3天洪水期数据均值填充
    flood_nan_idx = discharge[(discharge.isna()) & (flood_condition)].index
    for idx in flood_nan_idx:
        # 取前后3天窗口
        start_idx = max(0, idx - 3)
        end_idx = min(len(ts_data), idx + 4)
        window_mask = (ts_data.index >= start_idx) & (ts_data.index < end_idx)
        window_flood_mask = window_mask & flood_condition
        
        # 优先用窗口内洪水期数据，无则用整个窗口数据
        flood_window_data = discharge[window_flood_mask]
        if not flood_window_data.empty:
            discharge.loc[idx] = flood_window_data.mean()
        else:
            discharge.loc[idx] = discharge[window_mask].mean()
    
    ts_data["discharge_vol"] = discharge
    return ts_data

def load_static_data(catchment_id, static_dir, static_files):
    """加载并筛选单个流域的静态指标"""
    # 读取各静态文件
    static_dfs = {}
    for key, fname in static_files.items():
        file_path = os.path.join(static_dir, fname)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"静态文件不存在：{file_path}")
        static_dfs[key] = pd.read_csv(file_path, sep=STATIC_SEP)
    
    # 筛选当前流域数据
    static_data = pd.DataFrame({"gauge_id": [catchment_id]})
    
    # 地形属性：area、dpsbar
    topo_df = static_dfs["topo"][static_dfs["topo"]["gauge_id"] == catchment_id]
    static_data["area"] = topo_df["area"].values[0] if not topo_df.empty else np.nan
    static_data["dpsbar"] = topo_df["dpsbar"].values[0] if not topo_df.empty else np.nan
    
    # 气候指标：aridity
    clim_df = static_dfs["clim"][static_dfs["clim"]["gauge_id"] == catchment_id]
    static_data["aridity"] = clim_df["aridity"].values[0] if not clim_df.empty else np.nan
    
    # 土地覆盖：dom_land_cover
    land_df = static_dfs["land"][static_dfs["land"]["gauge_id"] == catchment_id]
    static_data["dom_land_cover"] = land_df["dom_land_cover"].values[0] if not land_df.empty else "unknown"
    
    # 土壤属性：tawc
    soil_df = static_dfs["soil"][static_dfs["soil"]["gauge_id"] == catchment_id]
    static_data["tawc"] = soil_df["tawc"].values[0] if not soil_df.empty else np.nan
    
    # 水文特征：baseflow_index
    hydro_df = static_dfs["hydro"][static_dfs["hydro"]["gauge_id"] == catchment_id]
    static_data["baseflow_index"] = hydro_df["baseflow_index"].values[0] if not hydro_df.empty else np.nan
    
    # 人类影响：benchmark_catch
    human_df = static_dfs["human"][static_dfs["human"]["gauge_id"] == catchment_id]
    static_data["benchmark_catch"] = human_df["benchmark_catch"].values[0] if not human_df.empty else "N"
    
    # 填充静态数据中的NaN（用同列均值）
    static_num_cols = ["area", "dpsbar", "aridity", "tawc", "baseflow_index"]
    for col in static_num_cols:
        if pd.isna(static_data[col]).iloc[0]:
            static_data[col] = static_dfs[
                "topo" if col in ["area", "dpsbar"] else
                "clim" if col == "aridity" else
                "soil" if col == "tawc" else "hydro"
            ][col].mean()
    
    return static_data

def preprocess_single_catchment(catchment_id, ts_data_dir, static_data_dir, output_dir):
    """处理单个流域的数据：时序+静态预处理→融合→保存"""
    # ---------------------- 步骤1：读取时序数据并筛选指标 ----------------------
    ts_file = os.path.join(ts_data_dir, f"CAMELS_GB_hydromet_timeseries_{catchment_id}_19701001-20150930.csv")
    if not os.path.exists(ts_file):
        print(f"警告：流域{catchment_id}的时序文件不存在，跳过→{ts_file}")
        return
    
    ts_df = pd.read_csv(ts_file, sep=TS_SEP)
    ts_cols = ["date", "precipitation", "peti", "temperature", "discharge_vol"]
    if not all(col in ts_df.columns for col in ts_cols[1:]):  # 检查指标列是否存在
        print(f"警告：流域{catchment_id}的时序文件缺少指标列，跳过")
        return
    
    ts_data = ts_df[ts_cols].copy()
    ts_data["date"] = pd.to_datetime(ts_data["date"], errors="coerce")  # 转换时间戳
    ts_data = ts_data.dropna(subset=["date"])  # 删除时间戳无效的行
    
    # ---------------------- 步骤2：时序数据预处理 ----------------------
    # 填充discharge_vol缺失值
    ts_data = fill_discharge_nan(ts_data)
    # 异常值处理
    for col in ["precipitation", "peti", "temperature", "discharge_vol"]:
        ts_data[col] = handle_outliers(ts_data[col])
    # 归一化（Min-Max [0,1]）
    ts_scaler = MinMaxScaler(feature_range=(0, 1))
    ts_normalized = ts_scaler.fit_transform(ts_data[["precipitation", "peti", "temperature", "discharge_vol"]])
    # 保存时序归一化参数
    np.save(os.path.join(output_dir, f"ts_scaler_{catchment_id}.npy"), 
            np.array([ts_scaler.data_min_, ts_scaler.data_max_]))
    
    # ---------------------- 步骤3：静态数据预处理 ----------------------
    static_data = load_static_data(catchment_id, static_data_dir, STATIC_FILES)
    # 分类型数据编码
    # 1. dom_land_cover：独热编码
    land_dummies = pd.get_dummies(static_data["dom_land_cover"], prefix="land")
    static_data = pd.concat([static_data, land_dummies], axis=1).drop("dom_land_cover", axis=1)
    # 2. benchmark_catch：Y→1，N→0
    static_data["benchmark_catch"] = static_data["benchmark_catch"].map({"Y": 1, "N": 0, "unknown": 0})
    # 数值型归一化
    static_num_cols = [col for col in static_data.columns if col != "gauge_id"]
    static_scaler = MinMaxScaler(feature_range=(0, 1))
    static_normalized = static_scaler.fit_transform(static_data[static_num_cols])
    # 保存静态归一化参数
    np.save(os.path.join(output_dir, f"static_scaler_{catchment_id}.npy"), 
            np.array([static_scaler.data_min_, static_scaler.data_max_]))
    static_normalized_df = pd.DataFrame(static_normalized, columns=static_num_cols)
    
    # ---------------------- 步骤4：数据融合（静态扩展时间维度+拼接） ----------------------
    T = len(ts_data)
    # 静态数据扩展为T行
    static_rep = static_normalized_df.iloc[0].repeat(T).values.reshape(T, -1)
    static_rep_df = pd.DataFrame(static_rep, columns=static_num_cols)
    # 拼接时序+静态数据
    input_data = pd.concat([
        pd.DataFrame(ts_normalized, columns=["precipitation", "peti", "temperature", "discharge_vol"]),
        static_rep_df
    ], axis=1)
    # 添加时间戳（可选，用于后续分析）
    input_data["date"] = ts_data["date"].values
    
    # ---------------------- 步骤5：保存最终输入数据 ----------------------
    input_data.to_csv(os.path.join(output_dir, f"model_input_{catchment_id}.csv"), index=False)
    print(f"流域{catchment_id}处理完成！输出文件：model_input_{catchment_id}.csv")

# ======================== 3. 主流程（批量处理多个流域）========================
if __name__ == "__main__":
    # 创建输出文件夹（不存在则创建）
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 批量处理每个流域
    print(f"开始处理{len(CATCHMENT_IDS)}个流域...")
    for catchment_id in tqdm(CATCHMENT_IDS, desc="处理进度"):
        preprocess_single_catchment(
            catchment_id=catchment_id,
            ts_data_dir=TS_DATA_DIR,
            static_data_dir=STATIC_DATA_DIR,
            output_dir=OUTPUT_DIR
        )
    
    print("所有流域处理完成！输出文件存放于：", OUTPUT_DIR)