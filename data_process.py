import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from tqdm import tqdm 

# ======================== 1. 配置参数（不变）========================
TS_DATA_DIR = "./data/CAMELS_GB_timeseries/"
STATIC_DATA_DIR = "./data/CAMELS_GB_static/"
OUTPUT_DIR = "./model_input_data/"
CATCHMENT_IDS = [22001, 22006, 32006, 39025, 42003, 45003, 51001, 54017, 75003, 79005]

STATIC_FILES = {
    "topo": "CAMELS_GB_topographic_attributes.csv",
    "clim": "CAMELS_GB_climatic_attributes.csv",
    "land": "CAMELS_GB_landcover_attributes.csv",
    "soil": "CAMELS_GB_soil_attributes.csv",
    "hydro": "CAMELS_GB_hydrologic_attributes.csv",
    "human": "CAMELS_GB_humaninfluence_attributes.csv"
}

STATIC_SEP = ","
TS_SEP = ","

# ======================== 2. 工具函数（删除所有图标）========================
def handle_outliers(series):
    mean = series.mean()
    std = series.std()
    lower_bound = mean - 3 * std
    series = series.where(series >= lower_bound, lower_bound)
    return series

def fill_discharge_nan(ts_data):
    flood_condition = ts_data["precipitation"] > 50
    discharge = ts_data["discharge_vol"].copy()
    
    non_flood_nan_idx = discharge[(discharge.isna()) & (~flood_condition)].index
    if len(non_flood_nan_idx) > 0:
        rolling_mean = discharge.rolling(window=7, center=True, min_periods=1).mean()
        discharge.loc[non_flood_nan_idx] = rolling_mean.loc[non_flood_nan_idx]
    
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

def load_static_data(catchment_id, static_dir, static_files):
    static_dfs = {}
    for key, fname in static_files.items():
        file_path = os.path.join(static_dir, fname)
        if not os.path.exists(file_path):
            print(f"警告：静态文件不存在→{file_path}，跳过该类指标")
            static_dfs[key] = pd.DataFrame()
            continue
        df = pd.read_csv(file_path, sep=STATIC_SEP)
        if df["gauge_id"].dtype != "int64":
            df["gauge_id"] = pd.to_numeric(df["gauge_id"], errors="coerce").fillna(-1).astype("int64")
        static_dfs[key] = df
    
    static_data = pd.DataFrame({"gauge_id": [catchment_id]})
    target_id = catchment_id
    
    # 地形指标
    if not static_dfs["topo"].empty:
        if target_id in static_dfs["topo"]["gauge_id"].values:
            topo_row = static_dfs["topo"][static_dfs["topo"]["gauge_id"] == target_id].iloc[0]
            static_data["area"] = topo_row["area"] if not pd.isna(topo_row["area"]) else np.nan
            static_data["dpsbar"] = topo_row["dpsbar"] if not pd.isna(topo_row["dpsbar"]) else np.nan
            print(f"topo文件 - area: {static_data['area'].iloc[0]:.2f}, dpsbar: {static_data['dpsbar'].iloc[0]:.2f}")
        else:
            static_data["area"] = np.nan
            static_data["dpsbar"] = np.nan
            print(f"topo文件 - 未找到流域{target_id}")
    else:
        static_data["area"] = np.nan
        static_data["dpsbar"] = np.nan
        print("topo文件 - 未加载")
    
    # 气候指标
    if not static_dfs["clim"].empty:
        if target_id in static_dfs["clim"]["gauge_id"].values:
            clim_row = static_dfs["clim"][static_dfs["clim"]["gauge_id"] == target_id].iloc[0]
            static_data["aridity"] = clim_row["aridity"] if not pd.isna(clim_row["aridity"]) else np.nan
            print(f"clim文件 - aridity: {static_data['aridity'].iloc[0]:.2f}")
        else:
            static_data["aridity"] = np.nan
            print(f"clim文件 - 未找到流域{target_id}")
    else:
        static_data["aridity"] = np.nan
        print("clim文件 - 未加载")
    
    # 土地覆盖
    if not static_dfs["land"].empty:
        if target_id in static_dfs["land"]["gauge_id"].values:
            land_row = static_dfs["land"][static_dfs["land"]["gauge_id"] == target_id].iloc[0]
            land_val = land_row["dom_land_cover"] if not pd.isna(land_row["dom_land_cover"]) else "unknown"
            static_data["dom_land_cover"] = land_val if land_val != "unknown" else np.nan
            print(f"land文件 - dom_land_cover: {static_data['dom_land_cover'].iloc[0]}")
        else:
            static_data["dom_land_cover"] = np.nan
            print(f"land文件 - 未找到流域{target_id}")
    else:
        static_data["dom_land_cover"] = np.nan
        print("land文件 - 未加载")
    
    # 土壤指标
    if not static_dfs["soil"].empty:
        if target_id in static_dfs["soil"]["gauge_id"].values:
            soil_row = static_dfs["soil"][static_dfs["soil"]["gauge_id"] == target_id].iloc[0]
            static_data["tawc"] = soil_row["tawc"] if not pd.isna(soil_row["tawc"]) else np.nan
            print(f"soil文件 - tawc: {static_data['tawc'].iloc[0]:.2f}")
        else:
            static_data["tawc"] = np.nan
            print(f"soil文件 - 未找到流域{target_id}")
    else:
        static_data["tawc"] = np.nan
        print("soil文件 - 未加载")
    
    # 水文特征
    if not static_dfs["hydro"].empty:
        if target_id in static_dfs["hydro"]["gauge_id"].values:
            hydro_row = static_dfs["hydro"][static_dfs["hydro"]["gauge_id"] == target_id].iloc[0]
            static_data["baseflow_index"] = hydro_row["baseflow_index"] if not pd.isna(hydro_row["baseflow_index"]) else np.nan
            print(f"hydro文件 - baseflow_index: {static_data['baseflow_index'].iloc[0]:.2f}")
        else:
            static_data["baseflow_index"] = np.nan
            print(f"hydro文件 - 未找到流域{target_id}")
    else:
        static_data["baseflow_index"] = np.nan
        print("hydro文件 - 未加载")
    
    # 人类影响
    if not static_dfs["human"].empty:
        if target_id in static_dfs["human"]["gauge_id"].values:
            human_row = static_dfs["human"][static_dfs["human"]["gauge_id"] == target_id].iloc[0]
            bench_val = human_row["benchmark_catch"] if not pd.isna(human_row["benchmark_catch"]) else np.nan
            static_data["benchmark_catch"] = bench_val if bench_val in ["Y", "N"] else np.nan
            print(f"human文件 - benchmark_catch: {static_data['benchmark_catch'].iloc[0]}")
        else:
            static_data["benchmark_catch"] = np.nan
            print(f"human文件 - 未找到流域{target_id}")
    else:
        static_data["benchmark_catch"] = np.nan
        print("human文件 - 未加载")
    
    return static_data

def preprocess_single_catchment(catchment_id, ts_data_dir, static_data_dir, output_dir):
    catchment_output_dir = os.path.join(output_dir, str(catchment_id) + "/")
    os.makedirs(catchment_output_dir, exist_ok=True)
    
    # ---------------------- 步骤1：时序数据读取 ----------------------
    ts_filename = f"CAMELS_GB_hydromet_timeseries_{catchment_id}_19701001-20150930.csv"
    ts_file_path = os.path.join(ts_data_dir, ts_filename)
    if not os.path.exists(ts_file_path):
        print(f"错误：流域{catchment_id}时序文件不存在→{ts_file_path}，跳过")
        return
    
    ts_df = pd.read_csv(ts_file_path, sep=TS_SEP)
    required_ts_cols = ["date", "precipitation", "peti", "temperature", "discharge_vol"]
    missing_ts_cols = [col for col in required_ts_cols[1:] if col not in ts_df.columns]
    if missing_ts_cols:
        print(f"警告：流域{catchment_id}时序文件缺少指标→{missing_ts_cols}，仅用现有指标")
        used_ts_cols = ["date"] + [col for col in required_ts_cols[1:] if col in ts_df.columns]
    else:
        used_ts_cols = required_ts_cols
    
    ts_data = ts_df[used_ts_cols].copy()
    ts_data["date"] = pd.to_datetime(ts_data["date"], errors="coerce")
    ts_data = ts_data.dropna(subset=["date"])
    
    valid_ts_cols = ["date"]
    for col in used_ts_cols[1:]:
        if not ts_data[col].isna().all():
            valid_ts_cols.append(col)
        else:
            print(f"警告：流域{catchment_id}时序指标{col}全为NaN，剔除")
    ts_data = ts_data[valid_ts_cols]
    if len(valid_ts_cols) < 2:
        print(f"错误：流域{catchment_id}无有效时序指标，跳过")
        return
    
    # ---------------------- 步骤2：时序数据预处理 ----------------------
    if "discharge_vol" in ts_data.columns:
        ts_data = fill_discharge_nan(ts_data)
    ts_num_cols = [col for col in valid_ts_cols if col != "date"]
    for col in ts_num_cols:
        ts_data[col] = handle_outliers(ts_data[col])
    
    ts_scaler = MinMaxScaler(feature_range=(0, 1))
    ts_normalized = ts_scaler.fit_transform(ts_data[ts_num_cols])
    np.save(
        os.path.join(catchment_output_dir, f"ts_scaler_{catchment_id}.npy"),
        np.array([ts_scaler.data_min_, ts_scaler.data_max_, ts_num_cols], dtype=object)
    )
    ts_normalized_df = pd.DataFrame(ts_normalized, columns=ts_num_cols)
    
    # ---------------------- 步骤3：静态数据预处理 ----------------------
    static_valid_cols = []
    static_processed = pd.DataFrame()
    
    try:
        print(f"\n流域{catchment_id}静态数据读取详情：")
        static_raw = load_static_data(catchment_id, static_data_dir, STATIC_FILES)
        
        static_clean_cols = ["gauge_id"]
        for col in static_raw.columns:
            if col == "gauge_id":
                continue
            if not pd.isna(static_raw[col].iloc[0]):
                static_clean_cols.append(col)
            else:
                print(f"警告：流域{catchment_id}静态指标{col}为NaN，剔除")
        static_clean = static_raw[static_clean_cols].copy()
        print(f"\n流域{catchment_id}清洁后静态数据：")
        print(static_clean.round(4))
        
        if len(static_clean_cols) <= 1:
            print(f"提示：流域{catchment_id}无有效静态指标，仅用时序数据训练")
        else:
            # 数值型指标处理
            static_num_cols = ["area", "dpsbar", "aridity", "tawc", "baseflow_index"]
            static_num_data = pd.DataFrame()
            for col in static_num_cols:
                if col in static_clean.columns:
                    val = float(static_clean[col].iloc[0])
                    static_num_data[col] = [val]
                    static_valid_cols.append(col)
            print(f"\n数值型静态指标：{static_num_data.columns.tolist()}，值：{[round(v,2) for v in static_num_data.iloc[0].tolist()]}")
            
            # 分类型指标处理
            static_cat_data = pd.DataFrame()
            if "dom_land_cover" in static_clean.columns:
                land_type = static_clean["dom_land_cover"].iloc[0]
                land_col_name = f"land_{land_type.replace(' ', '_').lower()}"
                static_cat_data[land_col_name] = [1]
                static_valid_cols.append(land_col_name)
            if "benchmark_catch" in static_clean.columns:
                bench_val = static_clean["benchmark_catch"].iloc[0]
                bench_code = 1 if bench_val == "Y" else 0
                static_cat_data["benchmark_catch"] = [bench_code]
                static_valid_cols.append("benchmark_catch")
            print(f"分类型静态指标：{static_cat_data.columns.tolist()}，值：{static_cat_data.iloc[0].tolist()}")
            
            # 拼接静态数据
            static_num_data.index = [0]
            static_cat_data.index = [0]
            static_processed = pd.concat(
                [static_num_data, static_cat_data],
                axis=1,
                verify_integrity=True
            )
            print(f"\n最终拼接静态数据：")
            print(static_processed.round(4))
    
    except Exception as e:
        print(f"错误：流域{catchment_id}静态数据处理异常→{str(e)}，仅用时序数据训练")
    
    # ---------------------- 步骤4：数据融合 ----------------------
    if not static_processed.empty:
        time_steps = len(ts_data)
        static_repeated = np.tile(static_processed.values, (time_steps, 1))
        static_repeated_df = pd.DataFrame(static_repeated, columns=static_valid_cols)
        final_input = pd.concat([ts_normalized_df, static_repeated_df], axis=1)
    else:
        final_input = ts_normalized_df
    
    final_input["date"] = ts_data["date"].values
    
    # ---------------------- 步骤5：保存与打印 ----------------------
    final_input_path = os.path.join(catchment_output_dir, f"model_input_{catchment_id}.csv")
    final_input.to_csv(final_input_path, index=False)
    if static_valid_cols:
        np.save(
            os.path.join(catchment_output_dir, f"static_cols_{catchment_id}.npy"),
            np.array(static_valid_cols)
        )
    
    print(f"\n流域{catchment_id}处理完成！")
    print(f"有效时序指标：{ts_num_cols}（共{len(ts_num_cols)}个）")
    print(f"有效静态指标：{static_valid_cols}（共{len(static_valid_cols)}个）")
    print(f"最终输入数据维度：{final_input.shape}（时间步×指标数）")
    if not static_processed.empty:
        print("静态指标预览（前3行）：")
        print(static_repeated_df.head(3).round(4))
    print(f"输出文件路径：{catchment_output_dir}\n")

# ======================== 3. 主流程（不变）========================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"开始处理{len(CATCHMENT_IDS)}个流域...\n")
    
    for catchment_id in tqdm(CATCHMENT_IDS, desc="整体处理进度", unit="流域"):
        preprocess_single_catchment(
            catchment_id=catchment_id,
            ts_data_dir=TS_DATA_DIR,
            static_data_dir=STATIC_DATA_DIR,
            output_dir=OUTPUT_DIR
        )
    
    print(f"\n所有流域处理完成！最终输入数据存放于：{OUTPUT_DIR}")