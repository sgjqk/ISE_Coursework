# === File: main_random.py ===
import os
import pandas as pd
import numpy as np

def load_dataset(file_path):
    """
    读取CSV并返回:
      - df: 数据表
      - config_cols: 配置列(最后一列为性能列)
      - perf_col: 性能列
    """
    df = pd.read_csv(file_path)
    config_cols = df.columns[:-1]
    perf_col = df.columns[-1]
    return df, config_cols, perf_col

def measure_with_noise(true_val, repeats=3, noise_scale=1e-4):
    """
    对性能值做“重复测量 + 噪声”
      - 重复 repeats 次, 每次给 true_val 加一个 N(0, noise_scale^2) 的噪声
      - 取平均值
      - 若出现负值则截断为0(可根据需求决定)
    """
    samples = []
    for _ in range(repeats):
        noisy_val = true_val + np.random.normal(0, noise_scale)
        if noisy_val < 0:
            noisy_val = 0
        samples.append(noisy_val)
    return np.mean(samples)

def evaluate_raw(config, df, config_cols, perf_col,
                 missing_value, dataset_name,
                 repeats=3, noise_scale=1e-4):
    """
    在原始数据表里查找 config 对应性能值；若找不到则用 missing_value。
    - 若 dataset_name in ["spear","storm","postgresql"] => 对性能值做“重复测量 + 噪声”
    - 否则 => 直接返回原值
    """
    mask = True
    for col, val in zip(config_cols, config):
        mask &= (df[col] == val)
    matched = df[mask]
    if matched.empty:
        true_val = missing_value
    else:
        true_val = matched[perf_col].iloc[0]

    # 仅在这三个数据集上引入噪声
    if dataset_name in ["spear","storm","postgresql"]:
        perf_val = measure_with_noise(true_val, repeats, noise_scale)
    else:
        perf_val = true_val

    return perf_val

def random_search_with_noise(file_path, budget, output_file,
                             repeats=3, noise_scale=1e-4):
    """
    - spear、storm => is_maximize=True + transform=1/(x+1e-9)
    - postgresql => 也加噪声, 但 is_maximize=False
    - 其它 => 不加噪声, is_maximize=False
    """
    df, config_cols, perf_col = load_dataset(file_path)
    dataset_name = os.path.basename(file_path).split('.')[0].lower()

    if dataset_name in ["spear", "storm"]:
        is_maximize = True
        missing_value = df[perf_col].max() * 2
        def transform_func(x):
            if x <= 0:
                x = 1e-9
            return 1.0 / (x + 1e-9)
    else:
        # 包含 postgresql + 其它
        is_maximize = False
        missing_value = df[perf_col].max() * 2
        transform_func = None

    if is_maximize:
        best_perf_trans = -np.inf
    else:
        best_perf_trans = np.inf

    best_config = None
    best_perf_raw = None

    search_records = []

    for _ in range(budget):
        # 1) 随机抽取配置
        chosen_config = []
        for col in config_cols:
            vals = df[col].unique()
            chosen_val = np.random.choice(vals)
            chosen_config.append(chosen_val)

        # 2) 查表 => 根据 dataset_name 是否噪声
        raw_perf = evaluate_raw(chosen_config, df, config_cols, perf_col,
                                missing_value, dataset_name,
                                repeats, noise_scale)

        # 3) 若 spear/storm => transform
        if transform_func is not None:
            perf_val = transform_func(raw_perf)
        else:
            perf_val = raw_perf

        # 4) 更新最优
        if is_maximize:
            if perf_val > best_perf_trans:
                best_perf_trans = perf_val
                best_perf_raw = raw_perf
                best_config = chosen_config
        else:
            if perf_val < best_perf_trans:
                best_perf_trans = perf_val
                best_perf_raw = raw_perf
                best_config = chosen_config

        # 5) 记录
        search_records.append(chosen_config + [raw_perf])

    columns = list(config_cols) + ["Performance"]
    pd.DataFrame(search_records, columns=columns).to_csv(output_file, index=False)

    return best_config, best_perf_raw

def main():
    datasets_folder = "datasets"
    output_folder = "random_search"
    os.makedirs(output_folder, exist_ok=True)

    # Key parameters
    num_runs = 20
    budget = 100
    repeats = 3
    noise_scale = 1e-4

    for run_idx in range(1, num_runs+1):
        print(f"\n=== Random Search (Noise) Run {run_idx} ===")
        for file_name in os.listdir(datasets_folder):
            if file_name.endswith(".csv"):
                file_path = os.path.join(datasets_folder, file_name)
                base_name = file_name.split('.')[0]
                out_file = os.path.join(output_folder, f"{base_name}_search_results_run{run_idx}.csv")

                best_cfg, best_perf = random_search_with_noise(
                    file_path, budget, out_file,
                    repeats=repeats, noise_scale=noise_scale
                )
                print(f"Dataset: {file_name}")
                print(f"  => Best Solution (Noisy) = {best_cfg}")
                print(f"  => Best Perf (Noisy)     = {best_perf}")

if __name__ == "__main__":
    main()
