import os
import pandas as pd
import numpy as np
from functools import partial
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


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


def measure_with_noise(true_val, repeats, noise_scale, dataset_name=""):
    """
    改进的噪声处理, 根据数据集调整噪声特性
    """
    # 根据数据集调整噪声参数
    if dataset_name == "postgresql":
        adjusted_noise = noise_scale * 0.5  # 降低PostgreSQL的噪声
    elif dataset_name == "spear":
        adjusted_noise = noise_scale * 0.3  # 大幅降低spear的噪声
    elif dataset_name == "storm":
        adjusted_noise = noise_scale * 0.4  # 降低storm的噪声
    else:
        adjusted_noise = noise_scale

    samples = []
    for _ in range(repeats):
        noisy_val = true_val + np.random.normal(0, adjusted_noise)
        if noisy_val < 0:
            noisy_val = 0
        samples.append(noisy_val)
    return np.mean(samples)


def evaluate_raw(config, df, config_cols, perf_col,
                 is_maximize, missing_value,
                 dataset_name,
                 repeats, noise_scale):
    """
    查表 => 若找不到则 missing_value
    => 根据数据集决定是否加噪声
    """
    # 1) 查表
    config_series = pd.Series(config)
    df_subset, config_series_aligned = df[config_cols].align(config_series, axis=1, copy=False)
    matched = df.loc[(df_subset == config_series_aligned).all(axis=1)]
    if matched.empty:
        true_val = missing_value
    else:
        true_val = matched[perf_col].iloc[0]

    # 2) 是否对该数据集加噪声
    if dataset_name in ["spear", "storm", "postgresql"]:
        perf_val = measure_with_noise(true_val, repeats, noise_scale, dataset_name)
    else:
        perf_val = true_val

    return perf_val


def get_transform_func(dataset_name):
    """
    获取针对特定数据集的变换函数
    """
    if dataset_name == "spear":
        def transform_func(x):
            if x <= 0:
                return 1e9  # 避免除零
            # 对于spear, 使用一个更平滑的变换
            return np.log(1 + 1.0 / x)

        return transform_func
    elif dataset_name == "storm":
        def transform_func(x):
            if x <= 0:
                return 1e9  # 避免除零
            # 对于storm, 使用一个更平滑的变换
            return np.sqrt(1.0 / x)

        return transform_func
    elif dataset_name == "postgresql":
        # 对于postgresql尝试一个非线性变换
        def transform_func(x):
            return np.log1p(x)  # log(1+x)

        return transform_func
    else:
        return None


def objective(params, df, config_cols, perf_col,
              is_maximize, missing_value, transform_func,
              dataset_name,
              repeats, noise_scale):
    """
    TPE内部用:
      - 先 evaluate_raw => 如果是特定数据集 => 加噪声 => transform => loss
    """
    raw_perf = evaluate_raw(params, df, config_cols, perf_col,
                            is_maximize, missing_value,
                            dataset_name,
                            repeats, noise_scale)

    if transform_func:
        perf_val = transform_func(raw_perf)
    else:
        perf_val = raw_perf

    loss = -perf_val if is_maximize else perf_val
    return {'loss': loss, 'status': STATUS_OK, 'raw_perf': raw_perf}


def get_missing_value(df, perf_col, is_maximize, dataset_name):
    """
    获取更合适的missing value
    """
    perf_max = df[perf_col].max()
    perf_min = df[perf_col].min()

    if dataset_name == "postgresql":
        return perf_max * 1.3 if not is_maximize else perf_min * 0.7
    elif dataset_name == "spear":
        return perf_max * 1.5 if not is_maximize else perf_min * 0.5
    elif dataset_name == "storm":
        return perf_max * 1.4 if not is_maximize else perf_min * 0.6
    else:
        return perf_max * 2 if not is_maximize else perf_min * 0.5


def get_sa_params(dataset_name):
    """
    获取针对特定数据集的SA参数
    """
    if dataset_name == "postgresql":
        return [
            {'initial_temp': 80, 'cooling_rate': 0.92, 'iterations': 200},
            {'initial_temp': 120, 'cooling_rate': 0.94, 'iterations': 250},
            {'initial_temp': 150, 'cooling_rate': 0.93, 'iterations': 300},
        ]
    elif dataset_name == "spear":
        return [
            {'initial_temp': 200, 'cooling_rate': 0.96, 'iterations': 300},
            {'initial_temp': 250, 'cooling_rate': 0.97, 'iterations': 350},
            {'initial_temp': 300, 'cooling_rate': 0.95, 'iterations': 400},
        ]
    elif dataset_name == "storm":
        return [
            {'initial_temp': 180, 'cooling_rate': 0.95, 'iterations': 250},
            {'initial_temp': 220, 'cooling_rate': 0.96, 'iterations': 300},
            {'initial_temp': 260, 'cooling_rate': 0.94, 'iterations': 350},
        ]
    else:
        return [
            {'initial_temp': 50, 'cooling_rate': 0.90, 'iterations': 100},
            {'initial_temp': 100, 'cooling_rate': 0.95, 'iterations': 100},
            {'initial_temp': 100, 'cooling_rate': 0.90, 'iterations': 150},
        ]


def simulated_annealing_search(initial_config, df, config_cols, perf_col,
                               is_maximize, missing_value,
                               initial_temp=100, cooling_rate=0.95, iterations=100,
                               transform_func=None,
                               dataset_name="",
                               repeats=3, noise_scale=1e-4):
    """
    改进的SA搜索
    """

    def get_transformed_perf(cfg):
        raw_val = evaluate_raw(cfg, df, config_cols, perf_col,
                               is_maximize, missing_value,
                               dataset_name,
                               repeats, noise_scale)
        return transform_func(raw_val) if transform_func else raw_val

    current_config = initial_config.copy()
    current_value = get_transformed_perf(current_config)
    best_config = current_config.copy()
    best_value = current_value

    T = initial_temp

    # 添加自适应邻域搜索
    adaptive_search_radius = 0.3  # 初始搜索半径比例

    for i in range(iterations):
        neighbor = current_config.copy()

        # 随着迭代进行, 调整搜索策略
        if i < iterations * 0.3:  # 前30%迭代
            # 多参数变异, 进行更广泛探索
            n_params_to_change = max(1, int(len(config_cols) * adaptive_search_radius))
            cols_to_change = np.random.choice(config_cols, n_params_to_change, replace=False)

            for col in cols_to_change:
                possible_vals = sorted(df[col].unique())
                candidate_vals = [v for v in possible_vals if v != current_config[col]]
                if candidate_vals:
                    neighbor[col] = np.random.choice(candidate_vals)
        else:
            # 单参数变异, 进行精细调整
            col = np.random.choice(config_cols)
            possible_vals = sorted(df[col].unique())
            current_idx = possible_vals.index(current_config[col])

            # 在临近值中选择
            radius = max(1, int(len(possible_vals) * adaptive_search_radius))
            min_idx = max(0, current_idx - radius)
            max_idx = min(len(possible_vals) - 1, current_idx + radius)

            candidate_indices = list(range(min_idx, max_idx + 1))
            if current_idx in candidate_indices:
                candidate_indices.remove(current_idx)

            if candidate_indices:
                new_idx = np.random.choice(candidate_indices)
                neighbor[col] = possible_vals[new_idx]

        # 若没有变化, 继续下一轮
        if neighbor == current_config:
            continue

        neighbor_value = get_transformed_perf(neighbor)

        delta = neighbor_value - current_value if is_maximize else current_value - neighbor_value
        if delta > 0:
            current_config, current_value = neighbor, neighbor_value
        else:
            acceptance_prob = np.exp(delta / T)
            if np.random.rand() < acceptance_prob:
                current_config, current_value = neighbor, neighbor_value

        if is_maximize:
            if current_value > best_value:
                best_value = current_value
                best_config = current_config.copy()
        else:
            if current_value < best_value:
                best_value = current_value
                best_config = current_config.copy()

        # 降温, 同时减小搜索半径
        T *= cooling_rate
        adaptive_search_radius *= 0.99

    return best_config, best_value


def tune_simulated_annealing(initial_config, df, config_cols, perf_col,
                             is_maximize, missing_value, param_grid, transform_func=None,
                             dataset_name="",
                             repeats=3, noise_scale=1e-4):
    """
    多组参数的SA, 选最优
    """
    best_cfg = None
    best_val = None
    for params in param_grid:
        cfg, val = simulated_annealing_search(
            initial_config, df, config_cols, perf_col,
            is_maximize, missing_value,
            initial_temp=params['initial_temp'],
            cooling_rate=params['cooling_rate'],
            iterations=params['iterations'],
            transform_func=transform_func,
            dataset_name=dataset_name,
            repeats=repeats,
            noise_scale=noise_scale
        )
        if best_cfg is None:
            best_cfg, best_val = cfg, val
        else:
            if is_maximize:
                if val > best_val:
                    best_cfg, best_val = cfg, val
            else:
                if val < best_val:
                    best_cfg, best_val = cfg, val
    return best_cfg, best_val


def run_tpe_for_dataset(file_path, budget=300, num_runs=20,
                        repeats=3, noise_scale=1e-4,
                        output_folder="tpe_results_improved"):
    """
    改进的TPE + SA:
      - 针对postgresql,spear,storm特别调整
      - 更合适的变换和噪声处理
    """
    os.makedirs(output_folder, exist_ok=True)
    df, config_cols, perf_col = load_dataset(file_path)

    dataset_name = os.path.basename(file_path).split('.')[0].lower()

    # 决定 is_maximize
    if dataset_name in ["spear", "storm"]:
        is_maximize = True
    else:
        is_maximize = False

    # 获取更合适的missing value
    missing_value = get_missing_value(df, perf_col, is_maximize, dataset_name)

    # 获取变换函数
    transform_func = get_transform_func(dataset_name)

    # 获取SA参数
    param_grid = get_sa_params(dataset_name)

    # TPE超参 - 针对特定数据集调整
    if dataset_name == "postgresql":
        tpe_algo = partial(
            tpe.suggest,
            n_startup_jobs=60,  # 增加初始随机探索
            gamma=0.3,  # 提高探索性
            n_EI_candidates=150
        )
    elif dataset_name == "spear":
        tpe_algo = partial(
            tpe.suggest,
            n_startup_jobs=80,  # 更多随机探索
            gamma=0.4,  # 提高探索性
            n_EI_candidates=200
        )
    elif dataset_name == "storm":
        tpe_algo = partial(
            tpe.suggest,
            n_startup_jobs=70,  # 更多随机探索
            gamma=0.35,  # 提高探索性
            n_EI_candidates=180
        )
    else:
        tpe_algo = partial(
            tpe.suggest,
            n_startup_jobs=30,
            gamma=0.2,
            n_EI_candidates=100
        )

    results_summary = {}

    for run_idx in range(1, num_runs + 1):
        trials = Trials()

        def objective_fn(params):
            return objective(params, df, config_cols, perf_col,
                             is_maximize, missing_value, transform_func, dataset_name,
                             repeats=repeats, noise_scale=noise_scale)

        # 创建搜索空间
        space = {col: hp.choice(col, sorted(df[col].unique())) for col in config_cols}

        # 运行TPE搜索
        best_dict = fmin(
            fn=objective_fn,
            space=space,
            algo=tpe_algo,
            max_evals=budget,
            trials=trials,
            verbose=False
        )

        # 保存搜索过程
        iteration_data = []
        for i, trial in enumerate(trials.trials):
            chosen_cfg = {}
            for col in config_cols:
                idx_list = trial['misc']['vals'][col]
                if idx_list and len(idx_list) > 0:
                    idx_chosen = idx_list[0]
                    unique_vals = sorted(df[col].unique())
                    chosen_cfg[col] = unique_vals[idx_chosen]
                else:
                    chosen_cfg[col] = None

            raw_perf = evaluate_raw(chosen_cfg, df, config_cols, perf_col,
                                    is_maximize, missing_value,
                                    dataset_name,
                                    repeats, noise_scale)
            row = [i] + [chosen_cfg[c] for c in config_cols] + [raw_perf]
            iteration_data.append(row)

        columns = ["Iteration"] + list(config_cols) + ["Performance"]
        df_out = pd.DataFrame(iteration_data, columns=columns)
        dataset_stem = os.path.basename(file_path).split('.')[0]
        csv_name = f"{dataset_stem}_tpe_run{run_idx}.csv"
        out_path = os.path.join(output_folder, csv_name)
        df_out.to_csv(out_path, index=False)

        # 解析 TPE 最优配置
        final_config = {}
        for col in config_cols:
            idx_chosen = best_dict[col]
            unique_vals = sorted(df[col].unique())
            final_config[col] = unique_vals[idx_chosen]

        final_perf = evaluate_raw(final_config, df, config_cols, perf_col,
                                  is_maximize, missing_value,
                                  dataset_name,
                                  repeats, noise_scale)

        # SA细化
        refined_cfg, refined_val_trans = tune_simulated_annealing(
            final_config, df, config_cols, perf_col,
            is_maximize, missing_value, param_grid, transform_func,
            dataset_name=dataset_name,
            repeats=repeats, noise_scale=noise_scale
        )
        refined_perf = evaluate_raw(refined_cfg, df, config_cols, perf_col,
                                    is_maximize, missing_value,
                                    dataset_name,
                                    repeats, noise_scale)

        print(f"[TPE(Noise) {dataset_stem} Run {run_idx}] best_dict={best_dict}")
        print(f"  => TPE final config: {final_config}, perf(noisy)={final_perf}")
        print(f"  => After SA: {refined_cfg}, perf(noisy)={refined_perf}")

        # CSV 最优
        if is_maximize:
            best_perf_csv = df_out["Performance"].max()
        else:
            best_perf_csv = df_out["Performance"].min()

        results_summary[f"run_{run_idx}"] = {
            "TPE Config": final_config,
            "TPE Perf(Noisy)": final_perf,
            "SA Config": refined_cfg,
            "SA Perf(Noisy)": refined_perf,
            "CSV Best Perf(Noisy)": best_perf_csv
        }

    return results_summary


def main():
    datasets_folder = "datasets"
    output_folder = "tpe_results_improved"
    os.makedirs(output_folder, exist_ok=True)

    # 超参
    budget = 300
    num_runs = 20
    repeats = 5  # 增加重复次数, 降低噪声影响
    noise_scale = 1e-4

    final_results = {}
    for file_name in os.listdir(datasets_folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(datasets_folder, file_name)
            dataset_name = file_name.split('.')[0]
            print(f"\nProcessing dataset: {dataset_name}")
            summary = run_tpe_for_dataset(
                file_path, budget, num_runs,
                repeats=repeats, noise_scale=noise_scale,
                output_folder=output_folder
            )
            final_results[dataset_name] = summary

    # 打印结果
    for ds, runs in final_results.items():
        print(f"\nDataset: {ds}")
        for run_id, info in runs.items():
            print(f"  {run_id}: TPE_Conf={info['TPE Config']}, TPEPerf={info['TPE Perf(Noisy)']}, "
                  f"SA_Conf={info['SA Config']}, SAPerf={info['SA Perf(Noisy)']}, "
                  f"CSV_BestPerf={info['CSV Best Perf(Noisy)']}")


if __name__ == "__main__":
    main()