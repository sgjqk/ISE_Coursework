import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def visualize_tpe_combined(dataset_name, results_folder="tpe_results_improved",
                           num_runs=20, max_iter=300,
                           save_folder="tpe_visualization_multi"):
    """
    对同一数据集 (dataset_name) 的多次TPE搜索结果, 进行:
      1) 平均曲线 + 标准差(多次运行的迭代过程)
      2) 最终最优性能的箱线图(多次运行的最终结果分布)

    要求文件命名格式: {dataset_name}_tpe_run{i}.csv, i=1..num_runs
    CSV含列: "Iteration", 以及 "Performance" (或可在脚本中根据情况适配)
    """

    os.makedirs(save_folder, exist_ok=True)

    # 用于判断最大化/最小化(此处假设最小化, 你可自行改写)
    # 如果你真有最大化数据, 可以加一个外部标记
    is_maximize = False

    # 用于存储多次运行的迭代性能, shape=[max_iter, num_runs]
    all_perf = np.full((max_iter, num_runs), np.nan)

    # 用于记录每次运行的"最终最优性能"
    final_bests = []

    for run_idx in range(1, num_runs + 1):
        file_name = f"{dataset_name}_tpe_run{run_idx}.csv"
        csv_path = os.path.join(results_folder, file_name)
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skip.")
            continue

        df = pd.read_csv(csv_path)
        # 假设df有 "Iteration" 列, "Performance" 列, 以及 config列
        # 先按 "Iteration" 排序(防止顺序乱)
        if "Iteration" not in df.columns:
            # 如果没有Iteration列, 只能用index当做迭代顺序
            df["Iteration"] = df.index

        df.sort_values(by="Iteration", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # 取前 max_iter 行
        df = df.iloc[:max_iter].copy()

        perf_values = df["Performance"].values

        # 放进 all_perf
        length = len(perf_values)
        if length < max_iter:
            all_perf[:length, run_idx - 1] = perf_values
        else:
            all_perf[:, run_idx - 1] = perf_values[:max_iter]

        # 找本次搜索的最优
        if is_maximize:
            best_perf = perf_values.max()
        else:
            best_perf = perf_values.min()
        final_bests.append(best_perf)

    # ========== (1) 平均曲线 + 标准差 ==========
    mean_perf = np.nanmean(all_perf, axis=1)
    std_perf = np.nanstd(all_perf, axis=1)

    # 找到全局最优(在所有run, iteration里)
    if is_maximize:
        global_best_perf = np.nanmax(all_perf)
        loc = np.where(all_perf == global_best_perf)
    else:
        global_best_perf = np.nanmin(all_perf)
        loc = np.where(all_perf == global_best_perf)

    # loc 可能包含多个索引, 取第一个
    if len(loc[0]) > 0:
        global_best_iter = loc[0][0]  # iteration index
    else:
        global_best_iter = 0

    plt.figure(figsize=(10, 6))
    iterations = np.arange(max_iter)
    plt.plot(iterations, mean_perf, color="blue", label="Average Performance")
    plt.fill_between(iterations, mean_perf - std_perf, mean_perf + std_perf,
                     color="blue", alpha=0.2, label="Std. Dev.")

    # 标红星
    plt.plot(global_best_iter, global_best_perf, marker="*", color="red",
             markersize=12, label="Global Best")

    plt.title(f"Average Curve + Global Best for {dataset_name} ({num_runs} runs)", fontsize=16)
    plt.xlabel("Search Iteration", fontsize=14)
    plt.ylabel("Performance", fontsize=14)
    plt.legend()
    # 保存图1
    avg_curve_img = os.path.join(save_folder, f"{dataset_name}_avg_curve.png")
    plt.savefig(avg_curve_img)
    plt.show()

    # ========== (2) 最终最优性能的箱线图 ==========
    plt.figure(figsize=(6, 6))
    plt.boxplot(final_bests, vert=True, tick_labels=[dataset_name])
    plt.ylabel("Best Performance", fontsize=14)
    plt.title(f"Final Best Distribution ({num_runs} runs) - {dataset_name}", fontsize=16)

    boxplot_img = os.path.join(save_folder, f"{dataset_name}_best_boxplot.png")
    plt.savefig(boxplot_img)
    plt.show()


def main():
    results_folder = "tpe_results_improved"
    save_folder = "tpe_visualization_multi"
    num_runs = 20
    max_iter = 300

    # 你想可视化的数据集列表
    datasets = ["7z", "Apache", "brotli", "LLVM", "PostgreSQL", "spear", "storm", "x264"]

    for ds in datasets:
        visualize_tpe_combined(ds, results_folder, num_runs, max_iter, save_folder)


if __name__ == "__main__":
    main()
