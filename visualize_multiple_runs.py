import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def visualize_combined(dataset_name, results_folder="search_results",
                       num_runs=10, max_iter=100, save_folder="visualization_results_multi"):
    """
    针对同一数据集 (dataset_name) 的多次搜索结果，做两种可视化:
      1) 平均曲线 + 标准差带 (多次运行的迭代过程)
      2) 最终最优性能的箱线图 (多次运行的最终结果分布)

    文件命名格式示例: dataset_name_search_results_run1.csv, run2.csv, ..., runN.csv
    每个CSV默认为: 100 行 (每行1次迭代), 有 "Performance" 列记录性能.
    """

    os.makedirs(save_folder, exist_ok=True)

    # 简单判定是最大化 or 最小化
    # 可按你实际需求改写, 这里示例: dataset名里含"---"则最大化
    if "---" in dataset_name.lower():
        is_maximize = True
    else:
        is_maximize = False

    # 用于存储多次运行的性能 (shape: [max_iter, num_runs])
    all_perf = np.full((max_iter, num_runs), np.nan)
    # 用于记录每次运行的 "最终最优性能"
    final_bests = []

    for run_idx in range(1, num_runs + 1):
        file_name = f"{dataset_name}_search_results_run{run_idx}.csv"
        csv_path = os.path.join(results_folder, file_name)

        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skip.")
            continue

        df = pd.read_csv(csv_path)
        # 只取前 max_iter 行, 防止行数不足或过多
        df = df.iloc[:max_iter].copy()

        perf_values = df["Performance"].values

        # 填入 all_perf
        length = len(perf_values)
        if length < max_iter:
            all_perf[:length, run_idx - 1] = perf_values
        else:
            all_perf[:, run_idx - 1] = perf_values[:max_iter]

        # 找到本次搜索的全局最优性能
        if is_maximize:
            best_val = np.nanmax(perf_values)
        else:
            best_val = np.nanmin(perf_values)
        final_bests.append(best_val)

    # ========== (1) 平均曲线 + 标准差 ==========
    mean_perf = np.nanmean(all_perf, axis=1)
    std_perf = np.nanstd(all_perf, axis=1)

    # 找到所有轮次的 全局最优(在10*max_iter里)
    if is_maximize:
        global_best_perf = np.nanmax(all_perf)
        global_best_loc = np.where(all_perf == global_best_perf)
    else:
        global_best_perf = np.nanmin(all_perf)
        global_best_loc = np.where(all_perf == global_best_perf)

    # global_best_loc 可能有多个, 取第一个即可
    if len(global_best_loc[0]) > 0:
        global_best_iter = global_best_loc[0][0]
    else:
        global_best_iter = 0

    # 画平均折线图
    plt.figure(figsize=(10,6))
    iterations = np.arange(max_iter)
    plt.plot(iterations, mean_perf, label="Average Performance", color="blue")
    plt.fill_between(iterations, mean_perf - std_perf, mean_perf + std_perf,
                     color="blue", alpha=0.2, label="Std. Dev.")

    # 在全局最优处标一个红星
    plt.plot(global_best_iter, global_best_perf, marker="*", color="red", markersize=12, label="Global Best")

    plt.title(f"Average Curve + Global Best for {dataset_name} ({num_runs} runs)", fontsize=16)
    plt.xlabel("Search Iteration", fontsize=14)
    plt.ylabel("Performance", fontsize=14)
    plt.legend()
    # 保存图1
    avg_img = os.path.join(save_folder, f"{dataset_name}_avg_curve.png")
    plt.savefig(avg_img)
    plt.show()

    # ========== (2) 最终最优性能的箱线图 ==========
    plt.figure(figsize=(6,6))
    plt.boxplot(final_bests, vert=True, tick_labels=[dataset_name])
    plt.ylabel("Best Performance", fontsize=14)
    plt.title(f"Final Best Distribution ({num_runs} runs) - {dataset_name}", fontsize=16)
    # 保存图2
    boxplot_img = os.path.join(save_folder, f"{dataset_name}_best_boxplot.png")
    plt.savefig(boxplot_img)
    plt.show()

def main():
    # 你要可视化的多个数据集名, 如 lab3 常见 "7z","Apache","brotli","LLVM","PostgreSQL","spear","storm","x264"
    datasets = ["7z", "Apache", "brotli", "LLVM", "PostgreSQL", "spear", "storm", "x264"]
    results_folder = "search_results"  # 存放多次运行结果的文件夹
    num_runs = 20                      # 你跑了10次
    max_iter = 100                     # 每次搜索的迭代次数

    for ds in datasets:
        visualize_combined(ds, results_folder, num_runs, max_iter)

if __name__ == "__main__":
    main()
