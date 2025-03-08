import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu


def load_final_bests(folder, dataset_name, file_suffix, num_runs=20, is_maximize=False):
    """
    从 folder 下加载 {dataset_name}_{file_suffix}_run{i}.csv (i=1..num_runs),
    读取其最终最优性能(若最小化则取 min, 若最大化则取 max).
    返回列表 final_bests, 长度=num_runs.

    :param folder: 文件夹路径,如 "search_results" 或 "tpe_results"
    :param dataset_name: 数据集名称,如 "7z"
    :param file_suffix: 标识算法的后缀,如 "search_results" 或 "tpe"
    :param num_runs: 运行次数
    :param is_maximize: 是否最大化问题,默认False(最小化)
    """
    final_bests = []

    for i in range(1, num_runs + 1):
        # 假设命名格式: "{dataset_name}_{file_suffix}_run{i}.csv"
        file_name = f"{dataset_name}_{file_suffix}_run{i}.csv"
        file_path = os.path.join(folder, file_name)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        df = pd.read_csv(file_path)
        perf = df["Performance"].values
        if is_maximize:
            best_perf = np.max(perf)
        else:
            best_perf = np.min(perf)
        final_bests.append(best_perf)

    return final_bests


def main():
    # 你想比较的多个数据集
    datasets = ["7z", "Apache", "brotli", "LLVM", "PostgreSQL", "spear", "storm", "x264"]
    # 文件夹
    random_folder = "E:\ISE-solution-main\ISE-solution-main\lab3\\random_search"
    tpe_folder = "E:\ISE-solution-main\ISE-solution-main\lab3_bayes\\tpe_results_improved"

    # 你跑了10次
    num_runs = 20
    # 是否最大化? (若某些系统是最大化,可单独处理)
    is_maximize = False

    for ds in datasets:
        # 加载随机搜索的最终最优性能
        random_bests = load_final_bests(random_folder, ds, "search_results", num_runs, is_maximize)
        # 加载TPE的最终最优性能
        tpe_bests = load_final_bests(tpe_folder, ds, "tpe", num_runs, is_maximize)

        # 转numpy array
        random_bests = np.array(random_bests)
        tpe_bests = np.array(tpe_bests)

        # 去掉可能因为文件缺失导致的空值
        random_bests = random_bests[~np.isnan(random_bests)]
        tpe_bests = tpe_bests[~np.isnan(tpe_bests)]

        # 若数量不足,继续
        if len(random_bests) < 2 or len(tpe_bests) < 2:
            print(f"{ds}: not enough data for statistical test.")
            continue

        # T-test
        t_stat, p_val_t = ttest_ind(random_bests, tpe_bests, equal_var=False)
        # Mann-Whitney U
        u_stat, p_val_u = mannwhitneyu(random_bests, tpe_bests, alternative='two-sided')

        print(f"\nDataset: {ds}")
        print(f"  Random bests: {random_bests}")
        print(f"  TPE bests:    {tpe_bests}")
        print(f"  T-test => t_stat={t_stat:.4f}, p_val={p_val_t:.4f}")
        print(f"  Mann-Whitney => U_stat={u_stat}, p_val={p_val_u:.4f}")

        # 可根据 p 值判断是否显著 (p<0.05)
        if p_val_u < 0.05:
            print("  => difference is statistically significant (Mann-Whitney, p<0.05)")
        else:
            print("  => difference is NOT significant (Mann-Whitney, p>=0.05)")


if __name__ == "__main__":
    main()
