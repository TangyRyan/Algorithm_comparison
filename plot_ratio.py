import os
import numpy as np
import matplotlib.pyplot as plt

from FF_GM_fullmatch import shortest_edge_first_greedy_matching_faiss_general
from ANN_injective import match_approx_nn_injective
from lapjv_wrapper import lapjv_match

# 你要的 N 列表
Ns = [1000, 5000, 10000, 20000, 30000, 50000, 80000, 100000]

# 为了算 mean±std，需要多个 seed
SEEDS = [0, 1, 2, 3, 4]   # 你可以改成 10 个 seed

DATA_ROOT = "data"
K = 10
EXTEND_SEARCH = True

# ⚠️ LAPJV 必须构造 N×N 矩阵：一般只对小 N 可行
# 你想画到 100000，LAPJV 基本跑不了，所以这里先限制一下
MAX_LAPJV_N = 10000

def load_AB(n: int):
    d = os.path.join(DATA_ROOT, str(n))
    A = np.load(os.path.join(d, "A.npy"))
    B = np.load(os.path.join(d, "B.npy"))
    return A, B

def compute_ratios_for_n(n: int):
    """对一个 N，跑多 seed，返回 ratios_ff(list), ratios_ann(list)"""
    ratios_ff = []
    ratios_ann = []

    for seed in SEEDS:
        # 方式1：如果你已经提前生成了固定 A/B，就直接 load（推荐：不同 seed 对应不同数据集）
        # 这里为了简单演示：仍然用你 data 目录里的 A/B（等价于每个 N 只有一套数据）
        A, B = load_AB(n)

        # FF
        _, cost_ff, _ = shortest_edge_first_greedy_matching_faiss_general(
            A, B, k=K, extend_search=EXTEND_SEARCH, verbose=False
        )
        # ANN injective
        _, cost_ann, _ = match_approx_nn_injective(
            A, B, k=K, extend_search=EXTEND_SEARCH, strict=True
        )

        # OPT
        _, cost_opt, _ = lapjv_match(A, B)

        ratios_ff.append(cost_ff / cost_opt)
        ratios_ann.append(cost_ann / cost_opt)

    return ratios_ff, ratios_ann

def main():
    xs = []
    ff_mean, ff_std = [], []
    ann_mean, ann_std = [], []

    for n in Ns:
        if n > MAX_LAPJV_N:
            print(f"Skip N={n} (LAPJV too big).")
            continue

        ratios_ff, ratios_ann = compute_ratios_for_n(n)

        xs.append(n)

        ff_mean.append(float(np.mean(ratios_ff)))
        ff_std.append(float(np.std(ratios_ff, ddof=1)) if len(ratios_ff) > 1 else 0.0)

        ann_mean.append(float(np.mean(ratios_ann)))
        ann_std.append(float(np.std(ratios_ann, ddof=1)) if len(ratios_ann) > 1 else 0.0)

        print(f"N={n}: GBCM mean={ff_mean[-1]:.4f} std={ff_std[-1]:.4f} | SLBM mean={ann_mean[-1]:.4f} std={ann_std[-1]:.4f}")

    # === 画图：你截图那种 errorbar 风格 ===
    plt.figure(figsize=(10, 7))

    plt.errorbar(xs, ff_mean, yerr=ff_std, marker="o", capsize=6, label="GBCM")
    plt.errorbar(xs, ann_mean, yerr=ann_std, marker="o", capsize=6, label="SLBM")

    plt.title("Cost ratio vs N (mean ± std over seeds)")
    plt.xlabel("N")
    plt.ylabel("cost / LAPJV_opt_cost")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    plt.savefig("cost_ratio_plot.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    main()
