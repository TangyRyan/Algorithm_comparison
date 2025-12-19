import os
import numpy as np
import matplotlib.pyplot as plt

from FF_GM_fullmatch import shortest_edge_first_greedy_matching_faiss_general
from ANN_injective import match_approx_nn_injective as match_approx_nn
from lapjv_wrapper import lapjv_match

Ns = [100, 2000, 3000, 4000, 5000]
SEEDS = [0, 1, 2, 3, 4]

def load_AB(seed, N):
    d = os.path.join("data_seeds", f"seed_{seed}", str(N))
    A = np.load(os.path.join(d, "A.npy"))
    B = np.load(os.path.join(d, "B.npy"))
    return A, B

def run_one(seed, N, k=10, extend_search=True):
    A, B = load_AB(seed, N)

    _, cost_gbcm, _ = shortest_edge_first_greedy_matching_faiss_general(
        A, B, k=k, extend_search=extend_search, verbose=False
    )
    _, cost_slbm, _ = match_approx_nn(A, B, k=k, extend_search=extend_search)
    _, cost_opt,  _ = lapjv_match(A, B)   # 精确解

    return cost_gbcm, cost_slbm, cost_opt

def main():
    gbcm_ratios = []
    slbm_ratios = []

    for N in Ns:
        r_g, r_s = [], []
        for seed in SEEDS:
            cost_g, cost_s, cost_opt = run_one(seed, N)
            r_g.append(cost_g / cost_opt)
            r_s.append(cost_s / cost_opt)

            print(f"N={N}, seed={seed}: opt={cost_opt:.6f}, "
                  f"gbcm={cost_g:.6f} (ratio {r_g[-1]:.6f}), "
                  f"slbm={cost_s:.6f} (ratio {r_s[-1]:.6f})")

        gbcm_ratios.append((np.mean(r_g), np.std(r_g)))
        slbm_ratios.append((np.mean(r_s), np.std(r_s)))

    # 画图：mean ± std
    xs = np.array(Ns)
    g_mean = np.array([m for m, s in gbcm_ratios])
    g_std  = np.array([s for m, s in gbcm_ratios])
    s_mean = np.array([m for m, s in slbm_ratios])
    s_std  = np.array([s for m, s in slbm_ratios])

    plt.figure(figsize=(8, 6))
    plt.errorbar(xs, g_mean, yerr=g_std, marker='o', capsize=4, label='GBCM')
    plt.errorbar(xs, s_mean, yerr=s_std, marker='o', capsize=4, label='SLBM')
    plt.xscale("log")
    plt.xlabel("N")
    plt.ylabel("cost / LAPJV_opt_cost")
    plt.title("Cost ratio vs N (mean ± std over seeds)")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("cost_ratio_small.png", dpi=200)
    print("saved -> cost_ratio_small.png")

if __name__ == "__main__":
    main()
