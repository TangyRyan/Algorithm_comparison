import os
import csv
import numpy as np

from FF_GM_fullmatch import shortest_edge_first_greedy_matching_faiss_general
from ANN_injective import match_approx_nn_injective
from lapjv_wrapper import lapjv_match

Ns = [1000, 5000, 10000, 20000, 30000, 50000, 80000, 100000]

DATA_ROOT = "data"
K = 10
EXTEND_SEARCH = True

# 关键：给 LAPJV 设置一个“最大可尝试 N”
# 你可以改大，但 N>=20000 在大多数机器上会爆内存（因为需要 N×N 距离矩阵）
MAX_LAPJV_N = 10000   # 建议先 10000，能跑再往上加

def load_AB(n: int):
    d = os.path.join(DATA_ROOT, str(n))
    A = np.load(os.path.join(d, "A.npy"))
    B = np.load(os.path.join(d, "B.npy"))
    return A, B

def main():
    out_csv = "cost_ratio_all.csv"
    rows = []

    for n in Ns:
        print(f"\n==== N={n} ====")
        A, B = load_AB(n)

        # FF (GBCM)
        _, cost_ff, t_ff = shortest_edge_first_greedy_matching_faiss_general(
            A, B, k=K, extend_search=EXTEND_SEARCH, verbose=False
        )

        # ANN (SLBM) - injective version
        _, cost_ann, t_ann = match_approx_nn_injective(
            A, B, k=K, extend_search=EXTEND_SEARCH, strict=True
        )

        cost_opt = None
        t_opt = None
        ratio_ff = None
        ratio_ann = None

        # LAPJV (Exact)
        if n <= MAX_LAPJV_N:
            try:
                _, cost_opt, t_opt = lapjv_match(A, B)
                ratio_ff = cost_ff / cost_opt
                ratio_ann = cost_ann / cost_opt
            except MemoryError:
                print("LAPJV failed: MemoryError (N×N matrix too large).")
            except Exception as e:
                print(f"LAPJV failed: {type(e).__name__}: {e}")
        else:
            print(f"Skip LAPJV for N={n} (n > MAX_LAPJV_N={MAX_LAPJV_N}).")

        print(f"FF  cost={cost_ff:.6f}, time={t_ff.get('total_time_s', float('nan')):.4f}s, ratio={ratio_ff}")
        print(f"ANN cost={cost_ann:.6f}, time={t_ann.get('total_time_s', float('nan')):.4f}s, ratio={ratio_ann}")
        if cost_opt is not None:
            print(f"OPT cost={cost_opt:.6f}, time={t_opt.get('total_time_s', float('nan')):.4f}s")

        rows.append({
            "N": n,
            "cost_ff": cost_ff,
            "time_ff_s": t_ff.get("total_time_s"),
            "cost_ann": cost_ann,
            "time_ann_s": t_ann.get("total_time_s"),
            "cost_lapjv": cost_opt,
            "time_lapjv_s": None if t_opt is None else t_opt.get("total_time_s"),
            "ratio_ff_over_opt": ratio_ff,
            "ratio_ann_over_opt": ratio_ann,
        })

    # 保存 CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nSaved -> {out_csv}")

if __name__ == "__main__":
    main()
