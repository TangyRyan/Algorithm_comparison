#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproducible benchmark for cost ratio = (algorithm_cost / LAPJV_opt_cost)

Key guarantees:
- Same SLBM implementation everywhere: ANN_injective.match_approx_nn_injective
- Same OPT implementation everywhere: lapjv_wrapper.lapjv_match (Euclidean cost)
- Deterministic-ish FAISS behavior: force 1 thread when available
- Seeded, per-seed datasets expected at: {DATA_ROOT}/seed_{seed}/{N}/A.npy and B.npy
  (falls back to {DATA_ROOT}/{N}/A.npy and B.npy if per-seed not found)
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from FF_GM_fullmatch import shortest_edge_first_greedy_matching_faiss_general
from ANN_injective import match_approx_nn_injective
from lapjv_wrapper import lapjv_match

# ---- Determinism helpers (best-effort) ----
def set_determinism(seed: int, faiss_threads: int = 1):
    np.random.seed(seed)
    try:
        import faiss  # type: ignore
        faiss.omp_set_num_threads(faiss_threads)
    except Exception:
        pass

# ---- Data loading ----
def load_AB(data_root: str, seed: int, n: int):
    """
    Preferred layout (recommended for mean±std over seeds):
      {data_root}/seed_{seed}/{n}/A.npy
      {data_root}/seed_{seed}/{n}/B.npy

    Backward-compatible fallback:
      {data_root}/{n}/A.npy
      {data_root}/{n}/B.npy
    """
    d1 = os.path.join(data_root, f"seed_{seed}", str(n))
    a1 = os.path.join(d1, "A.npy")
    b1 = os.path.join(d1, "B.npy")
    if os.path.exists(a1) and os.path.exists(b1):
        A = np.load(a1)
        B = np.load(b1)
        return A, B, d1

    d2 = os.path.join(data_root, str(n))
    a2 = os.path.join(d2, "A.npy")
    b2 = os.path.join(d2, "B.npy")
    if os.path.exists(a2) and os.path.exists(b2):
        A = np.load(a2)
        B = np.load(b2)
        return A, B, d2

    raise FileNotFoundError(
        f"Cannot find A.npy/B.npy for seed={seed}, N={n}. Tried:\n"
        f"  {a1}\n  {b1}\n  {a2}\n  {b2}\n"
    )

# ---- Core benchmark ----
def run_one_seed(A: np.ndarray, B: np.ndarray, k: int, extend_search: bool, strict: bool):
    # Ensure dtype consistency across machines
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)

    # GBCM / FF greedy
    _, cost_gbcm, _ = shortest_edge_first_greedy_matching_faiss_general(
        A, B, k=k, extend_search=extend_search, verbose=False
    )

    # SLBM (injective ANN)
    _, cost_slbm, _ = match_approx_nn_injective(
        A, B, k=k, extend_search=extend_search, strict=strict
    )

    # OPT (LAPJV)
    _, cost_opt, _ = lapjv_match(A, B)

    return float(cost_gbcm), float(cost_slbm), float(cost_opt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="data_seeds",
                    help="Dataset root. Default: data_seeds")
    ap.add_argument("--Ns", type=int, nargs="+", default=[1000, 5000, 10000],
                    help="List of N values.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                    help="Seeds to average over.")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--extend-search", action="store_true", default=True)
    ap.add_argument("--no-extend-search", dest="extend_search", action="store_false")
    ap.add_argument("--strict", action="store_true", default=True,
                    help="Enforce injective + full coverage in SLBM.")
    ap.add_argument("--xscale", choices=["linear", "log"], default="linear")
    ap.add_argument("--out", type=str, default="cost_ratio_repro.png")
    ap.add_argument("--save-metrics", type=str, default="cost_ratio_repro_metrics.npz",
                    help="Save raw costs/ratios to an .npz file for debugging.")
    ap.add_argument("--max-lapjv-n", type=int, default=10000,
                    help="Safety guard: skip N larger than this to avoid O(N^2) LAPJV blow-up.")
    args = ap.parse_args()

    xs = []
    gbcm_ratios, slbm_ratios = [], []
    # store raw for debugging
    raw = {}

    for n in args.Ns:
        if n > args.max_lapjv_n:
            print(f"[SKIP] N={n} > max_lapjv_n={args.max_lapjv_n} (LAPJV O(N^2) too big).")
            continue

        ratios_g, ratios_s = [], []
        costs_g, costs_s, costs_o = [], [], []
        src_dirs = set()

        for seed in args.seeds:
            set_determinism(seed, faiss_threads=1)
            A, B, src = load_AB(args.data_root, seed, n)
            src_dirs.add(src)

            c_g, c_s, c_o = run_one_seed(A, B, k=args.k, extend_search=args.extend_search, strict=args.strict)

            ratios_g.append(c_g / c_o)
            ratios_s.append(c_s / c_o)

            costs_g.append(c_g); costs_s.append(c_s); costs_o.append(c_o)

        xs.append(n)

        g_mean = float(np.mean(ratios_g))
        g_std  = float(np.std(ratios_g, ddof=1)) if len(ratios_g) > 1 else 0.0
        s_mean = float(np.mean(ratios_s))
        s_std  = float(np.std(ratios_s, ddof=1)) if len(ratios_s) > 1 else 0.0

        gbcm_ratios.append((g_mean, g_std))
        slbm_ratios.append((s_mean, s_std))

        raw[str(n)] = dict(
            src_dirs=sorted(src_dirs),
            seeds=list(args.seeds),
            cost_gbcm=costs_g,
            cost_slbm=costs_s,
            cost_opt=costs_o,
            ratio_gbcm=ratios_g,
            ratio_slbm=ratios_s,
        )

        print(f"N={n} | data={'; '.join(sorted(src_dirs))}")
        print(f"  GBCM ratio mean±std = {g_mean:.6f} ± {g_std:.6f}")
        print(f"  SLBM ratio mean±std = {s_mean:.6f} ± {s_std:.6f}")

    if not xs:
        raise RuntimeError("No N values were run. Check --Ns and --max-lapjv-n.")

    xs = np.array(xs, dtype=int)
    g_mean = np.array([m for m, s in gbcm_ratios], dtype=float)
    g_std  = np.array([s for m, s in gbcm_ratios], dtype=float)
    s_mean = np.array([m for m, s in slbm_ratios], dtype=float)
    s_std  = np.array([s for m, s in slbm_ratios], dtype=float)

    plt.figure(figsize=(10, 7))
    plt.errorbar(xs, g_mean, yerr=g_std, marker="o", capsize=6, label="GBCM")
    plt.errorbar(xs, s_mean, yerr=s_std, marker="o", capsize=6, label="SLBM (injective)")
    if args.xscale == "log":
        plt.xscale("log")
    plt.title("Cost ratio vs N (mean ± std over seeds)")
    plt.xlabel("N")
    plt.ylabel("cost / LAPJV_opt_cost")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"saved -> {args.out}")

    # Save debugging bundle
    np.savez(args.save_metrics, xs=xs, gbcm_mean=g_mean, gbcm_std=g_std,
             slbm_mean=s_mean, slbm_std=s_std, raw=raw)
    print(f"saved metrics -> {args.save_metrics}")

if __name__ == "__main__":
    main()
