#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproducible benchmark for cost ratio = (algorithm_cost / LAPJV_opt_cost)

What this script does (high level):
1) For each N in --Ns and for each seed in --seeds:
   - Load two point sets A,B from disk (preferred: per-seed directories).
   - Run three matchers to get total matching cost:
        (a) GBCM: shortest_edge_first_greedy_matching_faiss_general
        (b) SLBM: match_approx_nn_injective (injective ANN matching)
        (c) OPT : lapjv_match (LAPJV optimal assignment on Euclidean distance)
   - Compute ratios:
        ratio_gbcm = cost_gbcm / cost_opt
        ratio_slbm = cost_slbm / cost_opt
2) Aggregate mean ± std (over seeds) for each N.
3) Save:
   - A PNG plot (error bars over seeds)
   - A CSV (one row per (N, seed) + repeated mean/std columns)
   - An NPZ debug bundle (raw dict + arrays), optional

Key guarantees:
- Same SLBM implementation everywhere: ANN_injective.match_approx_nn_injective
- Same OPT implementation everywhere: lapjv_wrapper.lapjv_match (Euclidean cost)
- Best-effort deterministic FAISS behavior: force 1 thread when available
- Seeded, per-seed datasets expected at:
    {DATA_ROOT}/seed_{seed}/{N}/A.npy and B.npy
  (falls back to {DATA_ROOT}/{N}/A.npy and B.npy if per-seed not found)

New in v2 (per your request):
- All outputs go into --out-dir (default: cost_ratio/)
- Each run uses a unique run name (timestamp) to avoid overwriting:
    cost_ratio_<run>.png
    cost_ratio_<run>.csv
    cost_ratio_<run>.npz
"""

import os
import argparse
import csv
import datetime as _dt
import numpy as np
import matplotlib.pyplot as plt

from FF_GM_fullmatch import shortest_edge_first_greedy_matching_faiss_general
from ANN_injective import match_approx_nn_injective
from lapjv_wrapper import lapjv_match


# ---- Determinism helpers (best-effort) ----
def set_determinism(seed: int, faiss_threads: int = 1):
    """
    Best-effort reproducibility knobs:
    - numpy RNG seed
    - FAISS OpenMP threads set to 1 (if faiss is installed)
    """
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
    """
    Run the 3 solvers on one dataset (A,B) and return total costs.
    We cast to float32 to reduce cross-machine dtype drift.
    """
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)

    # (1) GBCM / FF greedy
    _, cost_gbcm, _ = shortest_edge_first_greedy_matching_faiss_general(
        A, B, k=k, extend_search=extend_search, verbose=False
    )

    # (2) SLBM (injective ANN)
    _, cost_slbm, _ = match_approx_nn_injective(
        A, B, k=k, extend_search=extend_search, strict=strict
    )

    # (3) OPT (LAPJV)
    _, cost_opt, _ = lapjv_match(A, B)

    return float(cost_gbcm), float(cost_slbm), float(cost_opt)


def _make_run_name(run_name: str | None):
    """
    If user doesn't provide a run name, generate a timestamp-based name
    so files are never overwritten between runs.
    """
    if run_name and run_name.strip():
        return run_name.strip()
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def main():
    ap = argparse.ArgumentParser()

    # Inputs
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

    # Plot
    ap.add_argument("--xscale", choices=["linear", "log"], default="linear")

    # Outputs (NEW)
    ap.add_argument("--out-dir", type=str, default="cost_ratio",
                    help="Output directory for PNG/CSV/NPZ. Default: cost_ratio")
    ap.add_argument("--run-name", type=str, default="",
                    help="Optional run name to prefix outputs. If empty, uses timestamp.")
    ap.add_argument("--out-name", type=str, default="cost_ratio",
                    help="Base name (without extension) for outputs. Default: cost_ratio")

    # Debug bundles
    ap.add_argument("--save-metrics", action="store_true", default=True,
                    help="Save NPZ debug bundle. Default: true.")
    ap.add_argument("--no-save-metrics", dest="save_metrics", action="store_false")

    # Safety
    ap.add_argument("--max-lapjv-n", type=int, default=10000,
                    help="Safety guard: skip N larger than this to avoid O(N^2) LAPJV blow-up.")

    args = ap.parse_args()

    run_name = _make_run_name(args.run_name)

    # Ensure output dir exists
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Unique output paths (never overwrite)
    png_path = os.path.join(out_dir, f"{args.out_name}_{run_name}.png")
    csv_path = os.path.join(out_dir, f"{args.out_name}_{run_name}.csv")
    npz_path = os.path.join(out_dir, f"{args.out_name}_{run_name}.npz")

    xs = []
    gbcm_ratios, slbm_ratios = [], []
    # raw dict for debugging (N -> per-seed lists)
    raw = {}

    for n in args.Ns:
        if n > args.max_lapjv_n:
            print(f"[SKIP] N={n} > max_lapjv_n={args.max_lapjv_n} (LAPJV O(N^2) too big).")
            continue

        ratios_g, ratios_s = [], []
        costs_g, costs_s, costs_o = [], [], []
        src_dirs = set()

        for seed in args.seeds:
            # Make results as stable as possible
            set_determinism(seed, faiss_threads=1)

            # Load data
            A, B, src = load_AB(args.data_root, seed, n)
            src_dirs.add(src)

            # Run solvers
            c_g, c_s, c_o = run_one_seed(A, B, k=args.k,
                                         extend_search=args.extend_search,
                                         strict=args.strict)

            # Ratios
            ratios_g.append(c_g / c_o)
            ratios_s.append(c_s / c_o)

            # Costs
            costs_g.append(c_g)
            costs_s.append(c_s)
            costs_o.append(c_o)

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
            gbcm_mean=g_mean,
            gbcm_std=g_std,
            slbm_mean=s_mean,
            slbm_std=s_std,
        )

        print(f"N={n} | data={'; '.join(sorted(src_dirs))}")
        print(f"  GBCM ratio mean±std = {g_mean:.6f} ± {g_std:.6f}")
        print(f"  SLBM ratio mean±std = {s_mean:.6f} ± {s_std:.6f}")

    if not xs:
        raise RuntimeError("No N values were run. Check --Ns and --max-lapjv-n.")

    # ---------- Save CSV (one file) ----------
    # One row per (N, seed) with repeated mean/std so you can filter/group easily later.
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "run_name", "data_root", "N", "seed",
            "src_dir",
            "k", "extend_search", "strict",
            "cost_gbcm", "cost_slbm", "cost_opt",
            "ratio_gbcm", "ratio_slbm",
            "gbcm_ratio_mean", "gbcm_ratio_std",
            "slbm_ratio_mean", "slbm_ratio_std",
        ])

        for n in xs:
            r = raw[str(n)]
            for i, seed in enumerate(r["seeds"]):
                w.writerow([
                    run_name, os.path.abspath(args.data_root), int(n), int(seed),
                    ";".join(r["src_dirs"]),
                    int(args.k), bool(args.extend_search), bool(args.strict),
                    float(r["cost_gbcm"][i]), float(r["cost_slbm"][i]), float(r["cost_opt"][i]),
                    float(r["ratio_gbcm"][i]), float(r["ratio_slbm"][i]),
                    float(r["gbcm_mean"]), float(r["gbcm_std"]),
                    float(r["slbm_mean"]), float(r["slbm_std"]),
                ])

    print(f"saved csv -> {csv_path}")

    # ---------- Plot ----------
    xs_arr = np.array(xs, dtype=int)
    g_mean_arr = np.array([m for m, s in gbcm_ratios], dtype=float)
    g_std_arr  = np.array([s for m, s in gbcm_ratios], dtype=float)
    s_mean_arr = np.array([m for m, s in slbm_ratios], dtype=float)
    s_std_arr  = np.array([s for m, s in slbm_ratios], dtype=float)

    plt.figure(figsize=(10, 7))
    plt.errorbar(xs_arr, g_mean_arr, yerr=g_std_arr, marker="o", capsize=6, label="GBCM")
    plt.errorbar(xs_arr, s_mean_arr, yerr=s_std_arr, marker="o", capsize=6, label="SLBM (injective)")
    if args.xscale == "log":
        plt.xscale("log")
    plt.title("Cost ratio vs N (mean ± std over seeds)")
    plt.xlabel("N")
    plt.ylabel("cost / LAPJV_opt_cost")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    print(f"saved plot -> {png_path}")

    # ---------- Save debugging bundle (NPZ) ----------
    if args.save_metrics:
        np.savez(npz_path,
                 run_name=run_name,
                 xs=xs_arr,
                 gbcm_mean=g_mean_arr, gbcm_std=g_std_arr,
                 slbm_mean=s_mean_arr, slbm_std=s_std_arr,
                 raw=raw)
        print(f"saved metrics -> {npz_path}")


if __name__ == "__main__":
    main()
