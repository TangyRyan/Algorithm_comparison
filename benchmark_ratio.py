import os
import numpy as np

from FF_GM_fullmatch import shortest_edge_first_greedy_matching_faiss_general
from ANN_injective import match_approx_nn_injective as match_approx_nn
from lapjv_wrapper import lapjv_match

def load_AB(root, N):
    d = os.path.join(root, str(N))
    A = np.load(os.path.join(d, "A.npy"))
    B = np.load(os.path.join(d, "B.npy"))
    return A, B

def run_one(N, data_root="data", k=10, extend_search=True):
    A, B = load_AB(data_root, N)

    _, cost_gbcm, t_gbcm = shortest_edge_first_greedy_matching_faiss_general(A, B, k=k, extend_search=extend_search)
    _, cost_slbm, t_slbm = match_approx_nn(A, B, k=k, extend_search=extend_search)

    _, cost_opt, t_opt = lapjv_match(A, B)

    print(f"N={N}")
    print(f"  LAPJV(opt) cost={cost_opt:.6f}, time={t_opt['total_time_s']:.3f}s")
    print(f"  GBCM      cost={cost_gbcm:.6f}, ratio={cost_gbcm/cost_opt:.6f}, time={t_gbcm['total_time_s']:.3f}s")
    print(f"  SLBM      cost={cost_slbm:.6f}, ratio={cost_slbm/cost_opt:.6f}, time={t_slbm['total_time_s']:.3f}s")

if __name__ == "__main__":
    # 你说主要看 5000：就跑它
    run_one(5000)

    # 可选 sanity：
    # run_one(1000)
    # run_one(2000)
