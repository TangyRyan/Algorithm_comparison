import time
import numpy as np
from lapjv import lapjv

def lapjv_match(A: np.ndarray, B: np.ndarray):
    """
    返回形式对齐你的两个文件：
      matching_pairs: List[(a_idx, b_idx)]
      final_cost: sum of Euclidean distances
      timing_stats: dict with *_s keys, including total_time_s
    """
    if A.shape != B.shape:
        raise ValueError("你的实验设定是 A,B 同维同数量点；若不等需要 padding，这里先直接报错。")

    # 对齐你文件：float32 + contiguous
    A = np.ascontiguousarray(A.astype(np.float32))
    B = np.ascontiguousarray(B.astype(np.float32))
    N, D = A.shape

    timing_stats = {}
    total_start = time.time()

    # 1) 构造 cost_matrix：欧氏距离矩阵（非常关键：必须是距离，不是距离平方）
    t0 = time.time()

    # 用公式避免 (N,N,2) 这种三维广播爆内存：
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    AA = np.sum(A * A, axis=1, keepdims=True)          # (N,1)
    BB = np.sum(B * B, axis=1, keepdims=True).T        # (1,N)
    C2 = AA + BB - 2.0 * (A @ B.T)                     # (N,N)
    np.maximum(C2, 0.0, out=C2)                        # 数值误差保护
    C = np.sqrt(C2, dtype=np.float32)                  # (N,N) 欧氏距离

    timing_stats["cost_matrix_build_s"] = time.time() - t0

    # 2) LAPJV 求解
    t1 = time.time()
    row_ind, col_ind, cost = lapjv(C)
    timing_stats["lapjv_solve_s"] = time.time() - t1

    # 3) Final Cost：按你文件的逻辑，显式求和（也方便 double-check）
    t2 = time.time()
    final_cost = float(np.sum(C[np.arange(N), row_ind]))
    timing_stats["final_cost_calc_s"] = time.time() - t2

    timing_stats["total_time_s"] = time.time() - total_start

    matching_pairs = [(i, int(row_ind[i])) for i in range(N)]
    return matching_pairs, final_cost, timing_stats
