import numpy as np
import faiss
import time
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def match_approx_nn(
        A: np.ndarray,
        B: np.ndarray,
        k: int = 10,
        extend_search: bool = True
) -> Tuple[List[Tuple[int, int]], float, Dict]:
    """
    Approximate Nearest Neighbor (ANN) matching:
    iterate A, pick nearest unused B among top-k candidates.
    Returns: matching_pairs, final_cost (sum of euclidean), timing_stats.
    """
    if A.shape[1] != B.shape[1]:
        raise ValueError("Input A and B must have same dimension.")

    N_A = A.shape[0]
    N_B = B.shape[0]
    D = A.shape[1]

    timing_stats = {}
    total_start_time = time.time()

    # Faiss needs float32
    A = np.ascontiguousarray(A.astype('float32'))
    B = np.ascontiguousarray(B.astype('float32'))

    # 1. Faiss k-NN (returns d^2)
    search_start_time = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] 1. Faiss k-NN search...")
    index = faiss.IndexFlatL2(D)
    index.add(B)
    distances_sq, indices = index.search(A, k)
    timing_stats['faiss_search_s'] = time.time() - search_start_time
    print(f"[{time.strftime('%H:%M:%S')}] Faiss search done. time: {timing_stats['faiss_search_s']:.4f}s")

    # 2. greedy assignment over top-k
    match_start_time = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] 2. greedy matching...")
    A_matched = np.zeros(N_A, dtype=bool)
    B_matched = np.zeros(N_B, dtype=bool)
    matching_pairs = []
    matched_dists_squared = []
    for i in range(N_A):
        if A_matched[i]:
            continue
        best_dist_sq = np.inf
        best_b_idx = -1
        for j_idx in range(k):
            b_idx = indices[i, j_idx]
            dist_sq = distances_sq[i, j_idx]
            if not B_matched[b_idx] and dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_b_idx = b_idx
        if best_b_idx != -1:
            A_matched[i] = True
            B_matched[best_b_idx] = True
            matching_pairs.append((i, int(best_b_idx)))
            matched_dists_squared.append(best_dist_sq)
    timing_stats['greedy_matching_s'] = time.time() - match_start_time
    print(f"[{time.strftime('%H:%M:%S')}] greedy done. matched {len(matching_pairs)} pairs. "
          f"time: {timing_stats['greedy_matching_s']:.4f}s")

    # 3. extend search if needed (one-to-one)
    if len(matching_pairs) < N_A and extend_search:
        extend_start_time = time.time()
        unmatched_A_indices = np.where(~A_matched)[0]
        unmatched_B_indices = np.where(~B_matched)[0]

        if len(unmatched_B_indices) > 0 and len(unmatched_A_indices) > 0:
            print(f"[{time.strftime('%H:%M:%S')}] 3. extend {len(unmatched_A_indices)} unmatched A points.")

            B_unmatched = B[unmatched_B_indices]
            index_unmatched = faiss.IndexFlatL2(D)
            index_unmatched.add(B_unmatched)

            A_unmatched_coords = A[unmatched_A_indices]

            # 搜 top-k（至少要 >1 才能在“最近邻已被占用”时 fallback）
            k_ext = min(k, len(unmatched_B_indices))
            dists_sq_unmatched, idx_in_unmatched = index_unmatched.search(A_unmatched_coords, k_ext)

            # 记录 unmatched_B 池里哪些已经被占用（局部）
            B_unmatched_taken = np.zeros(len(unmatched_B_indices), dtype=bool)

            for row_i, a_idx in enumerate(unmatched_A_indices):
                chosen = False
                for j in range(k_ext):
                    b_local = int(idx_in_unmatched[row_i, j])
                    if b_local < 0:
                        continue
                    if not B_unmatched_taken[b_local]:
                        B_unmatched_taken[b_local] = True

                        b_idx_original = int(unmatched_B_indices[b_local])
                        dist_squared = float(dists_sq_unmatched[row_i, j])

                        matching_pairs.append((int(a_idx), b_idx_original))
                        matched_dists_squared.append(dist_squared)

                        # 同步更新全局 matched，确保后续仍是一对一
                        A_matched[a_idx] = True
                        B_matched[b_idx_original] = True

                        chosen = True
                        break

                # 如果没选到（比如 unmatched_B 不够），就保持该 a_idx 未匹配
                if not chosen:
                    pass

            timing_stats['extend_search_s'] = time.time() - extend_start_time
            print(f"[{time.strftime('%H:%M:%S')}] extend done. total pairs: {len(matching_pairs)}. "
                  f"time: {timing_stats['extend_search_s']:.4f}s")
        else:
            print("Warning: no unmatched B available for extension.")

    # 4. final euclidean cost
    cost_calc_start = time.time()
    final_euclidean_cost = np.sum(np.sqrt(np.array(matched_dists_squared)))
    timing_stats['final_cost_calc_s'] = time.time() - cost_calc_start
    timing_stats['total_time_s'] = time.time() - total_start_time
    print(f"\n[{time.strftime('%H:%M:%S')}] --- total time: {timing_stats['total_time_s']:.4f}s ---")
    return matching_pairs, final_euclidean_cost, timing_stats


def generate_ring_data(N: int = 50, R1: float = 0.5, R2: float = 3.2,
                       sigma: float = 0.1, center_offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Generate two ring-shaped 2D point sets; parameters align with FF_GM_fullmatch for consistent plots."""
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radii_A = R1 + np.random.normal(0, sigma, N)
    A = np.zeros((N, 2))
    A[:, 0] = radii_A * np.cos(angles)
    A[:, 1] = radii_A * np.sin(angles)
    radii_B = R2 + np.random.normal(0, sigma, N)
    B = np.zeros((N, 2))
    B[:, 0] = radii_B * np.cos(angles) + center_offset
    B[:, 1] = radii_B * np.sin(angles)
    np.random.shuffle(B)
    return A.astype(np.float32), B.astype(np.float32)


def plot_matching_result(A: np.ndarray, B: np.ndarray, matching_pairs: list,
                         title: str = "ANN Greedy Matching on Two Rings"):
    """
    Visualize matching result; point size and aspect match FF_GM_fullmatch.
    """
    if A.shape[1] != 2 or B.shape[1] != 2:
        raise ValueError("Visualization only supports 2D points.")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(A[:, 0], A[:, 1], c='blue', s=1, label='Source samples')
    ax.scatter(B[:, 0], B[:, 1], c='red', s=1, label='Target samples')

    line_style = {'color': 'gray', 'alpha': 0.5, 'linewidth': 0.8, 'linestyle': '-'}
    for a_idx, b_idx in matching_pairs:
        x_start, y_start = A[a_idx]
        x_end, y_end = B[b_idx]
        ax.add_line(mlines.Line2D([x_start, x_end], [y_start, y_end], **line_style))

    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    source_legend = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                                  markersize=8, label='Source samples')
    target_legend = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                                  markersize=8, label='Target samples')
    ax.legend(handles=[source_legend, target_legend], loc='center')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


# --- example usage ---
if __name__ == '__main__':
    # ring demo; same scale/point size as FF_GM_fullmatch visualization
    N_SAMPLES = 1000
    A_ring, B_ring = generate_ring_data(
        N=N_SAMPLES,
        R1=0.5,
        R2=3.2,
        sigma=0.1,
        center_offset=0.1
    )
    print(f"generated A: {A_ring.shape}, B: {B_ring.shape}")

    K_CANDIDATES = 5
    match_result, total_cost, timings = match_approx_nn(
        A_ring,
        B_ring,
        k=K_CANDIDATES,
        extend_search=True
    )
    print(f"\nmatch total cost: {total_cost:.4f}")
    print(f"pairs: {len(match_result)}")

    plot_matching_result(
        A_ring,
        B_ring,
        match_result,
        title=f"ANN Greedy Matching on Rings (k={K_CANDIDATES}, Cost={total_cost:.2f})"
    )
