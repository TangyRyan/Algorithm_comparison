import numpy as np
import faiss
import heapq
import time
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import ot  # POT: Python Optimal Transport
from typing import List, Tuple, Dict, Optional


# ==========================================
# 1. 数据生成工具
# ==========================================
def generate_ring_data(N: int = 50, R1: float = 0.5, R2: float = 3.2,
                       sigma: float = 0.1, center_offset: float = 0.0):
    np.random.seed(42)
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


def plot_matching_result(A, B, matching_pairs, title, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(A[:, 0], A[:, 1], c='blue', s=5, label='Source', alpha=0.6)
    ax.scatter(B[:, 0], B[:, 1], c='red', s=5, label='Target', alpha=0.6)

    line_style = {'color': 'gray', 'alpha': 0.5, 'linewidth': 0.5}
    for a_idx, b_idx in matching_pairs:
        x_start, y_start = A[a_idx]
        x_end, y_end = B[b_idx]
        ax.plot([x_start, x_end], [y_start, y_end], **line_style)

    ax.set_title(title, fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)


# ==========================================
# 2. ANN Injective (Full Match Version)
#    来自 ANN_injective.py 的完整逻辑 (含兜底)
# ==========================================
def match_approx_nn_injective(
        A: np.ndarray,
        B: np.ndarray,
        k: int = 10,
        extend_search: bool = True,
        extend_k: Optional[int] = None,
        strict: bool = False  # 这里默认改为 False 以便在对比脚本中不中断，只报警告
) -> Tuple[List[Tuple[int, int]], float, Dict]:
    """
    Approximate Nearest Neighbor (ANN) Matching，保证得到 A -> B 的一对一映射：
    - A 中每个点都有且仅有一个 B 点（A 全覆盖）
    - 每个 B 点最多被一个 A 点使用（B 唯一）
    """
    if A.shape[1] != B.shape[1]:
        raise ValueError("输入点集 A 和 B 必须具有相同的维度 D。")

    N_A = A.shape[0]
    N_B = B.shape[0]
    D = A.shape[1]

    timing_stats: Dict[str, float] = {}
    total_start_time = time.time()

    # Faiss 需要 float32 类型
    A = np.ascontiguousarray(A.astype('float32'))
    B = np.ascontiguousarray(B.astype('float32'))

    # --- 1. Faiss k-NN 搜索 (返回 d^2) ---
    search_start_time = time.time()

    index = faiss.IndexFlatL2(D)
    index.add(B)

    distances_sq, indices = index.search(A, min(k, N_B))

    timing_stats['faiss_search_s'] = time.time() - search_start_time

    # --- 2. 逐行简单贪心匹配 ---
    match_start_time = time.time()

    A_matched = np.zeros(N_A, dtype=bool)
    B_matched = np.zeros(N_B, dtype=bool)

    matching_pairs: List[Tuple[int, int]] = []
    matched_dists_squared: List[float] = []

    for i in range(N_A):
        if A_matched[i]:
            continue

        best_dist_sq = np.inf
        best_b_idx = -1

        # 遍历候选近邻
        for j_idx in range(distances_sq.shape[1]):
            b_idx = int(indices[i, j_idx])
            dist_sq = float(distances_sq[i, j_idx])

            if not B_matched[b_idx] and dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_b_idx = b_idx

        if best_b_idx != -1:
            A_matched[i] = True
            B_matched[best_b_idx] = True
            matching_pairs.append((int(i), int(best_b_idx)))
            matched_dists_squared.append(best_dist_sq)

    timing_stats['greedy_matching_s'] = time.time() - match_start_time

    # --- 3. 补齐未匹配点：保证一对一 ---
    if len(matching_pairs) < N_A and extend_search:
        extend_start_time = time.time()

        unmatched_A_indices = np.where(~A_matched)[0]
        unmatched_B_indices = np.where(~B_matched)[0]

        if len(unmatched_B_indices) > 0:
            B_unmatched = B[unmatched_B_indices]
            index_unmatched = faiss.IndexFlatL2(D)
            index_unmatched.add(B_unmatched)

            A_unmatched_coords = A[unmatched_A_indices]

            k2 = extend_k
            if k2 is None:
                k2 = max(k, 50)
            k2 = int(min(max(1, k2), len(unmatched_B_indices)))

            # 先取 k2 个候选
            dists_sq_unmatched, idx_in_unmatched = index_unmatched.search(A_unmatched_coords, k2)

            # 为了降低冲突：优先处理“最容易匹配”(最近候选距离更小) 的点
            order = np.argsort(dists_sq_unmatched[:, 0])

            used_local_B = np.zeros(len(unmatched_B_indices), dtype=bool)

            for r in order:
                a_idx = int(unmatched_A_indices[r])

                chosen_local = -1
                chosen_dist = np.inf

                for j in range(k2):
                    b_local = int(idx_in_unmatched[r, j])
                    if not used_local_B[b_local]:
                        chosen_local = b_local
                        chosen_dist = float(dists_sq_unmatched[r, j])
                        break

                # 【核心修改点】如果在 k2 候选内都被占用：对该点做一次“全量候选”兜底
                if chosen_local == -1 and len(unmatched_B_indices) > k2:
                    d_full, idx_full = index_unmatched.search(A[a_idx:a_idx + 1], len(unmatched_B_indices))
                    for j in range(len(unmatched_B_indices)):
                        b_local = int(idx_full[0, j])
                        if not used_local_B[b_local]:
                            chosen_local = b_local
                            chosen_dist = float(d_full[0, j])
                            break

                if chosen_local != -1:
                    used_local_B[chosen_local] = True
                    b_idx_original = int(unmatched_B_indices[chosen_local])

                    A_matched[a_idx] = True
                    B_matched[b_idx_original] = True
                    matching_pairs.append((a_idx, b_idx_original))
                    matched_dists_squared.append(chosen_dist)

            timing_stats['extend_search_s'] = time.time() - extend_start_time

    # --- 4. 校验（可选）---
    if strict and len(matching_pairs) != N_A:
        print(f"[Warning] ANN Injective 未能完全覆盖: {len(matching_pairs)}/{N_A}")

    # --- 5. 计算最终总欧氏距离之和 (Cost) ---
    cost_calc_start = time.time()
    final_euclidean_cost = float(np.sum(np.sqrt(np.array(matched_dists_squared, dtype=np.float64))))
    timing_stats['final_cost_calc_s'] = time.time() - cost_calc_start
    total_time = time.time() - total_start_time
    timing_stats['total_time_s'] = total_time

    return matching_pairs, final_euclidean_cost, total_time


# ==========================================
# 3. FF-GM Full Match (Edge-based Greedy)
#    完全移植自 FF_GM_fullmatch.py
# ==========================================
def ff_gm_fullmatch(A, B, k=10, extend_search=True, verbose=False):
    N_A = A.shape[0]
    N_B = B.shape[0]
    D = A.shape[1]
    target_pairs = min(N_A, N_B)

    total_start_time = time.time()

    # 类型转换
    A = np.ascontiguousarray(A.astype('float32'))
    B = np.ascontiguousarray(B.astype('float32'))

    # --- 1. Faiss Search ---
    index = faiss.IndexFlatL2(D)
    index.add(B)
    distances_sq, indices = index.search(A, k)

    # --- 2. Build Min-Heap (Global Edges) ---
    min_heap = []
    for i in range(N_A):
        for j_idx in range(k):
            b_idx = indices[i, j_idx]
            dist_squared = distances_sq[i, j_idx]
            heapq.heappush(min_heap, (dist_squared, i, b_idx))

    # --- 3. Greedy Matching ---
    A_matched = np.zeros(N_A, dtype=bool)
    B_matched = np.zeros(N_B, dtype=bool)
    matching_pairs = []
    matched_dists_squared = []

    while min_heap and len(matching_pairs) < target_pairs:
        dist_squared, a_idx, b_idx = heapq.heappop(min_heap)
        if not A_matched[a_idx] and not B_matched[b_idx]:
            A_matched[a_idx] = True
            B_matched[b_idx] = True
            matching_pairs.append((int(a_idx), int(b_idx)))
            matched_dists_squared.append(dist_squared)

    # --- 4. Extend Search (补齐逻辑) ---
    if len(matching_pairs) < target_pairs and extend_search:
        rounds = 0
        max_rounds = 6
        k_ext = max(k, 20)

        while len(matching_pairs) < target_pairs and rounds < max_rounds:
            unmatched_A_indices = np.where(~A_matched)[0]
            unmatched_B_indices = np.where(~B_matched)[0]

            if len(unmatched_A_indices) == 0 or len(unmatched_B_indices) == 0:
                break

            k_eff = int(min(k_ext, len(unmatched_B_indices)))

            # 对未匹配部分重新建索引
            B_unmatched = B[unmatched_B_indices]
            index_unmatched = faiss.IndexFlatL2(D)
            index_unmatched.add(B_unmatched)

            A_unmatched_coords = A[unmatched_A_indices]
            dists_sq_unmatched, idxs_unmatched = index_unmatched.search(A_unmatched_coords, k_eff)

            extend_heap = []
            for ui, a_idx in enumerate(unmatched_A_indices):
                for t in range(k_eff):
                    b_local = int(idxs_unmatched[ui, t])
                    if b_local < 0: continue
                    dist_sq = float(dists_sq_unmatched[ui, t])
                    b_idx_original = int(unmatched_B_indices[b_local])
                    heapq.heappush(extend_heap, (dist_sq, int(a_idx), b_idx_original))

            before = len(matching_pairs)
            while extend_heap and len(matching_pairs) < target_pairs:
                dist_sq, a_idx, b_idx = heapq.heappop(extend_heap)
                if (not A_matched[a_idx]) and (not B_matched[b_idx]):
                    A_matched[a_idx] = True
                    B_matched[b_idx] = True
                    matching_pairs.append((int(a_idx), int(b_idx)))
                    matched_dists_squared.append(dist_sq)

            if len(matching_pairs) - before == 0:
                k_ext = min(len(unmatched_B_indices), max(k_eff * 2, k_eff + 10))
            else:
                k_ext = min(len(unmatched_B_indices), max(k_ext, k_eff * 2))
            rounds += 1

        # --- 4.F 强制兜底 (Forced Fallback) ---
        if len(matching_pairs) < target_pairs:
            unmatched_A_indices = np.where(~A_matched)[0]
            unmatched_B_indices = np.where(~B_matched)[0]

            B_rem = B[unmatched_B_indices]
            # 这里为了效率，对剩余点逐个找最近的
            for a_idx in unmatched_A_indices:
                if len(unmatched_B_indices) == 0: break

                # 为单个点找最近邻 (保证能找到)
                diff = B_rem - A[a_idx]
                d2_all = np.sum(diff ** 2, axis=1)
                pos = np.argmin(d2_all)

                b_idx = int(unmatched_B_indices[pos])
                A_matched[a_idx] = True
                B_matched[b_idx] = True
                matching_pairs.append((int(a_idx), b_idx))
                matched_dists_squared.append(d2_all[pos])

                unmatched_B_indices = np.delete(unmatched_B_indices, pos)
                B_rem = np.delete(B_rem, pos, axis=0)

    total_time = time.time() - total_start_time
    # 计算 Cost (Sum of Euclidean Distances)
    cost = np.sum(np.sqrt(matched_dists_squared))

    return matching_pairs, cost, total_time


# ==========================================
# 4. Exact OT (POT Library)
# ==========================================
def run_exact_ot(A, B):
    N = A.shape[0]
    start_time = time.time()

    # 计算距离矩阵
    M = ot.dist(A, B, metric='euclidean')
    a = np.ones((N,)) / N
    b = np.ones((N,)) / N

    # 求解 EMD
    P = ot.emd(a, b, M)
    total_time = time.time() - start_time

    # 解析结果
    rows, cols = np.where(P > 1e-8)
    pairs = list(zip(rows, cols))

    # 计算 Sum Cost
    cost_sum = 0.0
    for r, c in pairs:
        cost_sum += M[r, c]

    return pairs, cost_sum, total_time


# ==========================================
# 5. Partial OT (POT Library)
# ==========================================
def run_partial_ot(A, B, fraction=0.8):
    N = A.shape[0]
    start_time = time.time()

    M = ot.dist(A, B, metric='euclidean')
    a = np.ones((N,)) / N
    b = np.ones((N,)) / N

    # 求解 Partial OT (m = total mass to transport)
    P = ot.partial.partial_wasserstein(a, b, M, m=fraction)
    total_time = time.time() - start_time

    rows, cols = np.where(P > 1e-8)
    pairs = list(zip(rows, cols))

    cost_sum = 0.0
    for r, c in pairs:
        cost_sum += M[r, c]

    return pairs, cost_sum, total_time


# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # 1. 准备数据
    N_SAMPLES = 500
    print(f"Generating data with N={N_SAMPLES}...")
    # 使用稍微复杂的环形数据
    A, B = generate_ring_data(N=N_SAMPLES, R1=0.5, R2=3.5, sigma=0.1, center_offset=0.2)

    results = []

    # 2. ANN Injective (Full Match Version)
    print("Running ANN Injective (Full Match)...")
    # 注意：这里调用的是新版函数，默认开启 extend_search
    pairs_ann, cost_ann, time_ann = match_approx_nn_injective(A, B, k=10)
    results.append(("ANN Injective", len(pairs_ann), cost_ann, time_ann, pairs_ann))

    # 3. FF-GM Full Match (Edge-based Global)
    print("Running FF-GM FullMatch (Edge-based)...")
    pairs_ffgm, cost_ffgm, time_ffgm = ff_gm_fullmatch(A, B, k=10, extend_search=True, verbose=False)
    results.append(("FF-GM FullMatch", len(pairs_ffgm), cost_ffgm, time_ffgm, pairs_ffgm))

    # 4. Exact OT
    print("Running Exact OT (POT)...")
    pairs_ot, cost_ot, time_ot = run_exact_ot(A, B)
    results.append(("Exact OT", len(pairs_ot), cost_ot, time_ot, pairs_ot))

    # 5. Partial OT
    fraction = 0.8
    print(f"Running Partial OT (m={fraction})...")
    pairs_pot, cost_pot, time_pot = run_partial_ot(A, B, fraction=fraction)
    results.append((f"Partial OT ({fraction * 100:.0f}%)", len(pairs_pot), cost_pot, time_pot, pairs_pot))

    # 6. 打印统计表
    print("\n" + "=" * 85)
    print(f"{'Method':<25} | {'Matches':<10} | {'Total Cost (Sum)':<18} | {'Time (s)':<10}")
    print("-" * 85)
    for name, n_match, cost, t, _ in results:
        print(f"{name:<25} | {n_match:<10} | {cost:<18.4f} | {t:<10.4f}")
    print("=" * 85)

    # 7. 绘图
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    for idx, (name, n_match, cost, t, pairs) in enumerate(results):
        title = f"{name}\nN={n_match}, Cost={cost:.1f}, t={t:.3f}s"
        plot_matching_result(A, B, pairs, title, ax=axes[idx])

    plt.tight_layout()
    plt.show()