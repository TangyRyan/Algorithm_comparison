import numpy as np
import faiss
import time
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def match_approx_nn_injective(
        A: np.ndarray,
        B: np.ndarray,
        k: int = 10,
        extend_search: bool = True,
        extend_k: int | None = None,
        strict: bool = True
) -> Tuple[List[Tuple[int, int]], float, Dict]:
    """
    Approximate Nearest Neighbor (ANN) Matching，保证得到 A -> B 的一对一映射：
    - A 中每个点都有且仅有一个 B 点（A 全覆盖）
    - 每个 B 点最多被一个 A 点使用（B 唯一）
    - 若 N_A > N_B 且 strict=True，会直接报错（无法实现一对一全覆盖）

    逻辑仍沿用原文件：
    1) 先对全部 B 做 faiss kNN，逐行贪心挑“最近且未占用”的候选
    2) 如有未匹配 A，则只在未占用的 B 上做扩展匹配，但会显式避免冲突

    Args:
        A, B: 点集，shape 分别为 (N_A, D)、(N_B, D)
        k: 第一阶段候选数
        extend_search: 是否在第一阶段后补齐
        extend_k: 第二阶段候选数（None 则自动设为 max(k, 50) 且不超过未匹配 B 数）
        strict: 若无法完成一对一全覆盖是否报错

    Returns:
        matching_pairs: (A_idx, B_idx) 列表，长度应为 N_A（strict=True 时）
        final_euclidean_cost: 匹配欧氏距离总和
        timing_stats: 耗时统计
    """
    if A.shape[1] != B.shape[1]:
        raise ValueError("输入点集 A 和 B 必须具有相同的维度 D。")

    N_A = A.shape[0]
    N_B = B.shape[0]
    D = A.shape[1]

    if strict and N_A > N_B:
        raise ValueError(f"无法实现 A->B 的一对一全覆盖：N_A={N_A} > N_B={N_B}。")

    timing_stats: Dict[str, float] = {}
    total_start_time = time.time()

    # Faiss 需要 float32 类型
    A = np.ascontiguousarray(A.astype('float32'))
    B = np.ascontiguousarray(B.astype('float32'))

    # --- 1. Faiss k-NN 搜索 (返回 d^2) ---
    search_start_time = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] 1. Faiss $k$-NN 搜索...")

    index = faiss.IndexFlatL2(D)
    index.add(B)

    distances_sq, indices = index.search(A, min(k, N_B))

    timing_stats['faiss_search_s'] = time.time() - search_start_time
    print(f"[{time.strftime('%H:%M:%S')}] Faiss 搜索完成。耗时: {timing_stats['faiss_search_s']:.4f}s")

    # --- 2. 逐行简单贪心匹配 ---
    match_start_time = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] 2. 逐行简单贪心匹配...")

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
    print(f"[{time.strftime('%H:%M:%S')}] 贪心匹配完成。已匹配 {len(matching_pairs)} 对。耗时: {timing_stats['greedy_matching_s']:.4f}s")

    # --- 3. 补齐未匹配点：保证一对一 ---
    if len(matching_pairs) < N_A and extend_search:
        extend_start_time = time.time()

        unmatched_A_indices = np.where(~A_matched)[0]
        unmatched_B_indices = np.where(~B_matched)[0]

        if len(unmatched_B_indices) == 0:
            msg = "没有未匹配的 B 点可用于补齐。"
            if strict:
                raise RuntimeError(msg)
            print("警告：" + msg)
        else:
            print(f"[{time.strftime('%H:%M:%S')}] 3. 补齐 {len(unmatched_A_indices)} 个未匹配 A 点 (一对一)...")

            B_unmatched = B[unmatched_B_indices]
            index_unmatched = faiss.IndexFlatL2(D)
            index_unmatched.add(B_unmatched)

            A_unmatched_coords = A[unmatched_A_indices]

            k2 = extend_k
            if k2 is None:
                k2 = max(k, 50)
            k2 = int(min(max(1, k2), len(unmatched_B_indices)))

            # 先取 k2 个候选，再逐点挑“最近且未占用”的
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

                # 如果在 k2 候选内都被占用：对该点做一次“全量候选”兜底（只在未匹配集上）
                if chosen_local == -1 and len(unmatched_B_indices) > k2:
                    d_full, idx_full = index_unmatched.search(A[a_idx:a_idx + 1], len(unmatched_B_indices))
                    for j in range(len(unmatched_B_indices)):
                        b_local = int(idx_full[0, j])
                        if not used_local_B[b_local]:
                            chosen_local = b_local
                            chosen_dist = float(d_full[0, j])
                            break

                if chosen_local == -1:
                    msg = f"无法为 A[{a_idx}] 找到未占用的 B 点（可能 N_A>N_B 或 extend_k 太小）。"
                    if strict:
                        raise RuntimeError(msg)
                    print("警告：" + msg)
                    continue

                used_local_B[chosen_local] = True
                b_idx_original = int(unmatched_B_indices[chosen_local])

                A_matched[a_idx] = True
                B_matched[b_idx_original] = True
                matching_pairs.append((a_idx, b_idx_original))
                matched_dists_squared.append(chosen_dist)

            timing_stats['extend_search_s'] = time.time() - extend_start_time
            print(f"[{time.strftime('%H:%M:%S')}] 补齐完成。总匹配对数: {len(matching_pairs)}。耗时: {timing_stats['extend_search_s']:.4f}s")

    # --- 4. 校验“单射 + 全覆盖”条件 ---
    if strict:
        if len(matching_pairs) != N_A:
            raise RuntimeError(f"最终匹配对数 {len(matching_pairs)} != N_A={N_A}，未能实现 A 全覆盖。")
        b_ids = [b for _, b in matching_pairs]
        if len(set(b_ids)) != len(b_ids):
            raise RuntimeError("检测到重复使用的 B 点，未满足单射/一对一约束。")

    # --- 5. 计算最终总欧氏距离之和 (Cost) ---
    cost_calc_start = time.time()
    final_euclidean_cost = float(np.sum(np.sqrt(np.array(matched_dists_squared, dtype=np.float64))))
    timing_stats['final_cost_calc_s'] = time.time() - cost_calc_start
    timing_stats['total_time_s'] = time.time() - total_start_time

    print(f"\n[{time.strftime('%H:%M:%S')}] --- 整个算法总耗时: {timing_stats['total_time_s']:.4f}s ---")

    return matching_pairs, final_euclidean_cost, timing_stats


def generate_ring_data(N: int = 50, R1: float = 0.5, R2: float = 3.2,
                       sigma: float = 0.1, center_offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """生成两圈 2D 点集，参数与 FF_GM_fullmatch/ANN 可视化保持一致，便于对比。"""
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
                         title: str = "Injective ANN Matching on Two Rings"):
    """绘制匹配结果，点尺寸/比例与 FF_GM_fullmatch/ANN 版本保持一致。仅支持 2D 数据。"""
    if A.shape[1] != 2 or B.shape[1] != 2:
        raise ValueError("仅支持二维点的可视化。")

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


if __name__ == '__main__':
    # 可视化示例：使用环形数据，保证点尺寸与比例与其他文件一致
    N_SAMPLES = 1000
    A_ring, B_ring = generate_ring_data(
        N=N_SAMPLES,
        R1=0.5,
        R2=3.2,
        sigma=0.1,
        center_offset=0.1
    )

    print(f"生成的点集 A: {A_ring.shape}, B: {B_ring.shape}")

    K_CANDIDATES = 5
    match_result, total_cost, timings = match_approx_nn_injective(
        A_ring,
        B_ring,
        k=K_CANDIDATES,
        extend_search=True
    )

    print("\n--- 匹配结果摘要 ---")
    print(f"总匹配对数: {len(match_result)}")
    print(f"总欧氏距离 (Cost): {total_cost:.4f}")

    print("\n--- 耗时统计 ---")
    for key, value in timings.items():
        print(f"{key:<20}: {value:.4f}s")

    plot_matching_result(
        A_ring,
        B_ring,
        match_result,
        title=f"Injective ANN Matching on Rings (k={K_CANDIDATES}, Cost={total_cost:.2f})"
    )
