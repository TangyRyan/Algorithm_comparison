import numpy as np
import faiss
import heapq
import time
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Shortest Edge First Greedy Matching
# 注意：如果您的环境没有安装 faiss-cpu，请先安装：pip install faiss-cpu


def shortest_edge_first_greedy_matching_faiss_general(
        A: np.ndarray,
        B: np.ndarray,
        k: int = 10,
        extend_search: bool = True,
        verbose: bool = True
) -> Tuple[List[Tuple[int, int]], float, Dict]:
    """
    使用 Faiss 和最短边优先贪心策略实现任意维度 D 点集 A 和 B 之间的一一近似映射。

    该版本全程使用平方距离 (d^2) 驱动核心计算，适用于任意维度 D。

    Args:
        A (np.ndarray): 源 D 维点集，形状 (N_A, D)。
        B (np.ndarray): 目标 D 维点集，形状 (N_B, D)。
        k (int): 最近邻搜索的候选数量。
        extend_search (bool): 是否执行补齐搜索。

    Returns:
        Tuple[List[Tuple[int, int]], float, Dict]:
            - 匹配对 (A_idx, B_idx) 的列表
            - 匹配的欧氏距离之和 (Final Cost)
            - 包含各个阶段耗时的字典
    """
    if A.shape[1] != B.shape[1]:
        raise ValueError("输入点集 A 和 B 必须具有相同的维度 D。")

    N_A = A.shape[0]
    N_B = B.shape[0]
    D = A.shape[1]
    target_pairs = min(N_A, N_B)

    timing_stats = {}
    total_start_time = time.time()

    # Faiss 需要 float32 类型，确保数据兼容
    A = np.ascontiguousarray(A.astype('float32'))
    B = np.ascontiguousarray(B.astype('float32'))

    # --- 1. Faiss 索引构建与 $k$-NN 搜索 (返回 $d^2$) ---

    search_start_time = time.time()
    if verbose: print(f"[{time.strftime('%H:%M:%S')}] 1. Faiss 索引构建与 $k$-NN 搜索 (D={D})...")

    # IndexFlatL2: 适用于任意维度 D 的精确 L2 平方距离索引
    index = faiss.IndexFlatL2(D)
    index.add(B)

    # distances_sq 存储的是 $d^2$ (平方距离)
    distances_sq, indices = index.search(A, k)

    timing_stats['faiss_search_s'] = time.time() - search_start_time
    if verbose: print(f"[{time.strftime('%H:%M:%S')}] Faiss 搜索完成。耗时: {timing_stats['faiss_search_s']:.4f}s")

    # --- 2. 构建最小优先队列 (Min-Heap) - 使用 $d^2$ 作为优先级 ---

    heap_start_time = time.time()
    if verbose: print(f"[{time.strftime('%H:%M:%S')}] 2. 构建最小堆...")

    min_heap = []

    # 遍历所有 $N_A \times k$ 条候选边，并将其加入堆
    for i in range(N_A):
        for j_idx in range(k):
            b_idx = indices[i, j_idx]
            dist_squared = distances_sq[i, j_idx]

            # 堆中存储 (平方距离 $d^2$, A_idx, B_idx)
            heapq.heappush(min_heap, (dist_squared, i, b_idx))

    timing_stats['heap_build_s'] = time.time() - heap_start_time
    if verbose: print(f"[{time.strftime('%H:%M:%S')}] 最小堆构建完成。耗时: {timing_stats['heap_build_s']:.4f}s")

    # --- 3. 贪心匹配迭代 ---
    match_start_time = time.time()
    if verbose: print(f"[{time.strftime('%H:%M:%S')}] 3. 贪心匹配 ({len(min_heap)} 条边)...")

    A_matched = np.zeros(N_A, dtype=bool)
    B_matched = np.zeros(N_B, dtype=bool)

    matching_pairs = []
    matched_dists_squared = []  # 存储被接受匹配的 $d^2$

    while min_heap and len(matching_pairs) < target_pairs:
        dist_squared, a_idx, b_idx = heapq.heappop(min_heap)

        # 贪心决策：如果 A 点和 B 点都未被占用
        if not A_matched[a_idx] and not B_matched[b_idx]:
            # 接受匹配
            A_matched[a_idx] = True
            B_matched[b_idx] = True
            matching_pairs.append((int(a_idx), int(b_idx)))
            matched_dists_squared.append(dist_squared)  # 存储平方距离

    timing_stats['greedy_matching_s'] = time.time() - match_start_time
    if verbose: print(
        f"[{time.strftime('%H:%M:%S')}] 贪心匹配完成。已匹配 {len(matching_pairs)} 对。耗时: {timing_stats['greedy_matching_s']:.4f}s")

    # --- 4. 补齐未匹配点 (如果需要) ---
    #
    # 说明：
    #   前三步只在“每个 A 的 top-k 候选边”上做最短边优先的全局贪心匹配。
    #   当 k 较小且候选高度重叠时，稀疏候选图可能无法覆盖全部点，从而出现未匹配点。
    #
    # 这里的补齐策略：
    #   - 仅在未匹配集合 (A_unmatched, B_unmatched) 上继续做同样的 global-best 贪心；
    #   - 逐轮增大 k_ext（候选数），直到匹配补满或达到最大轮次；
    #   - 仍未补满时，使用“强制兜底”：对剩余 A 逐个在剩余 B 中找最近并立即占用，保证最终一一映射。

    if len(matching_pairs) < target_pairs and extend_search:
        extend_start_time = time.time()

        rounds = 0
        max_rounds = 6
        k_ext = max(k, 20)

        while len(matching_pairs) < target_pairs and rounds < max_rounds:
            unmatched_A_indices = np.where(~A_matched)[0]
            unmatched_B_indices = np.where(~B_matched)[0]

            if len(unmatched_A_indices) == 0 or len(unmatched_B_indices) == 0:
                break

            k_eff = int(min(k_ext, len(unmatched_B_indices)))
            if verbose:
                print(f"[{time.strftime('%H:%M:%S')}] 4.{rounds+1} 补齐轮次: "
                      f"unmatched A={len(unmatched_A_indices)}, unmatched B={len(unmatched_B_indices)}, k_ext={k_eff}")

            # 仅对未匹配的 B 点集构建 Faiss 索引
            B_unmatched = B[unmatched_B_indices]
            index_unmatched = faiss.IndexFlatL2(D)
            index_unmatched.add(B_unmatched)

            A_unmatched_coords = A[unmatched_A_indices]
            dists_sq_unmatched, idxs_unmatched = index_unmatched.search(A_unmatched_coords, k_eff)

            # 构建补齐候选边堆（全局最短边优先）
            extend_heap = []
            for ui, a_idx in enumerate(unmatched_A_indices):
                for t in range(k_eff):
                    b_local = int(idxs_unmatched[ui, t])
                    if b_local < 0:
                        continue
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

            gained = len(matching_pairs) - before
            if gained == 0:
                # 本轮没有增量：扩大候选数再试
                k_ext = min(len(unmatched_B_indices), max(k_eff * 2, k_eff + 10))
            else:
                # 有增量：通常也适当增大，减少下一轮“被抢占”导致的卡住
                k_ext = min(len(unmatched_B_indices), max(k_ext, k_eff * 2))

            rounds += 1

        # ---- 强制兜底：保证最终满配（必要时牺牲最优性）----
        if len(matching_pairs) < target_pairs:
            unmatched_A_indices = np.where(~A_matched)[0]
            unmatched_B_indices = np.where(~B_matched)[0]
            if verbose:
                print(f"[{time.strftime('%H:%M:%S')}] 4.F 兜底补齐: remaining A={len(unmatched_A_indices)}, "
                      f"remaining B={len(unmatched_B_indices)}")

            # 使用“逐个选最近并立即占用”的方式保证一对一
            # 对于 N<=几千的尾部规模非常快；对于更大规模也只作用于剩余部分。
            B_rem = B[unmatched_B_indices]
            for a_idx in unmatched_A_indices:
                if len(unmatched_B_indices) == 0:
                    break

                index_rem = faiss.IndexFlatL2(D)
                index_rem.add(B_rem)

                a_vec = A[int(a_idx)].reshape(1, -1)
                d2, pos = index_rem.search(a_vec, 1)
                pos = int(pos[0, 0])

                b_idx = int(unmatched_B_indices[pos])

                A_matched[int(a_idx)] = True
                B_matched[b_idx] = True
                matching_pairs.append((int(a_idx), b_idx))
                matched_dists_squared.append(float(d2[0, 0]))

                # 从剩余 B 中移除已占用点，保持一对一
                unmatched_B_indices = np.delete(unmatched_B_indices, pos)
                B_rem = np.delete(B_rem, pos, axis=0)

            # 最后保险：若仍未满（极少情况），按顺序配对补齐
            if len(matching_pairs) < target_pairs:
                remA = np.where(~A_matched)[0]
                remB = np.where(~B_matched)[0]
                nrem = min(len(remA), len(remB))
                for t in range(nrem):
                    a_idx = int(remA[t])
                    b_idx = int(remB[t])
                    A_matched[a_idx] = True
                    B_matched[b_idx] = True
                    diff = A[a_idx] - B[b_idx]
                    d2 = float(np.dot(diff, diff))
                    matching_pairs.append((a_idx, b_idx))
                    matched_dists_squared.append(d2)

        timing_stats['extend_search_s'] = time.time() - extend_start_time
        if verbose:
            print(f"[{time.strftime('%H:%M:%S')}] 补齐结束。总匹配对数: {len(matching_pairs)}。耗时: {timing_stats['extend_search_s']:.4f}s")
    # --- 5. 计算最终总欧氏距离之和 (Cost) ---

    cost_calc_start = time.time()
    # 核心步骤：对所有被选中的平方距离 $d^2$ 进行批量开方，然后求和
    # 这是整个计算流程中唯一进行 O(N) 次开方的地方，速度极快。
    final_euclidean_cost = np.sum(np.sqrt(np.array(matched_dists_squared)))

    timing_stats['final_cost_calc_s'] = time.time() - cost_calc_start

    total_time = time.time() - total_start_time
    timing_stats['total_time_s'] = total_time
    if verbose: print(f"\n[{time.strftime('%H:%M:%S')}] --- 整个算法总耗时: {total_time:.4f}s ---")

    return matching_pairs, final_euclidean_cost, timing_stats


def generate_ring_data(N: int = 50, R1: float = 0.5, R2: float = 3.2,
                       sigma: float = 0.1, center_offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    生成两个圆环状的 2D 点集 A 和 B。

    Args:
        N (int): 每个点集的样本数量。
        R1 (float): 源圆环 (A) 的平均半径。
        R2 (float): 目标圆环 (B) 的平均半径。
        sigma (float): 半径上的噪声强度 (控制点集的厚度)。
        center_offset (float): 目标圆环相对于源圆环中心的偏移量。

    Returns:
        tuple[np.ndarray, np.ndarray]: 点集 A 和点集 B，形状均为 (N, 2)。
    """

    # 角度是等间隔的，以模拟均匀分布的等势点
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

    # --- 1. 生成源点集 A (蓝色) ---
    # 添加高斯噪声到半径上
    radii_A = R1 + np.random.normal(0, sigma, N)

    A = np.zeros((N, 2))
    A[:, 0] = radii_A * np.cos(angles)  # x 坐标
    A[:, 1] = radii_A * np.sin(angles)  # y 坐标

    # --- 2. 生成目标点集 B (红色) ---
    radii_B = R2 + np.random.normal(0, sigma, N)

    B = np.zeros((N, 2))
    B[:, 0] = radii_B * np.cos(angles) + center_offset  # x 坐标，并添加偏移
    B[:, 1] = radii_B * np.sin(angles)

    # 随机打乱 B 的顺序，确保映射不是简单的索引对应
    np.random.shuffle(B)

    return A.astype(np.float32), B.astype(np.float32)


def plot_matching_result(A: np.ndarray, B: np.ndarray, matching_pairs: list,
                         title: str = "ANN Greedy Matching on Two Rings"):
    """
    绘制两个点集及其一对一匹配结果。

    Args:
        A (np.ndarray): 源点集 (N, 2)。
        B (np.ndarray): 目标点集 (N, 2)。
        matching_pairs (list): 匹配对列表，格式为 [(A_idx, B_idx), ...]。
        title (str): 图像标题。
    """

    fig, ax = plt.subplots(figsize=(8, 8))

    # --- 1. 绘制点集 ---

    # 绘制源点集 A (蓝色)
    ax.scatter(A[:, 0], A[:, 1], c='blue', s=1, label='Source samples')

    # 绘制目标点集 B (红色)
    ax.scatter(B[:, 0], B[:, 1], c='red', s=1, label='Target samples')

    # --- 2. 绘制匹配线条 ---

    # 设定线条颜色和透明度，以观察密集区域
    line_style = {'color': 'gray', 'alpha': 0.5, 'linewidth': 0.8, 'linestyle': '-'}

    for a_idx, b_idx in matching_pairs:
        # 获取匹配点的坐标
        x_start, y_start = A[a_idx]
        x_end, y_end = B[b_idx]

        # 绘制连接线
        line = mlines.Line2D([x_start, x_end], [y_start, y_end], **line_style)
        ax.add_line(line)

    # --- 3. 设置图表属性 ---

    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')  # 保持 x 和 y 轴比例一致

    # 创建图例句柄 (防止线条被包含在图例中)
    source_legend = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                                  markersize=8, label='Source samples')
    target_legend = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                                  markersize=8, label='Target samples')

    ax.legend(handles=[source_legend, target_legend], loc='center')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


# --- 示例使用 ---
if __name__ == '__main__':
    # # 规模设置：演示任意维度 D
    # N = 50000  # 10万个点
    # D = 2  # 128 维特征向量 (常见于高维数据)
    # K_CANDIDATES = 10  # 每个点只找 10 个最近邻
    #
    # print(f"--- 启动 Faiss 最短边优先贪心匹配 (D={D}) ---")
    # print(f"点集规模 N={N} (D={D})。候选数 k={K_CANDIDATES}")
    #
    # # 生成 128 维随机点集
    # A = np.random.rand(N, D)
    # B = np.random.rand(N, D)
    #
    # # 执行匹配
    # try:
    #     match_result, total_cost, timings = shortest_edge_first_greedy_matching_faiss_general(A, B, k=K_CANDIDATES,
    #                                                                                           extend_search=True)
    #
    #     print("\n--- 匹配结果摘要 ---")
    #     print(f"总匹配对数: {len(match_result)}")
    #     print(f"总欧氏距离 (Cost): {total_cost:.4f}")
    #
    #     print("\n--- 耗时统计 ---")
    #     for key, value in timings.items():
    #         print(f"{key:<20}: {value:.4f}s")
    #
    #     core_time = timings.get('faiss_search_s', 0) + timings.get('heap_build_s', 0) + timings.get('greedy_matching_s',
    #                                                                                                 0)
    #     print(f"\n核心计算耗时 (Faiss+堆): {core_time:.4f}s")
    #
    # except Exception as e:
    #     print(f"\n[错误] Faiss 算法执行失败，请检查 Faiss 是否安装正确或内存是否足够：{e}")

    N_SAMPLES = 1000
    A_ring, B_ring = generate_ring_data(
        N=N_SAMPLES,
        R1=0.5,
        R2=3.2,
        sigma=0.1,
        center_offset=0.1  # 增加轻微偏移，让问题更复杂
    )

    print(f"生成的点集 A: {A_ring.shape}, B: {B_ring.shape}")

    # --- 2. 执行 Faiss 贪心匹配 ---
    # 使用 K=5 作为候选数，实现近似匹配
    K_CANDIDATES = 5

    # 运行您的 Faiss 贪心匹配函数
    match_result, total_cost, timings = shortest_edge_first_greedy_matching_faiss_general(
        A_ring,
        B_ring,
        k=K_CANDIDATES,
        extend_search=True
    )

    print(f"\n匹配总成本 (欧氏距离和): {total_cost:.4f}")
    print(f"匹配对数: {len(match_result)}")

    # --- 3. 绘制结果 ---
    plot_matching_result(
        A_ring,
        B_ring,
        match_result,
        title=f"Fast Greedy Matching on Rings (k={K_CANDIDATES}, Cost={total_cost:.2f})"
    )
