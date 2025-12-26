import numpy as np
import faiss
import heapq
import time
from typing import List, Tuple, Dict


def shortest_edge_first_greedy_matching_faiss_general(
        A: np.ndarray,
        B: np.ndarray,
        k: int = 10,
        extend_search: bool = True,
        verbose: bool = True
) -> Tuple[List[Tuple[int, int]], float, Dict]:
    # 确保维度一致
    if A.shape[1] != B.shape[1]:
        raise ValueError("输入点集 A 和 B 必须具有相同的维度 D。")

    N_A = A.shape[0]
    N_B = B.shape[0]
    D = A.shape[1]

    timings = {}

    # --- 1. Faiss 索引构建与搜索 (GPU 加速) ---
    t0 = time.time()

    # 定义资源 (GPU)
    res = faiss.StandardGpuResources() if hasattr(faiss, 'StandardGpuResources') else None
    use_gpu = (faiss.get_num_gpus() > 0)

    # 构建索引
    if use_gpu:
        # GPU 模式
        # flat_config = faiss.GpuIndexFlatConfig()
        # index = faiss.GpuIndexFlatL2(res, D, flat_config)
        # 或者更通用的方式：
        cpu_index = faiss.IndexFlatL2(D)
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        if verbose: print("  [FF-GM] 使用 GPU 进行 Faiss 搜索")
    else:
        # CPU 模式
        index = faiss.IndexFlatL2(D)
        if verbose: print("  [FF-GM] 使用 CPU 进行 Faiss 搜索")

    index.add(B)

    # 搜索
    D_sq, I = index.search(A, k)
    timings['faiss_search_s'] = time.time() - t0

    # --- 2. 贪心匹配 (CPU 逻辑，难以并行) ---
    t1 = time.time()

    # 构建最小堆: (distance_sq, a_idx, b_idx)
    # 展平结果以便建堆
    # 预分配列表
    candidates = []
    for i in range(N_A):
        for j in range(k):
            b_idx = I[i, j]
            if b_idx >= 0:  # 有效索引
                d_val = D_sq[i, j]
                candidates.append((d_val, i, b_idx))

    heapq.heapify(candidates)
    timings['heap_build_s'] = time.time() - t1

    # --- 3. 逐出匹配 ---
    t2 = time.time()

    matched_pairs = []
    matched_A = set()
    matched_B = set()

    while candidates:
        d_val, a_idx, b_idx = heapq.heappop(candidates)

        if a_idx in matched_A or b_idx in matched_B:
            continue

        matched_pairs.append((a_idx, b_idx))
        matched_A.add(a_idx)
        matched_B.add(b_idx)

        # 提前结束条件
        if len(matched_pairs) == min(N_A, N_B):
            break

    timings['greedy_matching_s'] = time.time() - t2

    # --- 4. 补齐搜索 (如果需要) ---
    if extend_search and len(matched_pairs) < N_A:
        t3 = time.time()

        unmatched_A = [i for i in range(N_A) if i not in matched_A]
        if unmatched_A:
            # 这一步通常点很少，CPU 跑也很快，但也可以用 GPU
            # 为了简单，如果点多，Faiss 还可以复用

            # 找出所有未匹配的 B
            # 注意：如果 N_B 很大，直接构建未匹配 B 的索引可能慢
            # 这里采用简单策略：如果剩余未匹配 A 很少，直接暴力；如果多，重建索引

            # 只有当确实有剩余 B 时
            if len(matched_B) < N_B:
                # 获取未匹配 B 的索引列表
                # set 查找很快
                unmatched_B_indices = []
                # 优化：如果 N_B 巨大，遍历 range(N_B) 会慢。
                # 但目前没有更好办法，除非维护一个 available mask
                # 鉴于贪心算法特性，通常大部分 B 都还在

                # 在大数据集下，补齐阶段可能比较耗时。
                # 我们可以对 unmatched_A 再做一次全局搜索 (k=100)，看是否能捡漏
                # 这比重建 unmatched_B 索引要快

                # 策略 2.0: 再次搜索 A，这次找 K=100
                k_ext = 100
                D_sq_ext, I_ext = index.search(A[unmatched_A], k_ext)

                ext_candidates = []
                for idx_in_unmatched, a_real_idx in enumerate(unmatched_A):
                    for j in range(k_ext):
                        b_real_idx = I_ext[idx_in_unmatched, j]
                        if b_real_idx not in matched_B:
                            d_val = D_sq_ext[idx_in_unmatched, j]
                            ext_candidates.append((d_val, a_real_idx, b_real_idx))

                heapq.heapify(ext_candidates)

                while ext_candidates:
                    d_val, a_idx, b_idx = heapq.heappop(ext_candidates)
                    if a_idx in matched_A or b_idx in matched_B:
                        continue
                    matched_pairs.append((a_idx, b_idx))
                    matched_A.add(a_idx)
                    matched_B.add(b_idx)

        timings['extend_search_s'] = time.time() - t3

    # --- 5. 计算总 Cost ---
    # 这里的 Cost 是平方距离和，还是欧氏距离和？
    # 原函数名 shortest_edge_first... 通常基于欧氏距离
    # 但 Faiss 返回的是 L2 squared。
    # 为了保持一致性，我们计算匹配对的欧氏距离和

    total_cost = 0.0
    # 我们可以用 numpy 批量算，比循环快
    if matched_pairs:
        pairs_arr = np.array(matched_pairs)
        # A[pairs[:,0]], B[pairs[:,1]]
        # 注意：这里如果 A, B 很大，可能会占内存。分块或者直接循环。
        # 简单循环求和：
        # total_cost = np.sum(np.sqrt(np.sum((A[pairs_arr[:,0]] - B[pairs_arr[:,1]])**2, axis=1)))

        # 为了省内存，用原数据索引计算
        # 使用向量化计算
        p_A = A[pairs_arr[:, 0]]
        p_B = B[pairs_arr[:, 1]]
        dists = np.linalg.norm(p_A - p_B, axis=1)
        total_cost = np.sum(dists)

    timings['total_time_s'] = sum(timings.values())

    return matched_pairs, total_cost, timings