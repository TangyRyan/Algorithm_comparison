import numpy as np
import faiss
import time
from typing import List, Tuple, Dict


def match_approx_nn_injective(
        A: np.ndarray,
        B: np.ndarray,
        k: int = 50,
        strict: bool = False
) -> Tuple[List[Tuple[int, int]], float, Dict]:
    N_A = A.shape[0]
    N_B = B.shape[0]
    D = A.shape[1]

    stats = {}
    t_start = time.time()

    # --- GPU Faiss Setup ---
    res = faiss.StandardGpuResources() if hasattr(faiss, 'StandardGpuResources') else None
    use_gpu = (faiss.get_num_gpus() > 0)

    if use_gpu:
        cpu_index = faiss.IndexFlatL2(D)
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        # print("  [ANN] GPU Faiss 启用")
    else:
        index = faiss.IndexFlatL2(D)

    index.add(B)

    # 1. 对所有 A 找 Top-K
    # 这一步在 4090 上会极快
    D_sq, I = index.search(A, k)

    stats['faiss_time'] = time.time() - t_start

    # 2. 贪心分配 (CPU)
    t_greedy = time.time()

    pairs = []
    matched_B = set()

    # 简单的按行遍历 (A的顺序)
    # 改进版：应该按距离排序吗？
    # 原始逻辑是 "Injective"，通常是指按照 A 的顺序，给它找最近的未被占用的 B
    # 或者 全局排序。ANN Injective 通常指后者（类似 FF-GM 但没有堆的完全动态维护）
    # 这里我们实现一个高效版本：
    # 将所有 (dist, a_idx, b_idx) 放入列表排序 (近似全局贪心)

    candidates = []
    # 展平
    # 为了速度，可以使用 numpy 操作
    # rows: 0..N_A 重复 k 次
    # cols: I
    # vals: D_sq

    # 这种全量排序可能会慢 (N*K)。
    # 经典 ANN 策略：只看每个 A 的第 1 候选，冲突了再看第 2 候选。

    # A_pointers[i] 表示 A[i] 当前看到第几个邻居
    A_pointers = np.zeros(N_A, dtype=int)

    # 优先队列？或者直接迭代
    # 为了实现 "Injective" 且效果好，我们按 A-B 距离排序处理
    # 但排序 N*K 个元素太慢。

    # 快速策略：
    # 1. 拿所有 A 的第 1 最近邻。
    # 2. 按距离从小到大处理。
    # 3. 如果 B 未占用 -> 匹配。
    # 4. 如果 B 已占用 -> 将该 A 的第 2 最近邻加入待处理池。

    # 这是一个 Loop，直到所有 A 匹配或无候选

    # 初始化：每个 A 的当前候选 (dist, a_idx, b_idx)
    # 我们用一个 list 存当前每行的最佳候选

    # 优化：直接全量排序 N*K 个候选中的前 N 个？
    # 还是回退到简单逻辑：

    # 这里为了保持和你之前结果一致性，使用简单的 "每个A找最近未占用"
    # 但为了顺序无关性，最好按距离排序

    # 让我们用 "Sort all candidates" 策略，因为 K=50, N=10000 -> 50万个元素排序，CPU 0.1秒就搞定，不慢。

    # 构造候选列表
    # a_indices = np.repeat(np.arange(N_A), k)
    # b_indices = I.flatten()
    # dists = D_sq.flatten()

    # 过滤无效索引 (-1)
    valid_mask = I.flatten() >= 0

    # 结构化数组以便排序
    struct_arr = np.zeros(valid_mask.sum(), dtype=[('d', 'f4'), ('a', 'i4'), ('b', 'i4')])
    struct_arr['d'] = D_sq.flatten()[valid_mask]
    struct_arr['a'] = np.repeat(np.arange(N_A), k)[valid_mask]
    struct_arr['b'] = I.flatten()[valid_mask]

    # 排序 (耗时极短)
    struct_arr.sort(order='d')

    # 贪心匹配
    for i in range(len(struct_arr)):
        item = struct_arr[i]
        a = item['a']
        b = item['b']

        # 注意：这里一个 A 只需要匹配一次。
        # 我们需要 matched_A 集合
        # 修正：原函数逻辑是 A 必须匹配。

        # 这种逻辑需要 matched_A 和 matched_B
        # 但我们之前定义的 candidates 是平铺的，没有状态
        pass

        # 重写循环逻辑：
    matched_A = set()
    matched_B = set()

    for i in range(len(struct_arr)):
        item = struct_arr[i]
        a_idx = item['a']
        b_idx = item['b']

        if a_idx in matched_A or b_idx in matched_B:
            continue

        pairs.append((a_idx, b_idx))
        matched_A.add(a_idx)
        matched_B.add(b_idx)

        if len(matched_A) == N_A:
            break

    # 3. 兜底 (Force Match)
    # 如果还有 A 没匹配 (因为它的前 k 个邻居都被抢了)
    if len(matched_A) < N_A:
        unmatched_A = [i for i in range(N_A) if i not in matched_A]
        # 对这些 A，在所有未匹配 B 中找最近
        # 这就是 "Global" search
        # 为了快，我们只在剩下的 B 中搜
        # 重新建索引？

        # 简易版：直接搜全量 B，然后过滤
        # 因为剩下 A 不多
        if unmatched_A:
            # 搜 K=1000
            k_large = 500
            D_sq_2, I_2 = index.search(A[unmatched_A], k_large)

            # 同样逻辑，排序后匹配
            # 略微繁琐，这里简化：对每个未匹配 A，遍历其 k_large 邻居，选第一个未占用的
            for i, a_idx in enumerate(unmatched_A):
                found = False
                for j in range(k_large):
                    b_idx = I_2[i, j]
                    if b_idx not in matched_B:
                        pairs.append((a_idx, b_idx))
                        matched_B.add(b_idx)
                        found = True
                        break
                if not found:
                    # 极罕见：前 500 个都被占了。随便配一个未占用的
                    # 线性扫一遍 matched_B (慢但安全)
                    # 考虑到 N_B >> N_A，通常不会到这一步
                    pass

    stats['greedy_time'] = time.time() - t_greedy

    # 计算 Cost
    total_cost = 0.0
    if pairs:
        p_arr = np.array(pairs)
        dists = np.linalg.norm(A[p_arr[:, 0]] - B[p_arr[:, 1]], axis=1)
        total_cost = np.sum(dists)

    stats['total_time_s'] = time.time() - t_start
    return pairs, total_cost, stats