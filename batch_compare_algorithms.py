import numpy as np
import faiss
import heapq
import time
import json
import os
import csv
import matplotlib

# [配置] 强制使用非交互式后端，防止在服务器上报错
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from typing import List, Tuple, Dict

# ==========================================
# 0. 全局 GPU 资源初始化
# ==========================================
GLOBAL_GPU_RES = None
try:
    if hasattr(faiss, 'StandardGpuResources'):
        GLOBAL_GPU_RES = faiss.StandardGpuResources()
        # 预分配临时显存，优化性能
        GLOBAL_GPU_RES.setTempMemory(512 * 1024 * 1024)
    else:
        print("[Init] StandardGpuResources not found. Using CPU.")
except Exception as e:
    print(f"[Init] Failed to init GPU resources: {e}. Fallback to CPU.")
    GLOBAL_GPU_RES = None


# ==========================================
# 核心算法 1: ANN Injective (Sort + Fallback)
# [修改] 已增加来自 ANN_injective.py 的兜底逻辑
# ==========================================
def match_approx_nn_injective(
        A: np.ndarray,
        B: np.ndarray,
        k: int = 50
) -> Tuple[List[Tuple[int, int]], float, Dict]:
    N_A = A.shape[0]
    D = A.shape[1]
    stats = {}
    t_start = time.time()

    # --- GPU Setup ---
    use_gpu = (faiss.get_num_gpus() > 0) and (GLOBAL_GPU_RES is not None)

    if use_gpu:
        cpu_index = faiss.IndexFlatL2(D)
        # 复用全局 GPU 资源
        index = faiss.index_cpu_to_gpu(GLOBAL_GPU_RES, 0, cpu_index)
    else:
        index = faiss.IndexFlatL2(D)

    index.add(B)

    # 1. 第一轮: Top-K (50) 搜索
    D_sq, I = index.search(A, k)
    stats['faiss_time'] = time.time() - t_start

    t_greedy = time.time()
    valid_mask = I.flatten() >= 0

    # 2. 全量排序匹配 (Global Greedy)
    struct_arr = np.zeros(valid_mask.sum(), dtype=[('d', 'f4'), ('a', 'i4'), ('b', 'i4')])
    struct_arr['d'] = D_sq.flatten()[valid_mask]
    struct_arr['a'] = np.repeat(np.arange(N_A), k)[valid_mask]
    struct_arr['b'] = I.flatten()[valid_mask]

    struct_arr.sort(order='d')

    pairs = []
    matched_A = set()
    matched_B = set()

    for i in range(len(struct_arr)):
        item = struct_arr[i]
        a_idx, b_idx = item['a'], item['b']
        if a_idx in matched_A or b_idx in matched_B:
            continue
        pairs.append((a_idx, b_idx))
        matched_A.add(a_idx)
        matched_B.add(b_idx)
        if len(matched_A) == N_A:
            break

    # 3. [新增] 兜底策略 (Fallback / Extend Search)
    # 逻辑源自 ANN_injective.py
    if len(matched_A) < N_A:
        # print(f"  [ANN] 触发兜底: {N_A - len(matched_A)} 个点未匹配，扩大搜索范围...")
        unmatched_A = [i for i in range(N_A) if i not in matched_A]

        # 扩大搜索 K=500
        k_ext = 500
        # 确保数据内存连续
        A_unmatched = np.ascontiguousarray(A[unmatched_A])
        D_sq_ext, I_ext = index.search(A_unmatched, k_ext)

        # 采用 First-Fit 策略填补空缺
        for i, real_a_idx in enumerate(unmatched_A):
            for j in range(k_ext):
                b_idx = I_ext[i, j]
                # 如果该 Target 点未被占用，直接匹配
                if b_idx >= 0 and b_idx not in matched_B:
                    pairs.append((real_a_idx, b_idx))
                    matched_B.add(b_idx)
                    matched_A.add(real_a_idx)
                    break  # 找到一个就停止，处理下一个 A

    stats['greedy_time'] = time.time() - t_greedy

    # 4. 计算总 Cost
    total_cost = 0.0
    if pairs:
        p_arr = np.array(pairs)
        diff = A[p_arr[:, 0]] - B[p_arr[:, 1]]
        dists = np.linalg.norm(diff, axis=1)
        total_cost = np.sum(dists)

    stats['total_time_s'] = time.time() - t_start
    return pairs, total_cost, stats


# ==========================================
# 核心算法 2: FF-GM (Heap)
# ==========================================
def shortest_edge_first_greedy_matching_faiss_general(
        A: np.ndarray,
        B: np.ndarray,
        k: int = 50,
        extend_search: bool = True
) -> Tuple[List[Tuple[int, int]], float, Dict]:
    N_A = A.shape[0]
    D = A.shape[1]
    timings = {}
    t0 = time.time()

    # --- GPU Setup ---
    use_gpu = (faiss.get_num_gpus() > 0) and (GLOBAL_GPU_RES is not None)

    if use_gpu:
        cpu_index = faiss.IndexFlatL2(D)
        index = faiss.index_cpu_to_gpu(GLOBAL_GPU_RES, 0, cpu_index)
    else:
        index = faiss.IndexFlatL2(D)

    index.add(B)
    D_sq, I = index.search(A, k)
    timings['faiss_search_s'] = time.time() - t0

    t1 = time.time()
    candidates = []
    for i in range(N_A):
        for j in range(k):
            b_idx = I[i, j]
            if b_idx >= 0:
                d_val = D_sq[i, j]
                candidates.append((d_val, i, b_idx))

    heapq.heapify(candidates)
    timings['heap_build_s'] = time.time() - t1

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
        if len(matched_pairs) == N_A:
            break

    timings['greedy_matching_s'] = time.time() - t2

    if extend_search and len(matched_pairs) < N_A:
        t3 = time.time()
        unmatched_A = [i for i in range(N_A) if i not in matched_A]
        if unmatched_A:
            k_ext = 100
            try:
                A_unmatched = np.ascontiguousarray(A[unmatched_A])
                D_sq_ext, I_ext = index.search(A_unmatched, k_ext)

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
            except Exception as e:
                print(f"Warning: Extend search failed: {e}")

        timings['extend_search_s'] = time.time() - t3

    total_cost = 0.0
    if matched_pairs:
        pairs_arr = np.array(matched_pairs)
        diff = A[pairs_arr[:, 0]] - B[pairs_arr[:, 1]]
        dists = np.linalg.norm(diff, axis=1)
        total_cost = np.sum(dists)

    timings['total_time_s'] = sum(timings.values())
    return matched_pairs, total_cost, timings


# ==========================================
# 3. 批量处理工具
# ==========================================

def load_json_points(json_path: str) -> np.ndarray:
    """读取 JSON 并转换为 numpy 数组"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    points = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                row = [item.get('x', 0), item.get('y', 0)]
                if 'z' in item:
                    row.append(item['z'])
                points.append(row)
            elif isinstance(item, list):
                points.append(item)

    return np.ascontiguousarray(np.array(points, dtype=np.float32))


def visualize_one_pair(A, B, pairs, filename, title_suffix="", output_filename="visualization_sample.png"):
    """仅用于可视化第一对数据作为检查 (已针对大数据集优化)"""
    plt.figure(figsize=(10, 8))

    # 优化: 如果 Target 点太多，只随机画 5万个点作为背景
    if len(B) > 50000:
        idx = np.random.choice(len(B), 50000, replace=False)
        plt.scatter(B[idx, 0], B[idx, 1], c='lightgray', s=1, label='Target (Sampled)', alpha=0.5)
    else:
        plt.scatter(B[:, 0], B[:, 1], c='lightgray', s=5, label='Target', alpha=0.5)

    plt.scatter(A[:, 0], A[:, 1], c='red', s=5, label='Source', zorder=3, alpha=0.6)


    # 随机抽样画连线
    if len(pairs) > 0:
        num_draw = min(200, len(pairs))
        indices = np.random.choice(len(pairs), num_draw, replace=False)
        for i in indices:
            a_idx, b_idx = pairs[i]
            p1, p2 = A[a_idx], B[b_idx]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], c='blue', lw=1.2, alpha=0.9, zorder=10)

        plt.plot([], [], c='blue', label='Matches')

    plt.title(f"Matching: {filename} ({title_suffix})")
    plt.legend(loc='upper right')
    plt.tight_layout()

    # [修改] 使用传入的文件名保存
    plt.savefig(output_filename)
    plt.close()


def main():
    source_dir = 'source_points'
    target_dir = 'target_points'
    output_csv = 'benchmark_results.csv'

    if not os.path.exists(source_dir) or not os.path.exists(target_dir):
        print(f"Error: 请确保当前目录下存在 '{source_dir}' 和 '{target_dir}' 文件夹。")
        return

    files = [f for f in os.listdir(source_dir) if f.endswith('.json')]
    files.sort()

    if not files:
        print("Error: source_points 文件夹为空。")
        return

    print(f"找到 {len(files)} 个 Source 文件，开始批量测试...\n")

    # CSV Header
    csv_header = ['Filename', 'N_Source', 'N_Target',
                  'Sort_Time(s)', 'Sort_Cost',
                  'Heap_Time(s)', 'Heap_Cost']

    results = []

    for idx, filename in enumerate(files):
        s_path = os.path.join(source_dir, filename)
        t_path = os.path.join(target_dir, filename)

        if not os.path.exists(t_path):
            print(f"Warning: {filename} 在 target_points 中未找到对应文件，跳过。")
            continue

        try:
            A = load_json_points(s_path)
            B = load_json_points(t_path)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

        print(f"[{idx + 1}/{len(files)}] Processing {filename} (S:{len(A)}, T:{len(B)})...")

        # --- Algo 1: ANN Injective (Sort + Fallback) ---
        pairs1, cost1, stats1 = match_approx_nn_injective(A, B, k=50)
        time1 = stats1['total_time_s']

        # --- Algo 2: FF-GM (Heap) ---
        pairs2, cost2, stats2 = shortest_edge_first_greedy_matching_faiss_general(A, B, k=50)
        time2 = stats2['total_time_s']

        results.append([filename, len(A), len(B),
                        f"{time1:.4f}", f"{cost1:.2f}",
                        f"{time2:.4f}", f"{cost2:.2f}"])

        # 仅对第一个文件进行可视化检查
        # 建议改为 idx == 0，这样处理第一个文件时就会生成图片
        if idx == 0:
            print(f"  -> Generating visualization for {filename}...")

            # 1. 可视化算法 1 (Sort)
            visualize_one_pair(A, B, pairs1, filename, "Sort Algo", output_filename="viz_algo1_sort.png")

            # 2. 可视化算法 2 (Heap)
            visualize_one_pair(A, B, pairs2, filename, "Heap Algo", output_filename="viz_algo2_heap.png")

            print("  -> Saved: viz_algo1_sort.png & viz_algo2_heap.png")

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(results)

    print("-" * 60)
    print(f"测试完成！结果已保存至 {output_csv}")

    if results:
        times1 = [float(r[3]) for r in results]
        times2 = [float(r[5]) for r in results]
        avg_t1 = np.mean(times1)
        avg_t2 = np.mean(times2)
        print(f"平均耗时对比:")
        print(f"  Sort (ANN Injective): {avg_t1:.4f} s")
        print(f"  Heap (FF-GM)       : {avg_t2:.4f} s")
        if avg_t1 < avg_t2:
            print(f"  >> Sort 方法平均快 {(avg_t2 - avg_t1):.4f} 秒")


if __name__ == "__main__":
    main()