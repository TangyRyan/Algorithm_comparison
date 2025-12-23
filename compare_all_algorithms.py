import numpy as np
import time
import matplotlib.pyplot as plt
import ot  # POT: pip install POT
import os
import json
import warnings

# --- 导入你的算法模块 ---
try:
    import FF_GM_fullmatch
    import ANN_injective
    import lapjv_wrapper
except ImportError as e:
    print(f"错误: 无法导入模块，请检查目录下文件是否存在。\n详细信息: {e}")
    exit(1)


# ==========================================
# 1. 数据加载与预处理
# ==========================================
def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")

    print(f"正在读取: {os.path.basename(file_path)} ...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    points = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                if 'x' in item and 'y' in item:
                    points.append([item['x'], item['y']])
                elif 'lon' in item and 'lat' in item:
                    points.append([item['lon'], item['lat']])
                else:
                    vals = [v for v in item.values() if isinstance(v, (int, float))]
                    if len(vals) >= 2:
                        points.append(vals[:2])
            elif isinstance(item, list) and len(item) >= 2:
                points.append(item[:2])

    if len(points) == 0:
        raise ValueError("无法解析数据中的坐标")

    return np.array(points, dtype=np.float32)


def normalize_data(X):
    """
    归一化数据：(X - mean) / std
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0
    X_norm = (X - mean) / std
    return X_norm, mean, std


def align_datasets(A, B):
    n = min(len(A), len(B))
    return A[:n], B[:n]


# ==========================================
# 2. 算法封装
# ==========================================

def run_exact_ot(A, B):
    """ 精确最优传输 (EMD) """
    print("  [Exact OT] 计算距离矩阵...")
    start = time.time()

    M = ot.dist(A, B)
    M_max = M.max()
    if M_max > 0:
        M /= M_max

    n_A, n_B = len(A), len(B)
    a, b = np.ones((n_A,)) / n_A, np.ones((n_B,)) / n_B

    print("  [Exact OT] 求解 EMD...")
    P = ot.emd(a, b, M, numItermax=2000000)

    t = time.time() - start

    rows, cols = np.where(P > 1e-8)

    M_orig = ot.dist(A, B, metric='euclidean')
    cost = np.sum(P * M_orig) * len(A)  # 恢复到总Cost

    return list(zip(rows, cols)), cost, t


def run_partial_ot(A, B):
    """
    部分最优传输 (逻辑已修改为参考 compare_ann_ffgm_vs_ot.py)
    """
    n_A = len(A)
    n_B = len(B)

    # 动态计算 mass (m)
    # 如果 N_A == N_B，则尝试匹配所有点 (m=1.0)
    # 如果 N_A != N_B，则按比例匹配
    m_mass = min(n_A, n_B) / max(n_A, n_B)

    print(f"  [Partial OT] 正在计算 (m={m_mass:.4f})...")
    start = time.time()

    # 1. 计算距离矩阵
    M = ot.dist(A, B)
    M /= (M.max() + 1e-8)

    # 2. 构造分布
    a = np.ones((n_A,)) / n_A
    b = np.ones((n_B,)) / n_B

    P = None

    # 尝试求解 (参考 compare_ann_ffgm_vs_ot 的低 reg 策略)
    # 先尝试 reg=0.05 (高精度)，如果失败则回退到 0.1
    reg_candidates = [0.05, 0.1]

    for reg in reg_candidates:
        try:
            # 参考文件中的参数设置
            P = ot.partial.entropic_partial_wasserstein(
                a, b, M, m=m_mass, reg=reg, numItermax=1000, stopThr=1e-6, verbose=False
            )
            print(f"  Partial OT 求解成功 (reg={reg})")
            break
        except (MemoryError, UserWarning, Exception) as e:
            print(f"  Partial OT (reg={reg}) 失败或不收敛: {e}，尝试更宽松的参数...")
            continue

    if P is None:
        print("  Partial OT 所有尝试均失败。")
        return [], 0, 0

    t = time.time() - start

    # 3. 动态阈值过滤 (关键修改：参考 compare_ann_ffgm_vs_ot)
    # 避免硬编码的 1e-5 导致没有结果
    threshold = (1.0 / n_A) * 0.01  # 设定为平均概率的 1%
    rows, cols = np.where(P > threshold)

    # 4. 计算 Cost
    M_orig = ot.dist(A, B, metric='euclidean')
    # 注意：Partial OT 的 P 总和是 mass，为了便于对比，我们不进行归一化放大，
    # 而是直接计算传输的总代价。
    cost = np.sum(P * M_orig) * max(n_A, n_B)

    return list(zip(rows, cols)), cost, t


# ==========================================
# 3. 绘图
# ==========================================
def plot_matching_result(A, B, matching_pairs, title, ax):
    total = len(A)
    if total > 1000:
        indices = np.random.choice(total, 1000, replace=False)
    else:
        indices = np.arange(total)

    A_sub = A[indices]
    B_sub = B[indices]

    ax.scatter(A_sub[:, 0], A_sub[:, 1], c='blue', s=2, alpha=0.6, label='Source')
    ax.scatter(B_sub[:, 0], B_sub[:, 1], c='red', s=2, alpha=0.6, label='Target')

    pair_dict = {r: c for r, c in matching_pairs}

    for i in indices:
        if i in pair_dict:
            j = pair_dict[i]
            if j < len(B):
                ax.plot([A[i, 0], B[j, 0]], [A[i, 1], B[j, 1]], c='gray', alpha=0.3, lw=0.5)

    match_count = len(matching_pairs)
    match_rate = match_count / total * 100
    ax.set_title(f"{title}\nMatches: {match_count} ({match_rate:.1f}%)", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # --- 1. 配置路径 (请确认路径正确) ---
    SOURCE_FILE = r"D:\LZY_Project\CompareProject\source_points\2008_cancelled_flights.json"
    TARGET_FILE = r"D:\LZY_Project\CompareProject\target_points\2008_cancelled_flights.json"

    # 设置
    FORCE_FULL_SEARCH = True

    print(f"=== 算法对比实验 (Partial OT Logic Updated) ===")

    # --- 2. 加载 ---
    try:
        raw_A = load_data(SOURCE_FILE)
        raw_B = load_data(TARGET_FILE)
    except Exception as e:
        print(f"读取文件失败: {e}")
        exit()

    limit = 10000
    if len(raw_A) > limit:
        print(f"数据量 {len(raw_A)} > {limit}，截取前 {limit} 个点...")
        raw_A = raw_A[:limit]
        raw_B = raw_B[:limit]

    A, B = align_datasets(raw_A, raw_B)
    print(f"最终测试数据维度: {A.shape}")

    # --- 3. 归一化 ---
    print("正在进行数据归一化...")
    A_norm, _, _ = normalize_data(A)
    B_norm, _, _ = normalize_data(B)

    results = []

    # [1] LAPJV
    print(f"[1/5] LAPJV (Exact)...")
    try:
        pairs, cost, stats = lapjv_wrapper.lapjv_match(A_norm, B_norm)
        results.append(("LAPJV", pairs, cost, stats['total_time_s']))
    except Exception as e:
        print(f"Failed: {e}")
        results.append(("LAPJV", [], 0, 0))

    # [2] FF-GM
    print(f"[2/5] FF-GM...")
    try:
        k_val = len(B) if FORCE_FULL_SEARCH else 500
        k_val = min(k_val, len(B))
        print(f"  FF-GM k={k_val}...")
        pairs, cost, stats = FF_GM_fullmatch.shortest_edge_first_greedy_matching_faiss_general(
            A_norm, B_norm, k=k_val, extend_search=True, verbose=False
        )
        results.append((f"FF-GM", pairs, cost, stats['total_time_s']))
    except Exception as e:
        print(f"Failed: {e}")
        results.append(("FF-GM", [], 0, 0))

    # [3] ANN Injective
    print(f"[3/5] ANN Injective...")
    try:
        pairs, cost, stats = ANN_injective.match_approx_nn_injective(A_norm, B_norm, k=50, strict=False)
        results.append(("ANN", pairs, cost, stats['total_time_s']))
    except Exception as e:
        print(f"Failed: {e}")
        results.append(("ANN", [], 0, 0))

    # [4] Exact OT (EMD)
    print(f"[4/5] Exact OT (EMD)...")
    try:
        pairs, cost, t = run_exact_ot(A_norm, B_norm)
        results.append(("Exact OT", pairs, cost, t))
    except Exception as e:
        print(f"Failed: {e}")
        results.append(("Exact OT", [], 0, 0))

    # [5] Partial OT (Modified Logic)
    print(f"[5/5] Partial OT...")
    try:
        pairs, cost, t = run_partial_ot(A_norm, B_norm)
        results.append(("Partial OT", pairs, cost, t))
    except Exception as e:
        print(f"Failed: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Partial OT", [], 0, 0))

    # --- 5. 输出 ---
    print("\n" + "=" * 60)
    print(f"{'Method':<15} | {'Matches':<8} | {'Norm Cost':<12} | {'Time(s)':<8}")
    print("-" * 60)
    for name, pairs, cost, t in results:
        print(f"{name:<15} | {len(pairs):<8} | {cost:<12.2f} | {t:<8.2f}")
    print("=" * 60)

    print("正在绘图...")
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5))
    if len(results) == 1: axes = [axes]

    for ax, (name, pairs, cost, t) in zip(axes, results):
        plot_matching_result(A, B, pairs, f"{name}\nTime={t:.2f}s", ax=ax)

    plt.tight_layout()
    plt.show()