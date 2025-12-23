import numpy as np
import time
import matplotlib.pyplot as plt
import ot  # 需要安装 POT: pip install POT
import os
import json

# --- 导入你的算法模块 ---
try:
    import FF_GM_fullmatch
    import ANN_injective
    import lapjv_wrapper
except ImportError as e:
    print(f"错误: 无法导入模块，请检查目录下文件是否存在。\n详细信息: {e}")
    exit(1)


# ==========================================
# 1. 数据加载与对齐 (全量读取)
# ==========================================
def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")

    print(f"正在读取: {os.path.basename(file_path)} ...")
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 简单处理常见的 list 结构
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], list):
                points = np.array(data, dtype=np.float32)
            elif isinstance(data[0], dict):
                # 尝试提取 x,y 或 lon,lat
                sample = data[0]
                if 'x' in sample and 'y' in sample:
                    points = np.array([[d['x'], d['y']] for d in data], dtype=np.float32)
                elif 'lon' in sample and 'lat' in sample:
                    points = np.array([[d['lon'], d['lat']] for d in data], dtype=np.float32)
                else:
                    # 暴力提取前两个数值
                    vals = [[v for v in d.values() if isinstance(v, (int, float))][:2] for d in data]
                    points = np.array(vals, dtype=np.float32)
            else:
                raise ValueError("不支持的 JSON 格式")
        else:
            raise ValueError("JSON 必须是列表")
    elif file_path.endswith('.npy'):
        points = np.load(file_path).astype(np.float32)
    else:
        points = np.loadtxt(file_path, delimiter=',').astype(np.float32)
    return points


def align_datasets(A, B):
    # OT 和 LAPJV 通常需要维度一致（或者需要复杂的权重设置）
    # 这里为了公平对比，取交集数量
    n = min(len(A), len(B))
    return A[:n], B[:n]


# ==========================================
# 2. OT 算法封装 (修正版)
# ==========================================
def run_exact_ot(A, B):
    """ 计算精确的最优传输 (EMD) - 增加了最大迭代次数 """
    N = A.shape[0]
    start_time = time.time()

    # 1. 计算距离矩阵
    M = ot.dist(A, B, metric='euclidean')
    M /= M.max()  # 归一化维持数值稳定性

    a, b = np.ones((N,)) / N, np.ones((N,)) / N

    # 修正：增加 numItermax 到 100万 (默认是10万)
    # 这样可以消除 "numItermax reached" 的警告
    P = ot.emd(a, b, M, numItermax=2000000)

    total_time = time.time() - start_time

    # 解析结果
    rows, cols = np.where(P > 1e-8)
    pairs = list(zip(rows, cols))

    # 计算 Cost (使用原始距离)
    M_orig = ot.dist(A, B, metric='euclidean')
    cost_sum = np.sum(P * M_orig) * N

    return pairs, cost_sum, total_time


def run_partial_ot(A, B, mass=0.8):
    """
    计算部分最优传输 (Partial OT) - 极速版 (Fast Approximation)
    """
    N = A.shape[0]
    start_time = time.time()

    # 使用 float64 防止溢出
    A = A.astype(np.float64)
    B = B.astype(np.float64)

    # 1. 计算距离矩阵 (这是最耗时的一步，O(N^2))
    print("  [Partial OT] Calculating distance matrix...")
    M = ot.dist(A, B, metric='euclidean')

    # 2. 归一化
    max_val = M.max()
    if max_val > 0:
        M_norm = M / max_val
    else:
        M_norm = M

    a, b = np.ones((N,)) / N, np.ones((N,)) / N

    # 3. 极速参数设置
    # 直接使用较大的 reg (如 1.0)，这会极大加快收敛速度
    # 虽然精度略微降低，但在视觉对比任务中通常看不出区别
    fast_reg = 1.0

    print(f"  [Partial OT] Solving with reg={fast_reg} (Fast Mode)...")

    try:
        P = ot.partial.entropic_partial_wasserstein(
            a, b, M_norm, m=mass, reg=fast_reg, numItermax=500
        )
    except Exception as e:
        print(f"  [Partial OT] Error: {e}")
        return [], 0.0, time.time() - start_time

    # 检查是否计算失败
    if np.any(np.isnan(P)):
        print("  [Partial OT] Result contains NaN (Failed).")
        return [], 0.0, time.time() - start_time

    total_time = time.time() - start_time

    # 4. 过滤低概率连接
    # 由于 reg 较大，矩阵会比较稠密，我们需要更高的阈值来过滤噪声
    threshold = 1.0 / N * 0.1  # 提高阈值，只保留强连接
    rows, cols = np.where(P > threshold)
    pairs = list(zip(rows, cols))

    # 5. 计算 Cost
    cost_sum = np.sum(P * M) * N

    return pairs, cost_sum, total_time


# ==========================================
# 3. 绘图 (抽样显示 1000 点)
# ==========================================
def plot_matching_result(A, B, matching_pairs, title, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    total_points = len(A)
    # 只显示 1000 个点，避免重叠
    display_limit = 1000

    if total_points > display_limit:
        indices = np.random.choice(total_points, display_limit, replace=False)
    else:
        indices = np.arange(total_points)

    A_subset = A[indices]
    B_subset = B[indices]

    ax.scatter(A_subset[:, 0], A_subset[:, 1], c='blue', s=2, alpha=0.5, label='Source')
    ax.scatter(B_subset[:, 0], B_subset[:, 1], c='red', s=2, alpha=0.5, label='Target')

    # 转为字典加速查找
    pair_dict = {p[0]: p[1] for p in matching_pairs}

    line_style = {'color': 'gray', 'alpha': 0.4, 'linewidth': 0.5}

    lines_count = 0
    for a_idx in indices:
        if a_idx in pair_dict:
            b_idx = pair_dict[a_idx]
            # 确保 b_idx 也在范围内
            if b_idx < len(B):
                x_start, y_start = A[a_idx]
                x_end, y_end = B[b_idx]
                ax.plot([x_start, x_end], [y_start, y_end], **line_style)
                lines_count += 1

    # 在标题显示匹配率
    match_rate = len(matching_pairs) / total_points * 100
    ax.set_title(f"{title}\nMatched: {len(matching_pairs)} ({match_rate:.1f}%)", fontsize=9)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])


# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # ---------------- 配置 ----------------
    SOURCE_DIR = r"D:\LZY_Project\CompareProject\source_points"
    TARGET_DIR = r"D:\LZY_Project\CompareProject\target_points"
    SOURCE_FILENAME = "2008_cancelled_flights.json"
    TARGET_FILENAME = "2008_cancelled_flights.json"

    # 拼接路径
    SOURCE_FILE = os.path.join(SOURCE_DIR, SOURCE_FILENAME)
    TARGET_FILE = os.path.join(TARGET_DIR, TARGET_FILENAME)

    # Partial OT 的匹配比例 (0.5 到 0.9 之间通常效果明显)
    PARTIAL_MASS = 0.8
    # -------------------------------------

    print(f"=== 5种算法对比实验 ===")

    # 1. 加载数据
    try:
        raw_A = load_data(SOURCE_FILE)
        raw_B = load_data(TARGET_FILE)
    except Exception as e:
        print(f"加载失败: {e}")
        exit()

    A, B = align_datasets(raw_A, raw_B)
    print(f"数据维度: {A.shape} (已对齐)")

    # 警告：大数据量保护
    if len(A) > 6000:
        print("\n[警告] 数据量 > 6000。Exact OT (EMD) 和 LAPJV 可能会运行很慢 (>5分钟)。")
        print("如需快速测试，请手动在代码中截断数据 (例如 A = A[:3000])。\n")
        # 如果你想强制截断测试，取消下面这行的注释：
        # A, B = A[:3000], B[:3000]

    results = []

    # --- 1. LAPJV (Linear Assignment Problem) ---
    print(f"[1/5] LAPJV...")
    try:
        pairs, cost, stats = lapjv_wrapper.lapjv_match(A, B)
        results.append(("LAPJV (Exact)", pairs, cost, stats['total_time_s']))
    except Exception as e:
        print(f"  Failed: {e}")
        results.append(("LAPJV", [], 0, 0))

    # --- 2. FF-GM (Fast Greedy) ---
    print(f"[2/5] FF-GM...")
    try:
        n_match = min(len(A), len(B))
        k_min = 20
        k_max = 200
        k_adapt = int(np.sqrt(n_match))
        if k_adapt < k_min:
            k_adapt = k_min
        if k_adapt > k_max:
            k_adapt = k_max
        if k_adapt > n_match:
            k_adapt = n_match
        if k_adapt < 1:
            k_adapt = 1
        print(f"  FF-GM adaptive k: {k_adapt} (n={n_match}, min={k_min}, max={k_max})")
        pairs, cost, stats = FF_GM_fullmatch.shortest_edge_first_greedy_matching_faiss_general(
            A, B, k=k_adapt, extend_search=True, verbose=False
        )
        results.append(("FF-GM", pairs, cost, stats['total_time_s']))
    except Exception as e:
        print(f"  Failed: {e}")
        results.append(("FF-GM", [], 0, 0))

    # --- 3. ANN Injective ---
    print(f"[3/5] ANN Injective...")
    try:
        pairs, cost, stats = ANN_injective.match_approx_nn_injective(A, B, k=20, strict=False)
        results.append(("ANN Injective", pairs, cost, stats['total_time_s']))
    except Exception as e:
        print(f"  Failed: {e}")
        results.append(("ANN Injective", [], 0, 0))

    # --- 4. Exact OT (EMD) ---
    print(f"[4/5] Exact OT (EMD)...")
    try:
        pairs, cost, t = run_exact_ot(A, B)
        results.append(("Exact OT", pairs, cost, t))
    except Exception as e:
        print(f"  Failed: {e}")
        results.append(("Exact OT", [], 0, 0))

    # --- 5. Partial OT ---
    print(f"[5/5] Partial OT (m={PARTIAL_MASS})...")
    try:
        pairs, cost, t = run_partial_ot(A, B, mass=PARTIAL_MASS)
        results.append((f"Partial OT ({PARTIAL_MASS})", pairs, cost, t))
    except Exception as e:
        print(f"  Failed: {e}")
        results.append(("Partial OT", [], 0, 0))

    # ---------------- 表格输出 ----------------
    print("\n" + "=" * 80)
    print(f"{'Method':<20} | {'Matches':<8} | {'Cost (Sum)':<15} | {'Time (s)':<10}")
    print("-" * 80)
    for name, pairs, cost, t in results:
        print(f"{name:<20} | {len(pairs):<8} | {cost:<15.2f} | {t:<10.4f}")
    print("=" * 80)

    # ---------------- 绘图 (5 列) ----------------
    print("正在绘图...")
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))  # 宽图，容纳5个子图

    # 如果 results 少于 5 个 (比如有报错)，补齐
    if len(results) < 5:
        print("部分算法失败，仅显示成功部分")
        axes = axes[:len(results)]

    for ax, (name, pairs, cost, t) in zip(axes, results):
        plot_matching_result(A, B, pairs, f"{name}\nCost={cost:.0f}\nTime={t:.2f}s", ax=ax)

    plt.tight_layout()
    plt.show()
