import os
import json
import time
import numpy as np
import pandas as pd
import ot
import faiss  # 必须安装: pip install faiss-cpu
import warnings

# 忽略部分数值计算警告
warnings.filterwarnings("ignore")

# --- 尝试导入算法模块 ---
try:
    import FF_GM_fullmatch
    import ANN_injective
except ImportError as e:
    print(f"错误: 缺少必要的算法文件 (FF_GM_fullmatch.py 或 ANN_injective.py)。\n{e}")
    exit(1)

# ==========================================
# 1. 配置区域
# ==========================================
# 输入文件夹
SOURCE_DIR = "source_points"
TARGET_DIR = "target_points"

# 输出文件夹
RESULT_BASE_DIR = "results"
FFGM_OUT_DIR = os.path.join(RESULT_BASE_DIR, "FF_GM")
ANN_OUT_DIR = os.path.join(RESULT_BASE_DIR, "ANN")
OT_OUT_DIR = os.path.join(RESULT_BASE_DIR, "Partial_OT")

# 确保输出目录存在
for d in [FFGM_OUT_DIR, ANN_OUT_DIR, OT_OUT_DIR]:
    os.makedirs(d, exist_ok=True)

# 实验设置
REPEAT_TIMES = 3  # 每个算法跑3次取平均耗时
MAX_OT_CANDIDATES = 50  # Partial OT 预筛选：每个 Source 点找多少个 Target 邻居


# ==========================================
# 2. 数据处理工具
# ==========================================
def load_json_points(file_path):
    """读取 JSON，返回 (numpy坐标, 原始数据列表)"""
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    points = []

    for idx, item in enumerate(raw_data):
        # 兼容不同的坐标字段名
        x = item.get('x', item.get('lon', item.get('Longitude', None)))
        y = item.get('y', item.get('lat', item.get('Latitude', None)))

        if x is not None and y is not None:
            points.append([float(x), float(y)])

    return np.array(points, dtype=np.float32), raw_data


def normalize_points(X):
    """归一化：(X - mean) / std，对 OT 和 FF-GM 至关重要"""
    if len(X) == 0: return X, 0, 1
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0  # 防止除零
    return (X - mean) / std


def calculate_physical_cost(A_raw, B_raw, pairs):
    """
    计算匹配的物理总 Cost (基于原始坐标的欧氏距离之和)
    pairs: list of (source_idx, target_idx)
    """
    if not pairs:
        return 0.0

    total_cost = 0.0
    for s_idx, t_idx in pairs:
        # 确保索引不越界
        if s_idx < len(A_raw) and t_idx < len(B_raw):
            dist = np.linalg.norm(A_raw[s_idx] - B_raw[t_idx])
            total_cost += dist

    return total_cost


def save_result_json(algo_name, filename, target_raw_data, matching_pairs, source_raw_data, output_dir):
    """
    保存结果：在 target_raw_data 中注入 matched_source_id 和 label
    """
    import copy
    new_target_data = copy.deepcopy(target_raw_data)

    # 建立映射: target_idx (在 numpy 数组中的下标) -> source_idx
    # 注意：这里假设 load_json_points 提取的 numpy 数组顺序与 raw_data list 顺序一致
    # 且 raw_data 中没有被跳过的坏点（如果有坏点，需要额外的 index map，这里暂按理想情况处理）
    match_dict = {int(t_idx): int(s_idx) for s_idx, t_idx in matching_pairs}

    valid_idx = 0
    for i, item in enumerate(new_target_data):
        # 检查是否是有效点 (有坐标)
        x = item.get('x', item.get('lon', None))
        if x is None: continue

        if valid_idx in match_dict:
            src_idx = match_dict[valid_idx]
            if src_idx < len(source_raw_data):
                src_item = source_raw_data[src_idx]
                item['matched_source_id'] = src_item.get('id', 'N/A')
                item['matched_source_label'] = src_item.get('label', 'N/A')

        valid_idx += 1

    out_path = os.path.join(output_dir, filename)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(new_target_data, f, ensure_ascii=False, indent=2)


# ==========================================
# 3. 算法封装
# ==========================================

def run_ffgm_experiment(A_norm, B_norm):
    """运行 FF-GM (Full Search)"""
    # 针对 50万 数据，k=None (全量) 可能较慢，建议 k=1000~2000
    # 如果追求极致速度，可设 k=500
    k_val = min(len(B_norm), 1000)

    pairs, _, _ = FF_GM_fullmatch.shortest_edge_first_greedy_matching_faiss_general(
        A_norm, B_norm, k=k_val, extend_search=True, verbose=False
    )
    return pairs


def run_ann_experiment(A_norm, B_norm):
    """运行 ANN Injective"""
    pairs, _, _ = ANN_injective.match_approx_nn_injective(
        A_norm, B_norm, k=50, strict=False
    )
    return pairs


def run_partial_ot_large_scale(A_norm, B_norm):
    """
    【内存安全版 Partial OT】
    1. Faiss 预筛选 Top-K
    2. 局部 Partial OT
    """
    n_a = len(A_norm)
    n_b = len(B_norm)

    # --- 1. 预筛选 ---
    d = B_norm.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(B_norm)

    # 每个 Source 找 K 个 Target 候选
    K = MAX_OT_CANDIDATES
    _, I = index.search(A_norm, K)

    candidate_indices = np.unique(I)
    n_candidates = len(candidate_indices)
    cand_to_orig_map = candidate_indices
    B_subset = B_norm[candidate_indices]

    # --- 2. 运行 OT ---
    m_mass = min(n_a, n_candidates) / max(n_a, n_candidates)

    M = ot.dist(A_norm, B_subset)
    M /= (M.max() + 1e-8)

    a = np.ones((n_a,)) / n_a
    b = np.ones((n_candidates,)) / n_candidates

    # 使用较小的 reg 保证精度
    try:
        P = ot.partial.entropic_partial_wasserstein(
            a, b, M, m=m_mass, reg=0.05, numItermax=1000, stopThr=1e-5, verbose=False
        )
    except Exception:
        # 回退参数
        P = ot.partial.entropic_partial_wasserstein(
            a, b, M, m=m_mass, reg=0.1, numItermax=500, verbose=False
        )

    # --- 3. 解析结果 ---
    threshold = (1.0 / n_a) * 0.01
    rows, cols = np.where(P > threshold)

    final_pairs = []
    for r, c in zip(rows, cols):
        orig_target_idx = cand_to_orig_map[c]
        final_pairs.append((r, orig_target_idx))

    return final_pairs


# ==========================================
# 4. 主程序循环
# ==========================================
def main():
    if not os.path.exists(SOURCE_DIR) or not os.path.exists(TARGET_DIR):
        print(f"错误: 找不到目录 {SOURCE_DIR} 或 {TARGET_DIR}")
        return

    files = sorted([f for f in os.listdir(SOURCE_DIR) if f.endswith('.json')])
    if not files:
        print("未找到 JSON 文件。")
        return

    summary_records = []

    print(f"=== 开始处理 {len(files)} 组数据 ===")
    print(f"统计内容: 运行耗时 (ms) & 物理总Cost (原始距离和)")
    print("-" * 60)

    for idx, filename in enumerate(files):
        src_path = os.path.join(SOURCE_DIR, filename)
        tgt_path = os.path.join(TARGET_DIR, filename)

        if not os.path.exists(tgt_path):
            continue

        print(f"[{idx + 1}/{len(files)}] 正在处理: {filename}")

        # 加载原始数据
        A_raw_np, A_data = load_json_points(src_path)
        B_raw_np, B_data = load_json_points(tgt_path)

        # 归一化 (用于算法输入)
        A_norm = normalize_points(A_raw_np)
        B_norm = normalize_points(B_raw_np)

        tasks = [
            ("FF-GM", run_ffgm_experiment, FFGM_OUT_DIR),
            ("ANN", run_ann_experiment, ANN_OUT_DIR),
            ("Partial_OT", run_partial_ot_large_scale, OT_OUT_DIR)
        ]

        for algo_name, algo_func, out_dir in tasks:
            durations = []
            final_pairs = []

            # --- 1. 运行并计时 ---
            for i in range(REPEAT_TIMES):
                t0 = time.time()
                try:
                    pairs = algo_func(A_norm, B_norm)
                    durations.append((time.time() - t0) * 1000)  # ms
                    if i == REPEAT_TIMES - 1:  # 保留最后一次的结果
                        final_pairs = pairs
                except Exception as e:
                    print(f"    [Error] {algo_name}: {e}")
                    durations.append(0)

            avg_time = np.mean(durations) if durations else 0

            # --- 2. 计算物理 Cost (使用原始坐标) ---
            # 这是关键修改：不在算法内部算，而在外部用原始数据算
            physical_cost = calculate_physical_cost(A_raw_np, B_raw_np, final_pairs)

            # --- 3. 保存 JSON 结果 ---
            if final_pairs:
                save_result_json(algo_name, filename, B_data, final_pairs, A_data, out_dir)

            # --- 4. 记录汇总 ---
            summary_records.append({
                "Filename": filename,
                "Source_Count": len(A_raw_np),
                "Target_Count": len(B_raw_np),
                "Algorithm": algo_name,
                "Avg_Time_ms": round(avg_time, 2),
                "Total_Cost": round(physical_cost, 4),  # 新增 Cost 字段
                "Matched_Count": len(final_pairs)
            })

            print(
                f"    {algo_name:<12} | Time: {avg_time:8.2f} ms | Cost: {physical_cost:12.2f} | Matches: {len(final_pairs)}")

    # 保存汇总表
    df = pd.DataFrame(summary_records)
    # 调整列顺序，好看一点
    cols = ["Filename", "Algorithm", "Source_Count", "Target_Count", "Matched_Count", "Avg_Time_ms", "Total_Cost"]
    df = df[cols]

    csv_path = os.path.join(RESULT_BASE_DIR, "experiment_summary.csv")
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 60)
    print(f"完成! 汇总表已保存至: {csv_path}")


if __name__ == "__main__":
    main()