import os
import json
import time
import numpy as np
import pandas as pd
import ot
import faiss
import warnings
import copy
from datetime import datetime
import shutil
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- 尝试导入 CuPy (用于 POT 加速) ---
try:
    import cupy as cp
    import ot.backend

    # 检查 GPU 是否可用
    if cp.cuda.is_available():
        USE_CUPY = True
        print("[System] CuPy 已启用，Partial OT 将使用 GPU 加速！")
    else:
        USE_CUPY = False
        print("[System] CuPy 已安装但 CUDA 不可用，Partial OT 将使用 CPU。")
except ImportError:
    USE_CUPY = False
    print("[System] 未检测到 CuPy，Partial OT 将使用 CPU (建议 pip install cupy-cuda12x)。")

# 忽略警告
warnings.filterwarnings("ignore")

try:
    import FF_GM_fullmatch
    import ANN_injective
except ImportError as e:
    print(f"错误: 缺少必要的算法文件。\n{e}")
    exit(1)

# ================= 配置 =================
SOURCE_DIR = "source_points"
TARGET_DIR = "target_points"
RESULT_BASE_DIR = "results"
FFGM_OUT_DIR = os.path.join(RESULT_BASE_DIR, "FF_GM")
ANN_OUT_DIR = os.path.join(RESULT_BASE_DIR, "ANN")
OT_OUT_DIR = os.path.join(RESULT_BASE_DIR, "Partial_OT")
D3_DATA_DIR = os.path.join(RESULT_BASE_DIR, "D3_Data")  # 新增 D3 数据保存目录

for d in [FFGM_OUT_DIR, ANN_OUT_DIR, OT_OUT_DIR, D3_DATA_DIR]:
    os.makedirs(d, exist_ok=True)

REPEAT_TIMES = 3
SAMPLE_DATASETS = 3
SAMPLE_SEED = None
PLOT_ENABLED = True
PLOT_SIZE = 800
PLOT_DIR = os.path.join(RESULT_BASE_DIR, "plots")
MAX_OT_CANDIDATES = 5  # K=5 极速模式
MAX_OT_TARGET_SIZE = 50000


# ================= 工具函数 =================
def load_json_points(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    points = []
    for item in raw_data:
        x, y = None, None
        if isinstance(item, dict):
            x = item.get('x', item.get('lon', item.get('Longitude', None)))
            y = item.get('y', item.get('lat', item.get('Latitude', None)))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            x, y = item[0], item[1]
        if x is not None and y is not None:
            points.append([float(x), float(y)])
    return np.array(points, dtype=np.float32), raw_data


def _get_xy_from_item(item):
    if isinstance(item, dict):
        x = item.get('x', item.get('lon', item.get('Longitude', None)))
        y = item.get('y', item.get('lat', item.get('Latitude', None)))
    elif isinstance(item, (list, tuple)) and len(item) >= 2:
        x, y = item[0], item[1]
    else:
        return None

    if x is None or y is None:
        return None
    try:
        return float(x), float(y)
    except (TypeError, ValueError):
        return None


def parse_points_and_labels(raw_data, prefer_matched_label=False):
    points = []
    labels = []
    for item in raw_data:
        xy = _get_xy_from_item(item)
        if xy is None:
            continue
        x, y = xy

        label = "unmatched"
        if isinstance(item, dict):
            if prefer_matched_label and item.get("matched_source_label") is not None:
                label = item.get("matched_source_label")
            elif item.get("label") is not None:
                label = item.get("label")
            elif item.get("matched_source_id") is not None:
                label = item.get("matched_source_id")

        points.append([x, y])
        labels.append(str(label))

    return np.array(points, dtype=np.float32), labels


def normalize_points(X):
    if len(X) == 0: return X
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std


def calculate_physical_cost(A_raw, B_raw, pairs):
    if not pairs: return 0.0
    total_cost = 0.0
    if len(pairs) > 0:
        arr = np.array(pairs)
        diff = A_raw[arr[:, 0]] - B_raw[arr[:, 1]]
        total_cost = np.sum(np.sqrt(np.sum(diff ** 2, axis=1)))
    return total_cost


def save_result_json(algo_name, filename, target_raw_data, matching_pairs, source_raw_data, output_dir):
    match_dict = {int(t_idx): int(s_idx) for s_idx, t_idx in matching_pairs}
    final_output_list = []
    valid_idx = 0
    for item in target_raw_data:
        item_out = None
        is_valid = False
        if isinstance(item, dict) and (item.get('x') is not None or item.get('lon') is not None):
            item_out = copy.deepcopy(item)
            is_valid = True
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            item_out = {"x": item[0], "y": item[1]}
            is_valid = True

        if not is_valid:
            final_output_list.append(item)
            continue

        if valid_idx in match_dict:
            src_idx = match_dict[valid_idx]
            if src_idx < len(source_raw_data):
                src_item = source_raw_data[src_idx]
                s_id = src_item.get('id', 'N/A') if isinstance(src_item, dict) else 'N/A'
                s_label = src_item.get('label', 'N/A') if isinstance(src_item, dict) else 'N/A'
                item_out['matched_source_id'] = s_id
                item_out['matched_source_label'] = s_label

        final_output_list.append(item_out)
        valid_idx += 1

    out_path = os.path.join(output_dir, filename)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(final_output_list, f, ensure_ascii=False, indent=2)


# ================= 新增：D3 数据导出函数 =================
def save_d3_viz_data(filename, source_points_np, target_points_np, source_raw_data, all_algo_matches, output_dir):
    """
    生成专门供 D3.js 使用的 JSON 数据
    """
    # 1. 提取 Labels (对应 Source Points)
    labels = []
    # 确保和 load_json_points 的顺序一致
    for item in source_raw_data:
        xy = _get_xy_from_item(item)
        if xy is not None:
            l = "unknown"
            if isinstance(item, dict):
                l = item.get('label', item.get('id', 'unknown'))
            labels.append(str(l))

    # 2. 格式化 mappings
    # 我们需要一个数组，长度等于 source points 的数量。
    # array[i] = target_index (如果匹配) 或 -1 (如果未匹配)
    formatted_mappings = {}
    num_source = len(source_points_np)

    for algo_name, pairs in all_algo_matches.items():
        # 初始化全为 -1
        match_list = [-1] * num_source
        if pairs:
            for s_idx, t_idx in pairs:
                if s_idx < num_source:
                    match_list[s_idx] = int(t_idx)
        formatted_mappings[algo_name] = match_list

    # 3. 构建总数据结构
    export_data = {
        "filename": filename,
        "source_points": source_points_np.tolist(),
        "target_points": target_points_np.tolist(),
        "labels": labels,
        "mappings": formatted_mappings
    }

    # 保存
    # 去掉扩展名，加前缀
    base_name = os.path.splitext(filename)[0]
    out_name = f"viz_data_{base_name}.json"
    out_path = os.path.join(output_dir, out_name)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f)
    print(f"    [D3] 可视化数据已保存: {out_name}")


# ================= 算法封装 =================

def run_ffgm_experiment(A, B):
    k_val = min(len(B), 2000)
    pairs, _, _ = FF_GM_fullmatch.shortest_edge_first_greedy_matching_faiss_general(
        A, B, k=k_val, extend_search=True, verbose=False
    )
    return pairs


def run_ann_experiment(A, B):
    pairs, _, _ = ANN_injective.match_approx_nn_injective(A, B, k=50, strict=False)
    return pairs


def run_partial_ot_safe(A, B, filename):
    n_a = len(A)
    n_b = len(B)
    if n_b > MAX_OT_TARGET_SIZE and not USE_CUPY:
        print(f"    [跳过] Target过大且无GPU加速，跳过 Partial OT。")
        return []

    try:
        res = faiss.StandardGpuResources() if hasattr(faiss, 'StandardGpuResources') else None
        if faiss.get_num_gpus() > 0:
            cpu_index = faiss.IndexFlatL2(B.shape[1])
            index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            index = faiss.IndexFlatL2(B.shape[1])

        index.add(B)
        K = MAX_OT_CANDIDATES
        _, I = index.search(A, K)

        candidate_indices = np.unique(I)
        n_candidates = len(candidate_indices)
        cand_to_orig_map = candidate_indices
        B_subset = B[candidate_indices]

        a = np.ones((n_a,)) / n_a
        b = np.ones((n_candidates,)) / n_a
        m_mass = min(0.99, np.sum(b) - 0.001)

        if USE_CUPY:
            A_gpu = cp.array(A)
            B_sub_gpu = cp.array(B_subset)
            a_gpu = cp.array(a)
            b_gpu = cp.array(b)
            M_gpu = ot.dist(A_gpu, B_sub_gpu)
            M_gpu /= (cp.max(M_gpu) + 1e-8)
            P_gpu = ot.partial.entropic_partial_wasserstein(
                a_gpu, b_gpu, M_gpu, m=m_mass, reg=0.05, numItermax=500, stopThr=1e-7, verbose=False
            )
            P = cp.asnumpy(P_gpu)
        else:
            M = ot.dist(A, B_subset)
            M /= (M.max() + 1e-8)
            P = ot.partial.entropic_partial_wasserstein(
                a, b, M, m=m_mass, reg=0.05, numItermax=500, stopThr=1e-7, verbose=False
            )

        target_indices = np.argmax(P, axis=1)
        max_probs = np.max(P, axis=1)

        final_pairs = []
        for i in range(n_a):
            if max_probs[i] > 1e-9:
                c = target_indices[i]
                orig_target_idx = cand_to_orig_map[c]
                final_pairs.append((i, orig_target_idx))
        return final_pairs
    except Exception as e:
        print(f"    [异常] OT 运行出错: {e}")
        return []


# ================= 主程序 =================
def main():
    source_files = {f for f in os.listdir(SOURCE_DIR) if f.endswith('.json')}
    target_files = {f for f in os.listdir(TARGET_DIR) if f.endswith('.json')}
    all_files = sorted(source_files & target_files)
    files = all_files
    files = ["Person_activity.json"]

    csv_path = os.path.join(RESULT_BASE_DIR, "experiment_summary.csv")

    if not os.path.exists(csv_path):
        df_header = pd.DataFrame(
            columns=["Filename", "Algorithm", "Source_Count", "Target_Count", "Matched_Count", "Avg_Time_ms",
                     "Total_Cost"])
        df_header.to_csv(csv_path, index=False)

    if SAMPLE_DATASETS is not None and SAMPLE_DATASETS > 0 and SAMPLE_DATASETS < len(files):
        rng = np.random.default_rng(SAMPLE_SEED)
        indices = rng.choice(len(files), SAMPLE_DATASETS, replace=False)
        files = [files[i] for i in sorted(indices)]
        print(f"Sampled datasets: {len(files)} / {len(all_files)}")

    print(f"=== 开始处理 {len(files)} 组数据 (GPU 加速 + D3 导出版) ===")

    for idx, filename in enumerate(files):
        src_path = os.path.join(SOURCE_DIR, filename)
        tgt_path = os.path.join(TARGET_DIR, filename)
        if not os.path.exists(tgt_path): continue

        print(f"[{idx + 1}/{len(files)}] 检查: {filename}")

        # 加载
        A_raw_np, A_data = load_json_points(src_path)
        B_raw_np, B_data = load_json_points(tgt_path)
        # [修改] 不再对点坐标做归一化处理，直接使用原始坐标 (与 batch_compare_algorithms.py 保持一致)
        A = A_raw_np
        B = B_raw_np

        # 存储当前文件所有算法的匹配结果，用于 D3 导出
        current_file_matches = {}

        tasks = [
            ("FF-GM", run_ffgm_experiment, FFGM_OUT_DIR),
            ("ANN", run_ann_experiment, ANN_OUT_DIR),
            ("Partial_OT", run_partial_ot_safe, OT_OUT_DIR)
        ]

        for algo_name, algo_func, out_dir in tasks:
            durations = []
            final_pairs = []

            # 简单的重试逻辑
            for i in range(REPEAT_TIMES):
                t0 = time.time()
                try:
                    if algo_name == "Partial_OT":
                        pairs = algo_func(A, B, filename)
                    else:
                        pairs = algo_func(A, B)
                    durations.append((time.time() - t0) * 1000)
                    if i == REPEAT_TIMES - 1:
                        final_pairs = pairs
                except Exception as e:
                    print(f"    [错误] {algo_name}: {e}")
                    durations.append(0)

            # 记录结果到字典中，供 D3 导出使用
            current_file_matches[algo_name] = final_pairs

            # 计算指标并保存 CSV/JSON
            avg_time = np.mean(durations) if durations else 0
            physical_cost = calculate_physical_cost(A_raw_np, B_raw_np, final_pairs)

            if final_pairs:
                save_result_json(algo_name, filename, B_data, final_pairs, A_data, out_dir)

            record = {
                "Filename": filename,
                "Source_Count": len(A_raw_np),
                "Target_Count": len(B_raw_np),
                "Algorithm": algo_name,
                "Avg_Time_ms": round(avg_time, 2),
                "Total_Cost": round(physical_cost, 4),
                "Matched_Count": len(final_pairs)
            }
            pd.DataFrame([record]).to_csv(csv_path, mode='a', header=False, index=False)
            print(f"    {algo_name:<12} | Matches: {len(final_pairs)}")

        # === 关键步骤：本组数据处理完后，导出 D3 JSON ===
        save_d3_viz_data(filename, A_raw_np, B_raw_np, A_data, current_file_matches, D3_DATA_DIR)

    print("\n" + "=" * 60)
    print(f"全部完成! 结果已保存至: {csv_path}")
    print(f"D3 可视化数据已保存至: {D3_DATA_DIR}")


if __name__ == "__main__":
    main()