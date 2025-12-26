import os
import json
import time
import numpy as np
import ot
import faiss
import csv

# ================= 核心配置区域 =================

# 1. 文件夹路径配置
SOURCE_DIR = "source_points"
TARGET_DIR = "target_points"

# 2. 保护机制 & 归档
# 4090 显存很大，设置单边最大点数限制 (例如 100万)
MAX_POINTS_LIMIT = 1000000
ARCHIVE_FILE = "batch_results_50_files.csv"

# 3. OT 参数 (完全映射配置)
# 设为 1.0 代表尝试将 Source 的所有点都匹配到 Target 中
MASS_TO_TRANSPORT = 1.0
REG_STRENGTH = 0.05
CANDIDATE_K = 10

# ===========================================

# 尝试导入 cupy 以利用 4090 加速
try:
    import cupy as cp

    HAS_GPU = True
    print(f"✅ 检测到 GPU 环境 (Cupy), 将使用 RTX 4090 加速计算。")
except ImportError:
    HAS_GPU = False
    print("⚠️ 未检测到 cupy，将使用 CPU 运行。")


def load_json_points(file_path):
    if not os.path.exists(file_path):
        return np.array([])

    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️  JSON 解析错误: {os.path.basename(file_path)}")
            return np.array([])

    # ================= 兼容性增强 =================
    # 如果读取到的是字典，尝试自动提取内部的列表数据
    if isinstance(data, dict):
        # 1. 优先查找可能的键名
        possible_keys = ['target_points', 'source_points', 'points', 'data', 'target', 'coordinates']
        extracted = False
        for key in possible_keys:
            if key in data and isinstance(data[key], list):
                data = data[key]
                extracted = True
                break

        # 2. 如果没找到，尝试提取第一个是 list 类型的 Value
        if not extracted:
            for v in data.values():
                if isinstance(v, list) and len(v) > 0:
                    data = v
                    extracted = True
                    break
    # ============================================

    # 再次检查是否为列表，如果还不是，说明格式真的不对
    if not isinstance(data, list):
        # print(f"⚠️  警告: {os.path.basename(file_path)} 结构不包含列表数据")
        return np.array([])

    points = []
    for item in data:
        # 情况 A: 列表中的项是字典 (e.g. {"x":1, "y":2})
        if isinstance(item, dict):
            x = item.get('x', item.get('lon', item.get('Longitude', 0)))
            y = item.get('y', item.get('lat', item.get('Latitude', 0)))
            points.append([float(x), float(y)])
        # 情况 B: 列表中的项是列表/元组 (e.g. [1, 2])
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            x, y = item[0], item[1]
            points.append([float(x), float(y)])

    return np.array(points, dtype=np.float32)


def archive_result(data_dict):
    """将运行结果追加写入 CSV"""
    file_exists = os.path.isfile(ARCHIVE_FILE)
    with open(ARCHIVE_FILE, 'a', newline='') as f:
        fieldnames = ["timestamp", "dataset", "source_file", "n_source", "n_target",
                      "status", "device", "time_sec", "cost", "coverage", "m_value"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        row = {k: data_dict.get(k, "N/A") for k in fieldnames}
        writer.writerow(row)


def run_task_standalone(dataset_name, filename):
    src_path = os.path.join(SOURCE_DIR, filename)
    tgt_path = os.path.join(TARGET_DIR, filename)

    print(f"\n{'=' * 60}")
    print(f"🚀 任务启动: [{filename}]")

    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": dataset_name,
        "source_file": filename,
        "device": "GPU" if HAS_GPU else "CPU",
        "m_value": MASS_TO_TRANSPORT
    }

    # 1. 检查文件是否存在
    if not os.path.exists(tgt_path):
        print(f"❌ 跳过: Target 目录中未找到对应文件 {filename}")
        record["status"] = "Skipped: Target Missing"
        archive_result(record)
        return

    # 2. 加载数据
    A_raw = load_json_points(src_path)
    B_raw = load_json_points(tgt_path)
    n_a, n_b = len(A_raw), len(B_raw)

    record["n_source"] = n_a
    record["n_target"] = n_b
    print(f"   数据量 -> Source: {n_a}, Target: {n_b}")

    # 3. 保护机制
    if n_a == 0 or n_b == 0:
        record["status"] = "Skipped: Empty File"
        archive_result(record);
        return

    if n_a > MAX_POINTS_LIMIT or n_b > MAX_POINTS_LIMIT:
        record["status"] = "Skipped: Too Large"
        archive_result(record);
        return

    try:
        t_start = time.time()

        # 4. 预处理 (Z-score + Faiss)
        print(f"   [Step 1] 建立索引并筛选邻域 (K={CANDIDATE_K})...")
        A_norm = (A_raw - np.mean(A_raw, 0)) / (np.std(A_raw, 0) + 1e-8)
        B_norm = (B_raw - np.mean(B_raw, 0)) / (np.std(B_raw, 0) + 1e-8)

        d = A_norm.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(B_norm)

        _, I = index.search(A_norm, CANDIDATE_K)
        candidate_indices = np.unique(I)
        B_subset_norm = B_norm[candidate_indices]
        n_cand = len(candidate_indices)

        # 5. 准备 OT 数据 (GPU/CPU)
        if HAS_GPU:
            xp = cp
            A_gpu = xp.asarray(A_norm)
            B_sub_gpu = xp.asarray(B_subset_norm)
            M = xp.array(ot.dist(A_gpu, B_sub_gpu))
            a = xp.ones(n_a)
            b = xp.ones(n_cand)
        else:
            xp = np
            M = ot.dist(A_norm, B_subset_norm)
            a = np.ones(n_a)
            b = np.ones(n_cand)

        M /= (M.max() + 1e-8)

        # 6. 计算 m (完全映射逻辑)
        m_calc = int(n_a * MASS_TO_TRANSPORT)
        m_final = min(m_calc, n_cand)

        print(f"   [Step 2] 计算 Partial OT (m={m_final}/{n_a})...")
        P = ot.partial.entropic_partial_wasserstein(
            a, b, M, m=m_final, reg=REG_STRENGTH, numItermax=500
        )

        # 7. 解析结果
        if HAS_GPU: P = cp.asnumpy(P)

        max_probs = np.max(P, axis=1)
        target_indices_local = np.argmax(P, axis=1)
        matched_mask = max_probs > 1e-9

        rows = np.where(matched_mask)[0]
        cols_local = target_indices_local[matched_mask]
        cols_global = candidate_indices[cols_local]

        # 8. 计算物理 Cost
        final_cost = 0.0
        if len(rows) > 0:
            diff = A_raw[rows] - B_raw[cols_global]
            final_cost = np.sum(np.linalg.norm(diff, axis=1))

        elapsed = time.time() - t_start
        coverage = len(rows) / n_a

        record["status"] = "Success"
        record["time_sec"] = round(elapsed, 4)
        record["cost"] = round(final_cost, 4)
        record["coverage"] = round(coverage, 4)

        print(f"✅ 完成! 耗时: {elapsed:.2f}s | 覆盖率: {coverage * 100:.1f}% | Cost: {final_cost:.2f}")
        archive_result(record)

    except Exception as e:
        print(f"❌ 发生错误: {e}")
        record["status"] = f"Error: {str(e)}"
        archive_result(record)


if __name__ == "__main__":
    # 确保文件夹存在
    if not os.path.exists(SOURCE_DIR) or not os.path.exists(TARGET_DIR):
        print(f"错误: 找不到文件夹 '{SOURCE_DIR}' 或 '{TARGET_DIR}'")
        exit()

    # 1. 扫描所有 .json 文件
    all_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".json")]
    all_files.sort()  # 排序，保证顺序一致

    total_files = len(all_files)
    print(f"📂 扫描完毕: 在 '{SOURCE_DIR}' 中发现 {total_files} 个任务文件")

    # 2. 开始遍历循环
    for i, filename in enumerate(all_files):
        print(f"\nProcessing {i + 1}/{total_files}...")
        dataset_name = os.path.splitext(filename)[0]
        run_task_standalone(dataset_name, filename)

    print(f"\n🎉 所有 {total_files} 个文件的任务已全部结束。")
    print(f"📄 结果已保存在: {ARCHIVE_FILE}")

    if __name__ == "__main__":
        # 1. 路径检查 (保持与 batch_experiment_pics 一致的文件夹名)
        if not os.path.exists(SOURCE_DIR) or not os.path.exists(TARGET_DIR):
            print(f"错误: 找不到文件夹路径 '{SOURCE_DIR}' 或 '{TARGET_DIR}'")
            exit(1)

        # 2. 获取文件列表 (完全复制 batch_experiment_pics.py 的逻辑)
        # 使用集合 (set) 获取文件名，以便取交集
        source_files = {f for f in os.listdir(SOURCE_DIR) if f.endswith('.json')}
        target_files = {f for f in os.listdir(TARGET_DIR) if f.endswith('.json')}

        # 取交集并排序，确保只处理两边都有的文件，且顺序一致
        all_files = sorted(source_files & target_files)

        print(f"=== Debug 模式: 准备处理 {len(all_files)} 组数据 ===")
        print(f"📂 Source 目录: {SOURCE_DIR} (共 {len(source_files)} 个 json)")
        print(f"📂 Target 目录: {TARGET_DIR} (共 {len(target_files)} 个 json)")
        print(f"🔗 匹配成功 (交集): {len(all_files)} 个文件")
        print("-" * 60)

        # 3. 遍历执行
        for idx, filename in enumerate(all_files):
            print(f"\n[{idx + 1}/{len(all_files)}] 处理: {filename}")
            dataset_name = os.path.splitext(filename)[0]
            run_task_standalone(dataset_name, filename)

        print("\n" + "=" * 60)
        print(f"🎉 全部完成! 结果已保存至: {ARCHIVE_FILE}")