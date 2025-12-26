import numpy as np
import faiss
import heapq
import json
import os
import glob
from typing import List, Tuple

# ==========================================
# 1. 配置
# ==========================================
SOURCE_DIR_DEFAULT = 'source_points'
TARGET_DIR_DEFAULT = 'target_points'
OUTPUT_DIR = 'viz_results'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 初始化 GPU (如果可用)
GLOBAL_GPU_RES = None
try:
    if hasattr(faiss, 'StandardGpuResources'):
        GLOBAL_GPU_RES = faiss.StandardGpuResources()
        GLOBAL_GPU_RES.setTempMemory(256 * 1024 * 1024)
except Exception:
    pass


# ==========================================
# 2. 核心算法
# ==========================================
def match_approx_nn_injective(A: np.ndarray, B: np.ndarray, k: int = 50):
    # Algo 1: Sort
    N_A = A.shape[0]
    D = A.shape[1]

    use_gpu = (faiss.get_num_gpus() > 0) and (GLOBAL_GPU_RES is not None)
    index = faiss.IndexFlatL2(D)
    if use_gpu:
        index = faiss.index_cpu_to_gpu(GLOBAL_GPU_RES, 0, index)
    index.add(B)

    # Search
    D_sq, I = index.search(A, k)
    valid_mask = I.flatten() >= 0

    # Sort
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
        a, b = int(item['a']), int(item['b'])
        if a not in matched_A and b not in matched_B:
            pairs.append((a, b))
            matched_A.add(a)
            matched_B.add(b)
        if len(matched_A) == N_A: break

    # Fallback
    if len(matched_A) < N_A:
        unmatched_A = [i for i in range(N_A) if i not in matched_A]
        if unmatched_A:
            A_un = np.ascontiguousarray(A[unmatched_A])
            _, I_ext = index.search(A_un, min(500, len(B)))
            for i, real_a in enumerate(unmatched_A):
                for j in range(I_ext.shape[1]):
                    b = int(I_ext[i, j])
                    if b >= 0 and b not in matched_B:
                        pairs.append((real_a, b))
                        matched_A.add(real_a)
                        matched_B.add(b)
                        break
    return pairs


def match_approx_nn_heap(A: np.ndarray, B: np.ndarray, k: int = 50):
    # Algo 2: Heap
    N_A = A.shape[0]
    D = A.shape[1]

    use_gpu = (faiss.get_num_gpus() > 0) and (GLOBAL_GPU_RES is not None)
    index = faiss.IndexFlatL2(D)
    if use_gpu:
        index = faiss.index_cpu_to_gpu(GLOBAL_GPU_RES, 0, index)
    index.add(B)

    D_sq, I = index.search(A, k)

    candidates = []
    for i in range(N_A):
        for j in range(k):
            b = int(I[i, j])
            if b >= 0:
                candidates.append((float(D_sq[i, j]), i, b))
    heapq.heapify(candidates)

    pairs = []
    matched_A = set()
    matched_B = set()

    while candidates:
        _, a, b = heapq.heappop(candidates)
        if a not in matched_A and b not in matched_B:
            pairs.append((a, b))
            matched_A.add(a)
            matched_B.add(b)
        if len(pairs) == N_A: break

    # Fallback
    if len(pairs) < N_A:
        unmatched_A = [i for i in range(N_A) if i not in matched_A]
        if unmatched_A:
            A_un = np.ascontiguousarray(A[unmatched_A])
            D_ext, I_ext = index.search(A_un, min(100, len(B)))
            ext_cands = []
            for i, real_a in enumerate(unmatched_A):
                for j in range(I_ext.shape[1]):
                    b = int(I_ext[i, j])
                    if b not in matched_B:
                        ext_cands.append((float(D_ext[i, j]), real_a, b))
            heapq.heapify(ext_cands)
            while ext_cands:
                _, a, b = heapq.heappop(ext_cands)
                if a not in matched_A and b not in matched_B:
                    pairs.append((a, b))
                    matched_A.add(a)
                    matched_B.add(b)
    return pairs


# ==========================================
# 3. 数据处理 (修复版: 兼容 Dict 和 List 格式)
# ==========================================
def load_data_safe(s_path, t_path=None):
    # --- 1. Load Source ---
    with open(s_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 兼容处理: 有些 JSON 根节点是 {"points": [...]}
    if isinstance(data, dict):
        for key in ['points', 'data', 'nodes']:
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

    coords = []
    labels = []

    for item in data:
        # 情况 A: item 是字典 {"x": 1, "y": 2, "label": 0}
        if isinstance(item, dict):
            try:
                x = float(item.get('x', item.get('0', 0)))
                y = float(item.get('y', item.get('1', 0)))
                l = int(item.get('label', item.get('2', 0)))
                coords.append([x, y])
                labels.append(l)
            except:
                continue  # 数据坏了就跳过

        # 情况 B: item 是列表 [1.0, 2.0, 0] (修复你报错的关键部分)
        elif isinstance(item, (list, tuple)):
            if len(item) >= 2:
                # 假设前两个是 x, y
                x = float(item[0])
                y = float(item[1])
                # 如果有第三个，假设是 label
                l = int(item[2]) if len(item) > 2 else 0
                coords.append([x, y])
                labels.append(l)

    if not coords:
        # 如果没读到数据，创建一个空的防止报错
        print(f"Warning: No valid points found in {os.path.basename(s_path)}")
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), []

    A = np.array(coords, dtype=np.float32)

    # --- 2. Load or Generate Target ---
    B = None
    if t_path and os.path.exists(t_path):
        print(f"  -> Loading target from {os.path.basename(t_path)}")
        # 复用上面的读取逻辑读取 B
        with open(t_path, 'r', encoding='utf-8') as f:
            t_data = json.load(f)

        # 同样兼容 List/Dict
        if isinstance(t_data, dict):
            for key in ['points', 'data']:
                if key in t_data and isinstance(t_data[key], list):
                    t_data = t_data[key]
                    break

        b_coords = []
        for item in t_data:
            if isinstance(item, dict):
                b_coords.append([float(item.get('x', 0)), float(item.get('y', 0))])
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                b_coords.append([float(item[0]), float(item[1])])

        B = np.array(b_coords, dtype=np.float32)

    if B is None or len(B) == 0:
        print(f"  -> Generating SYNTHETIC target (Rotation+Noise)...")
        theta = np.radians(5)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        center = A.mean(axis=0) if len(A) > 0 else np.array([0, 0])
        noise = np.random.normal(0, 2.0, A.shape)
        # 稍微偏移一点
        B = (A - center).dot(R) + center + np.array([5.0, -5.0]) + noise
        B = B.astype(np.float32)

    return A, B, labels


# ==========================================
# 4. 生成 HTML
# ==========================================
def generate_html(A, B, labels, p1, p2, filename):
    if len(B) == 0: return

    radius = 800.0 / np.sqrt(len(B)) / 2.0

    all_p = np.vstack([A, B])
    min_x, max_x = all_p[:, 0].min(), all_p[:, 0].max()
    min_y, max_y = all_p[:, 1].min(), all_p[:, 1].max()
    pad_x = max((max_x - min_x) * 0.05, 1.0)
    pad_y = max((max_y - min_y) * 0.05, 1.0)

    bounds = {
        "minX": float(min_x - pad_x), "maxX": float(max_x + pad_x),
        "minY": float(min_y - pad_y), "maxY": float(max_y + pad_y)
    }

    # 构造数据
    viz_data = {
        "radius": float(radius),
        "bounds": bounds,
        "pointsA": [{"x": float(p[0]), "y": float(p[1]), "label": int(l)} for p, l in zip(A, labels)],
        "pointsB": [{"x": float(p[0]), "y": float(p[1]), "label": int(labels[i]) if i < len(labels) else 0} for i, p in
                    enumerate(B)],
        "links1": [{"source": int(u), "target": int(v)} for u, v in p1],
        "links2": [{"source": int(u), "target": int(v)} for u, v in p2]
    }

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
 body {{ font-family: 'Segoe UI', sans-serif; text-align: center; background: #f4f4f4; padding: 20px; }}
 .container {{ background: #fff; display: inline-block; padding: 20px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
 h2 {{ margin-top: 0; color: #333; }}
 .controls {{ margin-bottom: 15px; }}
 button {{ padding: 8px 16px; margin: 0 5px; cursor: pointer; border: 1px solid #ccc; background: white; border-radius: 4px; transition: 0.2s; }}
 button:hover {{ background: #eee; }}
 button.active {{ background: #007bff; color: #fff; border-color: #0056b3; }}
 #viz {{ border: 1px solid #eee; margin-top: 10px; }}
</style>
</head>
<body>
<div class="container">
 <h2>Matching Result: {filename}</h2>
 <div class="controls">
  <button id="b1" onclick="draw(1)" class="active">Algo 1 (Sort)</button>
  <button id="b2" onclick="draw(2)">Algo 2 (Heap)</button>
  <span style="font-size:12px; color:#888; margin-left:10px">Radius: {radius:.2f}px</span>
 </div>
 <div id="viz"></div>
</div>
<script>
 const data = {json.dumps(viz_data)};
 const w=800, h=800;
 const svg = d3.select("#viz").append("svg").attr("width",w).attr("height",h).style("background", "white");

 const x = d3.scaleLinear().domain([data.bounds.minX, data.bounds.maxX]).range([20, w-20]);
 const y = d3.scaleLinear().domain([data.bounds.minY, data.bounds.maxY]).range([h-20, 20]);
 const col = d3.scaleOrdinal(d3.schemeCategory10);

 const gL = svg.append("g"), gN = svg.append("g");

 // Draw Nodes
 gN.selectAll(".pA").data(data.pointsA).enter().append("circle")
   .attr("cx", d=>x(d.x)).attr("cy", d=>y(d.y)).attr("r", data.radius).attr("fill", d=>col(d.label));

 gN.selectAll(".pB").data(data.pointsB).enter().append("circle")
   .attr("cx", d=>x(d.x)).attr("cy", d=>y(d.y)).attr("r", data.radius).attr("fill", "#ddd")
   .attr("stroke", d=>col(d.label)).attr("stroke-width", Math.max(1, data.radius/2)).attr("opacity",0.7);

 function draw(id) {{
   d3.selectAll("button").classed("active", false); d3.select("#b"+id).classed("active", true);
   const lnk = (id===1)? data.links1 : data.links2;
   const c = (id===1)? "#28a745" : "#007bff";

   const l = gL.selectAll("line").data(lnk, d=>d.source+"-"+d.target);
   l.exit().remove();
   l.enter().append("line").merge(l)
    .attr("x1", d=>x(data.pointsA[d.source].x)).attr("y1", d=>y(data.pointsA[d.source].y))
    .attr("x2", d=>x(data.pointsB[d.target].x)).attr("y2", d=>y(data.pointsB[d.target].y))
    .attr("stroke", c).attr("stroke-width", Math.max(0.5, data.radius/2)).attr("stroke-opacity",0.6);
 }}
 draw(1);
</script>
</body>
</html>
    """
    out_name = os.path.join(OUTPUT_DIR, f"viz_{filename.replace('.json', '')}.html")
    with open(out_name, 'w', encoding='utf-8') as f: f.write(html)
    print(f"  -> Saved: {out_name}")


# ==========================================
# 5. 主程序
# ==========================================
def main():
    # 搜索策略
    search_path = SOURCE_DIR_DEFAULT
    json_files = glob.glob(os.path.join(search_path, "*.json"))

    if not json_files:
        print(f"Warning: No files in '{SOURCE_DIR_DEFAULT}'. Checking current directory...")
        search_path = "."
        json_files = glob.glob("*.json")
        json_files = [f for f in json_files if not f.startswith("viz_") and f != "package.json"]

    if not json_files:
        print("Error: No .json files found.")
        return

    print(f"Found {len(json_files)} datasets.")

    for f_path in json_files:
        fname = os.path.basename(f_path)
        print(f"\nProcessing {fname}...")

        t_path = os.path.join(TARGET_DIR_DEFAULT, fname)

        # 加载 (自动处理 字典 vs 列表 格式)
        A, B, labels = load_data_safe(f_path, t_path)

        if len(A) == 0:
            print(f"Skipping {fname} (empty or invalid).")
            continue

        p1 = match_approx_nn_injective(A, B)
        p2 = match_approx_nn_heap(A, B)

        generate_html(A, B, labels, p1, p2, fname)

    print(f"\nDone! Check '{OUTPUT_DIR}' folder.")


if __name__ == "__main__":
    main()