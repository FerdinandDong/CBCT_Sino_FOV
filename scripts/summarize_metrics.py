#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_metrics.py
汇总 metrics_report.txt，生成对比表格。
特性：
1. 自动去重 Baseline (Noisy/WCE)。
2. I2SB (SOTA) 沉底显示。
3. PConv 系列组内最优加 '*'。
4. I2SB 如果是全局最优也加 '*'。
"""

import os
import re
import datetime
from collections import defaultdict

# ================= 配置区域 =================

# 定义 SOTA 方法的文件夹名关键词（它将被单独放在最后）
# SOTA_KEYWORD = "i2sb_local_1step"
SOTA_KEYWORD = "i2sb_local_multi_NFE1"

ROOT_DIRS = [
    "outputs/i2sb_local_multi_NFE1",
    "outputs/pconv_ds2_L2L1_perc_edg_sty_lpips",
    "outputs/pconv_ds2_L2L1_perc_edg_sty",
    "outputs/pconv_ds2_L2L1_perc_edg",
    "outputs/pconv_ds2_L2L1_perc",
    "outputs/pconv_ds2_L2L1_edg_sty_lpips",
    "outputs/pconv_ds2_L2L1_edg_lpips",
    "outputs/pconv_ds2_L2L1",
    "outputs/pconv_ds2_L2"
]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
LOG_DIR = os.path.join(BASE_DIR, "logs")
REPORT_FILENAME = "metrics_report.txt"

# 指标方向 (True=越大越好, False=越小越好)
METRIC_BETTER_DIRECTION = {
    "RMSE": False, 
    "PSNR": True,   
    "SSIM": True,   
    "LPIPS": False  
}

# ================= 解析逻辑 =================

def parse_single_report(file_path):
    if not os.path.exists(file_path):
        # print(f"[WARN] Not found: {file_path}")
        return None

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data = [] 
    current_task = "Unknown"
    main_method_name = "Unknown"

    for line in lines:
        line = line.strip()
        if line.startswith("Method:") and "|" not in line:
            main_method_name = line.split(":", 1)[1].strip()
            continue
        if line.startswith("--- Task:"):
            current_task = line.replace("---", "").replace("Task:", "").strip()
            continue

        if "|" in line and "---" not in line and "Method" not in line and "Region" not in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 6:
                method_in_row = parts[0]
                region = parts[1]
                try:
                    vals = {
                        "RMSE": float(parts[2]),
                        "PSNR": float(parts[3]),
                        "SSIM": float(parts[4]),
                        "LPIPS": float(parts[5])
                    }
                except ValueError:
                    continue

                final_name = method_in_row
                is_baseline = False
                
                # 识别 Baseline
                if "Noisy" in method_in_row or "WCE" in method_in_row or "GT" in method_in_row:
                    is_baseline = True
                else:
                    # 使用文件夹名覆盖，确保唯一性
                    final_name = main_method_name

                data.append({
                    "task": current_task,
                    "method": final_name,
                    "region": region,
                    "metrics": vals,
                    "is_baseline": is_baseline
                })
    return data

def aggregate_data(root_list):
    database = defaultdict(lambda: defaultdict(dict))
    seen_baselines = set()

    for root in root_list:
        full_path = root if os.path.isabs(root) else os.path.join(BASE_DIR, root)
        rpt_path = os.path.join(full_path, REPORT_FILENAME)
        
        parsed = parse_single_report(rpt_path)
        if not parsed: continue
            
        for item in parsed:
            task = item['task']
            region = item['region']
            method = item['method']
            metrics = item['metrics']
            is_baseline = item['is_baseline']
            
            unique_key = f"{task}|{region}|{method}"

            if is_baseline:
                if unique_key not in seen_baselines:
                    database[task][region][method] = metrics
                    seen_baselines.add(unique_key)
            else:
                database[task][region][method] = metrics
    return database

def find_best_values(methods_dict):
    """找出给定字典中各项指标的最佳值"""
    if not methods_dict:
        return None
        
    best = {
        "RMSE": float('inf'), "PSNR": float('-inf'), 
        "SSIM": float('-inf'), "LPIPS": float('inf')
    }
    
    for _, vals in methods_dict.items():
        if vals["RMSE"] < best["RMSE"]: best["RMSE"] = vals["RMSE"]
        if vals["PSNR"] > best["PSNR"]: best["PSNR"] = vals["PSNR"]
        if vals["SSIM"] > best["SSIM"]: best["SSIM"] = vals["SSIM"]
        if vals["LPIPS"] < best["LPIPS"]: best["LPIPS"] = vals["LPIPS"]
    return best

def format_val(val, comparison_best, metric_name):
    """
    格式化数值。
    comparison_best: 用于对比的最佳值。如果 val 达到了这个最佳值，则加 *
    """
    if comparison_best is None:
        return f"{val:.4f}  "

    is_best = False
    if METRIC_BETTER_DIRECTION[metric_name]: # 越大越好
        if val >= comparison_best[metric_name] - 1e-5: is_best = True
    else: # 越小越好
        if val <= comparison_best[metric_name] + 1e-5: is_best = True
    
    s = f"{val:.4f}"
    if is_best:
        return f"{s} *"
    return f"{s}  "

# ================= 主程序 =================

def main():
    print(f"Scanning {len(ROOT_DIRS)} directories...")
    db = aggregate_data(ROOT_DIRS)
    
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(LOG_DIR, f"summary_metrics_{timestamp}.txt")
    
    lines = []
    lines.append("="*90)
    lines.append(f" EXPERIMENT SUMMARY REPORT")
    lines.append(f" Generated: {timestamp}")
    lines.append("="*90 + "\n")

    sorted_tasks = sorted(db.keys())
    
    for task in sorted_tasks:
        lines.append(f"\n>>> Task: {task}")
        
        regions = db[task].keys()
        region_order = {"Global": 0, "Inner": 1, "Outer": 2}
        sorted_regions = sorted(regions, key=lambda x: region_order.get(x, 99))
        
        for region in sorted_regions:
            methods_data = db[task][region] # {'Noisy': {...}, 'pconv...': {...}, 'i2sb...': {...}}
            if not methods_data: continue
            
            lines.append(f"\n  [Region: {region}]")
            
            # --- 1. 分组 ---
            # Group A: Baselines (Noisy, WCE)
            # Group B: SOTA (I2SB)
            # Group C: Others (PConv variants)
            
            group_baseline = {}
            group_sota = {}
            group_others = {} # 你的 PConv 系列
            
            for m_name, metrics in methods_data.items():
                if "Noisy" in m_name or "WCE" in m_name or "GT" in m_name:
                    group_baseline[m_name] = metrics
                elif SOTA_KEYWORD in m_name:
                    group_sota[m_name] = metrics
                else:
                    group_others[m_name] = metrics
            
            # --- 2. 计算 Best Values ---
            # 这里的逻辑是：
            # Others 组：自己跟自己比，标出组内最优（体现消融实验结果）
            # SOTA 组： 跟全局比，如果是全局第一，标出星号（体现 SOTA 地位或被超越）
            
            best_vals_others = find_best_values(group_others)
            
            # 全局最优 (用于给 SOTA 打星，或者给 Others 双重确认)
            # 如果你只想让 SOTA 在赢了 PConv 时才打星，可以用 find_best_values({**group_others, **group_sota})
            best_vals_global = find_best_values(methods_data)

            # --- 3. 打印表头 ---
            header = f"    {'Method':<40} | {'RMSE':<10} | {'PSNR':<10} | {'SSIM':<10} | {'LPIPS':<10}"
            lines.append("    " + "-"*86)
            lines.append(header)
            lines.append("    " + "-"*86)
            
            # --- 4. 打印 Baseline ---
            # 排序：Noisy 先，然后其他
            def baseline_sort(name): return (0 if "Noisy" in name else 1, name)
            
            for m_name in sorted(group_baseline.keys(), key=baseline_sort):
                vals = group_baseline[m_name]
                # Baseline 通常不打星
                row = f"    {m_name:<40} | {vals['RMSE']:.4f}     | {vals['PSNR']:.4f}     | {vals['SSIM']:.4f}     | {vals['LPIPS']:.4f}    "
                lines.append(row)
                
            # --- 5. 打印 Others (PConv 系列) ---
            # 排序：按名字字母序
            for m_name in sorted(group_others.keys()):
                vals = group_others[m_name]
                
                # 对比基准：best_vals_others (组内 PK)
                v_rmse = format_val(vals["RMSE"], best_vals_others, "RMSE")
                v_psnr = format_val(vals["PSNR"], best_vals_others, "PSNR")
                v_ssim = format_val(vals["SSIM"], best_vals_others, "SSIM")
                v_lpips = format_val(vals["LPIPS"], best_vals_others, "LPIPS")
                
                row = f"    {m_name:<40} | {v_rmse:<10} | {v_psnr:<10} | {v_ssim:<10} | {v_lpips:<10}"
                lines.append(row)
                
            # --- 6. 打印分割线 ---
            if group_sota:
                lines.append("    " + "-"*86)
                
                # --- 7. 打印 SOTA (沉底) ---
                for m_name in sorted(group_sota.keys()):
                    vals = group_sota[m_name]
                    
                    # 对比基准：best_vals_global (SOTA 要跟所有人 PK 才能拿星)
                    # 如果你不想让 SOTA 拿星，只想列出数值，可以把下面 best_vals_global 改为 None
                    v_rmse = format_val(vals["RMSE"], best_vals_global, "RMSE")
                    v_psnr = format_val(vals["PSNR"], best_vals_global, "PSNR")
                    v_ssim = format_val(vals["SSIM"], best_vals_global, "SSIM")
                    v_lpips = format_val(vals["LPIPS"], best_vals_global, "LPIPS")
                    
                    row = f"    {m_name:<40} | {v_rmse:<10} | {v_psnr:<10} | {v_ssim:<10} | {v_lpips:<10}"
                    lines.append(row)

            lines.append("    " + "-"*86)

    # 写入文件
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"\n[DONE] Summary saved to:\n  -> {out_file}")
    print("\n" + "\n".join(lines))

if __name__ == "__main__":
    main()