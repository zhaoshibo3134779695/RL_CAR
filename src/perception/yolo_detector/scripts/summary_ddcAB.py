#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取多个街区电动车检测结果文件，整合输出汇总报告
"""
import os
import re
import time
from pathlib import Path
from runtime_paths import RUNTIME_DDC_TXT, VOICE_RUNTIME_DIR, ensure_runtime_dirs

# ========== 配置区域 ==========
INPUT_DIR = str(RUNTIME_DDC_TXT)
INPUT_FILES = ["weiting_A.txt", "weiting_B.txt"]  # 违停文件列表
DDC_FILE = "ddczuizhong.txt"  # 电动车状态文件
OUTPUT_FILE = str(VOICE_RUNTIME_DIR / "ddc.txt")
WAIT_TIMEOUT = 20.0   # 阈值：最多等待秒数
WAIT_INTERVAL = 0.5   # 轮询间隔秒
# ==============================


def parse_violation_report(file_path):
    """
    解析单个街区违停报告文件
    :param file_path: 文件路径
    :return: dict 包含 {street: str, violations: int}，文件不存在时返回默认值
    """
    try:
        if not os.path.exists(file_path):
            # 从文件名推断街区（如 weiting_A.txt -> A街区）
            filename = os.path.basename(file_path)
            street_match = re.search(r'_([A-Z])\.txt', filename)
            street = street_match.group(1) if street_match else "未知"
            print(f"文件不存在: {file_path}，将{street}街区违停数计为0")
            return {
                'street': street,
                'violations': 0
            }
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # 提取街区名称（如 "A街区"）
        street_match = re.search(r'([A-Z])街区', content)
        street = street_match.group(1) if street_match else "未知"
        
        # 提取违停数量（如 "A街区有3辆电动车违停"）
        violation_match = re.search(r'有(\d+)辆电动车违停', content)
        violations = int(violation_match.group(1)) if violation_match else 0
        
        return {
            'street': street,
            'violations': violations
        }
    except Exception as e:
        print(f"解析文件 {file_path} 失败: {e}，将违停数计为0")
        # 尝试从文件名推断街区
        filename = os.path.basename(file_path)
        street_match = re.search(r'_([A-Z])\.txt', filename)
        street = street_match.group(1) if street_match else "未知"
        return {
            'street': street,
            'violations': 0
        }


def parse_ddc_status(file_path):
    """
    解析电动车状态文件
    :param file_path: 文件路径
    :return: dict 包含 {normal: int, fallen: int}，文件不存在时返回 {0, 0}
    """
    try:
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}，将电动车正常数和倒伏数计为0")
            return {'normal': 0, 'fallen': 0}
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # 提取正常数量（如 "电动车正常5辆"）
        normal_match = re.search(r'电动车正常(\d+)辆', content)
        normal = int(normal_match.group(1)) if normal_match else 0
        
        # 提取倒伏数量（如 "倒伏2辆"）
        fallen_match = re.search(r'倒伏(\d+)辆', content)
        fallen = int(fallen_match.group(1)) if fallen_match else 0
        
        return {
            'normal': normal,
            'fallen': fallen
        }
    except Exception as e:
        print(f"解析文件 {file_path} 失败: {e}，将电动车正常数和倒伏数计为0")
        return {'normal': 0, 'fallen': 0}


def wait_for_files(dir_path: str, required_files: list, timeout: float = WAIT_TIMEOUT, interval: float = WAIT_INTERVAL) -> bool:
    """
    轮询等待 required_files 在 dir_path 中全部存在；在 timeout 秒内若齐备返回 True，否则 False。
    注意：返回 False 不会中断流程，仅表示超时，后续逻辑会将缺失文件计为 0。
    """
    start = time.time()
    req_paths = [os.path.join(dir_path, f) for f in required_files]
    while time.time() - start < timeout:
        missing = [f for f, p in zip(required_files, req_paths) if not os.path.exists(p)]
        if not missing:
            print(f"检测到三个文件齐备：{', '.join(required_files)}，开始汇总。")
            return True
        time.sleep(interval)
    missing = [f for f, p in zip(required_files, req_paths) if not os.path.exists(p)]
    print(f"超过{int(timeout)}秒仍未齐备，缺失：{', '.join(missing) if missing else '无'}。将缺失文件计为0并继续。")
    return False


def merge_reports(input_files, ddc_file, output_file):
    """
    合并多个街区电动车检测报告
    :param input_files: 违停文件列表
    :param ddc_file: 电动车状态文件
    :param output_file: 输出汇总文件路径
    """
    reports = []
    
    # 读取所有违停文件（文件不存在时 parse_violation_report 会返回 violations=0）
    for file_name in input_files:
        file_path = os.path.join(INPUT_DIR, file_name)
        report = parse_violation_report(file_path)
        if report:
            reports.append(report)
            print(f"已读取: {file_name} - {report['street']}街区 {report['violations']}辆违停")
    
    # 读取电动车状态文件（支持无扩展名备选）
    ddc_path = os.path.join(INPUT_DIR, ddc_file)
    if not os.path.exists(ddc_path):
        ddc_path_alt = os.path.join(INPUT_DIR, "ddczuizhong")
        if os.path.exists(ddc_path_alt):
            ddc_path = ddc_path_alt
        else:
            print(f"电动车状态文件不存在: {ddc_file}（已尝试备选名），将正常数和倒伏数计为0")
    ddc_status = parse_ddc_status(ddc_path)
    print(f"已读取电动车状态: 正常{ddc_status['normal']}辆，倒伏{ddc_status['fallen']}辆")
    
    # 构建输出内容
    street_details = []
    for r in reports:
        if r['violations'] > 0:
            street_details.append(f"{r['street']}街区有{r['violations']}辆电动车违停")
        else:
            street_details.append(f"{r['street']}街区无违停")
    
    # 生成汇总文本
    if street_details:
        summary = f"{'，'.join(street_details)}，"
    else:
        summary = ""
    summary += f"停车区内电动车正常{ddc_status['normal']}辆，倒伏{ddc_status['fallen']}辆。"
    
    # 保存到文件
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"\n汇总报告已保存至: {output_file}")
        print(f"内容: {summary}")
        return output_file
    except Exception as e:
        print(f"保存汇总文件失败: {e}")
        return None


def main():
    print("=" * 60)
    print("AB街区电动车检测合并")
    print("=" * 60)

    # 等待三个文件齐备（最多 WAIT_TIMEOUT 秒）
    # 注意：wait_for_files 返回值不影响后续流程，仅作提示
    wait_for_files(
        INPUT_DIR,
        INPUT_FILES + [DDC_FILE],
        timeout=WAIT_TIMEOUT,
        interval=WAIT_INTERVAL
    )
    
    # 无论文件是否齐备，都继续生成汇总（缺失文件计为0）
    output_path = merge_reports(INPUT_FILES, DDC_FILE, OUTPUT_FILE)
    
    if output_path:
        try:
            output_dir = os.path.dirname(OUTPUT_FILE)
            new_file_path = os.path.join(output_dir, "10.txt")
            # 如果目标文件已存在，先删除
            if os.path.exists(new_file_path):
                os.remove(new_file_path)
            os.replace(output_path, new_file_path)
            print(f"已重命名文件: ddc.txt → 10.txt")
            print(f"新路径: {new_file_path}")
        except Exception as e:
            print(f"重命名失败: {e}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
ensure_runtime_dirs()
