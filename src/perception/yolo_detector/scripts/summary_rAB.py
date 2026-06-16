#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取多个街区检测结果文件，整合输出汇总报告
"""
import os
import re
from runtime_paths import RUNTIME_R, VOICE_RUNTIME_DIR, ensure_runtime_dirs

# ========== 配置区域 ==========
INPUT_DIR = str(RUNTIME_R)
INPUT_FILES = ["rA.txt", "rB.txt"]  # 要读取的文件列表
OUTPUT_FILE = str(VOICE_RUNTIME_DIR / "summary_rAB.txt")
# ==============================


def parse_report(file_path):
    """
    解析单个街区报告文件
    :param file_path: 文件路径
    :return: dict 包含 {street: str, total: int, outsiders: int}
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # 提取街区名称（如 "A街区"）
        street_match = re.search(r'([A-Z])街区', content)
        street = street_match.group(1) if street_match else "未知"
        
        # 提取总人数（如 "一共有4人"）
        total_match = re.search(r'一共有(\d+)人', content)
        total = int(total_match.group(1)) if total_match else 0
        
        # 提取非社区人员数（如 "发现1名非社区人员"）
        outsider_match = re.search(r'发现(\d+)名非社区人员', content)
        outsiders = int(outsider_match.group(1)) if outsider_match else 0
        
        return {
            'street': street,
            'total': total,
            'outsiders': outsiders
        }
    except Exception as e:
        print(f"解析文件 {file_path} 失败: {e}")
        return None


def merge_reports(input_files, output_file):
    """
    合并多个街区报告
    :param input_files: 输入文件路径列表
    :param output_file: 输出汇总文件路径
    """
    reports = []
    
    # 读取所有文件
    for file_name in input_files:
        file_path = os.path.join(INPUT_DIR, file_name)
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            continue
        
        report = parse_report(file_path)
        if report:
            reports.append(report)
            print(f"已读取: {file_name} - {report['street']}街 {report['total']}人，{report['outsiders']}名非社区人员")
    
    if not reports:
        print("没有有效的报告文件")
        return
    
    # 统计汇总
    total_people = sum(r['total'] for r in reports)
    
    # 构建输出内容
    street_details = []
    outsider_details = []
    
    for r in reports:
        street_details.append(f"{r['street']}街人数{r['total']}人")
        if r['outsiders'] > 0:
            outsider_details.append(f"{r['outsiders']}名非社区人员在{r['street']}街")
    
    # 生成汇总文本
    summary = f"社区内共有{total_people}人，其中{'，'.join(street_details)}"
    if outsider_details:
        summary += f"，发现{'，'.join(outsider_details)}"
    summary += "，图片已保存。"
    
    # 保存到文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"\n汇总报告已保存至: {output_file}")
        print(f"内容: {summary}")
    except Exception as e:
        print(f"保存汇总文件失败: {e}")


def main():
    print("=" * 60)
    print("AB街区检测合并")
    print("=" * 60)
    merge_reports(INPUT_FILES, OUTPUT_FILE)
    try:
        output_dir = os.path.dirname(OUTPUT_FILE)
        old_file_path = OUTPUT_FILE
        new_file_path = os.path.join(output_dir, "4.txt")
        
        if os.path.exists(old_file_path):
            # 如需覆盖已存在的 4.txt，先删除
            if os.path.exists(new_file_path):
                os.remove(new_file_path)
            os.rename(old_file_path, new_file_path)
            print(f"已重命名文件: summary_rAB.txt → 4.txt")
            print(f"新路径: {new_file_path}")
        else:
            print(f"文件不存在: {old_file_path}")
    except Exception as e:
        print(f"重命名失败: {e}")
    
    print("=" * 60)


if __name__ == '__main__':
    main()
ensure_runtime_dirs()
