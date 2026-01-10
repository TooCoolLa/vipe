#!/usr/bin/env python3
"""
分析相机位姿轨迹中的突跳（Jump）

用法:
    python analyze_jumps.py <pose_file.json> [options]

示例:
    python analyze_jumps.py data1/camera_poses.json
    python analyze_jumps.py data1/camera_poses_fused.json --threshold 3.5
    python analyze_jumps.py data1/camera_poses.json --output jumps_report.txt
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple


class JumpAnalyzer:
    """突跳分析器"""
    
    def __init__(self, pose_file: Path, sigma_multiplier: float = 3.0):
        """
        初始化分析器
        
        Args:
            pose_file: camera_poses.json文件路径
            sigma_multiplier: 突跳阈值的标准差倍数（默认3σ）
        """
        self.pose_file = pose_file
        self.sigma_multiplier = sigma_multiplier
        
        # 加载数据
        print(f"📂 加载数据: {pose_file}")
        with open(pose_file, 'r') as f:
            self.data = json.load(f)
        
        # 提取位置
        self.positions = self._extract_positions()
        
        # 计算帧间距离
        self.distances = np.linalg.norm(np.diff(self.positions, axis=0), axis=1)
        
        # 计算统计信息
        self.mean_dist = np.mean(self.distances)
        self.std_dist = np.std(self.distances)
        self.threshold = self.mean_dist + sigma_multiplier * self.std_dist
        
        # 识别突跳
        self.jump_mask = self.distances > self.threshold
        self.jump_indices = np.where(self.jump_mask)[0]
        
        print(f"✅ 加载完成: {len(self.positions)} 帧")
        print(f"📊 平均帧间距: {self.mean_dist:.4f}m, 标准差: {self.std_dist:.4f}m")
        print(f"🎯 突跳阈值: {self.threshold:.4f}m (均值 + {sigma_multiplier}σ)")
        print(f"⚠️  发现 {len(self.jump_indices)} 个突跳 ({len(self.jump_indices)/len(self.distances)*100:.2f}%)\n")
    
    def _extract_positions(self) -> np.ndarray:
        """从pose数据中提取位置"""
        positions = []
        for frame in self.data:
            matrix = np.array(frame['matrix'])
            pos = matrix[:3, 3]
            positions.append(pos)
        return np.array(positions)
    
    def group_consecutive_jumps(self, gap: int = 50) -> List[List[int]]:
        """
        将连续或接近的突跳分组
        
        Args:
            gap: 认为是连续的最大帧间隔
            
        Returns:
            突跳组列表，每组是帧索引列表
        """
        if len(self.jump_indices) == 0:
            return []
        
        groups = []
        current_group = [self.jump_indices[0]]
        
        for idx in self.jump_indices[1:]:
            if idx - current_group[-1] <= gap:
                current_group.append(idx)
            else:
                groups.append(current_group)
                current_group = [idx]
        groups.append(current_group)
        
        return groups
    
    def analyze_jump_group(self, group: List[int]) -> Dict:
        """
        分析单个突跳组的详细信息
        
        Args:
            group: 突跳帧索引列表
            
        Returns:
            分析结果字典
        """
        start_frame = group[0]
        end_frame = group[-1]
        
        # 统计信息
        group_distances = [self.distances[idx] for idx in group]
        max_jump = max(group_distances)
        min_jump = min(group_distances)
        avg_jump = np.mean(group_distances)
        
        # 获取前后上下文（±5帧）
        context_start = max(0, start_frame - 5)
        context_end = min(len(self.distances), end_frame + 6)
        context_distances = self.distances[context_start:context_end].tolist()
        
        # 计算位置变化
        pos_start = self.positions[start_frame]
        pos_end = self.positions[end_frame + 1]  # +1因为distance是diff
        total_displacement = np.linalg.norm(pos_end - pos_start)
        
        # 判断是否在子地图边界（假设每690帧一个子地图）
        nearest_boundary = round(start_frame / 690) * 690
        distance_to_boundary = abs(start_frame - nearest_boundary)
        at_boundary = distance_to_boundary <= 50
        
        return {
            'start_frame': start_frame,
            'end_frame': end_frame,
            'span': end_frame - start_frame + 1,
            'num_jumps': len(group),
            'max_jump': max_jump,
            'min_jump': min_jump,
            'avg_jump': avg_jump,
            'total_displacement': total_displacement,
            'context_start': context_start,
            'context_end': context_end,
            'context_distances': context_distances,
            'nearest_boundary': nearest_boundary,
            'distance_to_boundary': distance_to_boundary,
            'at_boundary': at_boundary,
            'jump_details': [(idx, self.distances[idx]) for idx in group]
        }
    
    def generate_report(self, output_file: Path = None) -> str:
        """
        生成详细的分析报告
        
        Args:
            output_file: 输出文件路径（可选）
            
        Returns:
            报告文本
        """
        lines = []
        
        # 标题
        lines.append("=" * 100)
        lines.append(" " * 30 + "🔍 相机轨迹突跳分析报告")
        lines.append("=" * 100)
        lines.append("")
        
        # 基本信息
        lines.append("📋 基本信息")
        lines.append("-" * 100)
        lines.append(f"文件路径: {self.pose_file}")
        lines.append(f"总帧数: {len(self.positions)}")
        lines.append(f"总行程: {np.sum(self.distances):.2f} 米")
        lines.append("")
        
        # 统计信息
        lines.append("📊 统计信息")
        lines.append("-" * 100)
        lines.append(f"平均帧间距: {self.mean_dist:.4f} 米")
        lines.append(f"标准差: {self.std_dist:.4f} 米")
        lines.append(f"最大帧间距: {np.max(self.distances):.4f} 米")
        lines.append(f"最小帧间距: {np.min(self.distances):.4f} 米")
        lines.append(f"突跳阈值 (均值+{self.sigma_multiplier}σ): {self.threshold:.4f} 米")
        lines.append("")
        
        # 突跳概览
        lines.append("⚠️  突跳概览")
        lines.append("-" * 100)
        lines.append(f"突跳总数: {len(self.jump_indices)} 个 ({len(self.jump_indices)/len(self.distances)*100:.2f}%)")
        
        if len(self.jump_indices) == 0:
            lines.append("\n✅ 未发现突跳，轨迹非常平滑！")
            report = "\n".join(lines)
            if output_file:
                output_file.write_text(report)
                print(f"📝 报告已保存到: {output_file}")
            return report
        
        # 按严重程度排序的前10个突跳
        top_jumps = sorted(enumerate(self.distances), key=lambda x: x[1], reverse=True)[:10]
        lines.append(f"\n最严重的10个突跳:")
        lines.append(f"  {'帧索引':<12} {'跳变距离':<15} {'位置':<40}")
        lines.append(f"  {'-'*70}")
        
        for idx, dist in top_jumps:
            if dist <= self.threshold:
                break
            pos = self.positions[idx]
            lines.append(f"  帧{idx:<9} {dist:>8.2f} 米       ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        
        lines.append("")
        
        # 突跳聚集区域分析
        jump_groups = self.group_consecutive_jumps(gap=50)
        lines.append(f"🔥 发现 {len(jump_groups)} 个突跳聚集区域")
        lines.append("=" * 100)
        
        # 按最大跳变排序
        sorted_groups = sorted(jump_groups, 
                              key=lambda g: max(self.distances[idx] for idx in g), 
                              reverse=True)
        
        for i, group in enumerate(sorted_groups, 1):
            info = self.analyze_jump_group(group)
            
            lines.append("")
            lines.append(f"区域 {i}: 帧 {info['start_frame']} - {info['end_frame']}")
            lines.append("-" * 100)
            lines.append(f"  跨度: {info['span']} 帧")
            lines.append(f"  突跳数量: {info['num_jumps']} 个")
            lines.append(f"  最大跳变: {info['max_jump']:.2f} 米")
            lines.append(f"  最小跳变: {info['min_jump']:.2f} 米")
            lines.append(f"  平均跳变: {info['avg_jump']:.2f} 米")
            lines.append(f"  总位移: {info['total_displacement']:.2f} 米")
            
            # 位置信息
            if info['at_boundary']:
                boundary_num = info['nearest_boundary'] // 690
                lines.append(f"  📍 位置: 靠近子地图边界{boundary_num} (距离边界{info['distance_to_boundary']}帧)")
            else:
                boundary_num = info['nearest_boundary'] // 690
                lines.append(f"  📍 位置: 远离边界 (最近边界{boundary_num}，距离{info['distance_to_boundary']}帧)")
            
            # 严重度评级
            if info['max_jump'] > 25:
                severity = "🔴🔴🔴🔴🔴 极其严重"
            elif info['max_jump'] > 15:
                severity = "🔴🔴🔴 非常严重"
            elif info['max_jump'] > 10:
                severity = "🟠🟠 严重"
            elif info['max_jump'] > 5:
                severity = "🟠 中等"
            else:
                severity = "🟡 轻微"
            lines.append(f"  ⚠️  严重度: {severity}")
            
            # 详细突跳列表
            lines.append(f"  突跳详情:")
            for j, (idx, dist) in enumerate(info['jump_details'][:10], 1):
                lines.append(f"    帧{idx}: {dist:.2f}米")
                if j == 10 and len(info['jump_details']) > 10:
                    lines.append(f"    ... (还有{len(info['jump_details'])-10}个)")
                    break
            
            # 前后上下文
            lines.append(f"  前后帧距离 (帧{info['context_start']}-{info['context_end']}):")
            for k, ctx_idx in enumerate(range(info['context_start'], info['context_end'])):
                if ctx_idx < len(self.distances):
                    dist = self.distances[ctx_idx]
                    is_jump = ctx_idx in group
                    marker = "💥" if is_jump else "  "
                    lines.append(f"    帧{ctx_idx}: {dist:.2f}米 {marker}")
        
        # 建议
        lines.append("")
        lines.append("=" * 100)
        lines.append("💡 分析与建议")
        lines.append("=" * 100)
        
        # 统计边界vs非边界
        groups_at_boundary = sum(1 for g in jump_groups 
                                if self.analyze_jump_group(g)['at_boundary'])
        groups_not_at_boundary = len(jump_groups) - groups_at_boundary
        
        lines.append(f"\n突跳分布:")
        lines.append(f"  • 子地图边界处: {groups_at_boundary} 个区域")
        lines.append(f"  • 子地图内部: {groups_not_at_boundary} 个区域")
        
        if groups_at_boundary > groups_not_at_boundary:
            lines.append(f"\n⚠️  大部分突跳在子地图边界，建议:")
            lines.append(f"  1. 检查子地图对齐算法")
            lines.append(f"  2. 调整RANSAC参数或启用位姿融合")
            lines.append(f"  3. 增加重叠区域帧数")
        else:
            lines.append(f"\n⚠️  大部分突跳在子地图内部，建议:")
            lines.append(f"  1. 检查原始SLAM轨迹质量")
            lines.append(f"  2. 查看问题区域的原始图像")
            lines.append(f"  3. 考虑重新运行SLAM，调整参数")
            lines.append(f"  4. 使用IMU或其他传感器辅助")
        
        # 最严重区域警告
        worst_group = sorted_groups[0]
        worst_info = self.analyze_jump_group(worst_group)
        lines.append(f"\n🚨 最严重区域警告:")
        lines.append(f"  帧{worst_info['start_frame']}-{worst_info['end_frame']}: 最大跳变{worst_info['max_jump']:.2f}米")
        lines.append(f"  这个区域需要优先处理！")
        
        lines.append("")
        lines.append("=" * 100)
        
        # 生成报告文本
        report = "\n".join(lines)
        
        # 保存到文件
        if output_file:
            output_file.write_text(report)
            print(f"📝 报告已保存到: {output_file}")
        
        return report
    
    def export_jump_list(self, output_file: Path):
        """
        导出突跳列表为CSV格式
        
        Args:
            output_file: 输出CSV文件路径
        """
        import csv
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame_Index', 'Jump_Distance_m', 'Position_X', 'Position_Y', 'Position_Z', 
                           'At_Boundary', 'Nearest_Boundary', 'Distance_to_Boundary'])
            
            for idx in self.jump_indices:
                dist = self.distances[idx]
                pos = self.positions[idx]
                
                nearest_boundary = round(idx / 690) * 690
                distance_to_boundary = abs(idx - nearest_boundary)
                at_boundary = distance_to_boundary <= 50
                
                writer.writerow([
                    idx, f"{dist:.4f}", 
                    f"{pos[0]:.4f}", f"{pos[1]:.4f}", f"{pos[2]:.4f}",
                    at_boundary, nearest_boundary, distance_to_boundary
                ])
        
        print(f"📊 突跳列表已导出到: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="分析相机位姿轨迹中的突跳",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  基本用法:
    python analyze_jumps.py data1/camera_poses.json
  
  自定义阈值 (4σ):
    python analyze_jumps.py data1/camera_poses.json --sigma 4.0
  
  保存报告:
    python analyze_jumps.py data1/camera_poses.json -o report.txt
  
  导出CSV:
    python analyze_jumps.py data1/camera_poses.json --csv jumps.csv
        """
    )
    
    parser.add_argument(
        'pose_file',
        type=Path,
        help='camera_poses.json文件路径'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=None,
        help='输出报告文件路径 (默认: 只打印到终端)'
    )
    
    parser.add_argument(
        '--csv',
        type=Path,
        default=None,
        help='导出突跳列表为CSV文件'
    )
    
    parser.add_argument(
        '--sigma',
        type=float,
        default=3.0,
        help='突跳阈值的标准差倍数 (默认: 3.0，即3σ)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='直接指定突跳阈值（米），覆盖--sigma参数'
    )
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not args.pose_file.exists():
        print(f"❌ 错误: 文件不存在: {args.pose_file}")
        return 1
    
    # 创建分析器
    analyzer = JumpAnalyzer(args.pose_file, sigma_multiplier=args.sigma)
    
    # 如果指定了阈值，覆盖自动计算的阈值
    if args.threshold is not None:
        analyzer.threshold = args.threshold
        analyzer.jump_mask = analyzer.distances > analyzer.threshold
        analyzer.jump_indices = np.where(analyzer.jump_mask)[0]
        print(f"🎯 使用自定义阈值: {analyzer.threshold:.4f}m")
        print(f"⚠️  重新识别到 {len(analyzer.jump_indices)} 个突跳\n")
    
    # 生成报告
    report = analyzer.generate_report(output_file=args.output)
    
    # 打印到终端（如果没有指定输出文件）
    if args.output is None:
        print(report)
    
    # 导出CSV
    if args.csv:
        analyzer.export_jump_list(args.csv)
    
    return 0


if __name__ == '__main__':
    exit(main())
