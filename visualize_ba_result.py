#!/usr/bin/env python3
"""可视化Bundle Adjustment优化结果"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_ba_results():
    """可视化BA优化结果"""
    
    data_dir = Path(__file__).parent / 'data'
    result = np.load(data_dir / 'simple_ba_result.npz')
    
    poses_init = result['poses_initial']
    poses_opt = result['poses_optimized']
    loss_history = result['loss_history']
    error_history = result['mean_error_history']
    
    n_frames = poses_init.shape[0]
    
    # 提取位置
    pos_init = poses_init[:, :3, 3]
    pos_opt = poses_opt[:, :3, 3]
    
    # 创建图表
    fig = plt.figure(figsize=(16, 10))
    
    # 1. 损失曲线
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(loss_history, 'b-', linewidth=2)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('BA Loss Convergence', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. 平均误差曲线
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(error_history, 'r-', linewidth=2)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Mean Reprojection Error (px)', fontsize=12)
    ax2.set_title('Mean Reprojection Error', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. 3D轨迹对比
    ax3 = plt.subplot(2, 3, 3, projection='3d')
    ax3.plot(pos_init[:, 0], pos_init[:, 1], pos_init[:, 2], 
             'b-o', label='Initial', linewidth=2, markersize=4, alpha=0.6)
    ax3.plot(pos_opt[:, 0], pos_opt[:, 1], pos_opt[:, 2], 
             'r-s', label='Optimized', linewidth=2, markersize=4, alpha=0.6)
    ax3.set_xlabel('X (m)', fontsize=10)
    ax3.set_ylabel('Y (m)', fontsize=10)
    ax3.set_zlabel('Z (m)', fontsize=10)
    ax3.set_title('Camera Trajectory 3D', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. XY平面轨迹
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(pos_init[:, 0], pos_init[:, 1], 'b-o', 
             label='Initial', linewidth=2, markersize=6, alpha=0.6)
    ax4.plot(pos_opt[:, 0], pos_opt[:, 1], 'r-s', 
             label='Optimized', linewidth=2, markersize=6, alpha=0.6)
    ax4.scatter(pos_init[0, 0], pos_init[0, 1], 
                c='green', s=100, marker='*', zorder=5, label='Start')
    ax4.set_xlabel('X (m)', fontsize=12)
    ax4.set_ylabel('Y (m)', fontsize=12)
    ax4.set_title('Camera Trajectory (Top View)', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.axis('equal')
    
    # 5. XZ平面轨迹
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(pos_init[:, 0], pos_init[:, 2], 'b-o', 
             label='Initial', linewidth=2, markersize=6, alpha=0.6)
    ax5.plot(pos_opt[:, 0], pos_opt[:, 2], 'r-s', 
             label='Optimized', linewidth=2, markersize=6, alpha=0.6)
    ax5.scatter(pos_init[0, 0], pos_init[0, 2], 
                c='green', s=100, marker='*', zorder=5, label='Start')
    ax5.set_xlabel('X (m)', fontsize=12)
    ax5.set_ylabel('Z (m)', fontsize=12)
    ax5.set_title('Camera Trajectory (Side View)', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.axis('equal')
    
    # 6. 位姿变化量
    ax6 = plt.subplot(2, 3, 6)
    pose_diff = np.linalg.norm(pos_opt - pos_init, axis=1)
    ax6.bar(range(n_frames), pose_diff * 1000, color='purple', alpha=0.7)
    ax6.set_xlabel('Frame Index', fontsize=12)
    ax6.set_ylabel('Position Change (mm)', fontsize=12)
    ax6.set_title('Per-Frame Position Changes', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存图片
    output_path = data_dir / 'ba_result_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"可视化已保存到: {output_path}")
    
    # 显示统计信息
    print(f"\n{'='*80}")
    print(f"Bundle Adjustment 优化结果统计")
    print(f"{'='*80}")
    print(f"帧数: {n_frames}")
    print(f"初始损失: {loss_history[0]:.6f}")
    print(f"最终损失: {loss_history[-1]:.6f}")
    print(f"损失减少: {(1 - loss_history[-1]/loss_history[0])*100:.2f}%")
    print(f"\n初始平均误差: {error_history[0]:.2f} px")
    print(f"最终平均误差: {error_history[-1]:.2f} px")
    print(f"误差减少: {(1 - error_history[-1]/error_history[0])*100:.2f}%")
    print(f"\n位姿变化统计:")
    print(f"  平均变化: {pose_diff.mean()*1000:.3f} mm")
    print(f"  最大变化: {pose_diff.max()*1000:.3f} mm (帧{pose_diff.argmax()})")
    print(f"  最小变化: {pose_diff.min()*1000:.3f} mm (帧{pose_diff.argmin()})")
    print(f"{'='*80}\n")
    
    # 打印前几帧的详细对比
    print("前5帧位置对比:")
    print(f"{'帧':>4s} {'初始X':>10s} {'初始Y':>10s} {'初始Z':>10s} | "
          f"{'优化X':>10s} {'优化Y':>10s} {'优化Z':>10s} | {'变化':>10s}")
    print("-" * 85)
    for i in range(min(5, n_frames)):
        print(f"{i:4d} "
              f"{pos_init[i,0]:10.4f} {pos_init[i,1]:10.4f} {pos_init[i,2]:10.4f} | "
              f"{pos_opt[i,0]:10.4f} {pos_opt[i,1]:10.4f} {pos_opt[i,2]:10.4f} | "
              f"{pose_diff[i]*1000:10.3f}mm")
    
    plt.show()


if __name__ == "__main__":
    visualize_ba_results()
