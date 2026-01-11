#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
独立BA模块使用示例
演示如何使用提取的BA模块进行位姿优化
"""

import numpy as np
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from vipe.slam.standalone_ba import standalone_bundle_adjustment, StandaloneBA


def example_basic_usage():
    """基本使用示例"""
    print("=" * 80)
    print("示例1: 基本使用")
    print("=" * 80)
    
    # 参数设置
    n_frames = 10
    h, w = 120, 160  # 使用较小的分辨率以加快速度
    
    # 1. 准备初始位姿（c2w格式）
    poses_c2w = np.eye(4)[None].repeat(n_frames, axis=0).astype(np.float32)
    for i in range(n_frames):
        # 相机沿x轴移动
        poses_c2w[i, 0, 3] = i * 0.1
        # 添加一些噪声
        poses_c2w[i, 0, 3] += np.random.randn() * 0.01
    
    print(f"初始位姿:")
    print(f"  帧数: {n_frames}")
    print(f"  第0帧位置: {poses_c2w[0, :3, 3]}")
    print(f"  第5帧位置: {poses_c2w[5, :3, 3]}")
    print(f"  第9帧位置: {poses_c2w[9, :3, 3]}")
    
    # 2. 准备深度图
    depths = np.ones((n_frames, h, w), dtype=np.float32) * 5.0
    # 添加一些深度变化
    for i in range(n_frames):
        depths[i] += np.random.randn(h, w) * 0.1
    
    print(f"\n深度图:")
    print(f"  形状: {depths.shape}")
    print(f"  平均深度: {depths.mean():.2f}m")
    
    # 3. 准备点对应关系
    # 模拟3组帧，每组有一些可见的特征点
    correspondences = []
    
    # 第一组: 帧0-3，5个点
    coords1 = np.random.rand(4, 5, h, w, 2).astype(np.float32)
    coords1[..., 0] *= w  # x坐标
    coords1[..., 1] *= h  # y坐标
    visible1 = np.random.rand(4, 5) > 0.2  # 80%可见性
    correspondences.append({
        'coords': coords1,
        'visible': visible1,
    })
    
    # 第二组: 帧3-7，8个点
    coords2 = np.random.rand(5, 8, h, w, 2).astype(np.float32)
    coords2[..., 0] *= w
    coords2[..., 1] *= h
    visible2 = np.random.rand(5, 8) > 0.3
    correspondences.append({
        'coords': coords2,
        'visible': visible2,
    })
    
    # 第三组: 帧7-9，6个点
    coords3 = np.random.rand(3, 6, h, w, 2).astype(np.float32)
    coords3[..., 0] *= w
    coords3[..., 1] *= h
    visible3 = np.random.rand(3, 6) > 0.2
    correspondences.append({
        'coords': coords3,
        'visible': visible3,
    })
    
    print(f"\n点对应关系:")
    print(f"  组数: {len(correspondences)}")
    print(f"  第1组: {coords1.shape[0]}帧, {coords1.shape[1]}点")
    print(f"  第2组: {coords2.shape[0]}帧, {coords2.shape[1]}点")
    print(f"  第3组: {coords3.shape[0]}帧, {coords3.shape[1]}点")
    print(f"  总帧数验证: {coords1.shape[0] + coords2.shape[0] + coords3.shape[0]} == {n_frames}")
    
    # 4. 相机内参
    intrinsics = np.array([500.0, 500.0, w/2, h/2], dtype=np.float32)
    
    print(f"\n相机内参:")
    print(f"  fx, fy: {intrinsics[0]:.1f}, {intrinsics[1]:.1f}")
    print(f"  cx, cy: {intrinsics[2]:.1f}, {intrinsics[3]:.1f}")
    
    # 5. 执行BA优化
    print(f"\n{'='*80}")
    print("开始Bundle Adjustment优化...")
    print(f"{'='*80}")
    
    optimized_poses = standalone_bundle_adjustment(
        poses_c2w=poses_c2w,
        metric_depths=depths,
        point_correspondences=correspondences,
        intrinsics=intrinsics,
        device='cuda',
        n_iters=5,
        fixed_frames=[0],  # 固定第一帧
        verbose=True,
    )
    
    # 6. 对比结果
    print(f"\n{'='*80}")
    print("优化结果对比:")
    print(f"{'='*80}")
    
    for i in [0, 5, 9]:
        pos_before = poses_c2w[i, :3, 3]
        pos_after = optimized_poses[i, :3, 3]
        delta = pos_after - pos_before
        print(f"帧{i}:")
        print(f"  优化前: [{pos_before[0]:.4f}, {pos_before[1]:.4f}, {pos_before[2]:.4f}]")
        print(f"  优化后: [{pos_after[0]:.4f}, {pos_after[1]:.4f}, {pos_after[2]:.4f}]")
        print(f"  变化量: [{delta[0]:.4f}, {delta[1]:.4f}, {delta[2]:.4f}]")


def example_with_class():
    """使用类接口的示例"""
    print("\n\n" + "=" * 80)
    print("示例2: 使用类接口")
    print("=" * 80)
    
    # 创建BA优化器实例
    ba_optimizer = StandaloneBA(
        device='cuda',
        n_iters=3,
        pose_damping=1e-4,
        pose_ep=0.1,
    )
    
    # 准备简单的测试数据
    n_frames = 5
    h, w = 60, 80
    
    poses = np.eye(4)[None].repeat(n_frames, axis=0).astype(np.float32)
    for i in range(n_frames):
        poses[i, 2, 3] = i * 0.2  # 沿z轴移动
    
    depths = np.ones((n_frames, h, w), dtype=np.float32) * 3.0
    
    correspondences = [{
        'coords': np.random.rand(n_frames, 3, h, w, 2).astype(np.float32) * [w, h],
        'visible': np.ones((n_frames, 3), dtype=bool),
    }]
    
    intrinsics = np.array([400.0, 400.0, w/2, h/2], dtype=np.float32)
    
    print(f"输入: {n_frames}帧, 分辨率{h}x{w}, {correspondences[0]['coords'].shape[1]}个点")
    
    # 执行优化
    result = ba_optimizer.optimize(
        poses_c2w=poses,
        metric_depths=depths,
        point_correspondences=correspondences,
        intrinsics=intrinsics,
        verbose=True,
    )
    
    print(f"\n优化完成，输出形状: {result.shape}")


def example_realistic_scenario():
    """更真实的场景示例"""
    print("\n\n" + "=" * 80)
    print("示例3: 更真实的场景")
    print("=" * 80)
    
    # 模拟一个相机在场景中移动并观察到多个3D点
    n_frames = 15
    h, w = 240, 320
    
    # 1. 生成真实的相机轨迹（圆形轨迹）
    radius = 2.0
    true_poses = np.eye(4)[None].repeat(n_frames, axis=0).astype(np.float32)
    
    for i in range(n_frames):
        theta = 2 * np.pi * i / n_frames
        # 圆形轨迹
        true_poses[i, 0, 3] = radius * np.cos(theta)
        true_poses[i, 2, 3] = radius * np.sin(theta)
        # 相机朝向圆心
        angle = theta + np.pi
        true_poses[i, 0, 0] = np.cos(angle)
        true_poses[i, 0, 2] = np.sin(angle)
        true_poses[i, 2, 0] = -np.sin(angle)
        true_poses[i, 2, 2] = np.cos(angle)
    
    # 2. 添加噪声得到初始猜测
    noisy_poses = true_poses.copy()
    for i in range(1, n_frames):  # 跳过第一帧
        noisy_poses[i, :3, 3] += np.random.randn(3) * 0.05  # 5cm噪声
    
    # 3. 生成深度图
    depths = np.ones((n_frames, h, w), dtype=np.float32) * 3.0
    
    # 4. 生成对应关系（模拟滑动窗口）
    window_size = 5
    n_points = 10
    correspondences = []
    
    for start_idx in range(0, n_frames - window_size + 1, 3):
        end_idx = start_idx + window_size
        n_frames_in_window = end_idx - start_idx
        
        coords = np.random.rand(n_frames_in_window, n_points, h, w, 2).astype(np.float32)
        coords[..., 0] *= w
        coords[..., 1] *= h
        
        # 模拟点的可见性（中间帧可见性更高）
        visible = np.zeros((n_frames_in_window, n_points), dtype=bool)
        for i in range(n_frames_in_window):
            prob = 1.0 - abs(i - n_frames_in_window // 2) / (n_frames_in_window // 2) * 0.5
            visible[i] = np.random.rand(n_points) < prob
        
        correspondences.append({
            'coords': coords,
            'visible': visible,
        })
    
    intrinsics = np.array([320.0, 320.0, w/2, h/2], dtype=np.float32)
    
    print(f"场景设置:")
    print(f"  相机轨迹: 圆形, 半径={radius}m, {n_frames}帧")
    print(f"  图像分辨率: {h}x{w}")
    print(f"  对应关系: {len(correspondences)}个窗口, 每个窗口{window_size}帧{n_points}点")
    
    print(f"\n初始误差:")
    position_errors = np.linalg.norm(noisy_poses[:, :3, 3] - true_poses[:, :3, 3], axis=1)
    print(f"  平均位置误差: {position_errors.mean():.4f}m")
    print(f"  最大位置误差: {position_errors.max():.4f}m")
    
    # 执行优化
    print(f"\n{'='*80}")
    print("执行BA优化...")
    print(f"{'='*80}")
    
    optimized_poses = standalone_bundle_adjustment(
        poses_c2w=noisy_poses,
        metric_depths=depths,
        point_correspondences=correspondences,
        intrinsics=intrinsics,
        device='cuda',
        n_iters=10,
        fixed_frames=[0],
        verbose=True,
    )
    
    # 计算优化后的误差
    print(f"\n优化后误差:")
    optimized_errors = np.linalg.norm(optimized_poses[:, :3, 3] - true_poses[:, :3, 3], axis=1)
    print(f"  平均位置误差: {optimized_errors.mean():.4f}m")
    print(f"  最大位置误差: {optimized_errors.max():.4f}m")
    print(f"\n改进:")
    print(f"  平均误差减少: {(position_errors.mean() - optimized_errors.mean()):.4f}m")
    print(f"  改进率: {(1 - optimized_errors.mean()/position_errors.mean())*100:.1f}%")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    try:
        example_basic_usage()
        example_with_class()
        example_realistic_scenario()
        
        print("\n\n" + "=" * 80)
        print("所有示例运行完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
