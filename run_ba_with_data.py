#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
使用data文件夹中的真实数据运行独立BA模块
输入: Co-Tracker tracks + DUSt3R重建结果
输出: 优化后的相机位姿
"""

import numpy as np
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from vipe.slam.standalone_ba import standalone_bundle_adjustment


def load_cotracker_data(track_file):
    """加载Co-Tracker跟踪数据"""
    print(f"加载Co-Tracker数据: {track_file}")
    data = np.load(track_file)
    tracks = data['tracks'][0]  # (n_frames, n_points, 2)
    visibility = data['visibility'][0]  # (n_frames, n_points)
    
    print(f"  - 帧数: {tracks.shape[0]}")
    print(f"  - 跟踪点数: {tracks.shape[1]}")
    print(f"  - 总可见次数: {visibility.sum()} / {visibility.size} ({visibility.sum()/visibility.size*100:.1f}%)")
    
    return tracks, visibility


def load_dust3r_data(folder):
    """加载DUSt3R重建结果"""
    print(f"\n加载DUSt3R数据: {folder}")
    
    poses = np.load(folder / 'camera_poses.npz')['camera_poses']
    points = np.load(folder / 'points.npz')['points']
    conf = np.load(folder / 'conf.npz')['conf']
    intrinsic = np.load(folder / 'intrinsic.npz')['intrinsic']
    images = np.load(folder / 'images.npz')['images']
    is_metric = np.load(folder / 'is_metric.npz')['is_metric']
    scale_factor = np.load(folder / 'scale_factor.npz')['scale_factor']
    
    print(f"  - 帧数: {poses.shape[0]}")
    print(f"  - 图像分辨率: {images.shape[1]}x{images.shape[2]}")
    print(f"  - 度量尺度: {'是' if is_metric[0] else '否'}")
    print(f"  - 尺度因子: {scale_factor[0]:.4f}")
    print(f"  - 平均置信度: {conf.mean():.3f}")
    print(f"  - 高置信度像素(>2.5): {(conf > 2.5).sum() / conf.size * 100:.1f}%")
    
    return {
        'poses': poses,
        'points': points,
        'conf': conf,
        'intrinsic': intrinsic,
        'images': images,
        'is_metric': is_metric,
        'scale_factor': scale_factor,
    }


def convert_cotracker_to_correspondences(tracks, visibility, image_shape):
    """
    将Co-Tracker格式转换为standalone_ba所需的格式
    
    Args:
        tracks: (n_frames, n_points, 2) - 2D轨迹
        visibility: (n_frames, n_points) - 可见性
        image_shape: (h, w) - 图像尺寸
    
    Returns:
        list of dict: point_correspondences格式
    """
    h, w = image_shape
    n_frames, n_points = tracks.shape[0], tracks.shape[1]
    
    # 创建coords数组 (n_frames, n_points, h, w, 2)
    # 由于Co-Tracker提供的是每个点的单一坐标，我们需要创建一个稀疏表示
    # 这里使用一个技巧：将坐标归一化到[0,1]，然后乘以(h,w)
    
    # 简化版本：直接使用tracks作为目标坐标
    # 在BA中，coords实际上是用来从feature map中查找target的
    # 对于真实数据，我们直接将tracks作为目标位置
    
    coords = np.zeros((n_frames, n_points, h, w, 2), dtype=np.float32)
    
    # 将每个点的坐标放在对应位置
    for frame_idx in range(n_frames):
        for point_idx in range(n_points):
            if visibility[frame_idx, point_idx]:
                x, y = tracks[frame_idx, point_idx]
                # 将坐标标准化到图像范围内
                x = np.clip(x, 0, w - 1)
                y = np.clip(y, 0, h - 1)
                
                # 在对应位置设置坐标
                ix, iy = int(x), int(y)
                if 0 <= ix < w and 0 <= iy < h:
                    coords[frame_idx, point_idx, iy, ix, 0] = x
                    coords[frame_idx, point_idx, iy, ix, 1] = y
    
    return [{
        'coords': coords,
        'visible': visibility,
    }]


def convert_cotracker_to_correspondences_sparse(tracks, visibility, image_shape, subsample_points=100):
    """
    将Co-Tracker格式转换为适合BA的稀疏格式
    
    Args:
        tracks: (n_frames, n_points, 2) 跟踪轨迹
        visibility: (n_frames, n_points) 可见性
        image_shape: (h, w) 图像尺寸
        subsample_points: 每帧最多使用多少个点（避免内存爆炸）
    """
    h, w = image_shape
    n_frames, n_points = tracks.shape[0], tracks.shape[1]
    
    # 选择质量最好的点（可见次数最多）
    visibility_counts = visibility.sum(axis=0)  # (n_points,)
    top_point_indices = np.argsort(visibility_counts)[::-1][:subsample_points]
    
    print(f"  - 从{n_points}个点中选择top {len(top_point_indices)}个点")
    print(f"  - 选中点的平均可见次数: {visibility_counts[top_point_indices].mean():.1f}/{n_frames}")
    
    # 只使用选中的点
    tracks_sub = tracks[:, top_point_indices, :]
    visibility_sub = visibility[:, top_point_indices]
    n_points_sub = len(top_point_indices)
    
    # 创建稀疏坐标图
    # 技巧：使用降采样的图像尺寸来减少内存
    downsample = 8  # 与DROID的feature map降采样因子一致
    h_down, w_down = h // downsample, w // downsample
    
    coords = np.zeros((n_frames, n_points_sub, h_down, w_down, 2), dtype=np.float32)
    
    # 为每个点创建一个稀疏的坐标场
    for frame_idx in range(n_frames):
        for point_idx in range(n_points_sub):
            if visibility_sub[frame_idx, point_idx]:
                x, y = tracks_sub[frame_idx, point_idx]
                
                # 降采样坐标
                x_down = x / downsample
                y_down = y / downsample
                
                # 裁剪到降采样图像范围
                x_down = np.clip(x_down, 0, w_down - 1)
                y_down = np.clip(y_down, 0, h_down - 1)
                
                # 在整个降采样图像上广播该点的坐标
                # BA会在对应位置查找depth并计算重投影
                coords[frame_idx, point_idx, :, :, 0] = x_down
                coords[frame_idx, point_idx, :, :, 1] = y_down
    
    print(f"  - coords形状: {coords.shape}")
    print(f"  - 内存占用: {coords.nbytes / 1024**2:.1f} MB")
    
    return [{
        'coords': coords,
        'visible': visibility_sub,
    }], downsample


def filter_by_confidence(tracks, visibility, conf, threshold=2.0):
    """根据置信度过滤跟踪点"""
    print(f"\n过滤低置信度点 (阈值={threshold})...")
    
    n_frames, n_points = tracks.shape[:2]
    h, w = conf.shape[1:3]
    
    filtered_visibility = visibility.copy()
    
    for frame_idx in range(n_frames):
        for point_idx in range(n_points):
            if visibility[frame_idx, point_idx]:
                x, y = tracks[frame_idx, point_idx]
                ix, iy = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
                
                # 检查该位置的置信度
                if conf[frame_idx, iy, ix] < threshold:
                    filtered_visibility[frame_idx, point_idx] = False
    
    n_before = visibility.sum()
    n_after = filtered_visibility.sum()
    print(f"  - 过滤前: {n_before} 可见点")
    print(f"  - 过滤后: {n_after} 可见点")
    print(f"  - 移除: {n_before - n_after} 点 ({(n_before - n_after) / n_before * 100:.1f}%)")
    
    return filtered_visibility


def extract_depths_from_points(points_3d):
    """从3D点云提取深度图"""
    # 深度是z坐标
    depths = points_3d[:, :, :, 2].astype(np.float32)
    return depths


def check_pose_format(poses):
    """检查并转换位姿格式"""
    print("\n检查位姿格式...")
    
    # 检查是否是c2w还是w2c
    # 一个简单的启发式：平移向量的符号
    sample_pose = poses[0]
    t = sample_pose[:3, 3]
    
    print(f"  - 第0帧位姿:")
    print(f"    旋转部分:\n{sample_pose[:3, :3]}")
    print(f"    平移向量: {t}")
    
    # DUSt3R通常输出c2w格式
    # 检查旋转矩阵是否正交
    R = sample_pose[:3, :3]
    is_orthogonal = np.allclose(R @ R.T, np.eye(3), atol=1e-3)
    det = np.linalg.det(R)
    
    print(f"    旋转矩阵正交性: {is_orthogonal}")
    print(f"    旋转矩阵行列式: {det:.6f}")
    
    if is_orthogonal and np.abs(det - 1.0) < 0.1:
        print(f"  → 位姿格式正确 (c2w)")
        return poses
    else:
        print(f"  ⚠ 位姿格式可能有问题")
        return poses


def main():
    print("=" * 80)
    print("使用真实数据运行独立BA模块")
    print("=" * 80)
    
    # 设置路径
    data_dir = Path(__file__).parent / 'data'
    
    # 选择要使用的数据
    cotracker_file = data_dir / 'co_tracker_data' / '0_40' / 'track.npz'
    dust3r_folder = data_dir / 'da3result' / '0_45'
    
    # 1. 加载数据
    print("\n" + "=" * 80)
    print("步骤1: 加载数据")
    print("=" * 80)
    
    tracks, visibility = load_cotracker_data(cotracker_file)
    dust3r_data = load_dust3r_data(dust3r_folder)
    
    # 2. 对齐数据
    print("\n" + "=" * 80)
    print("步骤2: 数据对齐")
    print("=" * 80)
    
    n_frames_cotracker = tracks.shape[0]
    n_frames_dust3r = dust3r_data['poses'].shape[0]
    
    print(f"Co-Tracker帧数: {n_frames_cotracker}")
    print(f"DUSt3R帧数: {n_frames_dust3r}")
    
    # 使用较小的帧数
    n_frames = min(n_frames_cotracker, n_frames_dust3r, 20)  # 只用前20帧以节省内存
    print(f"使用前 {n_frames} 帧（为节省GPU内存）")
    
    # 截取数据
    tracks = tracks[:n_frames]
    visibility = visibility[:n_frames]
    poses = dust3r_data['poses'][:n_frames]
    points = dust3r_data['points'][:n_frames]
    conf = dust3r_data['conf'][:n_frames]
    intrinsic_matrices = dust3r_data['intrinsic'][:n_frames]
    
    # 3. 数据预处理
    print("\n" + "=" * 80)
    print("步骤3: 数据预处理")
    print("=" * 80)
    
    # 检查位姿格式
    poses = check_pose_format(poses)
    
    # 提取深度
    print("\n提取深度图...")
    depths = extract_depths_from_points(points)
    print(f"  - 深度形状: {depths.shape}")
    print(f"  - 深度范围: [{depths.min():.3f}, {depths.max():.3f}] 米")
    print(f"  - 平均深度: {depths.mean():.3f} 米")
    
    # 根据置信度过滤点
    visibility_filtered = filter_by_confidence(
        tracks, visibility, conf, threshold=2.0
    )
    
    # 转换点对应关系
    print("\n转换点对应关系格式...")
    h, w = depths.shape[1:]
    correspondences, downsample_factor = convert_cotracker_to_correspondences_sparse(
        tracks, visibility_filtered, (h, w), subsample_points=50  # 减少点数以避免OOM
    )
    print(f"  - 对应关系组数: {len(correspondences)}")
    print(f"  - coords形状: {correspondences[0]['coords'].shape}")
    print(f"  - visible形状: {correspondences[0]['visible'].shape}")
    print(f"  - 降采样因子: {downsample_factor}")
    
    # 相应地降采样深度图
    print(f"\n降采样深度图以匹配coords...")
    h_down, w_down = h // downsample_factor, w // downsample_factor
    depths_down = np.zeros((n_frames, h_down, w_down), dtype=np.float32)
    for i in range(n_frames):
        import cv2
        depths_down[i] = cv2.resize(depths[i], (w_down, h_down), interpolation=cv2.INTER_LINEAR)
    print(f"  - 原始深度形状: {depths.shape}")
    print(f"  - 降采样深度形状: {depths_down.shape}")
    depths = depths_down
    
    # 准备相机内参 (使用第0帧)
    intrinsic_mat = intrinsic_matrices[0]
    intrinsics = np.array([
        intrinsic_mat[0, 0],  # fx
        intrinsic_mat[1, 1],  # fy
        intrinsic_mat[0, 2],  # cx
        intrinsic_mat[1, 2],  # cy
    ], dtype=np.float32)
    
    print(f"\n相机内参:")
    print(f"  - fx={intrinsics[0]:.2f}, fy={intrinsics[1]:.2f}")
    print(f"  - cx={intrinsics[2]:.2f}, cy={intrinsics[3]:.2f}")
    
    # 4. 执行Bundle Adjustment
    print("\n" + "=" * 80)
    print("步骤4: 执行Bundle Adjustment")
    print("=" * 80)
    
    print("\n初始位姿统计:")
    for i in [0, n_frames // 2, n_frames - 1]:
        t = poses[i, :3, 3]
        print(f"  帧{i}: 位置=({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})")
    
    try:
        print("\n开始优化...")
        print("-" * 80)
        
        # 使用类接口以访问更多参数
        from vipe.slam.standalone_ba import StandaloneBA
        ba_optimizer = StandaloneBA(
            device='cuda',
            n_iters=5,
            pose_damping=1e-3,
            pose_ep=0.1,
        )
        
        optimized_poses = ba_optimizer.optimize(
            poses_c2w=poses,
            metric_depths=depths,
            point_correspondences=correspondences,
            intrinsics=intrinsics,
            fixed_frames=[0],  # 固定第一帧
            verbose=True,
        )
        
        print("-" * 80)
        print("优化完成！")
        
        # 5. 分析结果
        print("\n" + "=" * 80)
        print("步骤5: 结果分析")
        print("=" * 80)
        
        # 计算位姿变化
        pose_changes = []
        for i in range(n_frames):
            t_before = poses[i, :3, 3]
            t_after = optimized_poses[i, :3, 3]
            change = np.linalg.norm(t_after - t_before)
            pose_changes.append(change)
        
        pose_changes = np.array(pose_changes)
        
        print(f"\n位姿变化统计:")
        print(f"  - 平均变化: {pose_changes.mean():.6f} 米")
        print(f"  - 最大变化: {pose_changes.max():.6f} 米 (帧{pose_changes.argmax()})")
        print(f"  - 最小变化: {pose_changes.min():.6f} 米 (帧{pose_changes.argmin()})")
        
        print(f"\n详细对比 (每5帧):")
        for i in range(0, n_frames, 5):
            t_before = poses[i, :3, 3]
            t_after = optimized_poses[i, :3, 3]
            delta = t_after - t_before
            print(f"  帧{i:2d}:")
            print(f"    优化前: ({t_before[0]:7.3f}, {t_before[1]:7.3f}, {t_before[2]:7.3f})")
            print(f"    优化后: ({t_after[0]:7.3f}, {t_after[1]:7.3f}, {t_after[2]:7.3f})")
            print(f"    变化量: ({delta[0]:7.4f}, {delta[1]:7.4f}, {delta[2]:7.4f}) | {np.linalg.norm(delta):.4f}m")
        
        # 6. 保存结果
        print("\n" + "=" * 80)
        print("步骤6: 保存结果")
        print("=" * 80)
        
        output_file = data_dir / 'optimized_poses.npz'
        np.savez(
            output_file,
            optimized_poses=optimized_poses,
            initial_poses=poses,
            pose_changes=pose_changes,
            intrinsics=intrinsics,
            n_frames=n_frames,
        )
        print(f"结果已保存到: {output_file}")
        
        print("\n" + "=" * 80)
        print("✓ 完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ 优化失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    exit_code = main()
    sys.exit(exit_code)
