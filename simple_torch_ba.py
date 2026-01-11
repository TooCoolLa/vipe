#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
基于PyTorch的简洁Bundle Adjustment实现
使用重投影误差的最小二乘优化，支持自动微分
直接处理Co-Tracker稀疏点跟踪数据
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, List, Tuple
from pathlib import Path


class SE3Parameter(nn.Module):
    """SE3位姿参数化（使用李代数表示）"""
    
    def __init__(self, poses_c2w: torch.Tensor):
        """
        Args:
            poses_c2w: (n, 4, 4) 初始位姿矩阵
        """
        super().__init__()
        n = poses_c2w.shape[0]
        
        # 使用6维李代数参数化: [tx, ty, tz, rx, ry, rz]
        # 初始化为零向量（表示单位变换的扰动）
        self.xi = nn.Parameter(torch.zeros(n, 6, dtype=torch.float32, device=poses_c2w.device))
        
        # 存储初始位姿作为参考
        self.register_buffer('poses_init', poses_c2w.clone())
    
    def forward(self) -> torch.Tensor:
        """
        将李代数参数转换为SE3矩阵
        Returns: (n, 4, 4) 位姿矩阵
        """
        n = self.xi.shape[0]
        poses = torch.zeros(n, 4, 4, dtype=self.xi.dtype, device=self.xi.device)
        
        for i in range(n):
            # 提取平移和旋转分量
            t = self.xi[i, :3]  # 平移
            omega = self.xi[i, 3:]  # 旋转轴角
            
            # 旋转轴角转旋转矩阵（Rodrigues公式）
            theta = torch.norm(omega)
            if theta < 1e-6:
                # 小角度近似
                R = torch.eye(3, dtype=omega.dtype, device=omega.device)
                R = R + self._skew_symmetric(omega)
            else:
                k = omega / theta
                K = self._skew_symmetric(k)
                R = torch.eye(3, dtype=omega.dtype, device=omega.device) + \
                    torch.sin(theta) * K + \
                    (1 - torch.cos(theta)) * (K @ K)
            
            # 构建SE3矩阵
            poses[i, :3, :3] = R
            poses[i, :3, 3] = t
            poses[i, 3, 3] = 1.0
            
            # 应用到初始位姿上
            poses[i] = poses[i] @ self.poses_init[i]
        
        return poses
    
    @staticmethod
    def _skew_symmetric(v: torch.Tensor) -> torch.Tensor:
        """构建反对称矩阵"""
        zero = torch.zeros(1, dtype=v.dtype, device=v.device)
        return torch.stack([
            torch.stack([zero[0], -v[2], v[1]]),
            torch.stack([v[2], zero[0], -v[0]]),
            torch.stack([-v[1], v[0], zero[0]])
        ])


class BundleAdjustment(nn.Module):
    """简洁的Bundle Adjustment优化器"""
    
    def __init__(
        self,
        poses_c2w: np.ndarray,
        point_tracks: np.ndarray,
        visibility: np.ndarray,
        depths: np.ndarray,
        intrinsics: np.ndarray,
        device: str = 'cuda',
    ):
        """
        Args:
            poses_c2w: (n_frames, 4, 4) c2w位姿矩阵
            point_tracks: (n_frames, n_points, 2) 2D点轨迹
            visibility: (n_frames, n_points) bool数组，点的可见性
            depths: (n_frames, h, w) 深度图
            intrinsics: (4,) [fx, fy, cx, cy]
            device: 'cuda' 或 'cpu'
        """
        super().__init__()
        
        self.device = torch.device(device)
        self.n_frames = poses_c2w.shape[0]
        self.n_points = point_tracks.shape[1]
        
        # 转换为torch tensor
        poses_c2w_torch = torch.from_numpy(poses_c2w).float().to(self.device)
        self.point_tracks = torch.from_numpy(point_tracks).float().to(self.device)
        self.visibility = torch.from_numpy(visibility).bool().to(self.device)
        self.depths_map = torch.from_numpy(depths).float().to(self.device)
        
        # 相机内参
        self.fx = torch.tensor(intrinsics[0], dtype=torch.float32, device=self.device)
        self.fy = torch.tensor(intrinsics[1], dtype=torch.float32, device=self.device)
        self.cx = torch.tensor(intrinsics[2], dtype=torch.float32, device=self.device)
        self.cy = torch.tensor(intrinsics[3], dtype=torch.float32, device=self.device)
        
        # 位姿参数（可优化）
        self.pose_param = SE3Parameter(poses_c2w_torch)
        
        # 为每个点初始化3D位置（使用第一个可见帧的深度）
        self.points_3d = nn.Parameter(self._initialize_3d_points())
        
        print(f"BA初始化:")
        print(f"  帧数: {self.n_frames}")
        print(f"  点数: {self.n_points}")
        print(f"  总观测数: {self.visibility.sum().item()}")
        print(f"  设备: {self.device}")
    
    def _initialize_3d_points(self) -> torch.Tensor:
        """使用第一个可见帧的深度初始化3D点"""
        points_3d = torch.zeros(self.n_points, 3, dtype=torch.float32, device=self.device)
        
        for point_idx in range(self.n_points):
            # 找到第一个可见的帧
            visible_frames = torch.where(self.visibility[:, point_idx])[0]
            if len(visible_frames) == 0:
                continue
            
            frame_idx = visible_frames[0].item()
            u, v = self.point_tracks[frame_idx, point_idx]
            
            # 从深度图中获取深度（双线性插值）
            h, w = self.depths_map.shape[1:]
            x = torch.clamp(u, 0, w - 1)
            y = torch.clamp(v, 0, h - 1)
            
            # 简单的最近邻采样
            x_int = torch.clamp(x.long(), 0, w - 1)
            y_int = torch.clamp(y.long(), 0, h - 1)
            depth = self.depths_map[frame_idx, y_int, x_int]
            
            # 反投影到3D（相机坐标系）
            x_cam = (x - self.cx) * depth / self.fx
            y_cam = (y - self.cy) * depth / self.fy
            z_cam = depth
            
            # 转换到世界坐标系
            pose_c2w = self.pose_param.poses_init[frame_idx]
            R = pose_c2w[:3, :3]
            t = pose_c2w[:3, 3]
            p_cam = torch.stack([x_cam, y_cam, z_cam])
            p_world = R @ p_cam + t
            
            points_3d[point_idx] = p_world
        
        return points_3d
    
    def project(self, points_3d: torch.Tensor, poses_c2w: torch.Tensor) -> torch.Tensor:
        """
        将3D点投影到图像平面
        
        Args:
            points_3d: (n_points, 3) 世界坐标系下的3D点
            poses_c2w: (n_frames, 4, 4) c2w位姿
        
        Returns:
            projected: (n_frames, n_points, 2) 投影的2D点
            depths: (n_frames, n_points) 深度值
            valid: (n_frames, n_points) 是否在相机前方
        """
        n_frames = poses_c2w.shape[0]
        n_points = points_3d.shape[0]
        
        projected = torch.zeros(n_frames, n_points, 2, device=self.device)
        depths = torch.zeros(n_frames, n_points, device=self.device)
        valid = torch.zeros(n_frames, n_points, dtype=torch.bool, device=self.device)
        
        for frame_idx in range(n_frames):
            # c2w转w2c
            pose_c2w = poses_c2w[frame_idx]
            R_c2w = pose_c2w[:3, :3]
            t_c2w = pose_c2w[:3, 3]
            
            # w2c变换
            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ t_c2w
            
            # 转换到相机坐标系
            points_cam = (R_w2c @ points_3d.T).T + t_w2c  # (n_points, 3)
            
            # 投影到图像平面
            x_cam = points_cam[:, 0]
            y_cam = points_cam[:, 1]
            z_cam = points_cam[:, 2]
            
            # 检查是否在相机前方
            valid[frame_idx] = z_cam > 0.1
            
            # 投影
            u = self.fx * x_cam / (z_cam + 1e-8) + self.cx
            v = self.fy * y_cam / (z_cam + 1e-8) + self.cy
            
            projected[frame_idx, :, 0] = u
            projected[frame_idx, :, 1] = v
            depths[frame_idx] = z_cam
        
        return projected, depths, valid
    
    def compute_reprojection_error(
        self,
        optimize_poses: bool = True,
        optimize_points: bool = True,
        use_huber: bool = True,
        huber_delta: float = 10.0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算重投影误差
        
        Args:
            use_huber: 是否使用Huber损失（对异常值鲁棒）
            huber_delta: Huber损失的阈值（像素）
        
        Returns:
            loss: 标量损失
            info: 包含详细信息的字典
        """
        # 获取当前位姿
        if optimize_poses:
            poses_c2w = self.pose_param()
        else:
            poses_c2w = self.pose_param.poses_init
        
        # 获取当前3D点
        if optimize_points:
            points_3d = self.points_3d
        else:
            points_3d = self.points_3d.detach()
        
        # 投影
        projected, depths, valid = self.project(points_3d, poses_c2w)
        
        # 计算重投影误差
        residuals = projected - self.point_tracks  # (n_frames, n_points, 2)
        
        # 只计算可见且有效的点
        mask = self.visibility & valid  # (n_frames, n_points)
        
        # L2误差
        squared_errors = (residuals ** 2).sum(dim=-1)  # (n_frames, n_points)
        errors = torch.sqrt(squared_errors + 1e-8)
        
        # 使用Huber损失（对大误差降权）
        if use_huber:
            huber_mask = errors <= huber_delta
            loss_per_point = torch.where(
                huber_mask,
                squared_errors,  # L2 for small errors
                2 * huber_delta * errors - huber_delta ** 2  # Linear for large errors
            )
        else:
            loss_per_point = squared_errors
        
        # 动态权重：根据误差大小降权（对异常值鲁棒）
        with torch.no_grad():
            median_error = errors[mask].median() if mask.any() else torch.tensor(1.0, device=self.device)
            weights = 1.0 / (1.0 + (errors / (median_error + 1e-8)) ** 2)
        
        # 应用mask和权重
        weighted_errors = loss_per_point * mask.float() * weights
        
        # 总损失
        loss = weighted_errors.sum() / (mask.sum() + 1e-8)
        
        # 统计信息
        info = {
            'loss': loss.item(),
            'n_observations': mask.sum().item(),
            'mean_error': errors[mask].mean().item() if mask.any() else 0.0,
            'median_error': errors[mask].median().item() if mask.any() else 0.0,
            'max_error': errors[mask].max().item() if mask.any() else 0.0,
            'inliers': (errors[mask] < huber_delta).sum().item() if mask.any() else 0,
        }
        
        return loss, info
    
    def optimize(
        self,
        n_iterations: int = 100,
        lr_pose: float = 1e-4,
        lr_points: float = 1e-3,
        optimize_poses: bool = True,
        optimize_points: bool = True,
        fixed_frames: Optional[List[int]] = None,
        verbose: bool = True,
    ) -> dict:
        """
        执行Bundle Adjustment优化
        
        Args:
            n_iterations: 迭代次数
            lr_pose: 位姿学习率
            lr_points: 3D点学习率
            optimize_poses: 是否优化位姿
            optimize_points: 是否优化3D点
            fixed_frames: 固定的帧索引列表
            verbose: 是否打印进度
        
        Returns:
            result: 包含优化结果的字典
        """
        if fixed_frames is None:
            fixed_frames = [0]  # 默认固定第一帧
        
        # 设置优化器
        params = []
        if optimize_poses:
            params.append({'params': self.pose_param.parameters(), 'lr': lr_pose})
        if optimize_points:
            params.append({'params': [self.points_3d], 'lr': lr_points})
        
        if not params:
            raise ValueError("至少需要优化位姿或3D点中的一个")
        
        optimizer = optim.Adam(params)
        # 使用学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=verbose
        )
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"开始Bundle Adjustment优化")
            print(f"{'='*80}")
            print(f"迭代次数: {n_iterations}")
            print(f"优化位姿: {optimize_poses} (lr={lr_pose})")
            print(f"优化3D点: {optimize_points} (lr={lr_points})")
            print(f"固定帧: {fixed_frames}")
            print(f"{'='*80}\n")
        
        history = {
            'loss': [],
            'mean_error': [],
            'max_error': [],
        }
        
        best_loss = float('inf')
        best_state = None
        
        for iteration in range(n_iterations):
            optimizer.zero_grad()
            
            # 计算损失
            loss, info = self.compute_reprojection_error(
                optimize_poses=optimize_poses,
                optimize_points=optimize_points,
            )
            
            # 反向传播
            loss.backward()
            
            # 固定指定帧的梯度
            if optimize_poses and fixed_frames:
                with torch.no_grad():
                    for frame_idx in fixed_frames:
                        if frame_idx < self.pose_param.xi.shape[0]:
                            self.pose_param.xi.grad[frame_idx] = 0
            
            # 梯度裁剪（防止爆炸）
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            # 更新参数
            optimizer.step()
            scheduler.step(loss)
            
            # 记录
            history['loss'].append(info['loss'])
            history['mean_error'].append(info['mean_error'])
            history['max_error'].append(info['max_error'])
            
            # 保存最佳状态
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {
                    'poses': self.pose_param().detach().cpu().numpy(),
                    'points_3d': self.points_3d.detach().cpu().numpy(),
                }
            
            # 打印进度
            if verbose and (iteration % 10 == 0 or iteration == n_iterations - 1):
                print(f"Iter {iteration:3d}: "
                      f"Loss={info['loss']:.6f}, "
                      f"Mean={info['mean_error']:.2f}px, "
                      f"Median={info['median_error']:.2f}px, "
                      f"Max={info['max_error']:.2f}px, "
                      f"Inliers={info['inliers']}/{info['n_observations']}")
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"优化完成！")
            print(f"最佳损失: {best_loss:.6f}")
            print(f"{'='*80}\n")
        
        return {
            'poses_c2w': best_state['poses'],
            'points_3d': best_state['points_3d'],
            'history': history,
            'final_loss': best_loss,
        }


def run_ba_with_data(
    poses_c2w: np.ndarray,
    point_tracks: np.ndarray,
    visibility: np.ndarray,
    depths: np.ndarray,
    intrinsics: np.ndarray,
    n_iterations: int = 100,
    optimize_poses: bool = True,
    optimize_points: bool = False,
    device: str = 'cuda',
    verbose: bool = True,
) -> dict:
    """
    便捷函数：运行Bundle Adjustment
    
    Args:
        poses_c2w: (n_frames, 4, 4) 初始位姿
        point_tracks: (n_frames, n_points, 2) 2D轨迹
        visibility: (n_frames, n_points) 可见性
        depths: (n_frames, h, w) 深度图
        intrinsics: (4,) [fx, fy, cx, cy]
        n_iterations: 优化迭代次数
        optimize_poses: 是否优化位姿
        optimize_points: 是否优化3D点（通常只优化位姿即可）
        device: 'cuda' 或 'cpu'
        verbose: 是否打印信息
    
    Returns:
        result: 包含优化结果的字典
    """
    ba = BundleAdjustment(
        poses_c2w=poses_c2w,
        point_tracks=point_tracks,
        visibility=visibility,
        depths=depths,
        intrinsics=intrinsics,
        device=device,
    )
    
    result = ba.optimize(
        n_iterations=n_iterations,
        optimize_poses=optimize_poses,
        optimize_points=optimize_points,
        verbose=verbose,
    )
    
    return result


if __name__ == "__main__":
    # 示例：使用data文件夹中的数据
    print("加载数据...")
    
    data_dir = Path(__file__).parent / 'data'
    
    # 加载Co-Tracker数据
    track_data = np.load(data_dir / 'co_tracker_data' / '0_40' / 'track.npz')
    tracks = track_data['tracks'][0]  # (40, 1452, 2)
    visibility = track_data['visibility'][0]  # (40, 1452)
    
    # 加载DUSt3R数据
    poses = np.load(data_dir / 'da3result' / '0_45' / 'camera_poses.npz')['camera_poses'][:40]
    points_3d_data = np.load(data_dir / 'da3result' / '0_45' / 'points.npz')['points'][:40]
    intrinsic_mat = np.load(data_dir / 'da3result' / '0_45' / 'intrinsic.npz')['intrinsic'][0]
    
    # 提取深度
    depths = points_3d_data[:, :, :, 2]
    
    # 准备内参
    intrinsics = np.array([
        intrinsic_mat[0, 0],  # fx
        intrinsic_mat[1, 1],  # fy
        intrinsic_mat[0, 2],  # cx
        intrinsic_mat[1, 2],  # cy
    ], dtype=np.float32)
    
    # 选择一部分点（避免内存问题）并使用质量更好的点
    n_points_to_use = 50  # 减少点数
    visibility_counts = visibility.sum(axis=0)
    
    # 只选择至少在20帧中可见的点
    good_points_mask = visibility_counts >= 20
    good_indices = np.where(good_points_mask)[0]
    
    if len(good_indices) > n_points_to_use:
        # 从好点中选择可见次数最多的
        good_visibility_counts = visibility_counts[good_indices]
        top_good_indices = good_indices[np.argsort(good_visibility_counts)[::-1][:n_points_to_use]]
    else:
        top_good_indices = good_indices
    
    tracks_subset = tracks[:, top_good_indices, :]
    visibility_subset = visibility[:, top_good_indices]
    
    print(f"\n使用数据:")
    print(f"  帧数: {poses.shape[0]}")
    print(f"  点数: {len(top_good_indices)}")
    print(f"  每个点平均可见: {visibility_subset.sum(axis=0).mean():.1f}帧")
    print(f"  图像尺寸: {depths.shape[1]}x{depths.shape[2]}")
    print(f"  可见观测: {visibility_subset.sum()}")
    
    # 运行BA优化
    result = run_ba_with_data(
        poses_c2w=poses,
        point_tracks=tracks_subset,
        visibility=visibility_subset,
        depths=depths,
        intrinsics=intrinsics,
        n_iterations=100,
        optimize_poses=True,
        optimize_points=False,  # 通常不优化3D点，只优化位姿
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=True,
    )
    
    # 显示结果
    print("\n优化结果:")
    print(f"  最终损失: {result['final_loss']:.6f}")
    print(f"  优化后位姿形状: {result['poses_c2w'].shape}")
    
    # 保存结果
    output_file = data_dir / 'simple_ba_result.npz'
    np.savez(
        output_file,
        poses_optimized=result['poses_c2w'],
        poses_initial=poses,
        loss_history=result['history']['loss'],
        mean_error_history=result['history']['mean_error'],
    )
    print(f"\n结果已保存到: {output_file}")
