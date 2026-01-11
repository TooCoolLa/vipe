# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Standalone Bundle Adjustment module
独立的BA模块，可以单独使用，不依赖DROID网络
"""

from typing import List, Tuple
import numpy as np
import torch
from einops import rearrange

from vipe.ext.lietorch import SE3
from vipe.utils.cameras import CameraType
from vipe.slam.ba.solver import Solver
from vipe.slam.ba.terms import DenseDepthFlowTerm
from vipe.slam.maths.retractor import PoseRetractor, DenseDispRetractor
from vipe.slam.ba.kernel import SparseBlockVector


class StandaloneBA:
    """
    独立的Bundle Adjustment优化器
    
    输入:
        - poses_c2w: (n, 4, 4) numpy array, 相机到世界的变换矩阵
        - metric_depths: (n, h, w) numpy array, 每帧的度量深度图
        - point_correspondences: List of dicts, 每个dict包含:
            - 'coords': (x, pointcount, h, w, 2) ndarray, 点的坐标
            - 'visible': (x, pointcount) bool ndarray, 点的可见性
            其中所有x的和等于n
        - intrinsics: (4,) numpy array, [fx, fy, cx, cy]
    
    输出:
        - optimized_poses_c2w: (n, 4, 4) numpy array, 优化后的位姿
    """
    
    def __init__(
        self,
        device: str = "cuda",
        n_iters: int = 3,
        pose_damping: float = 1e-3,
        pose_ep: float = 0.1,
        disp_damping: float = 1e-7,
        camera_type: CameraType = CameraType.PINHOLE,
    ):
        """
        初始化BA优化器
        
        Args:
            device: 计算设备 'cuda' 或 'cpu'
            n_iters: BA迭代次数
            pose_damping: 位姿阻尼系数
            pose_ep: 位姿epsilon
            disp_damping: 深度阻尼系数
            camera_type: 相机类型
        """
        self.device = torch.device(device)
        self.n_iters = n_iters
        self.pose_damping = pose_damping
        self.pose_ep = pose_ep
        self.disp_damping = disp_damping
        self.camera_type = camera_type
    
    def optimize(
        self,
        poses_c2w: np.ndarray,
        metric_depths: np.ndarray,
        point_correspondences: List[dict],
        intrinsics: np.ndarray,
        fixed_frames: List[int] = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        执行Bundle Adjustment优化
        
        Args:
            poses_c2w: (n, 4, 4) 相机到世界的位姿矩阵
            metric_depths: (n, h, w) 度量深度图
            point_correspondences: 点对应关系列表
            intrinsics: (4,) 相机内参 [fx, fy, cx, cy]
            fixed_frames: 固定不优化的帧索引列表，默认固定第0帧
            verbose: 是否打印优化信息
        
        Returns:
            optimized_poses_c2w: (n, 4, 4) 优化后的位姿
        """
        # 参数验证
        n_frames = poses_c2w.shape[0]
        assert poses_c2w.shape == (n_frames, 4, 4), f"poses shape should be (n,4,4), got {poses_c2w.shape}"
        assert metric_depths.shape[0] == n_frames, "depths and poses must have same number of frames"
        
        h, w = metric_depths.shape[1:]
        
        # 转换为torch张量
        poses_c2w_torch = torch.from_numpy(poses_c2w).float().to(self.device)
        metric_depths_torch = torch.from_numpy(metric_depths).float().to(self.device)
        intrinsics_torch = torch.from_numpy(intrinsics).float().to(self.device)
        
        # 转换c2w到w2c (BA使用世界到相机的变换)
        poses_w2c = self._c2w_to_w2c(poses_c2w_torch)
        
        # 转换深度到视差 (disparity = 1/depth)
        disps = torch.where(
            metric_depths_torch > 0,
            1.0 / metric_depths_torch,
            torch.zeros_like(metric_depths_torch)
        )
        
        # 构建边和目标
        edges_ii, edges_jj, target, weight = self._build_edges_from_correspondences(
            point_correspondences, h, w
        )
        
        if len(edges_ii) == 0:
            if verbose:
                print("Warning: No valid correspondences found, returning original poses")
            return poses_c2w
        
        # 执行优化
        optimized_poses_w2c = self._run_ba(
            poses_w2c=poses_w2c,
            disps=disps,
            edges_ii=edges_ii,
            edges_jj=edges_jj,
            target=target,
            weight=weight,
            intrinsics=intrinsics_torch,
            fixed_frames=fixed_frames if fixed_frames is not None else [0],
            verbose=verbose,
        )
        
        # 转换回c2w
        optimized_poses_c2w = self._w2c_to_c2w(optimized_poses_w2c)
        
        return optimized_poses_c2w.cpu().numpy()
    
    def _c2w_to_w2c(self, poses_c2w: torch.Tensor) -> torch.Tensor:
        """将c2w位姿转换为w2c"""
        n = poses_c2w.shape[0]
        poses_w2c = torch.zeros_like(poses_c2w)
        
        for i in range(n):
            R_c2w = poses_c2w[i, :3, :3]
            t_c2w = poses_c2w[i, :3, 3]
            
            # w2c = [R_c2w^T | -R_c2w^T @ t_c2w]
            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ t_c2w
            
            poses_w2c[i, :3, :3] = R_w2c
            poses_w2c[i, :3, 3] = t_w2c
            poses_w2c[i, 3, 3] = 1.0
        
        return poses_w2c
    
    def _w2c_to_c2w(self, poses_w2c: torch.Tensor) -> torch.Tensor:
        """将w2c位姿转换为c2w"""
        return self._c2w_to_w2c(poses_w2c)  # 互逆操作相同
    
    def _build_edges_from_correspondences(
        self,
        point_correspondences: List[dict],
        h: int,
        w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从点对应关系构建BA的边和目标
        
        Returns:
            edges_ii: (n_edges,) 源帧索引
            edges_jj: (n_edges,) 目标帧索引
            target: (n_edges, h*w, 2) 目标像素坐标
            weight: (n_edges, h*w, 2) 权重
        """
        all_ii = []
        all_jj = []
        all_targets = []
        all_weights = []
        
        frame_offset = 0
        
        for correspondence in point_correspondences:
            coords = correspondence['coords']  # (x, pointcount, h, w, 2)
            visible = correspondence['visible']  # (x, pointcount)
            
            x, pointcount, ch, cw, _ = coords.shape
            assert ch == h and cw == w, f"coords spatial dims ({ch},{cw}) must match depth ({h},{w})"
            
            # 为这组帧内的每个点建立边
            for point_idx in range(pointcount):
                # 找到该点可见的所有帧
                visible_frames = np.where(visible[:, point_idx])[0]
                
                if len(visible_frames) < 2:
                    continue  # 至少需要2帧才能建立约束
                
                # 在可见帧之间建立边（星型连接：第一个可见帧到其他所有可见帧）
                source_frame = visible_frames[0]
                source_global_idx = frame_offset + source_frame
                
                for target_local_idx in visible_frames[1:]:
                    target_global_idx = frame_offset + target_local_idx
                    
                    # 提取源帧和目标帧的坐标
                    source_coords = coords[source_frame, point_idx]  # (h, w, 2)
                    target_coords = coords[target_local_idx, point_idx]  # (h, w, 2)
                    
                    all_ii.append(source_global_idx)
                    all_jj.append(target_global_idx)
                    all_targets.append(target_coords)  # (h, w, 2)
                    all_weights.append(np.ones_like(target_coords))  # 默认权重为1
            
            frame_offset += x
        
        if len(all_ii) == 0:
            # 没有有效的对应关系
            return (
                torch.tensor([], dtype=torch.long, device=self.device),
                torch.tensor([], dtype=torch.long, device=self.device),
                torch.zeros((0, h*w, 2), device=self.device),
                torch.zeros((0, h*w, 2), device=self.device),
            )
        
        # 转换为torch张量
        edges_ii = torch.tensor(all_ii, dtype=torch.long, device=self.device)
        edges_jj = torch.tensor(all_jj, dtype=torch.long, device=self.device)
        
        # 堆叠并重塑target和weight
        target = torch.from_numpy(np.stack(all_targets, axis=0)).float().to(self.device)
        weight = torch.from_numpy(np.stack(all_weights, axis=0)).float().to(self.device)
        
        # 重塑为 (n_edges, h*w, 2)
        target = rearrange(target, 'n h w c -> n (h w) c')
        weight = rearrange(weight, 'n h w c -> n (h w) c')
        
        return edges_ii, edges_jj, target, weight
    
    def _run_ba(
        self,
        poses_w2c: torch.Tensor,
        disps: torch.Tensor,
        edges_ii: torch.Tensor,
        edges_jj: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
        intrinsics: torch.Tensor,
        fixed_frames: List[int],
        verbose: bool,
    ) -> torch.Tensor:
        """
        执行实际的BA优化
        
        Args:
            poses_w2c: (n, 4, 4) w2c位姿矩阵
            disps: (n, h, w) 视差图
            edges_ii, edges_jj: 边的索引
            target, weight: BA的目标和权重
            intrinsics: 相机内参
            fixed_frames: 固定的帧
            verbose: 是否打印信息
        
        Returns:
            optimized_poses_w2c: (n, 4, 4) 优化后的位姿
        """
        n_frames = poses_w2c.shape[0]
        h, w = disps.shape[1:]
        
        # 转换4x4矩阵为SE3的7维表示 [tx, ty, tz, qx, qy, qz, qw]
        poses_se3 = self._matrix_to_se3(poses_w2c)
        
        # 展平深度图
        disps_flat = rearrange(disps, 'n h w -> n (h w)')
        
        # 创建求解器
        solver = Solver(compute_energy=verbose)
        
        # 添加重投影约束项
        solver.add_term(
            DenseDepthFlowTerm(
                pose_i_inds=edges_ii,
                pose_j_inds=edges_jj,
                rig_i_inds=torch.zeros_like(edges_ii),  # 单目，只有1个视角
                rig_j_inds=torch.zeros_like(edges_jj),
                dense_disp_i_inds=edges_ii,  # 使用源帧的深度
                target=target,
                weight=weight * 0.001,  # 权重缩放
                intrinsics=intrinsics.unsqueeze(0),  # (1, 4)
                intrinsics_factor=1.0,  # 深度图已经是全分辨率
                rig=SE3.Identity(1).to(self.device),  # 单目，rig为单位矩阵
                image_size=(h, w),
                camera_type=self.camera_type,
            )
        )
        
        # 设置位姿优化参数
        fixed_indices = torch.tensor(fixed_frames, dtype=torch.long, device=self.device)
        solver.set_fixed("pose", fixed_indices)
        solver.set_retractor("pose", PoseRetractor())
        solver.set_damping("pose", damping=self.pose_damping, ep=self.pose_ep)
        
        # 设置深度优化参数
        solver.set_retractor("dense_disp", DenseDispRetractor())
        solver.set_damping(
            "dense_disp",
            damping=SparseBlockVector(
                inds=torch.arange(n_frames, device=self.device),
                data=torch.ones(n_frames, h*w, device=self.device) * self.disp_damping,
            ),
            ep=self.disp_damping,
        )
        solver.set_marginilized("dense_disp")  # 边缘化深度变量
        
        # 固定内参和rig
        solver.set_retractor("intrinsics", None)
        solver.set_fixed("intrinsics")
        solver.set_retractor("rig", None)
        solver.set_fixed("rig")
        
        # 迭代优化
        ba_energy = []
        for i in range(self.n_iters):
            cur_energy = solver.run_inplace(
                {
                    "pose": SE3(poses_se3),
                    "dense_disp": disps_flat,
                    "intrinsics": intrinsics.unsqueeze(0),
                    "rig": SE3.Identity(1).to(self.device),
                }
            )
            ba_energy.append(cur_energy)
        
        if verbose:
            print(f"BA optimization: {self.n_iters} iterations")
            print(f"  Initial energy: {ba_energy[0]:.6f}")
            print(f"  Final energy:   {ba_energy[-1]:.6f}")
            print(f"  Reduction:      {ba_energy[0] - ba_energy[-1]:.6f}")
        
        # 转换回4x4矩阵
        optimized_poses_w2c = self._se3_to_matrix(poses_se3)
        
        return optimized_poses_w2c
    
    def _matrix_to_se3(self, poses: torch.Tensor) -> torch.Tensor:
        """
        将4x4位姿矩阵转换为SE3的7维表示
        
        Args:
            poses: (n, 4, 4) 位姿矩阵
        
        Returns:
            se3: (n, 7) [tx, ty, tz, qx, qy, qz, qw]
        """
        from scipy.spatial.transform import Rotation
        
        n = poses.shape[0]
        se3 = torch.zeros(n, 7, device=poses.device, dtype=poses.dtype)
        
        for i in range(n):
            # 提取平移
            se3[i, :3] = poses[i, :3, 3]
            
            # 提取旋转并转换为四元数
            R = poses[i, :3, :3].cpu().numpy()
            rot = Rotation.from_matrix(R)
            quat = rot.as_quat()  # [qx, qy, qz, qw]
            se3[i, 3:] = torch.from_numpy(quat).to(poses.device)
        
        return se3
    
    def _se3_to_matrix(self, se3: torch.Tensor) -> torch.Tensor:
        """
        将SE3的7维表示转换为4x4位姿矩阵
        
        Args:
            se3: (n, 7) [tx, ty, tz, qx, qy, qz, qw]
        
        Returns:
            poses: (n, 4, 4) 位姿矩阵
        """
        from scipy.spatial.transform import Rotation
        
        n = se3.shape[0]
        poses = torch.eye(4, device=se3.device, dtype=se3.dtype).unsqueeze(0).repeat(n, 1, 1)
        
        for i in range(n):
            # 设置平移
            poses[i, :3, 3] = se3[i, :3]
            
            # 设置旋转
            quat = se3[i, 3:].cpu().numpy()  # [qx, qy, qz, qw]
            rot = Rotation.from_quat(quat)
            R = rot.as_matrix()
            poses[i, :3, :3] = torch.from_numpy(R).to(se3.device)
        
        return poses


def standalone_bundle_adjustment(
    poses_c2w: np.ndarray,
    metric_depths: np.ndarray,
    point_correspondences: List[dict],
    intrinsics: np.ndarray,
    device: str = "cuda",
    n_iters: int = 3,
    fixed_frames: List[int] = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    便捷函数：执行独立的Bundle Adjustment优化
    
    Args:
        poses_c2w: (n, 4, 4) numpy array, 相机到世界的变换矩阵
        metric_depths: (n, h, w) numpy array, 每帧的度量深度图
        point_correspondences: List of dicts, 每个dict包含:
            - 'coords': (x, pointcount, h, w, 2) ndarray, 点的坐标
            - 'visible': (x, pointcount) bool ndarray, 点的可见性
        intrinsics: (4,) numpy array, [fx, fy, cx, cy]
        device: 'cuda' 或 'cpu'
        n_iters: BA迭代次数
        fixed_frames: 固定不优化的帧索引列表，默认固定第0帧
        verbose: 是否打印优化信息
    
    Returns:
        optimized_poses_c2w: (n, 4, 4) numpy array, 优化后的位姿
    
    Example:
        >>> import numpy as np
        >>> 
        >>> # 准备数据
        >>> n_frames = 10
        >>> h, w = 480, 640
        >>> 
        >>> # 初始位姿（c2w）
        >>> poses = np.eye(4)[None].repeat(n_frames, axis=0)
        >>> for i in range(n_frames):
        >>>     poses[i, 0, 3] = i * 0.1  # 沿x轴移动
        >>> 
        >>> # 深度图
        >>> depths = np.ones((n_frames, h, w)) * 5.0  # 5米深度
        >>> 
        >>> # 点对应关系
        >>> correspondences = [{
        >>>     'coords': np.random.rand(5, 10, h, w, 2) * [w, h],  # 5帧，10个点
        >>>     'visible': np.random.rand(5, 10) > 0.3,  # 随机可见性
        >>> }]
        >>> 
        >>> # 相机内参
        >>> K = np.array([500.0, 500.0, 320.0, 240.0])
        >>> 
        >>> # 执行优化
        >>> optimized_poses = standalone_bundle_adjustment(
        >>>     poses, depths, correspondences, K, verbose=True
        >>> )
    """
    ba = StandaloneBA(device=device, n_iters=n_iters)
    return ba.optimize(
        poses_c2w=poses_c2w,
        metric_depths=metric_depths,
        point_correspondences=point_correspondences,
        intrinsics=intrinsics,
        fixed_frames=fixed_frames,
        verbose=verbose,
    )
