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

import argparse
import asyncio
import logging
import socket
import time

from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import viser
import viser.transforms as tf

from matplotlib import cm
from PIL import Image
from rich.logging import RichHandler

from vipe.utils.cameras import CameraType
from vipe.utils.depth import reliable_depth_mask_range
from vipe.utils.io import (
    ArtifactPath,
    read_depth_artifacts,
    read_intrinsics_artifacts,
    read_pose_artifacts,
    read_rgb_artifacts,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


@dataclass
class GlobalContext:
    artifacts: list[ArtifactPath]


_global_context: GlobalContext | None = None


@dataclass
class SceneFrameHandle:
    frame_handle: viser.FrameHandle
    frustum_handle: viser.CameraFrustumHandle
    pcd_handle: viser.PointCloudHandle | None = None
    frame_idx: int = 0
    is_loaded: bool = False

    def __post_init__(self):
        self.visible = False

    @property
    def visible(self) -> bool:
        return self.frame_handle.visible

    @visible.setter
    def visible(self, value: bool):
        self.frame_handle.visible = value
        self.frustum_handle.visible = value
        if self.pcd_handle is not None:
            self.pcd_handle.visible = value
    
    def unload(self):
        """Release resources for this frame."""
        if self.pcd_handle is not None:
            self.pcd_handle.remove()
            self.pcd_handle = None
        self.is_loaded = False


class ClientClosures:
    """
    All class methods automatically capture 'self', ensuring proper locals.
    """

    def __init__(self, client: viser.ClientHandle):
        self.client = client

        async def _run():
            try:
                await self.run()
            except asyncio.CancelledError:
                pass
            finally:
                self.cleanup()

        # Don't await to not block the rest of the coroutine.
        self.task = asyncio.create_task(_run())

        self.gui_playback_handle: viser.GuiFolderHandle | None = None
        self.gui_timestep: viser.GuiSliderHandle | None = None
        self.gui_framerate: viser.GuiSliderHandle | None = None
        self.scene_frame_handles: list[SceneFrameHandle] = []
        self.current_displayed_timestep: int = 0
        self.frame_metadata: list[dict] = []  # Store metadata instead of full data
        self.ring_buffer_size: int = 30  # Ring buffer size for smooth playback
        # Ring buffer cache - FIFO queue of frames
        self.frame_data_cache: dict[int, tuple[np.ndarray, np.ndarray | None, np.ndarray | None]] = {}
        self.ring_buffer_frames: list[int] = []  # Frame indices in buffer (FIFO order)
        self.preload_task: asyncio.Task | None = None  # Background preload task
        self.scene_root_handle: viser.FrameHandle | None = None  # Root frame for scene transformation
        self.is_playing: bool = False  # Playback state
        self.trajectory_frustums: list[tuple[viser.FrameHandle, viser.CameraFrustumHandle]] = []  # Historical frustums for trajectory (frame, frustum pairs)
        self.trajectory_update_task: asyncio.Task | None = None  # Debounced trajectory update task
        self.last_trajectory_frame: int = -1  # Last frame for which trajectory was updated

    async def stop(self):
        self.task.cancel()
        await self.task

    async def run(self):
        logger.info(f"Client {self.client.client_id} connected")

        all_artifacts = self.global_context().artifacts

        with self.client.gui.add_folder("Sample"):
            self.gui_id = self.client.gui.add_slider(
                "Artifact ID", min=0, max=len(all_artifacts) - 1, step=1, initial_value=0
            )
            gui_id_changer = self.client.gui.add_button_group(label="ID +/-", options=["Prev", "Next"])

            @gui_id_changer.on_click
            async def _(_) -> None:
                if gui_id_changer.value == "Prev":
                    self.gui_id.value = (self.gui_id.value - 1) % len(all_artifacts)
                else:
                    self.gui_id.value = (self.gui_id.value + 1) % len(all_artifacts)

            self.gui_name = self.client.gui.add_text("Artifact Name", "")
            self.gui_t_sub = self.client.gui.add_slider("Temporal subsample", min=1, max=16, step=1, initial_value=1)
            self.gui_s_sub = self.client.gui.add_slider("Spatial subsample", min=1, max=8, step=1, initial_value=2)
            self.gui_id.on_update(self.on_sample_update)
            self.gui_t_sub.on_update(self.on_sample_update)
            self.gui_s_sub.on_update(self.on_sample_update)

        with self.client.gui.add_folder("Scene"):
            self.gui_ring_buffer_size = self.client.gui.add_slider(
                "Ring buffer size", min=10, max=100, step=10, initial_value=30
            )
            
            @self.gui_ring_buffer_size.on_update
            async def _(_) -> None:
                self.ring_buffer_size = self.gui_ring_buffer_size.value
                # Reload buffer with new size
                await self._preload_ring_buffer(self.current_displayed_timestep)
            
            # Scene transformation controls
            gui_scene_transform_folder = self.client.gui.add_folder("Scene Transform")
            with gui_scene_transform_folder:
                self.gui_scene_position_x = self.client.gui.add_slider(
                    "Position X", min=-10.0, max=10.0, step=0.1, initial_value=0.0
                )
                self.gui_scene_position_y = self.client.gui.add_slider(
                    "Position Y", min=-10.0, max=10.0, step=0.1, initial_value=0.0
                )
                self.gui_scene_position_z = self.client.gui.add_slider(
                    "Position Z", min=-10.0, max=10.0, step=0.1, initial_value=0.0
                )
                self.gui_scene_scale = self.client.gui.add_slider(
                    "Scale", min=0.1, max=5.0, step=0.1, initial_value=1.0
                )
                gui_reset_transform = self.client.gui.add_button("Reset Transform")
                
                @self.gui_scene_position_x.on_update
                @self.gui_scene_position_y.on_update
                @self.gui_scene_position_z.on_update
                @self.gui_scene_scale.on_update
                async def _(_) -> None:
                    self._update_scene_transform()
                
                @gui_reset_transform.on_click
                async def _(_) -> None:
                    self.gui_scene_position_x.value = 0.0
                    self.gui_scene_position_y.value = 0.0
                    self.gui_scene_position_z.value = 0.0
                    self.gui_scene_scale.value = 1.0
                    self._update_scene_transform()
            
            self.gui_point_size = self.client.gui.add_slider(
                "Point size", min=0.0001, max=0.01, step=0.001, initial_value=0.001
            )

            # Update point cloud size
            @self.gui_point_size.on_update
            async def _(_) -> None:
                for frame_node in self.scene_frame_handles:
                    if frame_node.pcd_handle is not None:
                        frame_node.pcd_handle.point_size = self.gui_point_size.value

            self.gui_frustum_size = self.client.gui.add_slider(
                "Frustum size", min=0.01, max=0.5, step=0.01, initial_value=0.15
            )

            @self.gui_frustum_size.on_update
            async def _(_) -> None:
                for frame_node in self.scene_frame_handles:
                    frame_node.frustum_handle.scale = self.gui_frustum_size.value

            self.gui_colorful_frustum_toggle = self.client.gui.add_checkbox(
                "Colorful Frustum",
                initial_value=False,
            )

            @self.gui_colorful_frustum_toggle.on_update
            async def _(_) -> None:
                self._set_frustum_color(self.gui_colorful_frustum_toggle.value)

            self.gui_show_trajectory = self.client.gui.add_checkbox(
                "Show Camera Trajectory",
                initial_value=True,
            )

            @self.gui_show_trajectory.on_update
            async def _(_) -> None:
                self._update_trajectory_visibility()
            
            self.gui_trajectory_frustum_scale = self.client.gui.add_slider(
                "Trajectory Frustum Scale", min=0.01, max=0.3, step=0.01, initial_value=0.05
            )
            
            @self.gui_trajectory_frustum_scale.on_update
            async def _(_) -> None:
                # Update all trajectory frustums with new scale using debouncing
                self.last_trajectory_frame = -1  # Force rebuild
                if self.gui_show_trajectory.value:
                    await self._debounced_update_trajectory(self.current_displayed_timestep)

            self.gui_fov = self.client.gui.add_slider("FoV", min=30.0, max=120.0, step=1.0, initial_value=60.0)

            @self.gui_fov.on_update
            async def _(_) -> None:
                self.client.camera.fov = np.deg2rad(self.gui_fov.value)

            # Camera controls
            gui_camera_folder = self.client.gui.add_folder("Camera Control")
            with gui_camera_folder:
                self.gui_camera_position_x = self.client.gui.add_slider(
                    "Camera X", min=-20.0, max=20.0, step=0.5, initial_value=0.0
                )
                self.gui_camera_position_y = self.client.gui.add_slider(
                    "Camera Y", min=-20.0, max=20.0, step=0.5, initial_value=0.0
                )
                self.gui_camera_position_z = self.client.gui.add_slider(
                    "Camera Z", min=-20.0, max=20.0, step=0.5, initial_value=5.0
                )
                self.gui_camera_look_at_x = self.client.gui.add_slider(
                    "Look At X", min=-20.0, max=20.0, step=0.5, initial_value=0.0
                )
                self.gui_camera_look_at_y = self.client.gui.add_slider(
                    "Look At Y", min=-20.0, max=20.0, step=0.5, initial_value=0.0
                )
                self.gui_camera_look_at_z = self.client.gui.add_slider(
                    "Look At Z", min=-20.0, max=20.0, step=0.5, initial_value=0.0
                )
                gui_reset_camera = self.client.gui.add_button("Reset Camera")
                gui_look_at_origin = self.client.gui.add_button("Look at Origin")
                
                @self.gui_camera_position_x.on_update
                @self.gui_camera_position_y.on_update
                @self.gui_camera_position_z.on_update
                @self.gui_camera_look_at_x.on_update
                @self.gui_camera_look_at_y.on_update
                @self.gui_camera_look_at_z.on_update
                async def _(_) -> None:
                    self._update_camera()
                
                @gui_reset_camera.on_click
                async def _(_) -> None:
                    self.gui_camera_position_x.value = 0.0
                    self.gui_camera_position_y.value = 0.0
                    self.gui_camera_position_z.value = 5.0
                    self.gui_camera_look_at_x.value = 0.0
                    self.gui_camera_look_at_y.value = 0.0
                    self.gui_camera_look_at_z.value = 0.0
                    self._update_camera()
                
                @gui_look_at_origin.on_click
                async def _(_) -> None:
                    self.gui_camera_look_at_x.value = 0.0
                    self.gui_camera_look_at_y.value = 0.0
                    self.gui_camera_look_at_z.value = 0.0
                    self._update_camera()

            gui_snapshot = self.client.gui.add_button(
                "Snapshot",
                hint="Take a snapshot of the current scene",
            )

            # Async get_render does not work at the moment, we will put into thread loop.
            @gui_snapshot.on_click
            def _(_) -> None:
                current_artifact = self.global_context().artifacts[self.gui_id.value]
                file_name = f"{current_artifact.base_path.name}_{current_artifact.artifact_name}.png"
                snapshot_img = self.client.get_render(height=720, width=1280, transport_format="png")
                self.client.send_file_download(file_name, iio.imwrite("<bytes>", snapshot_img, extension=".png"))

        await self.on_sample_update(None)
        
        # Set initial camera position
        self._update_camera()

        while True:
            # Only auto-advance if playing and framerate > 0
            if self.is_playing and self.gui_framerate is not None and self.gui_framerate.value > 0:
                self._incr_timestep()
                await asyncio.sleep(1.0 / self.gui_framerate.value)
            else:
                await asyncio.sleep(0.1)  # Shorter sleep when paused for responsive controls

    async def on_sample_update(self, _):
        with self.client.atomic():
            self._rebuild_scene()
        self._rebuild_playback_gui()
        self._set_frustum_color(self.gui_colorful_frustum_toggle.value)

    def _set_frustum_color(self, colorful: bool):
        for frame_idx, frame_node in enumerate(self.scene_frame_handles):
            if not colorful:
                frame_node.frustum_handle.color = (0, 0, 0)
            else:
                # Use a rainbow color based on the frame index
                denom = len(self.scene_frame_handles) - 1
                rainbow_value = cm.jet(1.0 - frame_idx / denom)[:3]
                rainbow_value = tuple((int(c * 255) for c in rainbow_value))
                frame_node.frustum_handle.color = rainbow_value

    def _update_camera_trajectory(self, current_frame: int):
        """Update camera trajectory visualization up to current frame with frustums."""
        # Skip if trajectory is disabled or not changed
        if not self.gui_show_trajectory.value or len(self.frame_metadata) == 0:
            return
        
        if current_frame < 1:
            return  # Need at least 2 frames for trajectory
        
        # Skip if we already updated for this frame
        if current_frame == self.last_trajectory_frame:
            return
        
        self.last_trajectory_frame = current_frame
        
        # Clear old trajectory frustums
        for frame_handle, frustum in self.trajectory_frustums:
            frustum.remove()
            frame_handle.remove()
        self.trajectory_frustums.clear()
        
        # Get trajectory frustum scale
        trajectory_scale = self.gui_trajectory_frustum_scale.value
        spatial_subsample = self.gui_s_sub.value
        
        # Smart sampling: don't create frustum for EVERY frame if we have many
        # Show at most 50 trajectory frustums to keep it responsive
        max_trajectory_frustums = 50
        step = max(1, current_frame // max_trajectory_frustums)
        
        # Create frustums for sampled frames from 0 to current_frame-1 (excluding current)
        current_artifact = self.global_context().artifacts[self.gui_id.value]
        
        # We need to load RGB data for trajectory frames
        # To avoid performance issues, we'll load them in batches or use cached data
        for i in range(0, current_frame, step):
            if i >= len(self.frame_metadata):
                break
            
            metadata = self.frame_metadata[i]
            c2w = metadata['c2w']
            intr = metadata['intrinsics']
            camera_type = metadata['camera_type']
            
            # Get image if available in cache, otherwise skip or use placeholder
            if i in self.frame_data_cache:
                sampled_rgb, _, _ = self.frame_data_cache[i]
                # Create thumbnail
                frame_thumbnail = Image.fromarray(sampled_rgb)
                frame_thumbnail.thumbnail((100, 100), Image.Resampling.LANCZOS)
                thumbnail_array = np.array(frame_thumbnail)
            else:
                # Skip if not in cache to avoid loading delays
                # Or use a small placeholder
                thumbnail_array = None
            
            # Calculate frustum parameters
            pinhole_intr = camera_type.build_camera_model(intr).pinhole().intrinsics
            frame_height = int(pinhole_intr[3].item() * 2)
            frame_width = int(pinhole_intr[2].item() * 2)
            fov = 2 * np.arctan2(frame_height / 2, pinhole_intr[0].item())
            
            # Create frame handle first (like current frame does)
            frame_handle = self.client.scene.add_frame(
                f"/scene_root/trajectory/frame_{i}",
                axes_length=0.0,  # No axes for trajectory frames
                axes_radius=0.0,
                wxyz=tf.SO3.from_matrix(c2w[:3, :3]).wxyz,
                position=c2w[:3, 3],
            )
            
            # Create frustum as child of the frame (like current frame does)
            frustum = self.client.scene.add_camera_frustum(
                f"/scene_root/trajectory/frame_{i}/frustum",
                fov=fov,
                aspect=frame_width / frame_height,
                scale=trajectory_scale,
                image=thumbnail_array if thumbnail_array is not None else None,
            )
            
            # Set color based on position in trajectory (blue to red gradient)
            if thumbnail_array is None:
                t = i / max(current_frame, 1)
                frustum.color = (int(255 * t), 0, int(255 * (1 - t)))
            
            self.trajectory_frustums.append((frame_handle, frustum))
        
        logger.debug(f"Updated trajectory with {len(self.trajectory_frustums)} frustums (step={step})")
    
    def _update_trajectory_visibility(self):
        """Toggle trajectory visibility."""
        if self.gui_show_trajectory.value:
            # Reset last frame to force rebuild
            self.last_trajectory_frame = -1
            # Rebuild trajectory for current frame
            self._update_camera_trajectory(self.current_displayed_timestep)
        else:
            # Clear all trajectory frustums
            for frame_handle, frustum in self.trajectory_frustums:
                frustum.remove()
                frame_handle.remove()
            self.trajectory_frustums.clear()
            self.last_trajectory_frame = -1

    def _update_scene_transform(self):
        """Update the root scene transformation based on GUI controls."""
        if self.scene_root_handle is None:
            return
        
        position = np.array([
            self.gui_scene_position_x.value,
            self.gui_scene_position_y.value,
            self.gui_scene_position_z.value,
        ])
        scale = self.gui_scene_scale.value
        
        # Update root frame position and scale
        # Note: viser doesn't have direct scale, so we'd need to scale all children
        # For now, just update position
        self.scene_root_handle.position = position
        
        # If scale is needed, we'd have to recreate frames with scaled positions
        if abs(scale - 1.0) > 0.01:
            logger.warning("Scene scaling not yet fully implemented - only translation works")

    def _update_camera(self):
        """Update camera position and look-at target."""
        position = np.array([
            self.gui_camera_position_x.value,
            self.gui_camera_position_y.value,
            self.gui_camera_position_z.value,
        ])
        look_at = np.array([
            self.gui_camera_look_at_x.value,
            self.gui_camera_look_at_y.value,
            self.gui_camera_look_at_z.value,
        ])
        
        # Calculate camera orientation from position and look-at
        # Forward direction (camera looks in -Z direction in its own frame)
        forward = look_at - position
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-6:
            # Avoid division by zero
            return
        forward = forward / forward_norm
        
        # Up vector (world up is typically +Y or -Y depending on scene)
        world_up = np.array([0.0, 1.0, 0.0])
        
        # Right vector
        right = np.cross(forward, world_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            # Forward is parallel to up, use a different up vector
            world_up = np.array([1.0, 0.0, 0.0])
            right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        
        # Recompute up to make sure it's orthogonal
        up = np.cross(right, forward)
        
        # Build rotation matrix (camera looks in -Z, so forward is actually -Z)
        R = np.eye(3)
        R[:, 0] = right
        R[:, 1] = up
        R[:, 2] = -forward  # Camera looks in -Z direction
        
        # Convert to quaternion (wxyz format)
        wxyz = tf.SO3.from_matrix(R).wxyz
        
        # Update camera
        self.client.camera.position = position
        self.client.camera.wxyz = wxyz

    def _prepare_frame_metadata(self):
        """Prepare metadata for all frames without loading full data."""
        current_artifact = self.global_context().artifacts[self.gui_id.value]
        temporal_subsample: int = self.gui_t_sub.value
        
        self.frame_metadata = []
        
        poses = read_pose_artifacts(current_artifact.pose_path)[1].matrix().numpy()
        intrinsics_data = read_intrinsics_artifacts(
            current_artifact.intrinsics_path, 
            current_artifact.camera_type_path
        )
        
        frame_count = 0
        for frame_idx in range(len(poses)):
            if frame_idx % temporal_subsample != 0:
                continue
                
            self.frame_metadata.append({
                'original_idx': frame_idx,
                'display_idx': frame_count,
                'c2w': poses[frame_idx],
                'intrinsics': intrinsics_data[1][frame_idx],
                'camera_type': intrinsics_data[2][frame_idx],
            })
            frame_count += 1

    def _load_frame_data_batch(self, frame_indices: list[int]):
        """Load multiple frames in one pass through the data to avoid O(n²) complexity."""
        current_artifact = self.global_context().artifacts[self.gui_id.value]
        spatial_subsample: int = self.gui_s_sub.value
        
        # Filter out already cached frames
        indices_to_load = [idx for idx in frame_indices if idx not in self.frame_data_cache]
        if not indices_to_load:
            return
        
        logger.info(f"Loading {len(indices_to_load)} frames in batch: {indices_to_load[:5]}{'...' if len(indices_to_load) > 5 else ''}")
        
        # Map original frame indices we need
        original_indices = {self.frame_metadata[idx]['original_idx'] for idx in indices_to_load}
        
        # Single pass through RGB data
        rgb_frames = {}
        for frame_idx, (_, rgb) in enumerate(read_rgb_artifacts(current_artifact.rgb_path)):
            if frame_idx in original_indices:
                sampled_rgb = (rgb.cpu().numpy() * 255).astype(np.uint8)
                sampled_rgb = sampled_rgb[::spatial_subsample, ::spatial_subsample]
                rgb_frames[frame_idx] = (rgb, sampled_rgb)
            if len(rgb_frames) == len(original_indices):
                break
        
        # Single pass through depth data
        depth_frames = {}
        try:
            for frame_idx, (_, depth) in enumerate(read_depth_artifacts(current_artifact.depth_path)):
                if frame_idx in original_indices:
                    depth_frames[frame_idx] = depth
                if len(depth_frames) == len(original_indices):
                    break
        except FileNotFoundError:
            pass
        
        # Process and cache loaded frames
        for display_idx in indices_to_load:
            metadata = self.frame_metadata[display_idx]
            original_idx = metadata['original_idx']
            
            if original_idx not in rgb_frames:
                continue
            
            rgb, sampled_rgb = rgb_frames[original_idx]
            depth = depth_frames.get(original_idx)
            
            # Compute point cloud if depth available
            pcd, depth_mask = None, None
            if depth is not None:
                camera_type = metadata['camera_type']
                intr = metadata['intrinsics']
                camera_model = camera_type.build_camera_model(intr)
                
                frame_height, frame_width = rgb.shape[:2]
                disp_v, disp_u = torch.meshgrid(
                    torch.arange(frame_height).float()[::spatial_subsample],
                    torch.arange(frame_width).float()[::spatial_subsample],
                    indexing="ij",
                )
                if camera_type == CameraType.PANORAMA:
                    disp_v = disp_v / (frame_height - 1)
                    disp_u = disp_u / (frame_width - 1)
                disp = torch.ones_like(disp_v)
                pts, _, _ = camera_model.iproj_disp(disp, disp_u, disp_v)
                rays = pts[..., :3].numpy()
                if camera_type != CameraType.PANORAMA:
                    rays /= rays[..., 2:3]
                
                pcd = rays * depth.numpy()[::spatial_subsample, ::spatial_subsample, None]
                depth_mask = reliable_depth_mask_range(depth)[::spatial_subsample, ::spatial_subsample].numpy()
            
            self.frame_data_cache[display_idx] = (sampled_rgb, pcd, depth_mask)
        
        logger.info(f"Batch loading complete. Cache size: {len(self.frame_data_cache)}")

    async def _preload_ring_buffer(self, center_frame: int):
        """Preload ring buffer of frames around center frame for smooth playback."""
        if self.preload_task is not None and not self.preload_task.done():
            self.preload_task.cancel()
            try:
                await self.preload_task
            except asyncio.CancelledError:
                pass
        
        async def _do_preload():
            # Calculate initial ring buffer window
            total_frames = len(self.frame_metadata)
            if total_frames == 0:
                return
            
            buffer_start = max(0, center_frame - self.ring_buffer_size // 2)
            buffer_end = min(total_frames, buffer_start + self.ring_buffer_size)
            
            # Adjust if we hit the end
            if buffer_end - buffer_start < self.ring_buffer_size:
                buffer_start = max(0, buffer_end - self.ring_buffer_size)
            
            frames_to_load = list(range(buffer_start, buffer_end))
            
            logger.info(f"Preloading ring buffer: frames {buffer_start}-{buffer_end-1} (center: {center_frame})")
            
            # Load frames in batch
            self._load_frame_data_batch(frames_to_load)
            
            # Initialize/update ring buffer frame list - keep existing frames if possible
            new_buffer_frames = [f for f in frames_to_load if f in self.frame_data_cache]
            
            # Remove frames that are no longer in the new buffer window
            frames_to_remove = [f for f in self.ring_buffer_frames if f not in new_buffer_frames]
            for f in frames_to_remove:
                if f in self.frame_data_cache:
                    del self.frame_data_cache[f]
                if f < len(self.scene_frame_handles):
                    self.scene_frame_handles[f].unload()
            
            self.ring_buffer_frames = new_buffer_frames
            
            logger.info(f"Ring buffer updated with {len(self.ring_buffer_frames)} frames: {self.ring_buffer_frames[0]}-{self.ring_buffer_frames[-1] if self.ring_buffer_frames else 'empty'}")
        
        self.preload_task = asyncio.create_task(_do_preload())
        await self.preload_task

    def _add_frame_to_ring_buffer(self, frame_idx: int):
        """Add a frame to ring buffer, evicting oldest if full."""
        # Skip if already in buffer
        if frame_idx in self.ring_buffer_frames:
            logger.debug(f"Frame {frame_idx} already in buffer, skipping")
            return
        
        # Skip if out of range
        if frame_idx < 0 or frame_idx >= len(self.frame_metadata):
            logger.warning(f"Frame {frame_idx} out of range [0, {len(self.frame_metadata)-1}]")
            return
        
        # If buffer is full, evict oldest frame
        if len(self.ring_buffer_frames) >= self.ring_buffer_size:
            oldest_frame = self.ring_buffer_frames.pop(0)
            if oldest_frame in self.frame_data_cache:
                del self.frame_data_cache[oldest_frame]
            if oldest_frame < len(self.scene_frame_handles):
                self.scene_frame_handles[oldest_frame].unload()
            logger.info(f"Evicted frame {oldest_frame} from ring buffer (buffer full)")
        
        # Load the new frame
        self._load_frame_data_batch([frame_idx])
        
        # Add to buffer if successfully loaded
        if frame_idx in self.frame_data_cache:
            self.ring_buffer_frames.append(frame_idx)
            logger.info(f"Added frame {frame_idx} to ring buffer. Buffer size: {len(self.ring_buffer_frames)}/{self.ring_buffer_size}, Range: [{self.ring_buffer_frames[0] if self.ring_buffer_frames else 'N/A'}, {self.ring_buffer_frames[-1] if self.ring_buffer_frames else 'N/A'}]")
        else:
            logger.error(f"Failed to load frame {frame_idx} into ring buffer!")

    def _rebuild_scene(self):
        """Rebuild scene with lightweight placeholders - actual data loaded on demand."""
        current_artifact = self.global_context().artifacts[self.gui_id.value]
        
        # Clean up existing scene and caches
        for frame_node in self.scene_frame_handles:
            frame_node.unload()
        self.frame_data_cache.clear()
        self.ring_buffer_frames.clear()  # Also clear ring buffer frame list
        self.client.scene.reset()
        self.client.camera.fov = np.deg2rad(self.gui_fov.value)
        self.scene_frame_handles = []
        
        # Create root frame for scene transformation
        self.scene_root_handle = self.client.scene.add_frame(
            "/scene_root",
            wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            position=np.array([
                self.gui_scene_position_x.value,
                self.gui_scene_position_y.value,
                self.gui_scene_position_z.value,
            ]),
        )
        
        # Prepare metadata for all frames
        self._prepare_frame_metadata()
        
        if len(self.frame_metadata) == 0:
            return
        
        # Set up direction from first frame
        first_c2w = self.frame_metadata[0]['c2w']
        first_frame_y = first_c2w[:3, 1]
        self.client.scene.set_up_direction(-first_frame_y)
        
        # Create lightweight frame handles for all frames
        for metadata in self.frame_metadata:
            display_idx = metadata['display_idx']
            c2w = metadata['c2w']
            
            # Create frame handle (lightweight) - as child of scene_root
            frame_handle = self.client.scene.add_frame(
                f"/scene_root/frames/t{display_idx}",
                axes_length=0.05,
                axes_radius=0.005,
                wxyz=tf.SO3.from_matrix(c2w[:3, :3]).wxyz,
                position=c2w[:3, 3],
            )
            
            # Create placeholder frustum (no image yet)
            intr = metadata['intrinsics']
            camera_type = metadata['camera_type']
            pinhole_intr = camera_type.build_camera_model(intr).pinhole().intrinsics
            
            # Estimate frame dimensions from intrinsics
            frame_width = int(pinhole_intr[2].item() * 2)
            frame_height = int(pinhole_intr[3].item() * 2)
            fov = 2 * np.arctan2(frame_height / 2, pinhole_intr[0].item())
            
            frustum_handle = self.client.scene.add_camera_frustum(
                f"/scene_root/frames/t{display_idx}/frustum",
                fov=fov,
                aspect=frame_width / frame_height,
                scale=self.gui_frustum_size.value,
            )
            
            scene_frame = SceneFrameHandle(
                frame_handle=frame_handle,
                frustum_handle=frustum_handle,
                pcd_handle=None,
                frame_idx=display_idx,
                is_loaded=False,
            )
            self.scene_frame_handles.append(scene_frame)
        
        # Preload first frame
        if len(self.scene_frame_handles) > 0:
            # Initialize ring buffer at start
            asyncio.create_task(self._preload_ring_buffer(0))
            # For immediate display, load first frame synchronously
            self._ensure_frame_loaded(0)

    def _ensure_frame_loaded(self, display_idx: int):
        """Ensure a frame's data is loaded using FIFO ring buffer strategy."""
        frame = self.scene_frame_handles[display_idx]
        
        if frame.is_loaded:
            return
        
        # Check if frame is in cache
        if display_idx not in self.frame_data_cache:
            # Not in buffer, load it immediately and add to buffer
            logger.warning(f"Frame {display_idx} not in ring buffer (buffer: {len(self.ring_buffer_frames)} frames), emergency load")
            self._add_frame_to_ring_buffer(display_idx)
            
            if display_idx not in self.frame_data_cache:
                logger.error(f"Failed to load frame {display_idx} even after emergency load!")
                return
        
        sampled_rgb, pcd, depth_mask = self.frame_data_cache[display_idx]
        metadata = self.frame_metadata[display_idx]
        
        # Update frustum with actual image
        frame_height, frame_width = sampled_rgb.shape[:2]
        frame_thumbnail = Image.fromarray(sampled_rgb)
        frame_thumbnail.thumbnail((200, 200), Image.Resampling.LANCZOS)
        
        # Remove old frustum and create new one with image
        frame.frustum_handle.remove()
        
        intr = metadata['intrinsics']
        camera_type = metadata['camera_type']
        pinhole_intr = camera_type.build_camera_model(intr).pinhole().intrinsics
        fov = 2 * np.arctan2(frame_height / 2, pinhole_intr[0].item())
        
        frame.frustum_handle = self.client.scene.add_camera_frustum(
            f"/scene_root/frames/t{display_idx}/frustum",
            fov=fov,
            aspect=frame_width / frame_height,
            scale=self.gui_frustum_size.value,
            image=np.array(frame_thumbnail),
        )
        
        # Add point cloud if available
        if pcd is not None:
            pcd = pcd.reshape(-1, 3)
            rgb = sampled_rgb.reshape(-1, 3)
            if depth_mask is not None:
                depth_mask = depth_mask.reshape(-1)
                pcd = pcd[depth_mask]
                rgb = rgb[depth_mask]
            frame.pcd_handle = self.client.scene.add_point_cloud(
                name=f"/scene_root/frames/t{display_idx}/point_cloud",
                points=pcd,
                colors=rgb,
                point_size=self.gui_point_size.value,
                point_shape="rounded",
            )
        
        frame.is_loaded = True
    
    async def _debounced_update_trajectory(self, current_frame: int):
        """Debounced trajectory update to avoid excessive redraws during dragging."""
        # Cancel previous task if still running
        if self.trajectory_update_task is not None and not self.trajectory_update_task.done():
            self.trajectory_update_task.cancel()
            try:
                await self.trajectory_update_task
            except asyncio.CancelledError:
                pass
        
        async def _do_update():
            # Wait a bit to see if more updates come
            await asyncio.sleep(0.15)  # 150ms debounce
            self._update_camera_trajectory(current_frame)
        
        self.trajectory_update_task = asyncio.create_task(_do_update())
    
    async def _async_ensure_frame_loaded(self, display_idx: int):
        """Async version of _ensure_frame_loaded to avoid blocking UI."""
        frame = self.scene_frame_handles[display_idx]
        
        if frame.is_loaded:
            return
        
        # Check if frame is in cache
        if display_idx not in self.frame_data_cache:
            # Not in buffer, load it asynchronously
            logger.info(f"Frame {display_idx} not in ring buffer, loading asynchronously")
            await self._async_add_frame_to_ring_buffer(display_idx)
            
            if display_idx not in self.frame_data_cache:
                logger.error(f"Failed to load frame {display_idx}!")
                return
        
        # Now load it (this part is quick - just UI updates)
        self._ensure_frame_loaded(display_idx)
    
    async def _async_add_frame_to_ring_buffer(self, frame_idx: int):
        """Async version of _add_frame_to_ring_buffer."""
        # Skip if already in buffer
        if frame_idx in self.ring_buffer_frames:
            return
        
        # Skip if out of range
        if frame_idx < 0 or frame_idx >= len(self.frame_metadata):
            return
        
        # If buffer is full, evict oldest frame
        if len(self.ring_buffer_frames) >= self.ring_buffer_size:
            oldest_frame = self.ring_buffer_frames.pop(0)
            if oldest_frame in self.frame_data_cache:
                del self.frame_data_cache[oldest_frame]
            if oldest_frame < len(self.scene_frame_handles):
                self.scene_frame_handles[oldest_frame].unload()
        
        # Load the new frame asynchronously
        await asyncio.get_event_loop().run_in_executor(None, self._load_frame_data_batch, [frame_idx])
        
        # Add to buffer if successfully loaded
        if frame_idx in self.frame_data_cache:
            self.ring_buffer_frames.append(frame_idx)
            logger.debug(f"Added frame {frame_idx} to ring buffer")
        
        # No need to call _unload_distant_frames, ring buffer handles it
    
    def _unload_distant_frames(self, current_idx: int):
        """Deprecated - ring buffer now handles memory management."""
        pass

    def _incr_timestep(self):
        if self.gui_timestep is not None and len(self.scene_frame_handles) > 0:
            new_value = (self.gui_timestep.value + 1) % len(self.scene_frame_handles)
            logger.debug(f"Increment: {self.gui_timestep.value} -> {new_value}")
            self.gui_timestep.value = new_value

    def _decr_timestep(self):
        if self.gui_timestep is not None and len(self.scene_frame_handles) > 0:
            new_value = (self.gui_timestep.value - 1) % len(self.scene_frame_handles)
            logger.debug(f"Decrement: {self.gui_timestep.value} -> {new_value}")
            self.gui_timestep.value = new_value

    def _rebuild_playback_gui(self):
        current_artifact = self.global_context().artifacts[self.gui_id.value]
        self.gui_name.value = current_artifact.artifact_name
        if self.gui_playback_handle is not None:
            self.gui_playback_handle.remove()
        self.gui_playback_handle = self.client.gui.add_folder("Playback")

        with self.gui_playback_handle:
            self.gui_timestep = self.client.gui.add_slider(
                "Timeline", min=0, max=len(self.scene_frame_handles) - 1, step=1, initial_value=0
            )
            
            # Play/Pause control
            gui_play_pause = self.client.gui.add_button_group("Playback", options=["Play", "Pause"])
            
            @gui_play_pause.on_click
            async def _(_) -> None:
                if gui_play_pause.value == "Play":
                    self.is_playing = True
                    logger.info("Playback started")
                else:
                    self.is_playing = False
                    logger.info("Playback paused")
            
            # Frame control buttons
            gui_prev_button = self.client.gui.add_button("◀ Prev Frame")
            gui_next_button = self.client.gui.add_button("Next Frame ▶")
            
            @gui_prev_button.on_click
            async def _(_) -> None:
                logger.info("Prev button clicked")
                self._decr_timestep()
            
            @gui_next_button.on_click
            async def _(_) -> None:
                logger.info("Next button clicked")
                self._incr_timestep()
            
            self.gui_framerate = self.client.gui.add_slider("FPS", min=0, max=60, step=5.0, initial_value=10)

            self.current_displayed_timestep = self.gui_timestep.value

            @self.gui_timestep.on_update
            async def _(_) -> None:
                current_timestep = self.gui_timestep.value
                prev_timestep = self.current_displayed_timestep
                
                # First, toggle visibility immediately for responsive UI
                with self.client.atomic():
                    self.scene_frame_handles[current_timestep].visible = True
                    self.scene_frame_handles[prev_timestep].visible = False
                self.current_displayed_timestep = current_timestep
                
                # Then load frame data asynchronously (non-blocking)
                # Use create_task to avoid awaiting - let it run in background
                asyncio.create_task(self._async_ensure_frame_loaded(current_timestep))
                
                # Update camera trajectory with debouncing to avoid excessive redraws
                if self.gui_show_trajectory.value:
                    await self._debounced_update_trajectory(current_timestep)
                
                # Preload next frame into ring buffer (async, don't await)
                next_frame = current_timestep + 1
                if next_frame < len(self.scene_frame_handles):
                    asyncio.create_task(self._async_add_frame_to_ring_buffer(next_frame))

    def cleanup(self):
        logger.info(f"Client {self.client.client_id} disconnected")

    @classmethod
    def global_context(cls) -> GlobalContext:
        global _global_context
        assert _global_context is not None, "Global context not initialized"
        return _global_context


def get_host_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            # Doesn't even have to be reachable
            s.connect(("8.8.8.8", 1))
            internal_ip = s.getsockname()[0]
        except Exception:
            internal_ip = "127.0.0.1"
    return internal_ip


def run_viser(base_path: Path, port: int = 20540):
    # Get list of artifacts.
    logger.info(f"Loading artifacts from {base_path}")
    artifacts: list[ArtifactPath] = list(ArtifactPath.glob_artifacts(base_path, use_video=True))
    if len(artifacts) == 0:
        logger.error("No artifacts found. Exiting.")
        return

    global _global_context
    _global_context = GlobalContext(artifacts=sorted(artifacts, key=lambda x: x.artifact_name))

    server = viser.ViserServer(host=get_host_ip(), port=port, verbose=False)
    client_closures: dict[int, ClientClosures] = {}

    @server.on_client_connect
    async def _(client: viser.ClientHandle):
        client_closures[client.client_id] = ClientClosures(client)

    @server.on_client_disconnect
    async def _(client: viser.ClientHandle):
        # wait synchronously in this function for task to be finished.
        await client_closures[client.client_id].stop()
        del client_closures[client.client_id]

    while True:
        try:
            time.sleep(10.0)
        except KeyboardInterrupt:
            logger.info("Ctrl+C detected. Shutting down server...")
            break
    server.stop()


def main():
    parser = argparse.ArgumentParser(description="3D Visualizer")
    parser.add_argument("base_path", type=Path, help="Base path for the visualizer")
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=20540,
        help="Port number for the viser server.",
    )
    args = parser.parse_args()

    run_viser(args.base_path, args.port)


if __name__ == "__main__":
    main()
