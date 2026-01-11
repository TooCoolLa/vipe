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

from pathlib import Path

import click
import hydra

from vipe import get_config_path, make_pipeline
from vipe.streams.base import ProcessedVideoStream
from vipe.streams.raw_mp4_stream import RawMp4Stream
from vipe.streams.frame_dir_stream import FrameDirStream
from vipe.utils.logging import configure_logging
from vipe.utils.viser import run_viser
from vipe.cli.merge import align_submaps


@click.command()
@click.argument("video", type=click.Path(exists=True, path_type=Path), required=False)
@click.option(
    "--image-dir",
    type=click.Path(exists=True, path_type=Path),
    help="Directory containing image frames",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output directory (default: current directory)",
    default=Path.cwd() / "vipe_results",
)
@click.option("--pipeline", "-p", default="default", help="Pipeline configuration to use (default: 'default')")
@click.option("--visualize", "-v", is_flag=True, help="Enable visualization of intermediate results")
def infer(video: Path, image_dir: Path, output: Path, pipeline: str, visualize: bool):
    """Run inference on a video file or directory of images."""

    logger = configure_logging()

    # Validate that exactly one input source is provided
    if not video and not image_dir:
        click.echo("Error: Must provide either a video file or --image-dir", err=True)
        raise click.Abort()
    
    if video and image_dir:
        click.echo("Error: Cannot provide both video file and --image-dir", err=True)
        raise click.Abort()

    overrides = [f"pipeline={pipeline}", f"pipeline.output.path={output}", "pipeline.output.save_artifacts=true"]
    if visualize:
        overrides.append("pipeline.output.save_viz=true")
        overrides.append("pipeline.slam.visualize=true")
    else:
        overrides.append("pipeline.output.save_viz=false")

    # Set up stream configuration based on input type
    if image_dir:
        overrides.extend([
            "streams=frame_dir_stream",
            f"streams.base_path={image_dir}"
        ])
        input_path = image_dir
        input_desc = f"image directory {image_dir}"
    else:
        input_path = video
        input_desc = f"video {video}"

    with hydra.initialize_config_dir(config_dir=str(get_config_path()), version_base=None):
        args = hydra.compose("default", overrides=overrides)

    logger.info(f"Processing {input_desc}...")
    vipe_pipeline = make_pipeline(args.pipeline)

    if image_dir:
        # Use frame directory stream
        video_stream = ProcessedVideoStream(FrameDirStream(image_dir), []).cache(desc="Reading image frames")
    else:
        # Some input videos can be malformed, so we need to cache the videos to obtain correct number of frames.
        video_stream = ProcessedVideoStream(RawMp4Stream(video), []).cache(desc="Reading video stream")

    vipe_pipeline.run(video_stream)
    logger.info("Finished")


@click.command()
@click.argument("data_path", type=click.Path(exists=True, path_type=Path), default=Path.cwd() / "vipe_results")
@click.option("--port", "-p", default=20540, type=int, help="Port for the visualization server (default: 20540)")
def visualize(data_path: Path, port: int):
    run_viser(data_path, port)


@click.command()
@click.argument("root_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--overlap", "-n", default=10, type=int, help="Number of overlapping frames between submaps (default: 10)")
@click.option("--with-scale/--no-scale", default=False, help="Enable scale estimation in Umeyama alignment (default: False)")
@click.option("--use-ransac/--no-ransac", default=False, help="Use RANSAC for robust outlier rejection (default: False)")
@click.option("--ransac-threshold", default=1.0, type=float, help="RANSAC inlier threshold in meters (default: 1.0)")
@click.option("--ransac-iterations", default=1000, type=int, help="Number of RANSAC iterations (default: 1000)")
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output JSON file (default: <root_dir>/camera_poses.json)")
def merge(root_dir: Path, overlap: int, with_scale: bool, use_ransac: bool, ransac_threshold: float, ransac_iterations: int, output: Path):
    """Align multiple submaps into a unified global coordinate frame using Umeyama algorithm."""
    logger = configure_logging()
    
    if output is None:
        output = root_dir / "camera_poses.json"
    
    logger.info(f"Merging submaps from {root_dir}")
    logger.info(f"Overlap frames: {overlap}, Scale estimation: {with_scale}")
    logger.info(f"Use RANSAC: {use_ransac}")
    if use_ransac:
        logger.info(f"  RANSAC threshold: {ransac_threshold}m, iterations: {ransac_iterations}")
    logger.info(f"Output: {output}")
    
    try:
        align_submaps(
            root_dir, 
            overlap_frames=overlap, 
            with_scale=with_scale,
            use_ransac=use_ransac,
            ransac_threshold=ransac_threshold,
            ransac_iterations=ransac_iterations,
            output_file=output
        )
        logger.info("✓ Merge completed successfully")
    except Exception as e:
        logger.error(f"✗ Merge failed: {e}")
        raise click.Abort()


@click.group()
@click.version_option()
def main():
    """NVIDIA Video Pose Engine (ViPE) CLI"""
    pass


# Add subcommands
main.add_command(infer)
main.add_command(visualize)
main.add_command(merge)


if __name__ == "__main__":
    main()
