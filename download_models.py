#!/usr/bin/env python3
"""
DAV3 Pipeline Model Download Script (Python版本)
此脚本用于预先下载VIPE DAV3 pipeline运行所需的所有大模型文件
模型将被分类保存到 ./ckpt 文件夹中
"""

import os
import sys
from pathlib import Path
from urllib.request import urlretrieve


def download_file(url: str, dest: Path, desc: str = ""):
    """下载文件并显示进度"""
    print(f"  下载: {desc or url}")
    print(f"  目标: {dest}")
    
    if dest.exists():
        print(f"  ✓ 文件已存在，跳过")
        return True
    
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        def report_hook(count, block_size, total_size):
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                sys.stdout.write(f"\r  进度: {percent}%")
                sys.stdout.flush()
        
        urlretrieve(url, dest, reporthook=report_hook)
        print("\n  ✓ 下载完成")
        return True
    except Exception as e:
        print(f"\n  ✗ 下载失败: {e}")
        return False


def download_with_huggingface_hub(repo_id: str, local_dir: Path, desc: str = ""):
    """使用 huggingface_hub 下载整个仓库"""
    print(f"  下载: {desc or repo_id}")
    print(f"  目标: {local_dir}")
    
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False
        )
        print(f"  ✓ 下载完成")
        return True
    except ImportError:
        print(f"  ✗ 未安装 huggingface_hub，请运行: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"  ✗ 下载失败: {e}")
        return False


def download_with_gdown(url: str, dest: Path, desc: str = ""):
    """使用 gdown 从 Google Drive 下载"""
    print(f"  下载: {desc or url}")
    print(f"  目标: {dest}")
    
    if dest.exists():
        print(f"  ✓ 文件已存在，跳过")
        return True
    
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        import gdown
        gdown.download(url, str(dest), quiet=False, fuzzy=True)
        print(f"  ✓ 下载完成")
        return True
    except ImportError:
        print(f"  ✗ 未安装 gdown，请运行: pip install gdown")
        return False
    except Exception as e:
        print(f"  ✗ 下载失败: {e}")
        return False


def main():
    print("=" * 60)
    print("VIPE DAV3 Pipeline 模型下载脚本")
    print("=" * 60)
    print()
    
    # 基础目录
    ckpt_dir = Path("./ckpt")
    
    # 创建目录结构
    print("创建模型存储目录...")
    (ckpt_dir / "depth" / "dav3").mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "depth" / "priorda").mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "depth" / "metric3d").mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "depth" / "video_depth_anything").mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "slam" / "droid").mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "slam" / "superpoint").mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "track_anything" / "sam").mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "track_anything" / "aot").mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "track_anything" / "grounding_dino").mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "geocalib").mkdir(parents=True, exist_ok=True)
    print("✓ 目录创建完成\n")
    
    results = []
    
    # ========== 1. Depth Anything V3 (DAV3) ==========
    print("=" * 60)
    print("1. Depth Anything V3 模型")
    print("=" * 60)
    result = download_with_huggingface_hub(
        "depth-anything/DA3METRIC-LARGE",
        ckpt_dir / "depth" / "dav3" / "DA3METRIC-LARGE",
        "DAV3 主模型 (最重要)"
    )
    results.append(("DAV3 主模型", result))
    print()
    
    # ========== 2. Prior Depth Anything (PriorDA) ==========
    print("=" * 60)
    print("2. Prior Depth Anything 模型")
    print("=" * 60)
    
    # Frozen MDE
    result = download_file(
        "https://huggingface.co/Rain729/Prior-Depth-Anything/resolve/main/depth_anything_v2_vitb.pth",
        ckpt_dir / "depth" / "priorda" / "depth_anything_v2_vitb.pth",
        "Frozen MDE 模型 (vitb)"
    )
    results.append(("PriorDA Frozen MDE", result))
    print()
    
    # Prior-DA 微调模型
    result = download_file(
        "https://huggingface.co/Rain729/Prior-Depth-Anything/resolve/main/prior_depth_anything_vitb.pth",
        ckpt_dir / "depth" / "priorda" / "prior_depth_anything_vitb.pth",
        "Prior-DA 微调模型 (vitb)"
    )
    results.append(("PriorDA 微调模型", result))
    print()
    
    # ========== 3. DROID-SLAM ==========
    print("=" * 60)
    print("3. DROID-SLAM 模型")
    print("=" * 60)
    result = download_with_gdown(
        "https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view",
        ckpt_dir / "slam" / "droid" / "droid.pth",
        "DROID-SLAM 权重"
    )
    results.append(("DROID-SLAM", result))
    print()
    
    # ========== 4. SuperPoint ==========
    print("=" * 60)
    print("4. SuperPoint 模型")
    print("=" * 60)
    result = download_file(
        "https://github.com/rpautrat/SuperPoint/raw/refs/heads/master/weights/superpoint_v6_from_tf.pth",
        ckpt_dir / "slam" / "superpoint" / "superpoint_v6_from_tf.pth",
        "SuperPoint 权重"
    )
    results.append(("SuperPoint", result))
    print()
    
    # ========== 5. Track Anything 相关模型 ==========
    print("=" * 60)
    print("5. Track Anything 相关模型")
    print("=" * 60)
    
    # SAM
    result = download_file(
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        ckpt_dir / "track_anything" / "sam" / "sam_vit_b_01ec64.pth",
        "SAM ViT-B 模型"
    )
    results.append(("SAM", result))
    print()
    
    # AOT
    result = download_with_gdown(
        "https://drive.google.com/file/d/1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ/view",
        ckpt_dir / "track_anything" / "aot" / "R50_DeAOTL_PRE_YTB_DAV.pth",
        "AOT 模型"
    )
    results.append(("AOT", result))
    print()
    
    # Grounding DINO
    result = download_file(
        "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
        ckpt_dir / "track_anything" / "grounding_dino" / "groundingdino_swint_ogc.pth",
        "Grounding DINO 模型"
    )
    results.append(("Grounding DINO", result))
    print()
    
    # ========== 6. GeoCalib ==========
    print("=" * 60)
    print("6. GeoCalib 模型")
    print("=" * 60)
    result = download_with_huggingface_hub(
        "cvg/GeoCalib",
        ckpt_dir / "geocalib",
        "GeoCalib 模型"
    )
    results.append(("GeoCalib", result))
    print()
    
    # ========== 可选模型 ==========
    print("=" * 60)
    print("可选模型下载")
    print("=" * 60)
    
    # Metric3D
    response = input("是否下载 Metric3D 模型? (y/N): ").strip().lower()
    if response in ['y', 'yes']:
        print("\n下载 Metric3D 模型...")
        result = download_file(
            "https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_large_800k.pth",
            ckpt_dir / "depth" / "metric3d" / "metric_depth_vit_large_800k.pth",
            "Metric3D ViT-Large"
        )
        results.append(("Metric3D", result))
        print()
    
    # Video Depth Anything
    response = input("是否下载 Video Depth Anything 模型? (y/N): ").strip().lower()
    if response in ['y', 'yes']:
        print("\n下载 Video Depth Anything 模型...")
        result = download_file(
            "https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth",
            ckpt_dir / "depth" / "video_depth_anything" / "video_depth_anything_vitl.pth",
            "Video Depth Anything ViT-L"
        )
        results.append(("Video Depth Anything", result))
        print()
    
    # ========== 总结 ==========
    print("\n" + "=" * 60)
    print("下载总结")
    print("=" * 60)
    
    success_count = sum(1 for _, success in results if success)
    total_count = len(results)
    
    print(f"\n成功: {success_count}/{total_count}")
    print("\n各模型下载状态:")
    for name, success in results:
        status = "✓ 成功" if success else "✗ 失败"
        print(f"  {status} - {name}")
    
    print("\n" + "=" * 60)
    print("模型文件位置:")
    print("=" * 60)
    print(f"""
  - DAV3 模型:           {ckpt_dir / 'depth' / 'dav3'}
  - PriorDA 模型:        {ckpt_dir / 'depth' / 'priorda'}
  - DROID-SLAM 模型:     {ckpt_dir / 'slam' / 'droid'}
  - SuperPoint 模型:     {ckpt_dir / 'slam' / 'superpoint'}
  - SAM 模型:            {ckpt_dir / 'track_anything' / 'sam'}
  - AOT 模型:            {ckpt_dir / 'track_anything' / 'aot'}
  - Grounding DINO 模型: {ckpt_dir / 'track_anything' / 'grounding_dino'}
  - GeoCalib 模型:       {ckpt_dir / 'geocalib'}
    """)
    
    print("=" * 60)
    print("注意事项:")
    print("=" * 60)
    print("""
1. 某些模型需要安装额外的Python包:
   - pip install huggingface_hub
   - pip install gdown

2. DAV3 需要先安装深度估计包:
   - pip install --no-build-isolation -e .[dav3]

3. 确保有足够的磁盘空间 (约 20-30 GB)

4. 如果某些下载失败，可以重新运行此脚本
   已下载的文件会被跳过

5. 在中国大陆地区，可能需要设置代理或使用镜像源
    """)
    
    if success_count < total_count:
        print("\n警告: 部分模型下载失败，请检查错误信息并重试")
        return 1
    else:
        print("\n✓ 所有模型下载成功!")
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n用户中断下载")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
