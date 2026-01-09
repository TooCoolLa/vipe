#!/bin/bash
# DAV3 Pipeline Model Download Script
# 此脚本用于预先下载VIPE DAV3 pipeline运行所需的所有大模型文件
# 模型将被分类保存到 ./ckpt 文件夹中

set -e  # 遇到错误时退出

# 创建主目录结构
echo "创建模型存储目录..."
mkdir -p ./ckpt/{depth,slam,track_anything,geocalib}

# ========== Depth Anything V3 (DAV3) 模型 ==========
echo ""
echo "=========================================="
echo "1. 下载 Depth Anything V3 模型"
echo "=========================================="
mkdir -p ./ckpt/depth/dav3

# DAV3 主模型 - 这是最重要的模型
# 需要使用 huggingface-cli 下载
if command -v huggingface-cli &> /dev/null; then
    echo "使用 huggingface-cli 下载 DAV3 模型..."
    huggingface-cli download depth-anything/DA3METRIC-LARGE \
        --local-dir ./ckpt/depth/dav3/DA3METRIC-LARGE \
        --local-dir-use-symlinks False
else
    echo "警告: huggingface-cli 未安装，请先安装: pip install huggingface_hub[cli]"
    echo "或使用以下Python命令手动下载:"
    echo "from huggingface_hub import snapshot_download; snapshot_download('depth-anything/DA3METRIC-LARGE', local_dir='./ckpt/depth/dav3/DA3METRIC-LARGE')"
fi

# ========== Prior Depth Anything (PriorDA) 模型 ==========
echo ""
echo "=========================================="
echo "2. 下载 Prior Depth Anything 模型"
echo "=========================================="
mkdir -p ./ckpt/depth/priorda

# PriorDA使用的是DAV2模型和微调模型
# 冻结模型 (Frozen MDE)
echo "下载 Frozen MDE 模型 (vitb)..."
wget -c -P ./ckpt/depth/priorda \
    https://huggingface.co/Rain729/Prior-Depth-Anything/resolve/main/depth_anything_v2_vitb.pth

# 条件模型 (Conditioned MDE)
echo "下载 Conditioned MDE 模型 (vitb)..."
wget -c -P ./ckpt/depth/priorda \
    https://huggingface.co/Rain729/Prior-Depth-Anything/resolve/main/depth_anything_v2_vitb.pth

# Prior-DA 微调模型
echo "下载 Prior-DA 微调模型 (vitb)..."
wget -c -P ./ckpt/depth/priorda \
    https://huggingface.co/Rain729/Prior-Depth-Anything/resolve/main/prior_depth_anything_vitb.pth

# ========== DROID-SLAM 模型 ==========
echo ""
echo "=========================================="
echo "3. 下载 DROID-SLAM 模型"
echo "=========================================="
mkdir -p ./ckpt/slam/droid

echo "使用 gdown 下载 DROID-SLAM 权重..."
if command -v gdown &> /dev/null; then
    gdown "https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view" \
        -O ./ckpt/slam/droid/droid.pth --fuzzy
else
    echo "警告: gdown 未安装，请先安装: pip install gdown"
    echo "或手动下载: https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view"
fi

# ========== SuperPoint 模型 ==========
echo ""
echo "=========================================="
echo "4. 下载 SuperPoint 模型"
echo "=========================================="
mkdir -p ./ckpt/slam/superpoint

echo "下载 SuperPoint 权重..."
wget -c -P ./ckpt/slam/superpoint \
    https://github.com/rpautrat/SuperPoint/raw/refs/heads/master/weights/superpoint_v6_from_tf.pth

# ========== Track Anything 相关模型 ==========
echo ""
echo "=========================================="
echo "5. 下载 Track Anything 相关模型"
echo "=========================================="
mkdir -p ./ckpt/track_anything/{sam,aot,grounding_dino}

# Segment Anything Model (SAM)
echo "下载 SAM ViT-B 模型..."
wget -c -P ./ckpt/track_anything/sam \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# AOT (Associating Objects with Transformers)
echo "下载 AOT 模型..."
if command -v gdown &> /dev/null; then
    gdown "https://drive.google.com/file/d/1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ/view" \
        -O ./ckpt/track_anything/aot/R50_DeAOTL_PRE_YTB_DAV.pth --fuzzy
else
    echo "警告: gdown 未安装，跳过 AOT 模型下载"
fi

# Grounding DINO
echo "下载 Grounding DINO 模型..."
wget -c -P ./ckpt/track_anything/grounding_dino \
    https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth

# ========== GeoCalib 模型 ==========
echo ""
echo "=========================================="
echo "6. 下载 GeoCalib 模型"
echo "=========================================="
mkdir -p ./ckpt/geocalib

# GeoCalib 使用预训练的权重
# 需要从 GitHub 或 Hugging Face 下载
if command -v huggingface-cli &> /dev/null; then
    echo "使用 huggingface-cli 下载 GeoCalib 模型..."
    huggingface-cli download cvg/GeoCalib \
        --local-dir ./ckpt/geocalib \
        --local-dir-use-symlinks False || echo "注意: GeoCalib 模型可能需要手动下载"
else
    echo "注意: GeoCalib 模型可能需要从 https://github.com/cvg/GeoCalib 手动下载"
fi

# ========== 可选: Metric3D 模型 (如果需要) ==========
echo ""
echo "=========================================="
echo "7. [可选] 下载 Metric3D 模型"
echo "=========================================="
mkdir -p ./ckpt/depth/metric3d

read -p "是否下载 Metric3D 模型? (y/N): " download_metric3d
if [[ $download_metric3d =~ ^[Yy]$ ]]; then
    echo "下载 Metric3D ViT-Large 模型..."
    wget -c -P ./ckpt/depth/metric3d \
        https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_large_800k.pth
    
    echo "下载 Metric3D ViT-Small 模型..."
    wget -c -P ./ckpt/depth/metric3d \
        https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_small_800k.pth
else
    echo "跳过 Metric3D 模型下载"
fi

# ========== 可选: Video Depth Anything 模型 (如果需要) ==========
echo ""
echo "=========================================="
echo "8. [可选] 下载 Video Depth Anything 模型"
echo "=========================================="
mkdir -p ./ckpt/depth/video_depth_anything

read -p "是否下载 Video Depth Anything 模型? (y/N): " download_vda
if [[ $download_vda =~ ^[Yy]$ ]]; then
    echo "下载 Video Depth Anything ViT-L 模型..."
    wget -c -P ./ckpt/depth/video_depth_anything \
        https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth
else
    echo "跳过 Video Depth Anything 模型下载"
fi

# ========== 完成 ==========
echo ""
echo "=========================================="
echo "模型下载完成!"
echo "=========================================="
echo ""
echo "模型文件已保存到以下位置:"
echo "  - DAV3 模型:           ./ckpt/depth/dav3/"
echo "  - PriorDA 模型:        ./ckpt/depth/priorda/"
echo "  - DROID-SLAM 模型:     ./ckpt/slam/droid/"
echo "  - SuperPoint 模型:     ./ckpt/slam/superpoint/"
echo "  - SAM 模型:            ./ckpt/track_anything/sam/"
echo "  - AOT 模型:            ./ckpt/track_anything/aot/"
echo "  - Grounding DINO 模型: ./ckpt/track_anything/grounding_dino/"
echo "  - GeoCalib 模型:       ./ckpt/geocalib/"
echo ""
echo "注意事项:"
echo "1. 某些模型需要安装额外的工具 (huggingface-cli, gdown)"
echo "2. 如果下载失败，请检查网络连接或手动下载"
echo "3. DAV3 需要先安装: pip install --no-build-isolation -e .[dav3]"
echo "4. 确保有足够的磁盘空间 (约 20-30 GB)"
echo ""
echo "使用方法:"
echo "在运行 VIPE 时，模型会自动从相应目录加载"
echo "如果需要手动指定路径，请参考相关配置文件"
