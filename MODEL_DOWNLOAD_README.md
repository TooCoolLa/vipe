# VIPE DAV3 Pipeline 模型下载指南

本文档说明如何下载VIPE DAV3 pipeline所需的所有预训练模型。

## 快速开始

### 方法1: 使用 Python 脚本（推荐）

```bash
# 1. 安装必要的依赖
pip install huggingface_hub gdown wget

# 2. 运行下载脚本
python download_models.py
```

### 方法2: 使用 Shell 脚本（Linux/Mac）

```bash
# 1. 给脚本添加执行权限
chmod +x download_models.sh

# 2. 运行脚本
./download_models.sh
```

### 方法3: 使用 Git Bash（Windows）

```bash
# 在 Git Bash 中运行
bash download_models.sh
```

## 模型列表

### 核心模型（必需）

1. **Depth Anything V3 (DAV3)** - 主深度估计模型
   - 仓库: `depth-anything/DA3METRIC-LARGE`
   - 大小: ~5 GB
   - 位置: `./ckpt/depth/dav3/`

2. **Prior Depth Anything (PriorDA)** - 深度细化模型
   - 文件:
     - `depth_anything_v2_vitb.pth` (Frozen MDE)
     - `prior_depth_anything_vitb.pth` (微调模型)
   - 位置: `./ckpt/depth/priorda/`

3. **DROID-SLAM** - SLAM系统
   - 文件: `droid.pth`
   - 大小: ~300 MB
   - 位置: `./ckpt/slam/droid/`

4. **SuperPoint** - 特征点检测
   - 文件: `superpoint_v6_from_tf.pth`
   - 位置: `./ckpt/slam/superpoint/`

5. **Track Anything** - 目标跟踪
   - SAM: `sam_vit_b_01ec64.pth`
   - AOT: `R50_DeAOTL_PRE_YTB_DAV.pth`
   - Grounding DINO: `groundingdino_swint_ogc.pth`
   - 位置: `./ckpt/track_anything/`

6. **GeoCalib** - 相机标定
   - 仓库: `cvg/GeoCalib`
   - 位置: `./ckpt/geocalib/`

### 可选模型

1. **Metric3D** - 备用深度估计
   - 位置: `./ckpt/depth/metric3d/`

2. **Video Depth Anything** - 视频深度估计
   - 位置: `./ckpt/depth/video_depth_anything/`

## 目录结构

下载完成后，目录结构如下：

```
./ckpt/
├── depth/
│   ├── dav3/
│   │   └── DA3METRIC-LARGE/
│   ├── priorda/
│   │   ├── depth_anything_v2_vitb.pth
│   │   └── prior_depth_anything_vitb.pth
│   ├── metric3d/          # 可选
│   └── video_depth_anything/  # 可选
├── slam/
│   ├── droid/
│   │   └── droid.pth
│   └── superpoint/
│       └── superpoint_v6_from_tf.pth
├── track_anything/
│   ├── sam/
│   │   └── sam_vit_b_01ec64.pth
│   ├── aot/
│   │   └── R50_DeAOTL_PRE_YTB_DAV.pth
│   └── grounding_dino/
│       └── groundingdino_swint_ogc.pth
└── geocalib/
```

## 系统要求

- **磁盘空间**: 至少 20-30 GB 可用空间
- **网络**: 稳定的互联网连接
- **Python**: 3.8 或更高版本（如果使用 Python 脚本）

### Python 依赖

```bash
pip install huggingface_hub gdown wget
```

### 系统工具（Shell脚本需要）

- `wget` - 用于从 HTTP/HTTPS 下载
- `gdown` - 用于从 Google Drive 下载
- `huggingface-cli` - 用于从 Hugging Face Hub 下载

在 Ubuntu/Debian 上安装:
```bash
sudo apt-get install wget
pip install gdown huggingface_hub[cli]
```

## 使用方式

### 在 VIPE 中自动使用

默认情况下，VIPE 会从 PyTorch Hub 缓存目录加载模型。如果你想使用本地下载的模型，需要修改相应的代码或设置环境变量。

### 手动指定模型路径

你可以在代码中修改模型路径，例如：

```python
# 在 vipe/priors/depth/dav3.py 中
model = DepthAnything3.from_pretrained("./ckpt/depth/dav3/DA3METRIC-LARGE")
```

或者通过环境变量：

```bash
export TORCH_HOME=./ckpt
export HF_HOME=./ckpt
```

## 故障排除

### 下载速度慢

如果在中国大陆地区下载速度慢，可以：

1. 设置 Hugging Face 镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

2. 使用代理：
```bash
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
```

### Google Drive 下载失败

如果 `gdown` 下载失败，可以：

1. 手动访问链接并下载
2. 使用浏览器下载后放到对应目录

### Hugging Face 认证问题

某些模型可能需要 Hugging Face 账号认证：

```bash
huggingface-cli login
```

### 模型已存在

脚本会自动跳过已下载的文件，如果需要重新下载，请先删除对应文件。

## 验证下载

运行以下命令检查下载的文件：

```bash
# Linux/Mac
find ./ckpt -name "*.pth" -o -name "*.safetensors"

# Windows PowerShell
Get-ChildItem -Path ./ckpt -Recurse -Include *.pth,*.safetensors
```

## 常见问题

**Q: 是否必须下载所有模型？**

A: 取决于你的使用场景。对于基本的 DAV3 pipeline，核心模型是必需的。可选模型用于特定功能或备用方案。

**Q: 可以使用其他大小的模型（如 vits, vitl）吗？**

A: 可以。修改下载脚本中的模型名称即可。例如将 `vitb` 改为 `vitl` 以使用更大的模型。

**Q: 模型文件存储在哪里？**

A: 默认存储在项目根目录的 `./ckpt` 文件夹中。你可以修改脚本以使用不同的位置。

**Q: 如何更新模型？**

A: 删除旧的模型文件，重新运行下载脚本即可。

## 更多信息

- VIPE 项目: [GitHub](https://github.com/your-repo/vipe)
- Depth Anything V3: https://github.com/ByteDance-Seed/Depth-Anything-3
- Prior Depth Anything: https://github.com/SpatialVision/Prior-Depth-Anything
- DROID-SLAM: https://github.com/princeton-vl/DROID-SLAM
- GeoCalib: https://github.com/cvg/GeoCalib

## 许可证

各个模型遵循其原始仓库的许可证。请查看 THIRD_PARTY_LICENSES.md 了解详情。
