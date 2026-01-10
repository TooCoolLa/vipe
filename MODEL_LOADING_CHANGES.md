# 模型加载路径修改说明

## 修改内容

已修改以下文件，使所有模型优先从本地 `./ckpt` 目录加载，如果本地文件不存在才会从网络下载：

### 1. Depth Anything V3 (DAV3)
**文件**: `vipe/priors/depth/dav3.py`

**修改**:
- 优先查找 `./ckpt/depth/dav3/DA3METRIC-LARGE/`
- 如果不存在，则从 Hugging Face Hub 下载

```python
# 新的加载逻辑
local_model_path = Path("./ckpt/depth/dav3/DA3METRIC-LARGE")
if local_model_path.exists():
    self.model = DepthAnything3.from_pretrained(str(local_model_path))
else:
    self.model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
```

### 2. Prior Depth Anything (PriorDA)
**文件**: `vipe/priors/depth/priorda/priorda.py`

**修改**:
- Frozen MDE: 优先查找 `./ckpt/depth/priorda/depth_anything_v2_vitb.pth`
- Conditioned MDE: 优先查找 `./ckpt/depth/priorda/depth_anything_v2_vitb.pth`
- 微调模型: 优先查找 `./ckpt/depth/priorda/prior_depth_anything_vitb.pth`
- 如果不存在，则从 Hugging Face Hub 下载

### 3. DROID-SLAM / Pi3X
**文件**: `vipe/slam/networks/droid_net.py` 和 `vipe/slam/networks/pi3x_net.py`

**修改**:
- droid模块已被替换为pi3x模块（来自 https://github.com/yyfz/Pi3）
- 接口保持不变，输入输出兼容
- 优先查找 `./ckpt/slam/pi3x/pi3x.pth` 或 `./ckpt/slam/droid/droid.pth`
- 如果不存在，则从 Google Drive 下载原始 DROID 权重

### 4. SuperPoint
**文件**: `vipe/slam/networks/superpoint.py`

**修改**:
- 优先查找 `./ckpt/slam/superpoint/superpoint_v6_from_tf.pth`
- 如果不存在，则从 GitHub 下载

### 5. Track Anything - SAM
**文件**: `vipe/priors/track_anything/__init__.py`

**修改**:
- 优先查找 `./ckpt/track_anything/sam/sam_vit_b_01ec64.pth`
- 如果不存在，则从 Facebook AI 下载

### 6. Track Anything - AOT
**文件**: `vipe/priors/track_anything/__init__.py`

**修改**:
- 优先查找 `./ckpt/track_anything/aot/R50_DeAOTL_PRE_YTB_DAV.pth`
- 如果不存在，则从 Google Drive 下载

### 7. Track Anything - Grounding DINO
**文件**: `vipe/priors/track_anything/detector.py`

**修改**:
- 优先查找 `./ckpt/track_anything/grounding_dino/groundingdino_swint_ogc.pth`
- 如果不存在，则从 Hugging Face Hub 下载

## 使用方式

### 方式1: 使用下载脚本预先下载所有模型（推荐）

```bash
# 运行下载脚本
python download_models.py

# 然后正常运行 VIPE
python run.py streams=your_stream pipeline=dav3
```

### 方式2: 首次运行时自动下载

如果 `./ckpt` 目录不存在模型文件，程序会自动从网络下载到默认位置（torch.hub 缓存目录），但不会保存到 `./ckpt` 目录。

### 方式3: 手动放置模型文件

如果你已经有这些模型文件，可以手动放置到相应目录：

```
./ckpt/
├── depth/
│   ├── dav3/
│   │   └── DA3METRIC-LARGE/
│   └── priorda/
│       ├── depth_anything_v2_vitb.pth
│       └── prior_depth_anything_vitb.pth
├── slam/
│   ├── droid/
│   │   └── droid.pth
│   ├── pi3x/
│   │   └── pi3x.pth
│   └── superpoint/
│       └── superpoint_v6_from_tf.pth
└── track_anything/
    ├── sam/
    │   └── sam_vit_b_01ec64.pth
    ├── aot/
    │   └── R50_DeAOTL_PRE_YTB_DAV.pth
    └── grounding_dino/
        └── groundingdino_swint_ogc.pth
```

## 加载日志

运行时，你会看到类似以下的日志信息：

```
Loading DAV3 model from local path: ./ckpt/depth/dav3/DA3METRIC-LARGE
Loading Frozen MDE model from local path: ./ckpt/depth/priorda/depth_anything_v2_vitb.pth
Loading DROID-SLAM model from local path: ./ckpt/slam/droid/droid.pth
Loading SuperPoint model from local path: ./ckpt/slam/superpoint/superpoint_v6_from_tf.pth
Loading SAM model from local path: ./ckpt/track_anything/sam/sam_vit_b_01ec64.pth
Loading AOT model from local path: ./ckpt/track_anything/aot/R50_DeAOTL_PRE_YTB_DAV.pth
Loading Grounding DINO model from local path: ./ckpt/track_anything/grounding_dino/groundingdino_swint_ogc.pth
```

如果模型不在本地，则会显示：

```
Downloading DAV3 model from Hugging Face Hub...
Downloading DROID-SLAM model from Google Drive...
```

## 优点

1. **离线运行**: 预先下载后可以完全离线运行
2. **节省时间**: 避免每次运行时重新下载
3. **可控性**: 明确知道模型文件的位置
4. **版本管理**: 可以手动管理模型版本
5. **网络优化**: 可以在网络条件好的时候批量下载

## 注意事项

1. 确保 `./ckpt` 目录结构正确
2. 模型文件名必须与代码中指定的名称完全匹配
3. DAV3 模型是目录格式，不是单个文件
4. 所有路径都是相对于项目根目录的相对路径
5. 如果模型文件损坏，删除后重新下载

## 回退到原始行为

如果你想使用原始的下载行为（每次从网络加载），只需：

1. 删除或重命名 `./ckpt` 目录
2. 或者修改相应的代码文件，移除本地路径检查逻辑

## 测试

运行以下命令测试模型加载是否正常：

```bash
# 测试基本运行
python run.py --help

# 测试 DAV3 pipeline
python run.py streams=your_video pipeline=dav3

# 查看模型加载日志
python run.py streams=your_video pipeline=dav3 2>&1 | grep "Loading"
```
