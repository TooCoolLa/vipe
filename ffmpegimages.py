import os
import sys
import shutil
import subprocess
import argparse
import glob

def run_ffmpeg_extraction(video_path, temp_dir):
    """
    使用 ffmpeg 将视频解压为图片序列到临时目录
    """
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # 图片文件名依然保持补零 (000001.jpg)，这是为了在该文件夹内排序方便
    output_pattern = os.path.join(temp_dir, "%06d.jpg")
    
    print(f"[1/3] 正在提取全量帧到临时目录: {temp_dir} ...")
    
    cmd = [
        "ffmpeg", 
        "-i", video_path, 
        "-start_number", "0", 
        "-q:v", "2", 
        "-r","30",
        output_pattern,
        "-loglevel", "error", 
        "-stats"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("Error: FFmpeg 执行失败，请检查视频文件路径或是否安装了 FFmpeg。")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: 未找到 FFmpeg 命令，请确保已安装并添加到环境变量。")
        sys.exit(1)

def distribute_frames(temp_dir, out_root, chunk_size, overlap):
    """
    读取临时目录的帧，按滑动窗口复制到分段目录
    """
    # 获取所有图片并排序
    all_frames = sorted(glob.glob(os.path.join(temp_dir, "*.jpg")))
    total_frames = len(all_frames)
    
    if total_frames == 0:
        print("Error: 未提取到任何帧。")
        sys.exit(1)

    print(f"[2/3] 总帧数: {total_frames}，开始分段处理...")
    print(f"      配置: Chunk={chunk_size}, Overlap={overlap}")

    stride = chunk_size - overlap
    current_start = 0

    while current_start < total_frames:
        current_end = current_start + chunk_size
        
        # 获取当前段的文件
        segment_files = all_frames[current_start:current_end]
        
        if not segment_files:
            break

        actual_end_index = current_start + len(segment_files)
        
        # ---------------------------------------------------------
        # 修改点：文件夹命名不再补零
        # 例如: 0_100, 90_190, 1050_1150
        # ---------------------------------------------------------
        seg_folder_name = f"{current_start}_{actual_end_index}"
        seg_folder_path = os.path.join(out_root, seg_folder_name,'images')
        
        if not os.path.exists(seg_folder_path):
            os.makedirs(seg_folder_path)

        # 复制文件
        for src_path in segment_files:
            filename = os.path.basename(src_path)
            dst_path = os.path.join(seg_folder_path, filename)
            shutil.copy2(src_path, dst_path)

        # 更新起点
        current_start += stride
        
        sys.stdout.write(f"\r      已生成分段: {seg_folder_name:<20}") # <20 用于清除之前的长文件名残留显示
        sys.stdout.flush()

    print("\n")

def main():
    parser = argparse.ArgumentParser(description="视频滑动窗口切片工具")
    # 使用标准的短/长选项（长选项以双短横线开头），并将必要参数设为必填
    parser.add_argument('-v', '--video_path', type=str, required=True, help="输入视频文件路径")
    parser.add_argument('-d', '--out_root', type=str, required=True, help="输出根目录路径")
    parser.add_argument('-c', '--chunk_size', type=int, required=True, help="每个分段的帧数 (Chunk Size)")
    parser.add_argument('-o', '--overlap', type=int, required=True, help="分段间的重叠帧数 (Overlap)")

    args = parser.parse_args()

    # 1. 参数校验
    if not os.path.isfile(args.video_path):
        print(f"Error: 视频文件不存在 -> {args.video_path}")
        return

    if args.overlap >= args.chunk_size:
        print(f"Error: Overlap ({args.overlap}) 必须小于 Chunk Size ({args.chunk_size})")
        return

    # 2. 准备临时路径
    temp_dir = os.path.join(args.out_root, "temp_raw_frames_DELETE_ME")
    
    try:
        # Step 1: 解压
        run_ffmpeg_extraction(args.video_path, temp_dir)
        # Step 2: 分发
        distribute_frames(temp_dir, args.out_root, args.chunk_size, args.overlap)
        # Step 3: 清理
        print(f"[3/3] 清理临时文件...")
        shutil.rmtree(temp_dir)
        print(f"完成！输出目录: {args.out_root}")

    except KeyboardInterrupt:
        print("\n用户中断操作，正在尝试清理临时文件...")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        print("已清理。")

if __name__ == "__main__":
    main()