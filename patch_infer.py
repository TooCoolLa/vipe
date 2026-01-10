import os
import argparse
import subprocess
import sys

def run_batch_inference(root_directory):
    # 1. 检查根目录是否存在
    if not os.path.exists(root_directory):
        print(f"❌ 错误: 找不到目录 {root_directory}")
        sys.exit(1)

    # 2. 获取并排序所有子目录
    # 过滤掉非目录文件和 .ipynb_checkpoints
    subdirs = [
        d for d in os.listdir(root_directory) 
        if os.path.isdir(os.path.join(root_directory, d)) and d != ".ipynb_checkpoints"
    ]
    
    # 尝试按数字排序 (针对 start_end 格式)
    try:
        subdirs.sort(key=lambda x: int(x.split('_')[0]))
    except:
        subdirs.sort()

    print(f"📂 在 '{root_directory}' 下找到 {len(subdirs)} 个子目录，准备处理...")

    # 3. 设置环境变量 (export MPLBACKEND=Agg)
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    # 4. 循环处理
    success_count = 0
    fail_count = 0

    for folder_name in subdirs:
        rootpath = os.path.join(root_directory, folder_name)
        input_path = os.path.join(rootpath, "images")
        output_path = os.path.join(rootpath, "result") 
        
        print(f"\n[处理中] ----------------------------------------")
        print(f"输入: {folder_name}")
        
        # 构建命令
        cmd = [
            "vipe", "infer",
            "--image-dir", input_path,
            "--pipeline", "dav3",
            "--output", output_path
        ]

        try:
            # check=True 遇到错误会抛出异常
            subprocess.run(cmd, env=env, check=True)
            print(f"✅ 完成")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"❌ 失败 (错误码: {e.returncode})")
            fail_count += 1
        except FileNotFoundError:
            print("❌ 致命错误: 找不到 'vipe' 命令，请检查环境安装。")
            sys.exit(1)

    print(f"\n========================================")
    print(f"全部结束: 成功 {success_count} 个, 失败 {fail_count} 个")

if __name__ == "__main__":
    # 定义参数解析
    parser = argparse.ArgumentParser(description="批量运行 vipe infer 脚本")
    
    # 添加 -d 参数
    parser.add_argument("-d", "--dir", required=True, help="包含子目录数据的父目录路径")
    
    args = parser.parse_args()
    
    run_batch_inference(args.dir)