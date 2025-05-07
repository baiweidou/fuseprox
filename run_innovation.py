# -*- coding:utf-8 -*-
import os
import subprocess
import argparse

def run_experiment(config_path):
    """
    运行指定配置文件的实验
    """
    print(f"开始运行实验: {config_path}")
    cmd = f"python train.py --config {config_path}"
    subprocess.run(cmd, shell=True)
    print(f"实验完成: {config_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行多模态融合创新点实验")
    parser.add_argument("--config", type=str, default="configs/Innovation.yaml", 
                        help="配置文件路径")
    args = parser.parse_args()
    
    # 运行实验
    run_experiment(args.config)
    
    print("所有实验已完成！") 