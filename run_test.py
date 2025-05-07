# -*- coding:utf-8 -*-
import os
import subprocess

if __name__ == "__main__":
    # 使用简化版配置文件
    config_path = "configs/test_innovation.yaml"
    
    print(f"开始运行测试实验: {config_path}")
    cmd = f"python train.py --config {config_path}"
    subprocess.run(cmd, shell=True)
    print(f"测试实验完成: {config_path}")
    
    print("测试完成！") 