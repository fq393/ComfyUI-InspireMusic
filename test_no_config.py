#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试不使用配置文件的 InspireMusic 模型初始化
"""

import os
import sys
import logging

# 添加项目路径到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_no_config_file():
    """测试不使用配置文件的模型初始化"""
    try:
        from inspiremusic.cli.inference import InspireMusicModel
        
        # 使用一个假的模型目录进行测试
        model_dir = "/tmp/test_model"
        os.makedirs(model_dir, exist_ok=True)
        
        # 创建必要的模型文件（空文件用于测试）
        required_files = ['llm.pt', 'flow.pt']
        for file in required_files:
            file_path = os.path.join(model_dir, file)
            with open(file_path, 'w') as f:
                f.write("# dummy file for testing")
        
        # 创建必要的目录
        os.makedirs(os.path.join(model_dir, 'music_tokenizer'), exist_ok=True)
        os.makedirs(os.path.join(model_dir, 'wavtokenizer'), exist_ok=True)
        
        print("测试1: 使用配置文件（默认行为）")
        try:
            model1 = InspireMusicModel(
                model_name="test",
                model_dir=model_dir,
                use_config_file=True,  # 使用配置文件
                gpu=-1  # 使用 CPU
            )
            print("✗ 使用配置文件测试失败（预期的，因为没有配置文件）")
        except Exception as e:
            print(f"✓ 使用配置文件测试按预期失败: {e}")
        
        print("\n测试2: 不使用配置文件（新功能）")
        try:
            model2 = InspireMusicModel(
                model_name="test",
                model_dir=model_dir,
                use_config_file=False,  # 不使用配置文件
                gpu=-1  # 使用 CPU
            )
            print("✓ 不使用配置文件测试成功！")
            return True
        except Exception as e:
            print(f"✗ 不使用配置文件测试失败: {e}")
            return False
            
    except ImportError as e:
        print(f"✗ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"✗ 测试过程中出现错误: {e}")
        return False
    finally:
        # 清理测试文件
        import shutil
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)

if __name__ == "__main__":
    print("开始测试不使用配置文件的功能...")
    success = test_no_config_file()
    if success:
        print("\n🎉 测试成功！现在可以不使用配置文件来初始化 InspireMusic 模型了。")
    else:
        print("\n❌ 测试失败，请检查代码修改。")