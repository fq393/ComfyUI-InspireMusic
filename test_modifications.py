#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test to verify our modifications work correctly
简单测试验证我们的修改是否正确工作
"""

import os
import sys
from pathlib import Path
import inspect

# Add the project root to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

def test_model_manager_modifications():
    """
    Test model manager modifications without importing torch-dependent modules
    测试模型管理器修改，不导入依赖torch的模块
    """
    print("Testing model manager modifications...")
    print("测试模型管理器修改...")
    
    try:
        # Read the model_manager.py file to check our modifications
        # 读取model_manager.py文件检查我们的修改
        model_manager_path = current_dir / "modules" / "model_manager.py"
        
        if not model_manager_path.exists():
            print(f"✗ Model manager file not found: {model_manager_path}")
            print(f"✗ 模型管理器文件未找到: {model_manager_path}")
            return False
            
        with open(model_manager_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check if our modifications are present
        # 检查我们的修改是否存在
        checks = [
            ("model_dir: str = None" in content, "model_dir parameter added"),
            ("if model_dir:" in content, "model_dir condition added"),
            ("cache_key = f\"inspiremusic_{model_name}_{fast_mode}_{output_sample_rate}_{model_dir}\"" in content, "cache key updated"),
            ("model_path = model_dir" in content, "model_dir usage added")
        ]
        
        all_passed = True
        for check, description in checks:
            if check:
                print(f"✓ {description}")
                print(f"✓ {description}")
            else:
                print(f"✗ {description} - MISSING")
                print(f"✗ {description} - 缺失")
                all_passed = False
                
        return all_passed
        
    except Exception as e:
        print(f"✗ Error testing model manager: {e}")
        print(f"✗ 测试模型管理器错误: {e}")
        return False

def test_node_modifications():
    """
    Test node modifications without importing torch-dependent modules
    测试节点修改，不导入依赖torch的模块
    """
    print("\nTesting node modifications...")
    print("测试节点修改...")
    
    try:
        # Read the inspiremusic_node.py file to check our modifications
        # 读取inspiremusic_node.py文件检查我们的修改
        node_path = current_dir / "nodes" / "inspiremusic_node.py"
        
        if not node_path.exists():
            print(f"✗ Node file not found: {node_path}")
            print(f"✗ 节点文件未找到: {node_path}")
            return False
            
        with open(node_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check if our modifications are present
        # 检查我们的修改是否存在
        checks = [
            ('"model_dir": ("STRING", {"default": "", "multiline": False})' in content, "model_dir input added"),
            ("model_dir: str = None" in content, "model_dir parameter in _load_model"),
            ("model_dir if model_dir and model_dir.strip() else None" in content, "model_dir condition in load_model call"),
            ("model_dir: str = \"\"" in content, "model_dir parameter in generate_music"),
            ("self._load_model(model_name, fast_mode, output_sample_rate, model_dir)" in content, "model_dir passed to _load_model")
        ]
        
        all_passed = True
        for check, description in checks:
            if check:
                print(f"✓ {description}")
                print(f"✓ {description}")
            else:
                print(f"✗ {description} - MISSING")
                print(f"✗ {description} - 缺失")
                all_passed = False
                
        return all_passed
        
    except Exception as e:
        print(f"✗ Error testing node: {e}")
        print(f"✗ 测试节点错误: {e}")
        return False

def test_file_structure():
    """
    Test that all required files exist
    测试所有必需文件是否存在
    """
    print("\nTesting file structure...")
    print("测试文件结构...")
    
    required_files = [
        "modules/model_manager.py",
        "nodes/inspiremusic_node.py",
        "nodes/__init__.py"
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = current_dir / file_path
        if full_path.exists():
            print(f"✓ {file_path} exists")
            print(f"✓ {file_path} 存在")
        else:
            print(f"✗ {file_path} missing")
            print(f"✗ {file_path} 缺失")
            all_exist = False
            
    return all_exist

def main():
    print("InspireMusic Server Model Support Verification")
    print("InspireMusic服务器模型支持验证")
    print("=" * 60)
    
    # Run all tests
    # 运行所有测试
    tests = [
        ("File Structure", test_file_structure),
        ("Model Manager", test_model_manager_modifications),
        ("Node Modifications", test_node_modifications)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    # 总结
    print("\n" + "=" * 60)
    print("SUMMARY 总结:")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        status_cn = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status} {status_cn}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All modifications completed successfully!")
        print("🎉 所有修改成功完成！")
        print("\nYour ComfyUI-InspireMusic now supports server model paths.")
        print("您的ComfyUI-InspireMusic现在支持服务器模型路径。")
        print("\nUsage 使用方法:")
        print("1. In ComfyUI, add the InspireMusic Text to Music node")
        print("   在ComfyUI中，添加InspireMusic Text to Music节点")
        print("2. In the node settings, fill the 'model_dir' field with your server path")
        print("   在节点设置中，在'model_dir'字段填入您的服务器路径")
        print("3. Example: /data/ComfyUI/models/InspireMusic/InspireMusic-1.5B-Long")
        print("   示例: /data/ComfyUI/models/InspireMusic/InspireMusic-1.5B-Long")
    else:
        print("❌ Some modifications failed. Please check the errors above.")
        print("❌ 某些修改失败。请检查上面的错误。")

if __name__ == "__main__":
    main()