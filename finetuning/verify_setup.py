#!/usr/bin/env python3
"""验证 LoRA 微调环境配置"""

import os
import json
import sys
from pathlib import Path

def check_directory_structure():
    """检查目录结构"""
    print("\n📁 检查目录结构...")
    base = Path(__file__).parent

    required_dirs = [
        "LLaMA-Factory",
        "data",
        "output",
    ]

    required_files = [
        "train_lora_sft_A_single.yaml",
        "train_lora_sft_B_single.yaml",
        "train_lora_sft_A_multi.yaml",
        "train_lora_sft_B_multi.yaml",
    ]

    all_ok = True
    for d in required_dirs:
        path = base / d
        if path.exists():
            print(f"  ✅ {d}/")
        else:
            print(f"  ❌ {d}/ (缺失)")
            all_ok = False

    for f in required_files:
        path = base / f
        if path.exists():
            print(f"  ✅ {f}")
        else:
            print(f"  ❌ {f} (缺失)")
            all_ok = False

    # 检查模型
    model_path = base.parent / "models" / "Qwen2.5-Coder-7B-Instruct"
    if model_path.exists():
        size_gb = sum(f.stat().st_size for f in model_path.glob('**/*') if f.is_file()) / 1024**3
        print(f"  ✅ models/Qwen2.5-Coder-7B-Instruct/ ({size_gb:.2f} GB)")
    else:
        print(f"  🔄 models/Qwen2.5-Coder-7B-Instruct/ (下载中...)")

    return all_ok

def check_python_packages():
    """检查 Python 依赖"""
    print("\n📦 检查 Python 依赖...")

    required_packages = [
        "transformers",
        "torch",
        "peft",
        "datasets",
        "llamafactory",
    ]

    all_ok = True
    for pkg in required_packages:
        try:
            __import__(pkg.replace("-", "_"))
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ❌ {pkg} (未安装)")
            all_ok = False

    return all_ok

def check_dataset_registration():
    """检查数据集注册"""
    print("\n📋 检查数据集注册...")

    dataset_info_path = Path(__file__).parent / "LLaMA-Factory" / "data" / "dataset_info.json"

    if not dataset_info_path.exists():
        print(f"  ❌ dataset_info.json 不存在")
        return False

    with open(dataset_info_path) as f:
        dataset_info = json.load(f)

    required_datasets = [
        "sft_A_single",
        "sft_B_single",
        "sft_A_multi",
        "sft_B_multi",
    ]

    all_ok = True
    for ds in required_datasets:
        if ds in dataset_info:
            print(f"  ✅ {ds}")
        else:
            print(f"  ❌ {ds} (未注册)")
            all_ok = False

    return all_ok

def check_yaml_configs():
    """检查 YAML 配置文件"""
    print("\n⚙️  检查 YAML 配置文件...")

    try:
        import yaml
    except ImportError:
        print("  ⚠️  PyYAML 未安装，跳过详细检查")
        return True

    base = Path(__file__).parent
    yaml_files = [
        "train_lora_sft_A_single.yaml",
        "train_lora_sft_B_single.yaml",
        "train_lora_sft_A_multi.yaml",
        "train_lora_sft_B_multi.yaml",
    ]

    all_ok = True
    for yaml_file in yaml_files:
        path = base / yaml_file
        if not path.exists():
            print(f"  ❌ {yaml_file} (不存在)")
            all_ok = False
            continue

        try:
            with open(path) as f:
                config = yaml.safe_load(f)

            # 检查关键参数
            if "model_name_or_path" in config:
                print(f"  ✅ {yaml_file}")
            else:
                print(f"  ⚠️  {yaml_file} (缺少 model_name_or_path)")
        except Exception as e:
            print(f"  ❌ {yaml_file} (解析失败: {e})")
            all_ok = False

    return all_ok

def check_gpu():
    """检查 GPU 可用性"""
    print("\n🎮 检查 GPU...")

    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"  ✅ 检测到 {gpu_count} 张 GPU")
            for i in range(gpu_count):
                name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"     - GPU {i}: {name} ({memory:.1f} GB)")
            return True
        else:
            print(f"  ⚠️  未检测到 GPU（将使用 CPU，速度很慢）")
            return False
    except Exception as e:
        print(f"  ⚠️  无法检查 GPU: {e}")
        return False

def main():
    print("=" * 50)
    print("LoRA 微调环境验证 (Qwen2.5-Coder-7B-Instruct)")
    print("=" * 50)

    checks = [
        ("目录结构", check_directory_structure),
        ("Python 依赖", check_python_packages),
        ("数据集注册", check_dataset_registration),
        ("YAML 配置", check_yaml_configs),
        ("GPU 可用", check_gpu),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"  ❌ 检查 {name} 出错: {e}")
            results.append((name, False))

    # 总结
    print("\n" + "=" * 50)
    print("检查总结:")
    print("=" * 50)

    all_passed = True
    for name, result in results:
        status = "✅ 通过" if result else "⚠️  警告/失败"
        print(f"  {status} - {name}")
        if not result:
            all_passed = False

    print()
    if all_passed:
        print("🎉 所有检查都通过了！可以开始训练")
        print("\n使用命令开始训练:")
        print("  cd finetuning")
        print("  source ../venv/bin/activate")
        print("  llamafactory-cli train train_lora_sft_A_single.yaml")
        return 0
    else:
        print("⚠️  有些检查未通过，请解决后再训练")
        return 1

if __name__ == "__main__":
    sys.exit(main())
