"""
代码介绍：
将/gemini/platform/public/aigc/fys/separation/ckpt/speechtokenizer_ckpt&datas 下的目录软链接到对应以下目录的对应内容中，
/gemini/platform/public/aigc/fys/separation/TeleAI_MG_FM/SpeechTokenizer-main-copy

要求：
1. 不存在目录自动创建
2. 存在目录自动链接相应内容
3. 目录结构保持一致
"""

import os
import sys
from pathlib import Path

def create_symlinks_recursive(source_path, target_path, indent=0):
    """递归创建软链接
    
    Args:
        source_path: 源路径（文件或目录）
        target_path: 目标路径
        indent: 缩进级别，用于打印
    """
    prefix = "  " * indent
    
    try:
        # 如果目标路径已经存在
        if target_path.exists() or target_path.is_symlink():
            if target_path.is_symlink():
                # 如果已经是软链接，检查是否指向正确的位置
                existing_link = target_path.resolve()
                if existing_link == source_path.resolve():
                    print(f"{prefix}已存在正确的软链接: {target_path.name}")
                    return
                else:
                    print(f"{prefix}删除旧的软链接: {target_path.name}")
                    target_path.unlink()
            elif target_path.is_dir() and source_path.is_dir():
                # 如果目标是真实目录且源也是目录，递归处理目录内容
                print(f"{prefix}目标目录已存在，进入目录处理内容: {target_path.name}")
                for sub_item in source_path.iterdir():
                    sub_source = sub_item
                    sub_target = target_path / sub_item.name
                    create_symlinks_recursive(sub_source, sub_target, indent + 1)
                return
            else:
                # 如果是文件或其他类型，跳过
                print(f"{prefix}警告：目标路径已存在且不是软链接，跳过: {target_path.name}")
                return
        
        # 创建软链接
        if os.name == 'nt':  # Windows系统
            if source_path.is_dir():
                os.symlink(source_path, target_path, target_is_directory=True)
            else:
                os.symlink(source_path, target_path)
        else:  # Unix/Linux系统
            os.symlink(source_path, target_path)
        
        print(f"{prefix}成功创建软链接: {target_path.name} -> {source_path}")
        
    except PermissionError:
        print(f"{prefix}权限错误：无法创建软链接 {target_path.name}，请使用管理员权限运行")
    except OSError as e:
        print(f"{prefix}创建软链接失败 {target_path.name}: {e}")
    except Exception as e:
        print(f"{prefix}未知错误 {target_path.name}: {e}")


def create_symlinks():
    """创建软链接，将源目录下的内容链接到目标目录"""
    
    # 定义源目录和目标目录
    source_dir = Path("/gemini/platform/public/aigc/fys/separation/ckpt/speechtokenizer_ckpt&datas")
    target_dir = Path("/gemini/platform/public/aigc/fys/separation/TeleAI_MG_FM/SpeechTokenizer-main-copy")
    
    # 检查源目录是否存在
    if not source_dir.exists():
        print(f"错误：源目录不存在: {source_dir}")
        sys.exit(1)
    
    # 如果目标目录不存在，创建它
    if not target_dir.exists():
        print(f"创建目标目录: {target_dir}")
        target_dir.mkdir(parents=True, exist_ok=True)
    
    # 遍历源目录下的所有文件和子目录
    for item in source_dir.iterdir():
        source_path = item
        target_path = target_dir / item.name
        create_symlinks_recursive(source_path, target_path, indent=0)
    
    print("\n软链接创建完成！")

def main():
    """主函数"""
    print("开始创建软链接..")
    print(f"源目录: /gemini/platform/public/aigc/fys/separation/ckpt/speechtokenizer_ckpt&datas")
    print(f"目标目录: /gemini/platform/public/aigc/fys/separation/TeleAI_MG_FM/SpeechTokenizer-main-copy")
    print("-" * 80)
    
    create_symlinks()

if __name__ == "__main__":
    main()