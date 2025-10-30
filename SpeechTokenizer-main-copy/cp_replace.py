"""
from /gemini/platform/public/aigc/mah_1/mah/unitok/SpeechTokenizer-main 
to :/gemini/platform/public/aigc/fys/separation/TeleAI_MG_FM/SpeechTokenizer-main-copy
将内容不同的文件进行替换，相同的就不换了，

ckpt不换

如果替换的内容是软链接，则直接替换软链接指向的内容

忽略的目录：
checkpoints
exps_1
exps_semantic
exps_semanticgen
exps_semanticgen_dit
mel_plots

"""

import os
import shutil
import filecmp
from pathlib import Path


def resolve_symlink(path):
    """解析软链接，返回实际文件路径"""
    if os.path.islink(path):
        return os.path.realpath(path)
    return path


def should_skip_file(file_path):
    """判断是否应该跳过该文件（例如ckpt文件）"""
    # 忽略的文件扩展名列表
    ignored_extensions = [
        '.ckpt',
        '.pt',
        '.pth',
        '.pyc',
        '.pyo',
    ]
    
    # 忽略的关键词（路径中包含这些词的文件）
    ignored_keywords = [
        'ckpt',
    ]
    
    file_path_lower = file_path.lower()
    
    # 检查文件扩展名
    for ext in ignored_extensions:
        if file_path.endswith(ext):
            return True
    
    # 检查关键词
    for keyword in ignored_keywords:
        if keyword in file_path_lower:
            return True
    
    return False


def should_skip_directory(rel_path):
    """判断路径是否在忽略的目录中"""
    # 忽略的目录列表
    ignored_dirs = [
        ".vscode",
        'checkpoints',
        'exps_1',
        'exps_semantic',
        'exps_semanticgen',
        'exps_semanticgen_dit',
        'mel_plots'
    ]
    
    # 检查路径中是否包含任何忽略的目录
    path_parts = Path(rel_path).parts
    for ignored_dir in ignored_dirs:
        if ignored_dir in path_parts:
            return True
    return False


def copy_different_files(src_dir, dst_dir, dry_run=False):
    """
    比较两个目录，将内容不同的文件从src_dir复制到dst_dir
    
    Args:
        src_dir: 源目录
        dst_dir: 目标目录
        dry_run: 是否为试运行模式（只打印不实际复制）
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    
    if not src_path.exists():
        print(f"错误：源目录不存在: {src_dir}")
        return
    
    if not dst_path.exists():
        print(f"警告：目标目录不存在: {dst_dir}")
        return
    
    copied_count = 0
    skipped_count = 0
    same_count = 0
    ignored_dir_count = 0
    
    # 遍历源目录中的所有文件
    for src_file in src_path.rglob('*'):
        if src_file.is_file():
            # 计算相对路径
            rel_path = src_file.relative_to(src_path)
            dst_file = dst_path / rel_path
            
            # 检查是否在忽略的目录中
            if should_skip_directory(rel_path):
                ignored_dir_count += 1
                continue
            
            # 检查是否应该跳过
            if should_skip_file(str(rel_path)):
                print(f"跳过: {rel_path} (ckpt文件)")
                skipped_count += 1
                continue
            
            # 解析软链接
            actual_src_file = resolve_symlink(str(src_file))
            
            # 如果目标文件不存在
            if not dst_file.exists():
                print(f"新文件: {rel_path}")
                if not dry_run:
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(actual_src_file, dst_file)
                copied_count += 1
            else:
                # 解析目标文件的软链接
                actual_dst_file = resolve_symlink(str(dst_file))
                
                # 比较文件内容
                try:
                    if not filecmp.cmp(actual_src_file, actual_dst_file, shallow=False):
                        print(f"替换: {rel_path} (内容不同)")
                        if not dry_run:
                            shutil.copy2(actual_src_file, dst_file)
                        copied_count += 1
                    else:
                        same_count += 1
                except Exception as e:
                    print(f"警告：比较文件失败 {rel_path}: {e}")
                    # 发生错误时也尝试复制
                    if not dry_run:
                        shutil.copy2(actual_src_file, dst_file)
                    copied_count += 1
    
    print("\n" + "="*50)
    print(f"处理完成!")
    print(f"复制/替换的文件数: {copied_count}")
    print(f"内容相同跳过的文件数: {same_count}")
    print(f"ckpt跳过的文件数: {skipped_count}")
    print(f"忽略目录中的文件数: {ignored_dir_count}")
    print("="*50)


if __name__ == "__main__":
    # 配置源目录和目标目录
    SOURCE_DIR = "/gemini/platform/public/aigc/mah_1/mah/unitok/SpeechTokenizer-main"
    TARGET_DIR = "/gemini/platform/public/aigc/fys/separation/TeleAI_MG_FM/SpeechTokenizer-main-copy"
    
    # 设置为True进行试运行（只打印不实际复制）
    DRY_RUN = False
    
    print(f"源目录: {SOURCE_DIR}")
    print(f"目标目录: {TARGET_DIR}")
    print(f"模式: {'试运行（不实际复制）' if DRY_RUN else '实际执行'}")
    print("="*50)
    
    # 执行复制
    copy_different_files(SOURCE_DIR, TARGET_DIR, dry_run=DRY_RUN)