#!/usr/bin/env python3
"""
检查 metadata.scp 中的 JSON 文件，找出有问题的文件（transcripts 格式异常）
对有问题的文件进行修复，并保存备份到_old结尾的文件
"""
import json
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
import sys

def repair_json_file(metadata_path):
    """
    修复 JSON 文件，去除不符合格式的字符串
    跳过备份，直接修复原文件
    返回修复后的 JSON 对象，如果修复失败返回 None
    """
    try:
        # 读取并解析原始文件内容
        with open(metadata_path, 'r', encoding='utf-8') as f:
            content = f.read()

        try:
            av_info = json.loads(content)
        except json.JSONDecodeError:
            # 如果解析失败，尝试找到有效的 JSON 对象
            av_info = extract_valid_json(content)
            if av_info is None:
                return None
        
        # 清理 transcripts 数组中的非对象元素 #especially for avdeepfake1M++
        if isinstance(av_info, dict) and 'transcripts' in av_info:
            if isinstance(av_info['transcripts'], list):
                # 只保留对象类型的元素，过滤掉字符串
                av_info['transcripts'] = [
                    item for item in av_info['transcripts'] 
                    if isinstance(item, dict)
                ]
        
        # 保存修复后的文件
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(av_info, f, ensure_ascii=False, indent=2)
        
        return av_info
        
    except Exception as e:
        print(f"  修复失败: {str(e)}")
        return None


def extract_valid_json(content):
    """
    从内容中提取有效的 JSON 对象
    """
    # 找到第一个 { 和最后一个 }
    start_idx = content.find('{')
    end_idx = content.rfind('}')
    
    if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
        return None
    
    try:
        json_str = content[start_idx:end_idx+1]
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def check_single_file(json_path, fix=False, verbose=False):
    """
    检查并可选修复单个 JSON 文件
    
    Args:
        json_path: JSON文件路径
        fix: 是否修复问题
        verbose: 是否输出详细调试信息
        
    Returns:
        dict: {
            'status': 'ok' | 'error' | 'fixed',
            'issue': str,  # 问题描述
            'av_info': dict | None,  # 解析后的JSON对象
            'details': {  # 详细统计信息（verbose时）
                'obj_count': int,
                'str_count': int,
                'other_count': int,
                'invalid_items': list
            }
        }
    """
    result = {
        'status': 'ok',
        'issue': None,
        'av_info': None,
        'details': {}
    }
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"检查文件: {json_path}")
        print('='*80)
    
    if not Path(json_path).exists():
        result['status'] = 'error'
        result['issue'] = '文件不存在'
        if verbose:
            print(f"❌ 文件不存在: {json_path}")
        return result
    
    # 尝试加载JSON
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            av_info = json.load(f)
        if verbose:
            print("✓ JSON 格式正常，可以直接加载")
    except json.JSONDecodeError as e:
        result['status'] = 'error'
        result['issue'] = f'JSON解析失败: {str(e)}'
        if verbose:
            print(f"❌ JSON 解析失败: {str(e)}")
        
        if fix:
            if verbose:
                print("\n尝试修复...")
            av_info = repair_json_file(json_path)
            if av_info is not None:
                result['status'] = 'fixed'
                result['issue'] = 'JSON解析失败 -> 已修复'
                result['av_info'] = av_info
                if verbose:
                    print("✓ 修复成功")
            else:
                if verbose:
                    print("❌ 修复失败")
                return result
        else:
            return result
    except Exception as e:
        result['status'] = 'error'
        result['issue'] = f'JSON读取失败: {str(e)}'
        if verbose:
            print(f"❌ 文件读取失败: {str(e)}")
        return result
    
    # 检查 transcripts 字段
    if 'transcripts' not in av_info:
        result['status'] = 'error'
        result['issue'] = '缺失 transcripts 字段'
        result['av_info'] = av_info
        return result
    
    transcripts = av_info['transcripts']
    
    # 检查格式
    if isinstance(transcripts, str):
        result['status'] = 'error'
        result['issue'] = f'transcripts 是字符串而非列表'
        result['av_info'] = av_info
        return result
    
    if not isinstance(transcripts, list):
        result['status'] = 'error'
        result['issue'] = f'transcripts 类型异常: {type(transcripts).__name__}'
        result['av_info'] = av_info
        return result
    
    if len(transcripts) == 0:
        result['av_info'] = av_info
        return result
    
    # 扫描整个列表
    obj_count = 0
    str_count = 0
    other_count = 0
    invalid_items = []
    
    for idx, item in enumerate(transcripts):
        if isinstance(item, dict):
            obj_count += 1
            if 'word' not in item:
                invalid_items.append({
                    'index': idx,
                    'type': 'dict_missing_word',
                    'value': list(item.keys())
                })
        elif isinstance(item, str):
            str_count += 1
            invalid_items.append({
                'index': idx,
                'type': 'string',
                'value': item
            })
        else:
            other_count += 1
            invalid_items.append({
                'index': idx,
                'type': type(item).__name__,
                'value': str(item)
            })
    
    result['details'] = {
        'obj_count': obj_count,
        'str_count': str_count,
        'other_count': other_count,
        'invalid_items': invalid_items
    }
    
    if verbose:
        print(f"\n📊 Transcripts 分析:")
        print(f"   总数: {len(transcripts)}")
        print(f"   对象数: {obj_count}")
        print(f"   字符串数: {str_count}")
        print(f"   其他类型数: {other_count}")
    
    if invalid_items:
        # 找到第一个非字典元素或缺失word的字典
        first_non_dict = next((item for item in invalid_items if item['type'] != 'dict_missing_word'), None)
        first_bad_dict = next((item for item in invalid_items if item['type'] == 'dict_missing_word'), None)
        
        if first_non_dict:
            result['status'] = 'error'
            result['issue'] = f"transcripts 列表包含非字典元素: {first_non_dict['type']}"
        elif first_bad_dict:
            result['status'] = 'error'
            result['issue'] = f"transcripts 列表中的字典缺失 word 字段: {first_bad_dict['value']}"
        
        result['av_info'] = av_info
        
        if verbose:
            print(f"\n⚠️ 发现 {len(invalid_items)} 个问题元素:")
            for item in invalid_items[:10]:
                print(f"   [索引 {item['index']}] 类型: {item['type']}, 值: {str(item['value'])[:50]}")
            if len(invalid_items) > 10:
                print(f"   ... 还有 {len(invalid_items) - 10} 个")
        
        if fix:
            if verbose:
                print("\n开始修复...")
            av_info_fixed = repair_json_file(json_path)
            if av_info_fixed is not None:
                result['status'] = 'fixed'
                result['issue'] = result['issue'] + ' -> 已修复'
                result['av_info'] = av_info_fixed
                if verbose:
                    removed = len(transcripts) - len(av_info_fixed.get('transcripts', []))
                    print(f"✓ 已移除 {removed} 个非对象元素")
                    print(f"✓ 已保存修复后的文件")
    else:
        result['av_info'] = av_info
        if verbose:
            print("\n✓ 所有 transcripts 元素格式正常")
    
    if verbose:
        print('='*80 + '\n')
    
    return result


def check_metadata_files(metadata_scp, output_file=None, fix_files=False):
    """
    扫描 metadata.scp 中的所有 JSON 文件，找出有问题的文件
    如果 fix_files=True，则对问题文件进行修复并保存备份
    """
    problematic_files = []
    fixed_files = []
    
    # 读取 metadata.scp
    utt2metadata = {}
    with open(metadata_scp) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                uttid, path = parts
            elif len(parts) == 1:
                uttid, path = None, parts[0]
            else:
                continue
            utt2metadata[uttid if uttid is not None else path] = path
    
    print(f"总共需要检查 {len(utt2metadata)} 个文件")
    print(f"修复模式: {'已启用' if fix_files else '已禁用'}")
    print("="*80)
    
    for uttid, metadata_path in tqdm(utt2metadata.items()):
        result = check_single_file(metadata_path, fix=fix_files, verbose=False)
        
        if result['status'] == 'error':
            problematic_files.append({
                'uttid': uttid,
                'path': metadata_path,
                'issue': result['issue']
            })
        elif result['status'] == 'fixed':
            problematic_files.append({
                'uttid': uttid,
                'path': metadata_path,
                'issue': result['issue'].split(' -> ')[0]  # 原始问题
            })
            fixed_files.append({
                'uttid': uttid,
                'path': metadata_path,
                'issue': result['issue']  # 包含修复状态
            })
    
    print("="*80)
    print(f"\n检查完成！")
    print(f"总共找到 {len(problematic_files)} 个有问题的文件")
    if fix_files:
        print(f"其中 {len(fixed_files)} 个文件已成功修复")
    print()
    
    if problematic_files:
        # 按 issue 类型分组
        issues_by_type = {}
        for item in problematic_files:
            issue = item['issue']
            if issue not in issues_by_type:
                issues_by_type[issue] = []
            issues_by_type[issue].append(item)
        
        # 输出统计
        print("问题分类统计:")
        for issue, items in sorted(issues_by_type.items(), key=lambda x: -len(x[1])):
            print(f"  {len(items):6d} 个文件 - {issue}")
        
        print("\n详细列表:")
        print("-"*80)
        for item in problematic_files:
            print(f"UttID: {item['uttid']}")
            print(f"Path: {item['path']}")
            print(f"Issue: {item['issue']}")
            print("-"*80)
        
        # 保存到文件
        if output_file:
            with open(output_file, 'w') as f:
                f.write(f"检查时间: {metadata_scp}\n")
                f.write(f"总共找到 {len(problematic_files)} 个有问题的文件\n")
                if fix_files:
                    f.write(f"其中 {len(fixed_files)} 个文件已成功修复\n\n")
                else:
                    f.write("\n")
                f.write("问题分类统计:\n")
                for issue, items in sorted(issues_by_type.items(), key=lambda x: -len(x[1])):
                    f.write(f"  {len(items):6d} 个文件 - {issue}\n")
                f.write("\n" + "="*80 + "\n")
                f.write("详细列表:\n")
                f.write("="*80 + "\n")
                for item in problematic_files:
                    f.write(f"UttID: {item['uttid']}\n")
                    f.write(f"Path: {item['path']}\n")
                    f.write(f"Issue: {item['issue']}\n")
                    f.write("-"*80 + "\n")
                if fixed_files:
                    f.write("\n" + "="*80 + "\n")
                    f.write("已修复的文件:\n")
                    f.write("="*80 + "\n")
                    for item in fixed_files:
                        f.write(f"UttID: {item['uttid']}\n")
                        f.write(f"Path: {item['path']}\n")
                        f.write(f"Status: {item['issue']}\n")
                        f.write("-"*80 + "\n")
            print(f"\n详细列表已保存到: {output_file}")
    else:
        print("✓ 所有文件都正常！")

def debug_repair_json_file(json_path):
    """
    Debug 函数：测试 repair_json_file 函数是否有效
    显示修复前后的对比
    """
    print(f"\n调试修复函数: {json_path}")
    print("="*80)
    
    if not Path(json_path).exists():
        print(f"❌ 文件不存在: {json_path}")
        return
    
    # # 检查 _old 文件是否已存在
    # backup_path = json_path + '_old'
    # if Path(backup_path).exists():
    #     print(f"⚠️ _old 文件已存在，跳过修复")
    #     print(f"   备份文件: {backup_path}")
    #     return
    
    # 1. 显示修复前的状态
    print("\n📋 修复前分析:")
    print("-"*80)
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            content_before = f.read()
        
        av_info_before = json.loads(content_before)
        print("✓ 文件可以直接解析（可能是之前修复过的）")
        status_before = "success"
    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析失败: {str(e)}")
        status_before = "failed"
        # 显示有问题的行
        lines = content_before.split('\n')
        print(f"   文件总行数: {len(lines)}")
    
    # 2. 调用修复函数
    print("\n🔧 开始修复...")
    print("-"*80)
    av_info_after = repair_json_file(json_path)
    
    if av_info_after is None:
        print("❌ 修复失败")
        return
    
    print("✓ 修复函数执行完成")
    
    # 3. 显示修复后的状态
    print("\n📋 修复后分析:")
    print("-"*80)
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            content_after = f.read()
        
        av_info_after = json.loads(content_after)
        print("✓ 修复后文件可以直接解析")
        status_after = "success"
    except json.JSONDecodeError as e:
        print(f"❌ 修复后仍然无法解析: {str(e)}")
        status_after = "failed"
    
    # 4. 对比 transcripts 字段
    if av_info_after and 'transcripts' in av_info_after:
        transcripts = av_info_after['transcripts']
        print(f"\n📊 Transcripts 字段分析:")
        print(f"   总数: {len(transcripts)}")
        
        # 统计类型
        obj_count = sum(1 for item in transcripts if isinstance(item, dict))
        str_count = sum(1 for item in transcripts if isinstance(item, str))
        other_count = len(transcripts) - obj_count - str_count
        
        print(f"   对象数: {obj_count}")
        print(f"   字符串数: {str_count}")
        print(f"   其他类型数: {other_count}")
        
        if str_count > 0 or other_count > 0:
            print(f"\n❌ 修复不完全，还有 {str_count + other_count} 个非对象元素")
            # 显示这些问题元素
            for idx, item in enumerate(transcripts):
                if not isinstance(item, dict):
                    print(f"   [索引 {idx}] 类型: {type(item).__name__}, 值: {str(item)[:50]}")
        else:
            print(f"\n✅ 修复成功！所有 transcripts 元素都是对象")
    
    # 5. 总结
    print("\n" + "="*80)
    print("修复总结:")
    print(f"  修复前状态: {status_before}")
    print(f"  修复后状态: {status_after}")
    # print(f"  备份文件: {backup_path}")
    print(f"  修复文件: {json_path}")
    print("="*80 + "\n")


def debug_json_file(json_path, fix=False):
    """
    Debug 函数：针对某一个 JSON 文件进行详细分析和修复
    使用统一的 check_single_file 函数，启用详细输出
    """
    result = check_single_file(json_path, fix=fix, verbose=True)
    return result


def main():


    parser = argparse.ArgumentParser(description="检查和修复 metadata.scp 中的问题文件")

    # check 命令
    parser.add_argument('--metadata_scp', type=str, required=True,
                              help='metadata.scp 文件路径')
    parser.add_argument('--output', type=str, default=None,
                              help='save details when checking (optional)')
    parser.add_argument('--fix', action='store_true', default=True,
                              help='自动修复问题文件，并保存备份到_old结尾的文件')
    
    args = parser.parse_args()

    check_metadata_files(args.metadata_scp, args.output, fix_files=args.fix)


    # parser = argparse.ArgumentParser(description="检查和修复 metadata.scp 中的问题文件")
    # subparsers = parser.add_subparsers(dest='command', help='命令', default="debug")
    
    # # check 命令
    # check_parser = subparsers.add_parser('check', help='扫描并检查所有文件')
    # check_parser.add_argument('--metadata_scp', type=str, required=True,
    #                           help='metadata.scp 文件路径')
    # check_parser.add_argument('--output', type=str, default=None,
    #                           help='输出文件路径 (可选)')
    # check_parser.add_argument('--fix', action='store_true',
    #                           help='自动修复问题文件，并保存备份到_old结尾的文件')
    
    # # debug 命令
    # debug_parser = subparsers.add_parser('debug', help='调试单个 JSON 文件')
    # debug_parser.add_argument('--json_path', type=str, required=True, default='',
    #                           help='JSON 文件路径')
    # debug_parser.add_argument('--fix', action='store_true',
    #                           help='自动修复问题')
    
    # args = parser.parse_args()
    
    # if args.command == 'check':
    #     check_metadata_files(args.metadata_scp, args.output, fix_files=args.fix)
    # elif args.command == 'debug':
    #     debug_json_file(args.json_path, fix=args.fix)
    # else:
    #     parser.print_help()

# def main_single():

if __name__ == '__main__':
    # main()
    
    json_path = '/export/fs05/arts/dataset/AV-Deepfake1M-PlusPlus/train/lrs3/pY6ulSP70HY/00017/00017.json'
    # json_path = sys.argv[1]
    debug_json_file(json_path, fix=True)
    
    
