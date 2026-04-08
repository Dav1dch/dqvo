import torch
import numpy as np
from collections import OrderedDict
from pathlib import Path
import difflib

def compare_checkpoints(checkpoint_path1, checkpoint_path2, 
                       compare_model=True, compare_optimizer=True,
                       detailed=False, verbose=True):
    """
    全面比较两个checkpoint文件
    
    Args:
        checkpoint_path1: 第一个checkpoint文件路径
        checkpoint_path2: 第二个checkpoint文件路径
        compare_model: 是否比较模型参数
        compare_optimizer: 是否比较优化器状态
        detailed: 是否进行详细比较（包括数值差异）
        verbose: 是否打印详细信息
    """
    # 加载checkpoint文件
    ckpt1 = torch.load(checkpoint_path1, map_location='cpu')
    ckpt2 = torch.load(checkpoint_path2, map_location='cpu')
    
    print("=" * 80)
    print(f"检查点比较: {Path(checkpoint_path1).name} vs {Path(checkpoint_path2).name}")
    print("=" * 80)
    
    # 提取各部分
    model_state1 = extract_model_state_dict(ckpt1)
    model_state2 = extract_model_state_dict(ckpt2)
    
    optimizer_state1 = extract_optimizer_state_dict(ckpt1)
    optimizer_state2 = extract_optimizer_state_dict(ckpt2)
    
    # 比较模型参数
    if compare_model and model_state1 and model_state2:
        print("\n" + "=" * 80)
        print("模型参数比较")
        print("=" * 80)
        compare_state_dicts(model_state1, model_state2, "模型", detailed, verbose)
    
    # 比较优化器状态
    if compare_optimizer and optimizer_state1 and optimizer_state2:
        print("\n" + "=" * 80)
        print("优化器状态比较")
        print("=" * 80)
        compare_optimizer_states(optimizer_state1, optimizer_state2, detailed, verbose)
    
    # 比较其他元数据
    print("\n" + "=" * 80)
    print("其他元数据比较")
    print("=" * 80)
    compare_metadata(ckpt1, ckpt2)

def extract_model_state_dict(checkpoint):
    """从checkpoint中提取模型state_dict"""
    if isinstance(checkpoint, dict):
        # 尝试多种常见的模型state_dict键名
        for key in ['state_dict', 'model_state_dict', 'model', 'net', 'weights']:
            if key in checkpoint:
                state_dict = checkpoint[key]
                if isinstance(state_dict, (dict, OrderedDict)):
                    return state_dict
        # 如果没有特定键，检查是否是state_dict本身
        if all(isinstance(k, str) for k in checkpoint.keys()):
            return checkpoint
    elif isinstance(checkpoint, torch.nn.Module):
        return checkpoint.state_dict()
    return None

def extract_optimizer_state_dict(checkpoint):
    """从checkpoint中提取优化器state_dict"""
    if isinstance(checkpoint, dict):
        # 尝试多种常见的优化器state_dict键名
        for key in ['optimizer_state_dict', 'optimizer', 'optim']:
            if key in checkpoint:
                return checkpoint[key]
    return None

def compare_state_dicts(state_dict1, state_dict2, name="", detailed=False, verbose=True):
    """比较两个state_dict"""
    params1 = set(state_dict1.keys())
    params2 = set(state_dict2.keys())
    
    if verbose:
        print(f"{name}参数总数: {len(params1)} vs {len(params2)}")
    
    # 1. 独有参数
    only_in_1 = params1 - params2
    only_in_2 = params2 - params1
    
    if only_in_1:
        print(f"\n只在第一个{name}中的参数 ({len(only_in_1)}个):")
        for i, param in enumerate(sorted(list(only_in_1))[:10]):
            shape = state_dict1[param].shape
            dtype = state_dict1[param].dtype
            print(f"  {i+1:3d}. {param} (形状: {shape}, 类型: {dtype})")
        if len(only_in_1) > 10:
            print(f"  ... 还有{len(only_in_1)-10}个参数")
    
    if only_in_2:
        print(f"\n只在第二个{name}中的参数 ({len(only_in_2)}个):")
        for i, param in enumerate(sorted(list(only_in_2))[:10]):
            shape = state_dict2[param].shape
            dtype = state_dict2[param].dtype
            print(f"  {i+1:3d}. {param} (形状: {shape}, 类型: {dtype})")
        if len(only_in_2) > 10:
            print(f"  ... 还有{len(only_in_2)-10}个参数")
    
    # 2. 共同参数
    common_params = params1 & params2
    if common_params:
        print(f"\n共同参数数量: {len(common_params)}")
        
        # 检查形状差异
        shape_diffs = []
        for param in common_params:
            shape1 = state_dict1[param].shape
            shape2 = state_dict2[param].shape
            if shape1 != shape2:
                shape_diffs.append((param, shape1, shape2))
        
        if shape_diffs:
            print(f"形状不同的参数 ({len(shape_diffs)}个):")
            for i, (param, shape1, shape2) in enumerate(shape_diffs[:5]):
                print(f"  {i+1:3d}. {param}")
                print(f"      第一个形状: {shape1}")
                print(f"      第二个形状: {shape2}")
            if len(shape_diffs) > 5:
                print(f"  ... 还有{len(shape_diffs)-5}个形状不同的参数")
        else:
            print("所有共同参数形状相同")
        
        # 检查类型差异
        type_diffs = []
        for param in common_params:
            dtype1 = state_dict1[param].dtype
            dtype2 = state_dict2[param].dtype
            if dtype1 != dtype2:
                type_diffs.append((param, dtype1, dtype2))
        
        if type_diffs:
            print(f"\n数据类型不同的参数 ({len(type_diffs)}个):")
            for i, (param, dtype1, dtype2) in enumerate(type_diffs[:5]):
                print(f"  {i+1:3d}. {param}: {dtype1} vs {dtype2}")
        
        # 详细比较数值差异
        if detailed and common_params and not shape_diffs:
            compare_parameter_values(state_dict1, state_dict2, common_params)
    
    # 3. 统计信息
    print("\n" + "-" * 40)
    print(f"统计摘要:")
    print(f"  独有参数: {len(only_in_1)} (第一个) / {len(only_in_2)} (第二个)")
    print(f"  共同参数: {len(common_params)}")
    if common_params:
        print(f"  形状差异: {len(shape_diffs)}")
        print(f"  类型差异: {len(type_diffs)}")

def compare_optimizer_states(optim_state1, optim_state2, detailed=False, verbose=True):
    """比较两个优化器状态"""
    print("优化器类型:", type(optim_state1).__name__, "vs", type(optim_state2).__name__)
    
    # 检查基本结构
    if 'state' not in optim_state1 or 'state' not in optim_state2:
        print("警告: 优化器状态结构异常，缺少'state'键")
        return
    
    # 比较param_groups
    print("\nparam_groups比较:")
    pg1 = optim_state1.get('param_groups', [])
    pg2 = optim_state2.get('param_groups', [])
    
    print(f"param_groups数量: {len(pg1)} vs {len(pg2)}")
    
    # 比较每个param_group
    for i in range(min(len(pg1), len(pg2))):
        print(f"\n  param_group {i}:")
        
        # 比较键
        keys1 = set(pg1[i].keys())
        keys2 = set(pg2[i].keys())
        
        common_keys = keys1 & keys2
        only_in_1 = keys1 - keys2
        only_in_2 = keys2 - keys1
        
        if only_in_1:
            print(f"    只在第一个中的键: {sorted(only_in_1)}")
        if only_in_2:
            print(f"    只在第二个中的键: {sorted(only_in_2)}")
        
        # 比较共同键的值
        for key in sorted(common_keys):
            val1 = pg1[i][key]
            val2 = pg2[i][key]
            
            if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
                if torch.equal(val1, val2):
                    print(f"    {key}: 相同 (张量)")
                else:
                    print(f"    {key}: 不同 (张量)")
                    if detailed:
                        diff = torch.abs(val1 - val2)
                        print(f"      最大差异: {diff.max().item():.6f}, 平均差异: {diff.mean().item():.6f}")
            elif val1 == val2:
                print(f"    {key}: 相同 ({val1})")
            else:
                print(f"    {key}: 不同 ({val1} vs {val2})")
    
    # 比较state中的参数状态
    print("\nstate比较:")
    state1 = optim_state1['state']
    state2 = optim_state2['state']
    
    # 获取参数id
    param_ids1 = set(state1.keys())
    param_ids2 = set(state2.keys())
    
    print(f"跟踪的参数数量: {len(param_ids1)} vs {len(param_ids2)}")
    
    # 独有参数id
    only_in_1_ids = param_ids1 - param_ids2
    only_in_2_ids = param_ids2 - param_ids1
    
    if only_in_1_ids:
        print(f"只在第一个中跟踪的参数id: {sorted(list(only_in_1_ids))[:10]}")
        if len(only_in_1_ids) > 10:
            print(f"  ... 还有{len(only_in_1_ids)-10}个")
    
    if only_in_2_ids:
        print(f"只在第二个中跟踪的参数id: {sorted(list(only_in_2_ids))[:10]}")
        if len(only_in_2_ids) > 10:
            print(f"  ... 还有{len(only_in_2_ids)-10}个")
    
    # 共同参数id
    common_ids = param_ids1 & param_ids2
    if common_ids:
        print(f"\n共同跟踪的参数id数量: {len(common_ids)}")
        
        # 比较每个共同参数的状态
        if detailed:
            print("\n共同参数状态详细比较:")
            for i, param_id in enumerate(sorted(list(common_ids))[:5]):  # 只显示前5个
                state1_param = state1[param_id]
                state2_param = state2[param_id]
                
                print(f"\n  参数 {param_id}:")
                
                # 比较状态键
                keys1 = set(state1_param.keys())
                keys2 = set(state2_param.keys())
                
                common_keys = keys1 & keys2
                only_in_1_keys = keys1 - keys2
                only_in_2_keys = keys2 - keys1
                
                if only_in_1_keys:
                    print(f"    只在第一个中的状态键: {sorted(only_in_1_keys)}")
                if only_in_2_keys:
                    print(f"    只在第二个中的状态键: {sorted(only_in_2_keys)}")
                
                # 比较共同键
                for key in sorted(common_keys):
                    val1 = state1_param[key]
                    val2 = state2_param[key]
                    
                    if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
                        if val1.shape != val2.shape:
                            print(f"    {key}: 形状不同 ({val1.shape} vs {val2.shape})")
                        elif torch.equal(val1, val2):
                            print(f"    {key}: 相同 (形状: {val1.shape})")
                        else:
                            diff = torch.abs(val1 - val2)
                            print(f"    {key}: 不同 (形状: {val1.shape})")
                            print(f"      最大差异: {diff.max().item():.6f}, 平均差异: {diff.mean().item():.6f}")
                    elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                        if val1 == val2:
                            print(f"    {key}: 相同 ({val1})")
                        else:
                            print(f"    {key}: 不同 ({val1} vs {val2})")
                    else:
                        print(f"    {key}: 类型不同或无法比较")
        
        # 检查是否有动量缓冲区等常见状态
        if common_ids:
            sample_id = next(iter(common_ids))
            if sample_id in state1:
                sample_state = state1[sample_id]
                print(f"\n状态键示例 (参数 {sample_id}): {list(sample_state.keys())}")

def compare_parameter_values(state_dict1, state_dict2, common_params):
    """比较参数的数值差异"""
    print("\n参数数值差异分析:")
    
    diffs = []
    for param in sorted(common_params):
        tensor1 = state_dict1[param].float()
        tensor2 = state_dict2[param].float()
        
        # 确保形状相同
        if tensor1.shape != tensor2.shape:
            continue
        
        # 计算差异指标
        diff_abs = torch.abs(tensor1 - tensor2)
        mae = diff_abs.mean().item()
        max_diff = diff_abs.max().item()
        
        # 计算相对差异
        with torch.no_grad():
            denominator = torch.abs(tensor1) + 1e-8
            relative_diff = (diff_abs / denominator).mean().item()
        
        # 检查是否完全相同
        if torch.equal(tensor1, tensor2):
            continue
        
        diffs.append((param, mae, max_diff, relative_diff, tensor1.numel()))
    
    if diffs:
        print("有明显数值差异的参数:")
        diffs.sort(key=lambda x: x[1], reverse=True)  # 按MAE降序排序
        
        for i, (param, mae, max_diff, rel_diff, numel) in enumerate(diffs[:10]):  # 显示前10个
            print(f"  {i+1:2d}. {param}")
            print(f"      MAE: {mae:.6e}, 最大差异: {max_diff:.6e}, 相对差异: {rel_diff:.6e}")
            print(f"      参数数量: {numel}")
        
        # 计算总体统计
        total_mae = sum(d[1] for d in diffs) / len(diffs)
        total_max = max(d[2] for d in diffs)
        print(f"\n总体差异统计:")
        print(f"  平均MAE: {total_mae:.6e}")
        print(f"  最大差异: {total_max:.6e}")
        print(f"  有差异的参数数量: {len(diffs)}/{len(common_params)}")
    else:
        print("所有共同参数的数值完全相同")

def compare_metadata(ckpt1, ckpt2):
    """比较checkpoint中的其他元数据"""
    if not isinstance(ckpt1, dict) or not isinstance(ckpt2, dict):
        print("checkpoint不是字典格式，无法比较元数据")
        return
    
    # 提取非模型、非优化器的键
    common_keys = set(ckpt1.keys()) & set(ckpt2.keys())
    
    # 排除已比较的键
    excluded_keys = {'state_dict', 'model_state_dict', 'model', 'net', 'weights',
                     'optimizer_state_dict', 'optimizer', 'optim'}
    
    metadata_keys = [k for k in common_keys if k not in excluded_keys]
    
    if metadata_keys:
        print("元数据比较:")
        for key in sorted(metadata_keys):
            val1 = ckpt1[key]
            val2 = ckpt2[key]
            
            if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
                if torch.equal(val1, val2):
                    print(f"  {key}: 相同 (张量, 形状: {val1.shape})")
                else:
                    print(f"  {key}: 不同 (张量)")
            elif val1 == val2:
                print(f"  {key}: 相同 ({val1})")
            else:
                print(f"  {key}: 不同 ({val1} vs {val2})")
        
        # 检查独有元数据
        only_in_1 = set(ckpt1.keys()) - set(ckpt2.keys()) - excluded_keys
        only_in_2 = set(ckpt2.keys()) - set(ckpt1.keys()) - excluded_keys
        
        if only_in_1:
            print(f"\n只在第一个checkpoint中的元数据: {sorted(only_in_1)}")
        if only_in_2:
            print(f"只在第二个checkpoint中的元数据: {sorted(only_in_2)}")
    else:
        print("没有其他元数据可比较")

def analyze_optimizer_state(optimizer_state):
    """分析优化器状态的详细结构"""
    if not optimizer_state:
        print("没有优化器状态")
        return
    
    print("优化器状态分析:")
    print("-" * 40)
    
    if 'param_groups' in optimizer_state:
        print(f"param_groups数量: {len(optimizer_state['param_groups'])}")
        
        for i, group in enumerate(optimizer_state['param_groups']):
            print(f"\nparam_group {i}:")
            for key, value in group.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: 张量, 形状: {value.shape}, 类型: {value.dtype}")
                elif isinstance(value, list):
                    print(f"  {key}: 列表, 长度: {len(value)}")
                    if key == 'params' and value:
                        print(f"    参数ID: {value[:10]}" + ("..." if len(value) > 10 else ""))
                else:
                    print(f"  {key}: {value}")
    
    if 'state' in optimizer_state:
        state = optimizer_state['state']
        print(f"\nstate中跟踪的参数数量: {len(state)}")
        
        if state:
            # 分析第一个参数的状态
            first_param_id = next(iter(state.keys()))
            first_state = state[first_param_id]
            
            print(f"\n参数 {first_param_id} 的状态示例:")
            for key, value in first_state.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: 张量, 形状: {value.shape}, 类型: {value.dtype}")
                else:
                    print(f"  {key}: {value}")
            
            # 统计常见状态键
            all_state_keys = set()
            for param_state in state.values():
                all_state_keys.update(param_state.keys())
            
            print(f"\n所有状态键: {sorted(all_state_keys)}")

def find_param_mappings(checkpoint_path1, checkpoint_path2, threshold=0.7):
    """尝试自动找到参数名称之间的映射关系"""
    ckpt1 = torch.load(checkpoint_path1, map_location='cpu')
    ckpt2 = torch.load(checkpoint_path2, map_location='cpu')
    
    state_dict1 = extract_model_state_dict(ckpt1)
    state_dict2 = extract_model_state_dict(ckpt2)
    
    if not state_dict1 or not state_dict2:
        print("无法提取模型参数")
        return
    
    params1 = list(state_dict1.keys())
    params2 = list(state_dict2.keys())
    
    print("尝试自动匹配相似参数名:")
    print(f"参数数量: {len(params1)} vs {len(params2)}")
    print(f"相似度阈值: {threshold}")
    
    mappings = []
    matched_params2 = set()
    
    for param1 in sorted(params1):
        best_match = None
        best_score = 0
        best_shape_match = False
        
        for param2 in sorted(params2):
            if param2 in matched_params2:
                continue
            
            # 计算名称相似度
            score = difflib.SequenceMatcher(None, param1, param2).ratio()
            
            # 检查形状是否匹配
            shape1 = state_dict1[param1].shape
            shape2 = state_dict2[param2].shape
            shape_match = shape1 == shape2
            
            # 综合评分（形状匹配更重要）
            final_score = score * 0.4 + (1.0 if shape_match else 0.0) * 0.6
            
            if final_score > best_score:
                best_score = final_score
                best_match = param2
                best_shape_match = shape_match
        
        if best_match and best_score >= threshold:
            matched_params2.add(best_match)
            shape1 = state_dict1[param1].shape
            shape2 = state_dict2[best_match].shape
            shape_info = f"形状: {shape1}" if shape1 == shape2 else f"形状不同 ({shape1} vs {shape2})"
            
            mappings.append((param1, best_match, best_score, shape_info))
    
    if mappings:
        print(f"\n找到 {len(mappings)} 个可能匹配:")
        print("-" * 80)
        
        for param1, param2, score, shape_info in sorted(mappings, key=lambda x: x[2], reverse=True):
            print(f"{param1}")
            print(f"  → {param2}")
            print(f"    相似度: {score:.3f}, {shape_info}")
            print()
    else:
        print("没有找到足够的匹配")
    
    return mappings

# 使用示例
if __name__ == "__main__":
    # 示例1: 完整比较
    print("示例1: 完整比较两个checkpoint")
    compare_checkpoints(
        # checkpoint_path1="checkpoint1.pth",
        # checkpoint_path2="checkpoint2.pth",
        checkpoint_path1="./checkpoints/Exp41-copy/checkpoint_best.pth",
        checkpoint_path2="./checkpoints/Exp48/checkpoint_last.pth",
        compare_model=True,
        compare_optimizer=True,
        detailed=True,
        verbose=True
    )
    
    # 示例2: 只比较模型参数
    print("\n" + "=" * 80)
    print("示例2: 只比较模型参数")
    compare_checkpoints(
        checkpoint_path1="./checkpoints/Exp41-copy/checkpoint_best.pth",
        checkpoint_path2="./checkpoints/Exp48/checkpoint_last.pth",
        compare_model=True,
        compare_optimizer=False,
        detailed=False,
        verbose=True
    )
    
    # 示例3: 分析优化器状态
    print("\n" + "=" * 80)
    print("示例3: 分析优化器状态")
    ckpt = torch.load("./checkpoints/Exp48/checkpoint_last.pth", map_location='cpu')
    optimizer_state = extract_optimizer_state_dict(ckpt)
    if optimizer_state:
        analyze_optimizer_state(optimizer_state)
    
    # # 示例4: 尝试自动匹配参数
    # print("\n" + "=" * 80)
    # print("示例4: 自动匹配参数")
    # find_param_mappings(
    #     # checkpoint_path1="checkpoint1.pth",
    #     # checkpoint_path2="checkpoint2.pth",
    #     checkpoint_path1="./checkpoints/Exp41-copy/checkpoint_best.pth",
    #     checkpoint_path2="./checkpoints/Exp48/checkpoint_best.pth",
    #     threshold=0.6
    # )