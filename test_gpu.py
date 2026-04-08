#!/usr/bin/env python3
"""
综合性能测试脚本：对比 GPU / CPU / IO 性能
使用方法：python benchmark.py [--skip-gpu] [--skip-cpu] [--skip-io] [--output file.txt]
"""

import argparse
import sys
import subprocess
import time
import os
import platform
import torch
import numpy as np

# ------------------------------------------------------------
# 系统信息收集
# ------------------------------------------------------------
def get_system_info():
    info = {}
    info["python"] = sys.version.replace("\n", " ")
    info["torch"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()
    info["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else "N/A"
    info["gpu_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if torch.cuda.is_available():
        info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(info["gpu_count"])]
    else:
        info["gpu_names"] = []
    
    # 获取驱动版本和更多 GPU 信息（通过 nvidia-smi）
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,name,pci.bus_id", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            info["nvidia_driver"] = lines[0].split(",")[0].strip() if lines else "N/A"
            info["nvidia_gpus"] = [line.split(",")[1].strip() for line in lines if line]
        else:
            info["nvidia_driver"] = "N/A"
            info["nvidia_gpus"] = []
    except:
        info["nvidia_driver"] = "N/A"
        info["nvidia_gpus"] = []
    
    # CPU 信息
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                cpuinfo = f.read()
            model = [line.split(":")[1].strip() for line in cpuinfo.split("\n") if "model name" in line]
            if model:
                info["cpu_model"] = model[0]
            else:
                info["cpu_model"] = "Unknown"
            # 核心数
            info["cpu_cores"] = os.cpu_count()
        except:
            info["cpu_model"] = platform.processor()
            info["cpu_cores"] = os.cpu_count()
    else:
        info["cpu_model"] = platform.processor()
        info["cpu_cores"] = os.cpu_count()
    
    return info

# ------------------------------------------------------------
# GPU 计算性能测试
# ------------------------------------------------------------
def test_gpu_compute(device, size=10000, iterations=50):
    """测试矩阵乘法的吞吐量 (TFLOPS)"""
    print(f"\n[GPU Compute] Testing on {device} ...")
    torch.cuda.set_device(device)
    a = torch.randn(size, size, device=device, dtype=torch.float32)
    b = torch.randn(size, size, device=device, dtype=torch.float32)
    
    # 预热
    for _ in range(10):
        torch.mm(a, b)
    torch.cuda.synchronize()
    
    # 计时
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        c = torch.mm(a, b)
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)  # milliseconds
    
    total_ops = 2 * size * size * size * iterations  # 每个矩阵乘法的浮点操作数（近似）
    tflops = (total_ops / (elapsed_ms / 1000)) / 1e12
    print(f"  Matrix multiply {size}x{size} x {iterations} iterations")
    print(f"  Time: {elapsed_ms:.2f} ms -> {tflops:.2f} TFLOPS (FP32)")
    
    # 测试 FP16 混合精度（如果支持）
    if torch.cuda.is_available() and torch.cuda.get_device_capability(device)[0] >= 7:
        a16 = a.half()
        b16 = b.half()
        # 预热
        for _ in range(10):
            torch.mm(a16, b16)
        torch.cuda.synchronize()
        start.record()
        for _ in range(iterations):
            c16 = torch.mm(a16, b16)
        end.record()
        torch.cuda.synchronize()
        elapsed_ms_fp16 = start.elapsed_time(end)
        tflops_fp16 = (total_ops / (elapsed_ms_fp16 / 1000)) / 1e12
        print(f"  FP16 time: {elapsed_ms_fp16:.2f} ms -> {tflops_fp16:.2f} TFLOPS")
    return tflops

# ------------------------------------------------------------
# GPU 显存带宽测试
# ------------------------------------------------------------
def test_gpu_mem_bw(device, size_gb=1, iterations=100):
    """测试 GPU 内部显存拷贝带宽 (GB/s)"""
    print(f"\n[GPU Memory Bandwidth] Testing on {device} ...")
    torch.cuda.set_device(device)
    # 分配两个大小相同的大张量
    num_elements = int(size_gb * 1024**3 / 4)  # 4 bytes per float
    a = torch.randn(num_elements, device=device, dtype=torch.float32)
    b = torch.zeros_like(a)
    
    # 预热
    for _ in range(10):
        b.copy_(a)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        b.copy_(a)
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    
    total_bytes = a.numel() * a.element_size() * iterations * 2  # 读 + 写
    bandwidth_gb_s = (total_bytes / (elapsed_ms / 1000)) / 1e9
    print(f"  Copy {size_gb:.1f} GB x {iterations} times")
    print(f"  Time: {elapsed_ms:.2f} ms -> {bandwidth_gb_s:.2f} GB/s")
    return bandwidth_gb_s

# ------------------------------------------------------------
# PCIe 传输带宽测试
# ------------------------------------------------------------
def test_pcie_bw(device, size_gb=1, iterations=50):
    """测试 CPU <-> GPU 传输带宽"""
    print(f"\n[PCIe Bandwidth] Testing on {device} ...")
    torch.cuda.set_device(device)
    num_elements = int(size_gb * 1024**3 / 4)
    cpu_tensor = torch.randn(num_elements, dtype=torch.float32)
    gpu_tensor = torch.empty_like(cpu_tensor, device=device)
    
    # 预热
    for _ in range(5):
        gpu_tensor.copy_(cpu_tensor)
        cpu_tensor.copy_(gpu_tensor.cpu())
    torch.cuda.synchronize()
    
    # CPU -> GPU
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        gpu_tensor.copy_(cpu_tensor)
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    bytes_transferred = cpu_tensor.nbytes * iterations
    bw_cpu2gpu = (bytes_transferred / (elapsed_ms / 1000)) / 1e9
    print(f"  CPU -> GPU: {size_gb:.1f} GB x {iterations} -> {bw_cpu2gpu:.2f} GB/s")
    
    # GPU -> CPU
    start.record()
    for _ in range(iterations):
        cpu_tensor.copy_(gpu_tensor.cpu())
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    bw_gpu2cpu = (bytes_transferred / (elapsed_ms / 1000)) / 1e9
    print(f"  GPU -> CPU: {size_gb:.1f} GB x {iterations} -> {bw_gpu2cpu:.2f} GB/s")
    return bw_cpu2gpu, bw_gpu2cpu

# ------------------------------------------------------------
# CPU 性能测试
# ------------------------------------------------------------
def test_cpu():
    """测试 CPU 单核和多核浮点性能"""
    print("\n[CPU Performance]")
    import multiprocessing as mp
    
    # 单核浮点运算
    def single_core_work(iterations=200):
        total = 0.0
        for i in range(iterations):
            total += np.sum(np.random.rand(1000)) * 1.0
        return total
    
    # 多核并行
    def multi_core_work(iterations, n_cores):
        with mp.Pool(n_cores) as pool:
            results = pool.map(single_core_work, [iterations] * n_cores)
        return sum(results)
    
    # 单核测试
    start = time.perf_counter()
    single_core_work(iterations=200)
    elapsed_single = time.perf_counter() - start
    print(f"  Single core time: {elapsed_single:.2f} s")
    
    # 多核测试（使用所有核心）
    cores = os.cpu_count()
    if cores > 1:
        start = time.perf_counter()
        multi_core_work(iterations=50, n_cores=cores)
        elapsed_multi = time.perf_counter() - start
        print(f"  Multi core ({cores} cores) time: {elapsed_multi:.2f} s")
        speedup = elapsed_single / elapsed_multi
        print(f"  Speedup: {speedup:.2f}x")
    
    # 简单整数运算测试
    start = time.perf_counter()
    total = 0
    for i in range(10_000_000):
        total += i
    elapsed_int = time.perf_counter() - start
    print(f"  Integer ops (10M additions): {elapsed_int:.3f} s")
    return elapsed_single

# ------------------------------------------------------------
# 磁盘 I/O 测试（可选）
# ------------------------------------------------------------
def test_io(tmp_file="/tmp/benchmark_io.bin", size_mb=500):
    """测试顺序读写速度"""
    print(f"\n[I/O Performance]")
    try:
        size = size_mb * 1024 * 1024
        data = os.urandom(size)
        
        # 写测试
        start = time.perf_counter()
        with open(tmp_file, "wb") as f:
            f.write(data)
        write_time = time.perf_counter() - start
        write_speed = size_mb / write_time
        print(f"  Write {size_mb} MB: {write_time:.2f} s -> {write_speed:.2f} MB/s")
        
        # 读测试
        start = time.perf_counter()
        with open(tmp_file, "rb") as f:
            f.read()
        read_time = time.perf_counter() - start
        read_speed = size_mb / read_time
        print(f"  Read  {size_mb} MB: {read_time:.2f} s -> {read_speed:.2f} MB/s")
        
        os.remove(tmp_file)
    except Exception as e:
        print(f"  I/O test failed: {e}")
    return

# ------------------------------------------------------------
# 主函数
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Performance benchmark for GPU/CPU/IO")
    parser.add_argument("--skip-gpu", action="store_true", help="Skip GPU tests")
    parser.add_argument("--skip-cpu", action="store_true", help="Skip CPU tests")
    parser.add_argument("--skip-io", action="store_true", help="Skip I/O tests")
    parser.add_argument("--output", type=str, help="Save results to file")
    args = parser.parse_args()
    
    # 收集系统信息
    info = get_system_info()
    print("=" * 60)
    print("System Information")
    print("=" * 60)
    for k, v in info.items():
        if isinstance(v, list):
            print(f"{k}:")
            for item in v:
                print(f"  - {item}")
        else:
            print(f"{k}: {v}")
    
    # 开始测试
    print("\n" + "=" * 60)
    print("Starting benchmarks...")
    print("=" * 60)
    
    results = {}
    
    if not args.skip_gpu and torch.cuda.is_available():
        # 对每个 GPU 进行测试（如果多个）
        for i in range(info["gpu_count"]):
            print(f"\n--- Testing GPU {i}: {info['gpu_names'][i]} ---")
            results[f"gpu{i}_compute"] = test_gpu_compute(i)
            results[f"gpu{i}_mem_bw"] = test_gpu_mem_bw(i)
            results[f"gpu{i}_pcie"] = test_pcie_bw(i)
    elif not args.skip_gpu and not torch.cuda.is_available():
        print("\n[Warning] CUDA not available, skipping GPU tests.")
    
    if not args.skip_cpu:
        results["cpu_perf"] = test_cpu()
    
    if not args.skip_io:
        test_io()
    
    # 保存结果
    if args.output:
        with open(args.output, "w") as f:
            f.write("Performance Benchmark Results\n")
            f.write("=" * 60 + "\n")
            for k, v in results.items():
                f.write(f"{k}: {v}\n")
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
