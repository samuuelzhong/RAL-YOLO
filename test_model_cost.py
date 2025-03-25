import psutil
import threading
import torch
import time
import os
import cv2
import pynvml
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from RAL_YOLO_main import catch_video
from scipy import stats

# 确保NVML可用
try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

def monitor_resources(interval=0.1):
    """资源监控线程函数"""
    stop_event = threading.Event()
    cpu_usage = []
    mem_usage = []
    power_usage = []  # 新增：GPU功耗监控
    peak_mem = 0
    process = psutil.Process(os.getpid())

    # 初始化GPU功耗监控
    nvml_initialized = False
    if HAS_NVML:
        try:
            pynvml.nvmlInit()
            nvml_initialized = True
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as e:
            print(f"NVML初始化失败: {str(e)}")

    def monitoring_loop():
        nonlocal peak_mem, nvml_initialized
        while not stop_event.is_set():
            try:
                # 获取内存和CPU使用情况
                mem = process.memory_info().rss
                cpu = process.cpu_percent() / psutil.cpu_count()
                mem_mb = mem / (1024 ** 2)

                # 记录CPU和内存数据
                cpu_usage.append(cpu)
                mem_usage.append(mem_mb)
                if mem_mb > peak_mem:
                    peak_mem = mem_mb

                # 获取GPU功耗
                if nvml_initialized:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # 转换为瓦特
                    power_usage.append(power)
            except (psutil.NoSuchProcess, pynvml.NVMLError) as e:
                break
            stop_event.wait(interval)

        # 关闭NVML连接
        if nvml_initialized:
            pynvml.nvmlShutdown()

    monitor_thread = threading.Thread(target=monitoring_loop)
    monitor_thread.daemon = True
    monitor_thread.start()

    return stop_event, cpu_usage, mem_usage, peak_mem, power_usage

def get_video_frame_count(video_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    # 获取帧数
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 释放资源
    cap.release()

    return frame_count

def run_performance_test(test_func, *args, **kwargs):
    """执行性能测试的包装函数"""
    # 初始化GPU统计
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        start_energy = 0
        if HAS_NVML:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                pynvml.nvmlShutdown()
            except Exception as e:
                print(f"能耗监控不可用: {str(e)}")
    else:
        print("CUDA不可用，无法测量GPU内存")

    # 启动监控线程
    stop_event, cpu_usage, mem_usage, peak_mem, power_usage = monitor_resources()

    # 记录开始时间和帧数
    start_time = time.time()
    total_frames = 0  # 需要根据实际视频帧数设置

    try:
        # 执行目标函数
        result = test_func(*args, **kwargs)
        total_frames = get_video_frame_count(video_path)
    finally:
        # 记录结束时间和能耗
        end_time = time.time()
        total_time = end_time - start_time

        # 停止监控线程
        stop_event.set()
        time.sleep(0.5)

        # 获取最终能耗数据
        energy_consumption = 0
        if torch.cuda.is_available() and HAS_NVML:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                end_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                energy_consumption = (end_energy - start_energy) / 1000000  # 转换为千焦
                pynvml.nvmlShutdown()
            except Exception as e:
                print(f"能耗数据获取失败: {str(e)}")

        # 计算GPU内存使用
        gpu_peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

        # 生成增强报告
        print("\n" + "=" * 40)
        print("性能测试报告:")
        print(f"推理延迟: {total_time / total_frames * 1000:.2f} ms")
        print(f"平均CPU使用率: {sum(cpu_usage) / len(cpu_usage):.2f}%" if cpu_usage else "0%")
        print(f"平均内存占用: {sum(mem_usage) / len(mem_usage):.2f} MB" if mem_usage else "0 MB")
        print(f"峰值内存占用: {max(mem_usage):.2f} MB")

        if torch.cuda.is_available():
            print(f"GPU峰值内存占用: {gpu_peak / (1024 ** 2):.2f} MB")
            if power_usage:
                print(f"平均GPU功耗: {sum(power_usage) / len(power_usage):.2f} W")
                print(f"峰值GPU功耗: {max(power_usage):.2f} W" if power_usage else "N/A")
            if energy_consumption > 0:
                print(f"总能耗: {energy_consumption:.2f} kJ")
                print(f"能效比: {total_frames / energy_consumption:.2f} 帧/千焦")
        print("=" * 40 + "\n")


    return {
        "inference_time": total_frames / total_time * 1000,
        "cpu_usage": sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0,
        "peak_mem": max(mem_usage) if mem_usage else 0,
        "peak_gpu_power": max(power_usage) if power_usage else 0,
        "gpu_peak_mem": gpu_peak / (1024 ** 2) if torch.cuda.is_available() else 0,
        "energy_efficiency": total_frames / energy_consumption if energy_consumption > 0 else 0
    }

import json

# 保存 results 数据到 JSON 文件
def save_results_to_json(results, filename="cost_results.json"):
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filename}")

# 从 JSON 文件加载数据
def load_results_from_json(filename="cost_results.json"):
    with open(filename, "r") as f:
        results = json.load(f)
    return results


def analyze_data(column_data):
    results = []
    # 基本统计量
    median = np.median(column_data)
    q1, q3 = np.percentile(column_data, [25, 75])
    iqr = q3 - q1
    std = np.std(column_data, ddof=1)  # 样本标准差

    # 置信区间（95% CI，使用t分布）
    n = len(column_data)
    se = std / np.sqrt(n)  # 标准误
    ci_low, ci_high = stats.t.interval(0.95, df=n - 1, loc=median, scale=se)

    results.append({
        'median': median,
        'iqr': (q1, q3, iqr),
        'ci': (ci_low, ci_high),
        'std': std
    })

    return results


def print_results(results):
    for i, res in enumerate(results):
        print(f"Column {i + 1} Analysis:")
        print(f"  Median: {res['median']:.4f}")
        print(f"  IQR: {res['iqr'][2]:.4f} (Q1={res['iqr'][0]:.4f}, Q3={res['iqr'][1]:.4f})")
        print(f"  95% CI: [{res['ci'][0]:.4f}, {res['ci'][1]:.4f}]")
        print(f"  Std Dev: {res['std']:.4f}\n")


# 使用示例
if __name__ == "__main__":
    '''
    # 测试参数示例
    video_path = r"E:\project\RAL-YOLO\test\ROCKFALL\bare_test.mp4"
    output_path = "cost_test.mp4"
    weights_path = f"runs\\train\\yolov8n\\weights\\best.pt"

    # 重复实验48次
    num_experiments = 48
    results = []

    for i in range(num_experiments):
        print(f"Running experiment {i + 1}/{num_experiments}")
        result = run_performance_test(
            catch_video,
            video_path,
            output_path,
            weights_path,
            kernel_size=3,
            lateFN=5,
            latethres=4,
            colorjitter=False
        )
        results.append(result)
    # 保存数据
    save_results_to_json(results)
    '''

    # 加载数据
    loaded_results = load_results_from_json()
    # 提取数据
    inference_time = np.array([(1/(result["inference_time"]/1000))*1000 for result in loaded_results])
    print(inference_time)
    cpu_usages = np.array([result["cpu_usage"] for result in loaded_results])
    peak_mems = np.array([result["peak_mem"] for result in loaded_results])
    peak_gpu_powers = np.array([result["peak_gpu_power"] for result in loaded_results])
    gpu_peak_mems = np.array([result["gpu_peak_mem"] for result in loaded_results])
    energy_efficiencies = np.array([result["energy_efficiency"] for result in loaded_results])
    # total = [inference_time, cpu_usages, peak_mems, peak_gpu_powers, gpu_peak_mems, energy_efficiencies]
    total = [inference_time, peak_mems, gpu_peak_mems, energy_efficiencies]
    # categories = ['Inference latency', 'CPU usage', 'Peak memory', 'GPU usage', 'GPU power usage', 'Energy efficiency']
    categories = ['Inference latency', 'Peak memory', 'Peak GPU memory', 'Energy efficiency']
    # colors = ["#8e93af","#d7a6b3","#e8cda5","#9dc1c5","#D8BFD8","#7B68EE"]
    colors = ["#8e93af", "#d7a6b3", "#e8cda5", "#9dc1c5"]
    # 生成小提琴图
    # 创建画布
    # 设置全局字体为 Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 24  # 设置默认字体大小
    plt.figure(figsize=(10, 6))

    # 绘制小提琴图
    for i, (data, label, color) in enumerate(zip(total, categories, colors)):
        ax = plt.subplot(1, 4, i + 1)  # 1 行 6 列，第 i+1 个子图

        # 绘制小提琴图
        violin = ax.violinplot(dataset=data, showmeans=False, showmedians=False, showextrema=False, widths=0.9)
        # 设置颜色
        for pc in violin['bodies']:
            pc.set_facecolor(color)
            #pc.set_edgecolor('black')
            pc.set_alpha(0.5)
        # 设置中位数和均值的颜色
        # violin['cmedians'].set_color('red')
        # violin['cmeans'].set_color('green')

        # 绘制箱型图
        box = ax.boxplot(data, positions=[1], widths=0.2, patch_artist=True, showfliers=False,
                         # COLOR
                         capprops=dict(color=color),
                         whiskerprops=dict(color=color),
                         medianprops=dict(color="#f1f5f9"),
                         flierprops=dict(markeredgecolor=color),)
        for box_element in box['boxes']:
            box_element.set_facecolor(color)
            box_element.set_edgecolor('white')
            box_element.set_alpha(0.8)

        # 绘制散点
        x = np.random.normal(1, 0.05, size=len(data))  # 添加随机抖动以避免重叠
        ax.scatter(x, data, color=color, edgecolor="#888888", s=20, zorder=2)

        # 设置标题
        ax.set_title(label, fontsize=12)

        # 隐藏上、下、左的轴线
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # 隐藏 x 轴刻度和标签
        ax.set_xticks([])
        ax.set_xlabel('')

        # 设置网格线
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    # 调整布局
    plt.tight_layout()
    plt.savefig('box_violins/model_cost.tiff')
    plt.show()

    for i, element in enumerate(total):
        results = analyze_data(element)
        print(categories[i],'\n')
        print_results(results)