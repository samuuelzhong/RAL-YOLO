# -*- coding: utf-8 -*-
import streamlit as st
import tempfile
import torch
from RAL_YOLO_main import *
from torch.profiler import profile, record_function, ProfilerActivity

import psutil  # 用于获取系统资源使用情况

def print_computational_cost_report(results):
    """
    打印计算成本报告。

    参数:
        results: measure_computational_cost 返回的字典。
    """
    print("=" * 50)
    print("Computational Cost Report")
    print("=" * 50)
    print(f"Device: {results['device']}")
    print(f"Memory Usage: {results['memory_usage_mb']:.2f} MB")
    print(f"Inference Time: {results['inference_time_sec']:.6f} sec")
    if results['power_usage_w'] is not None:
        print(f"Power Usage: {results['power_usage_w']:.2f} W")
    print("\nProfiler Summary:")
    print(results['profiler_summary'])
    print("=" * 50)

# 参数设置侧边栏
def object_detector_ui():
    st.sidebar.markdown("### 第二步:调整动态感知参数:")
    kernel_size = st.sidebar.slider(
        "空间视野范围",
        min_value=3,  # 起始值改为3
        max_value=7,  # 保持7结尾
        value=3,  # 默认值保持3
        step=2  # 设置步长为2
    )

    lateFN = st.sidebar.slider(
        "时间感知步长",
        min_value=5,  # 起始值改为5
        max_value=8,  # 保持8结尾
        value=5,  # 默认值保持5
        step=1  # 步长设置为1
    )
    return kernel_size, lateFN

# 主应用逻辑
def run_the_app():
    st.sidebar.markdown("### 第一步：选择本地视频文件（mp4/avi/mov）")
    uploaded_file = st.sidebar.file_uploader("上传文件", type=["mp4", "avi", "mov"])

    tmp_file_path = None
    if uploaded_file is not None:
        # 保存上传文件到临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(uploaded_file.read())
            tmp_file_path = tmp.name

    kernel_size, lateFN = object_detector_ui()

    left_column, right_column = st.columns(2)
    _, middle_column, _ = st.sidebar.columns(3)

    if middle_column.button('开始检测') and tmp_file_path is not None:
        try:
            with st.spinner('正在处理视频，请稍候...'):
                output_path = 'result.mp4'
                #process_video(tmp_file_path, output_path)
                weights_path = f"runs\\train\\yolov8n\\weights\\best.pt"
                catch_video(
                    video_index=tmp_file_path,
                    output_video_path=output_path,
                    weights=weights_path,
                    kernel_size=kernel_size,  # (2 * kernel_size + 1)
                    lateFN=lateFN,
                    latethres=lateFN - 1,
                    # outgt_txt='save_txt\\turf_test_gt_test\\',
                    # outpred_txt=f'save_txt\\{slope_type}_test_pred_cj({cj_seed})_{model}\\',
                    # colorjitter=colorjitter,
                    # cj_seed=cj_seed,
                    labelimgformat=False,
                )


                # 显示结果
                with left_column:
                    st.markdown('### 原始视频')
                    st.video(tmp_file_path)

                with right_column:
                    st.markdown('### 检测结果')
                    train_demo = open(output_path, 'rb')
                    video_bytes = train_demo.read()
                    st.video(video_bytes)

            # 清理临时文件
            os.unlink(tmp_file_path)
            if os.path.exists(output_path):
                pass
        except Exception as e:
            st.error(f"处理出错: {str(e)}")
    elif tmp_file_path is None:
        st.warning("请先上传视频文件")


# 主程序
def main():
    st.set_page_config(page_title="RAL-YOLO 落石检测系统", layout="wide")

    st.sidebar.title("控制面板")
    app_mode = st.sidebar.selectbox("选择模式",
                                    ["运行程序", "使用说明", "关于"])

    if app_mode == "运行程序":
        st.title("RAL-YOLO 视频落石检测系统")
        run_the_app()
    elif app_mode == "使用说明":
        st.markdown("""
        # 使用说明
        1. 在侧边栏上传视频文件（支持mp4/avi/mov格式）
        2. 调整动态感知模块的参数\n
            空间范围：数值越小，则对相机晃动鲁棒性越低且计算成本更小\n
            时间步长：数值越小，则对落石运动越敏感且计算成本更小
        3. 点击开始检测按钮
        4. 查看左右分栏的对比结果
        """)
    elif app_mode == "关于":
        st.markdown("""
        # 关于
        当前边坡落石检测的智能模型存在两大局限：易受静态岩石干扰导致误报，且无法实现落石影响区域分割。\n
        针对这些问题，本项目提出RAL-YOLO（Rockfall Alert Launcher - You Only Look Once），\n
        这是一种基于YOLOv8的检测框架。该框架包含两大创新模块：\n
            运动状态分类模块（Motion Classification，MC）：精准识别岩石静态/动态状态；\n
            道路追踪与映射模块（Road Tracking and Mapping，RTM）：基于道路区域实现落石影响范围精确分割。\n
        RAL-YOLO在岩石静态-动态状态分类与影响区域分割任务中性能显著优于基线模型，可覆盖大部分落石灾害\n
        ## 团队成员
        算法、可视化平台与数据集设计：钟方源（项目负责人）\n
        数据采集：崔文博，李天翊，卢富斌，唐诩博，历阳，史雨轩，周宗翰，谢广林\n
        项目指导：庞锐（副教授）\n
        版本：1.0
        """)


if __name__ == "__main__":
    main()