import pandas as pd
import matplotlib.pyplot as plt

# 读取 Excel 文件
file_path = 'sota_contras.xlsx'
df = pd.read_excel(file_path)
for ver in ['8', '5', '9', '10', '11']:
    # 筛选 YOLOv5 和 YOLOv8 系列的数据
    yolo_data = df[df['Model'].str.contains(f'YOLOv{ver}')]

    # 提取参数量和 Rock 值
    yolo_params = yolo_data['Parameters']
    np_params = yolo_params.values

    yolo_rock = yolo_data['Rock']
    np_rock = yolo_rock.values

    yolo_road = yolo_data['Road']
    np_road = yolo_road.values

    # 创建图形
    plt.figure(figsize=(10, 6))

    # 绘制
    plt.plot(np_params, np_rock, marker='o', label=f'YOLOv{ver}')
    plt.plot(np_params, np_road, marker='o', label=f'YOLOv{ver}')
    # 添加图例
    plt.legend()

    # 添加标题和轴标签
    plt.title('Values vs. Parameters')
    plt.xlabel('Parameters (Millions)')
    plt.xlim([0, 90])
    plt.ylabel('Rock Value')
    plt.ylim([0, 1])
    # 显示网格
    plt.grid(True)
    # 显示图形
    plt.show()
    correlation_coefficient_rock = yolo_params.corr(yolo_rock)
    correlation_coefficient_road = yolo_params.corr(yolo_road)
    # 计算标准差
    std_rock_dev = yolo_rock.std()
    std_road_dev = yolo_road.std()
    # 计算变差系数（标准差除以均值）
    cv_rock = std_rock_dev / yolo_rock.mean()
    cv_road = std_road_dev / yolo_road.mean()

    print(f"YOLOv{ver} 系列的参数量与 Rock 值之间的相关系数为: {correlation_coefficient_rock:.4f}")
    print(f"YOLOv{ver} 系列的参数量与 Road 值之间的相关系数为: {correlation_coefficient_road:.4f}")
    print(f"YOLOv{ver} Rock标准差为: {std_rock_dev:.4f}")
    print(f"YOLOv{ver} Road标准差为: {std_road_dev:.4f}")
    print(f"YOLOv{ver} Rock变差系数为: {cv_rock:.4f}")
    print(f"YOLOv{ver} Road变差系数为: {cv_road:.4f}")