import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

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

with open(r'largesample_txt\dic_test.txt', 'r') as f:

    '''dic format: [seed, map, mp, mr]'''

    dic_map50 = []
    dic_mp = []
    dic_mr = []
    for i, line in enumerate(f):
        line = line.strip()
        line = line.split(':')
        slope_type, seeds = line[0].split('_')
        seeds = seeds[3:-1]
        seeds = '0' if seeds=='None' else seeds
        value = line[1].strip()
        map50, mp, mr = value[2:-1].split(',')

        dic_map50.append(float(map50))
        dic_mp.append(float(mp))
        dic_mr.append(float(mr))

print(dic_map50,dic_mp,dic_mr)
# 数据
data = {
    'map50': dic_map50,
    'precision': dic_mp,
    'recall': dic_mr}

# 转换为DataFrame
df = pd.DataFrame(data)

total = [dic_map50, dic_mp, dic_mr]
categories = ['MAP50', 'P', 'R']
colors = ["#00008B","#4169E1","#6495ED"]
# 生成小提琴图
# 创建画布
# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 24  # 设置默认字体大小
plt.figure(figsize=(7.5, 6))

# 绘制小提琴图
for i, (data, label, color) in enumerate(zip(total, categories, colors)):
    ax = plt.subplot(1, 3, i + 1)  # 1 行 6 列，第 i+1 个子图

    # 绘制小提琴图
    violin = ax.violinplot(dataset=data, showmeans=False, showmedians=False, showextrema=False, widths=0.9)
    # 设置颜色
    for pc in violin['bodies']:
        pc.set_facecolor(color)
        #pc.set_edgecolor('black')
        pc.set_alpha(0.7 if i <= 5 else 0.8)
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
        box_element.set_alpha(1 if i <= 5 else 1)

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
plt.savefig('box_violins/map.tiff')
plt.show()

for i, element in enumerate(total):
    results = analyze_data(element)
    print(categories[i], '\n')
    print_results(results)