from utils.evaluate_visualization import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_density_combined(all_true_series_list, pred_series_list_list, pred_labels_list, output_dir, alpha=0.6, no_grid=False,
                 custom_legend=None):
    """
    Plot density distributions for true values and predictions in a 2x2 grid, with custom legend.

    Parameters:
    - all_true_series_list: List of true data series (each element can be a list of pandas Series).
    - pred_series_list_list: List of lists of predicted data series (each element is a list of pandas Series).
    - pred_labels_list: List of lists of prediction labels for the predictions.
    - output_dir: Directory where the plot should be saved.
    - alpha: Alpha (transparency) for the plot.
    - no_grid: Whether to show gridlines in the plot.
    - custom_legend: A dictionary of custom legends for each subplot.
                      Example: {'(a)': ['True', 'Pred1'], '(b)': ['True', 'Pred2']}, etc.
    """
    # 创建 2x2 的子图布局
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    plt.rcParams.update({'font.size': 12})

    # 将子图扁平化，方便迭代
    axs = axs.flatten()

    # 绘制每个子图
    print("正在绘制所有真实数据和预测数据的密度分布...")
    for i, (combined_true, pred_series_list, pred_labels) in enumerate(
            zip(all_true_series_list, pred_series_list_list, pred_labels_list)):
        # 合并所有的真实数据
        combined_true_data = pd.concat(combined_true).dropna()
        if not combined_true_data.empty:
            sns.kdeplot(combined_true_data, label='True (Combined)', color='orange', linewidth=1.5, alpha=alpha,
                        fill=True, ax=axs[i])
        else:
            print(f"⚠️ 文件 {i} 中未找到真实的有效数据，跳过绘制 True 分布。")

        # 绘制预测数据
        for pred_series, label in zip(pred_series_list, pred_labels):

            if not pred_series.empty:
                sns.kdeplot(pred_series, label=f'Pred ({label})', linewidth=1.5, alpha=alpha, fill=True, ax=axs[i])
            else:
                print(f"⚠️ 文件 {label} 中未找到预测的有效数据，跳过绘制。")

        # 在子图上添加标注 (a), (b), (c), (d)
        axs[i].text(0.05, 0.95, f'({chr(97 + i)})', transform=axs[i].transAxes, fontsize=12, verticalalignment='top')

        # 设置每个子图的标签
        axs[i].set(xlabel='Value', ylabel='Density')
        if not no_grid:
            axs[i].grid(True, linestyle='--', alpha=0.6)

        # 如果 custom_legend 传入了对应的子图图例，设置对应图例
        if custom_legend and f'({chr(97 + i)})' in custom_legend:
            legend_labels = custom_legend[f'({chr(97 + i)})']
            axs[i].legend(legend_labels, loc='upper right')

    # 设置总标题
    plt.suptitle('Density Distribution:True Value vs All Predictions', fontsize=16)

    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 保存图像
    out_path = os.path.join(output_dir, 'all_predictions_combined_density.png')
    try:
        plt.savefig(out_path)
        print(f"Saved combined plot: {out_path}")
    except Exception as e:
        print(f"❌ 无法保存图形 {out_path}: {e}")

    plt.close()

CSI300_csv_path = './output/MAA/沪深300_processed'
Methanol_csv_path = './output/MAA/甲醇_processed'
Rebar_csv_path = './output/MAA/螺纹钢_processed'
Soybean_csv_path = './output/MAA/美大豆期货_processed'

data_path=[Soybean_csv_path, Rebar_csv_path, Methanol_csv_path,CSI300_csv_path]
all_true_series_list=[]
pred_series_list_list=[]
pred_labels_list=[]
for path in data_path:
    csv_paths = glob.glob(os.path.join(path, '*.csv'))
    all_true_series, pred_series_list, pred_labels=read_and_collect_data(csv_paths)
    all_true_series_list.append(all_true_series)
    pred_series_list_list.append(pred_series_list)
    pred_labels_list.append(pred_labels)

output_dir = './output'
plot_density_combined(all_true_series_list, pred_series_list_list, pred_labels_list, output_dir, alpha=0.4, no_grid=True)