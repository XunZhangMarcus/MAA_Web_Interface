import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
import glob

def plot_true_pred_density(df, output_dir, filename, alpha=0.5, no_grid=False):
    """
    为每个 CSV 文件绘制真实值和预测值的密度估计图
    Args:
        df (pd.DataFrame): 包含 'true' 和 'pred' 列的 DataFrame
        output_dir (str): 保存图形的目录
        filename (str): 原始 CSV 文件名（不含扩展名）
        alpha (float): 面积图的透明度
        no_grid (bool): 是否取消网格线
    """
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(10, 6))

    # 绘制真实值和预测值的 KDE 面积图
    sns.kdeplot(df['true'].dropna(), label='True', color='orange',
                linewidth=0.8, alpha=alpha, fill=True)  # 更改 linewidth
    sns.kdeplot(df['pred'].dropna(), label='Pred', color='blue',
                linewidth=0.8, alpha=alpha, fill=True)  # 更改 linewidth

    # plt.title(f'Density: True vs Pred ({filename})', fontsize=16)
    ax = plt.gca()
    ax.set(xlabel=None, ylabel=None)
    # plt.xlabel('Value', fontsize=14)
    # plt.ylabel('Density', fontsize=14)
    plt.legend()
    if not no_grid:
        plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(output_dir, f'{filename}_true_pred_density.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='为每个 CSV 文件绘制真实值和预测值的密度估计图')
    parser.add_argument('--input_dir', type=str, default='true2pred',
                        help='包含 CSV 文件的目录')
    parser.add_argument('--output_dir', type=str, default='outputs_vis',
                        help='保存图形的目录')
    parser.add_argument('--alpha', type=float, default=0.4,
                        help='面积图的透明度')
    parser.add_argument('--no_grid', default=True,
                        help='添加该参数则取消网格线')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    csv_paths = glob.glob(os.path.join(args.input_dir, '*.csv'))
    if not csv_paths:
        print(f"❌ 未在目录 {args.input_dir} 中找到 CSV 文件。")
        exit(1)

    # for path in csv_paths:
    #     filename = os.path.splitext(os.path.basename(path))[0]
    #     try:
    #         df = pd.read_csv(path)
    #     except Exception as e:
    #         print(f"❌ 无法读取文件 {path}: {e}")
    #         continue
    #
    #     if 'true' not in df.columns or 'pred' not in df.columns:
    #         print(f"⚠️ 文件 {path} 中缺少 'true' 或 'pred' 列，跳过。")
    #         continue
    #
    #     plot_true_pred_density(df, args.output_dir, filename,
    #                            alpha=args.alpha, no_grid=args.no_grid)
    # --- 修改开始 ---

    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(10, 6))  # 在循环外创建一次图形

    all_true_series = []
    pred_series_list = []
    pred_labels = []

    # 第一次循环：读取所有文件，收集数据
    print("正在读取并收集数据...")
    for path in csv_paths:
        filename = os.path.splitext(os.path.basename(path))[0]
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"❌ 无法读取文件 {path}: {e}")
            continue

        if 'true' not in df.columns or 'pred' not in df.columns:
            print(f"⚠️ 文件 {path} 中缺少 'true' 或 'pred' 列,跳过。")
            continue

        # 收集真实值和预测值数据
        all_true_series.append(df['true'].dropna())
        pred_series_list.append(df['pred'].dropna())
        pred_labels.append(filename)  # 使用文件名作为预测分布的标签

    if not all_true_series:
        print("❌ 未在任何文件中找到有效数据。")
        plt.close()  # 关闭之前创建的空 figure
        exit(1)

    # 合并所有真实值数据并绘制其总体的密度分布
    combined_true = pd.concat(all_true_series).dropna()
    if not combined_true.empty:
        # 使用你提到的更好看的样式：细边界线 (linewidth=1.5)，半透明填充 (alpha=args.alpha)
        sns.kdeplot(combined_true, label='True (Combined)', color='orange',
                    linewidth=1.5, alpha=args.alpha, fill=True)
    else:
        print("⚠️ 未找到真实的有效数据，跳过绘制 True 分布。")

    # 绘制每个文件的预测值密度分布
    print("正在绘制所有预测分布...")
    # seaborn 会自动为不同的曲线选择颜色
    # 如果你想控制颜色，可以使用 seaborn.color_palette 或手动指定
    for pred_series, label in zip(pred_series_list, pred_labels):
        if not pred_series.empty:
            # 同样使用细边界线和半透明填充
            sns.kdeplot(pred_series, label=f'Pred ({label})',
                        linewidth=1.5, alpha=args.alpha, fill=True)  # 让 seaborn 自动选择颜色
        else:
            print(f"⚠️ 文件 {label} 中未找到预测的有效数据，跳过绘制。")

    # --- 完成绘图设置 ---
    ax = plt.gca()
    ax.set(xlabel='Value', ylabel='Density')  # 添加轴标签
    # 可以选择添加一个总的标题
    plt.title('Density Distribution: Combined True vs All Predictions', fontsize=16)

    plt.legend()  # 添加图例，显示每个预测对应的文件名
    if not args.no_grid:
        plt.grid(True, linestyle='--', alpha=0.6)  # 可以让网格线更柔和

    plt.tight_layout()

    # 在循环外保存一次图形
    out_path = os.path.join(args.output_dir, 'all_predictions_combined_density.png')
    try:
        plt.savefig(out_path)
        print(f"Saved combined plot: {out_path}")
    except Exception as e:
        print(f"❌ 无法保存图形 {out_path}: {e}")

    plt.close()  # 关闭图形

    # --- 修改结束 ---