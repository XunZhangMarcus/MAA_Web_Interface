import argparse
import torch
from time_series_maa import MAA_time_series


# 创建命令行参数解析器
def parse_args():
    parser = argparse.ArgumentParser(description="Run MAA Time Series Model")

    # 添加命令行参数
    parser.add_argument('--ckpt_dir', type=str, required=True, help="Checkpoint directory path")
    parser.add_argument('--output_dir', type=str, default='output', help="Directory to save output")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the processed data file")
    parser.add_argument('--start_row', type=int, required=True, help="Start row for data processing")
    parser.add_argument('--end_row', type=int, required=True, help="End row for data processing")
    parser.add_argument('--target_columns', type=int, nargs='+', default=[1], help="Columns for target variables")
    parser.add_argument('--feature_columns', type=int, nargs='+', default=list(range(1, 19)),
                        help="Columns for feature variables")
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--train_split', type=float, default=0.7, help="Train-test split ratio")
    parser.add_argument('--do_distill_epochs', type=int, default=1, help="Number of distill epochs")
    parser.add_argument('--cross_finetune_epochs', type=int, default=5, help="Number of cross finetune epochs")
    parser.add_argument('--initial_learning_rate', type=float, default=2e-5, help="Initial learning rate")
    parser.add_argument('--precise', type=str, choices=['float32', 'float64'], default='float32',
                        help="Precision of the model")
    parser.add_argument('--seed', type=int, default=3407, help="Random seed")
    parser.add_argument('--num_cls', type=int, default=3, help="Number of classes for classification")
    parser.add_argument('--generators_names', type=str, nargs='+', default=['gru', 'lstm', 'transformer'],
                        help="List of generators names")
    parser.add_argument('--discriminators_names', type=str, nargs='+', default=None,
                        help="List of discriminators names (optional)")

    return parser.parse_args()


# 主执行函数
def main():
    # 解析命令行参数
    args = parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 根据命令行参数初始化 MAA_time_series 类
    gca = MAA_time_series(args, N_pairs=3, batch_size=args.batch_size, num_epochs=args.num_epochs,
                          generators_names=args.generators_names, discriminators_names=args.discriminators_names,
                          ckpt_dir=args.ckpt_dir, output_dir=args.output_dir,
                          window_sizes=[5, 10, 15], initial_learning_rate=args.initial_learning_rate,
                          train_split=args.train_split, do_distill_epochs=args.do_distill_epochs,
                          cross_finetune_epochs=args.cross_finetune_epochs,
                          precise=torch.float32 if args.precise == 'float32' else torch.float64,
                          device=device, seed=args.seed, ckpt_path=args.ckpt_dir, gan_weights=None)

    # 处理数据
    gca.process_data(args.data_path, args.start_row, args.end_row, args.target_columns, [args.feature_columns] * 3,
                     log_diff=False)

    # 初始化数据加载器
    gca.init_dataloader()

    # 初始化模型
    gca.init_model(args.num_cls)

    # 进行预测
    gca.pred()

    # 提取训练集和测试集的预测值
    # train_preds = results["train_mse_per_target"]
    # test_preds = results["test_mse_per_target"]
    #
    # # 打印预测值
    # print("Train Predictions:", train_preds)
    # print("Test Predictions:", test_preds)


if __name__ == "__main__":
    main()
