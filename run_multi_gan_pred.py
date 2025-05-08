import argparse
import os
import pandas as pd
import torch
import numpy as np
from time_series_maa import MAA_time_series
from utils.logger import setup_experiment_logging
import logging
import json

def load_prediction_config(config_path):
    """加载预测配置文件"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logging.error(f"加载配置文件失败: {str(e)}")
        raise

def run_prediction(args):
    """运行预测的主函数"""
    try:
        # 设置日志
        logger = setup_experiment_logging(args.output_dir, vars(args))
        
        # 初始化MAA模型
        gca = MAA_time_series(args, args.N_pairs, args.batch_size, args.num_epochs,
                            args.generators, args.discriminators,
                            args.ckpt_dir, args.output_dir,
                            args.window_sizes,
                            ckpt_path=args.ckpt_path,
                            initial_learning_rate=args.lr,
                            train_split=args.train_split,
                            do_distill_epochs=args.distill_epochs,
                            cross_finetune_epochs=args.cross_finetune_epochs,
                            device=args.device,
                            seed=args.random_seed)

        # 处理数据
        gca.process_data(args.data_path, args.start_timestamp, args.end_timestamp,
                        args.target_columns[0], args.feature_groups, args.log_diff)
        
        # 初始化数据加载器
        gca.init_dataloader()
        
        # 初始化模型
        gca.init_model(args.num_classes)
        
        # 加载模型权重
        gca.load_model()
        
        # 进行预测
        results = gca.pred()
        
        # 保存预测结果
        output_file = os.path.join(args.output_dir, "prediction_results.csv")
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        
        logging.info(f"预测结果已保存到: {output_file}")
        
        return results

    except Exception as e:
        logging.error(f"预测过程出错: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='运行MAA模型预测')
    
    # 基本参数
    parser.add_argument('--data_path', type=str, required=True, help='输入数据文件路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--ckpt_dir', type=str, required=True, help='检查点目录')
    parser.add_argument('--ckpt_path', type=str, default='latest', help='检查点路径')
    
    # 模型参数
    parser.add_argument('--N_pairs', type=int, default=3, help='GAN对数量')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--window_sizes', type=int, nargs='+', default=[5, 10, 15], help='窗口大小列表')
    parser.add_argument('--num_classes', type=int, default=3, help='分类数量')
    
    # 特征和目标列
    parser.add_argument('--feature_groups', type=list, required=True, help='特征组列表')
    parser.add_argument('--target_columns', type=list, required=True, help='目标列列表')
    
    # 其他参数
    parser.add_argument('--device', type=int, nargs='+', default=[0], help='使用的GPU设备')
    parser.add_argument('--random_seed', type=int, default=3407, help='随机种子')
    parser.add_argument('--log_diff', action='store_true', help='是否使用对数差分')
    parser.add_argument('--start_timestamp', type=int, default=31, help='起始时间戳')
    parser.add_argument('--end_timestamp', type=int, default=-1, help='结束时间戳')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行预测
    results = run_prediction(args)
    
    # 打印预测结果
    print("\n预测结果摘要:")
    print(f"预测结果已保存到: {os.path.join(args.output_dir, 'prediction_results.csv')}")

if __name__ == "__main__":
    main() 