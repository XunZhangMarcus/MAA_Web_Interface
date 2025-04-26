from time_series_maa import GCA_time_series
import torch

# 初始化参数
args = None  # 根据实际情况传入参数
N_pairs = 3  # 生成器或对抗器的个数
batch_size = 32
num_epochs = 100
generators_names = ['gru', 'lstm', 'transformer']
discriminators_names = None
ckpt_dir = 'ckpt/20250425_102916' #NEEDED CKPT PATH
output_dir = 'output'
window_sizes = [5, 10, 15]
initial_learning_rate = 2e-5
train_split = 0.7
do_distill_epochs = 1
cross_finetune_epochs = 5
precise = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 3407
ckpt_path = "./ckpt/20250425_102916"
gan_weights = None

# 实例化 GCA_time_series 类
gca = GCA_time_series(args, N_pairs, batch_size, num_epochs,
                      generators_names, discriminators_names,
                      ckpt_dir, output_dir,
                      window_sizes,
                      initial_learning_rate,
                      train_split,
                      do_distill_epochs, cross_finetune_epochs,
                      precise,
                      device,
                      seed,
                      ckpt_path,
                      gan_weights)

# 处理数据
data_path = 'database/processed_美元指数_day.csv'
start_row = 1060
end_row = 4284
target_columns = [1]
feature_columns_list = [list(range(1, 19)),list(range(1, 19)),list(range(1, 19)),]
log_diff = True
gca.process_data(data_path, start_row, end_row, target_columns, feature_columns_list, log_diff)

# 初始化数据加载器
gca.init_dataloader()

# 初始化模型
num_cls = 3  # 分类数
gca.init_model(num_cls)

# 进行预测
results = gca.pred()

# 提取训练集和测试集的预测值
train_preds = results["train_mse_per_target"]
test_preds = results["test_mse_per_target"]

# 打印预测值
print("Train Predictions:", train_preds)
print("Test Predictions:", test_preds)