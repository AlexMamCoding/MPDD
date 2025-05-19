from datetime import datetime  # 导入datetime模块，用于处理日期和时间
import os  # 导入os模块，用于与操作系统进行交互，如文件和目录操作
import json  # 导入json模块，用于处理JSON数据
import time  # 导入time模块，用于处理时间相关的操作
import argparse  # 导入argparse模块，用于解析命令行参数
import torch  # 导入PyTorch库，用于深度学习相关的操作
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score  # 从sklearn库中导入评估指标计算函数
from torch.utils.data import DataLoader  # 从PyTorch的data模块中导入DataLoader，用于数据加载和批处理
from train_val_split import train_val_split1, train_val_split2  # 从自定义模块中导入训练集和验证集划分函数
from models.our.our_model import ourModel  # 从自定义模块中导入自定义的模型类
from dataset import *  # 从自定义的dataset模块中导入所有内容（具体功能需看该模块定义）
from utils.logger import get_logger  # 从自定义的logger模块中导入获取日志记录器的函数
import numpy as np  # 导入NumPy库，用于数值计算


class Opt:
    def __init__(self, config_dict):
        # 将配置字典中的键值对更新到对象的属性中
        self.__dict__.update(config_dict)


def load_config(config_file):
    with open(config_file, 'r') as f:
        # 读取配置文件并将其解析为JSON格式返回
        return json.load(f)


def eval(model, val_loader, device):
    model.eval()  # 将模型设置为评估模式
    total_emo_pred = []  # 用于存储所有预测的情感标签
    total_emo_label = []  # 用于存储所有真实的情感标签

    with torch.no_grad():  # 不计算梯度，用于评估阶段
        for i, data in enumerate(val_loader):  # 遍历验证数据加载器中的数据
            # 打印批次索引，方便追踪
            # print(f"\n批次 {i+1}/{len(val_loader)}")

            for k, v in data.items():
                data[k] = v.to(device)  # 将数据移动到指定设备上（如GPU或CPU）
            model.set_input(data)  # 将输入数据设置到模型中
            model.test()  # 对输入数据进行测试
             # 检查模型输出
            # print(f"model.emo_pred 形状: {model.emo_pred.shape}")  # 应是 [batch_size, num_classes]
            # print(f"model.emo_pred 示例: {model.emo_pred[0].cpu().numpy()}")  # 应是概率分布
            
            # emo_pred = model.emo_pred.argmax(dim=1).cpu().numpy()  # 获取预测的情感标签并转换为NumPy数组
            # emo_label = data['emo_label'].cpu().numpy()  # 获取真实的情感标签并转换为NumPy数组
             # 处理多时间步预测：对时间步维度取平均后再求类别预测
            # model.emo_pred 形状为 [batch_size, sequence_length, num_classes]
            emo_pred_time_avg = model.emo_pred.mean(dim=1)  # 对时间步维度求平均，形状变为 [batch_size, num_classes]
            
            emo_pred = emo_pred_time_avg.argmax(dim=1).cpu().numpy()  # 获取预测的情感标签
            
            emo_label = data['emo_label'].cpu().numpy()  # 获取真实的情感标签
            # 将概率转换为类别索引（关键修改）
             # 打印预测和标签的形状与内容
            # print(f"emo_pred 形状: {emo_pred.shape}")
            # print(f"emo_pred 示例: {emo_pred[:5]}")  # 应是 [0, 1, 1, 0, ...]
            # print(f"emo_label 形状: {emo_label.shape}")
            # print(f"emo_label 示例: {emo_label[:5]}")
            
            
            total_emo_pred.append(emo_pred)  # 将预测标签添加到列表中
            total_emo_label.append(emo_label)  # 将真实标签添加到列表中

    total_emo_pred = np.concatenate(total_emo_pred)  # 将所有预测标签连接成一个数组
    total_emo_label = np.concatenate(total_emo_label)  # 将所有真实标签连接成一个数组

    emo_acc_unweighted = accuracy_score(total_emo_label, total_emo_pred, sample_weight=None)  # 计算未加权的准确率
    class_counts = np.bincount(total_emo_label)  # 获取每个类别的样本数量
    sample_weights = 1 / (class_counts[total_emo_label] + 1e-6)  # 计算每个样本的权重，避免除零错误
    emo_acc_weighted = accuracy_score(total_emo_label, total_emo_pred, sample_weight=sample_weights)  # 计算加权的准确率

    emo_f1_weighted = f1_score(total_emo_label, total_emo_pred, average='weighted')  # 计算加权的F1值
    emo_f1_unweighted = f1_score(total_emo_label, total_emo_pred, average='macro')  # 计算未加权的F1值
    emo_cm = confusion_matrix(total_emo_label, total_emo_pred)  # 计算混淆矩阵

    return total_emo_label, total_emo_pred, emo_acc_weighted, emo_acc_unweighted, emo_f1_weighted, emo_f1_unweighted, emo_cm


def train_model(train_json, model, audio_path='', video_path='', max_len=5,
                best_model_name='best_model.pth', seed=None, args = None):
    """
    This is the traing function
    """
    # 检查 args 是否为空（确保参数已传递）
    if args is None:
        raise ValueError("必须传递 args 参数")
    
    logger.info(f'personalized features used：{args.personalized_features_file}')  # 记录使用的个性化特征文件
    num_epochs = args.num_epochs  # 获取训练的轮数
    device = args.device  # 获取使用的设备（如GPU或CPU）
    print(f"device: {device}")  # 打印使用的设备信息
    model.to(device)  # 将模型移动到指定设备上

    # split training and validation set
    # data = json.load(open(train_json, 'r'))
    if args.track_option == 'Track1':
        # 根据Track1的方式划分训练集和验证集
        train_data, val_data, train_category_count, val_category_count = train_val_split1(train_json, val_ratio=0.1, random_seed=seed)
    elif args.track_option == 'Track2':
        # 根据Track2的方式划分训练集和验证集
        train_data, val_data, train_category_count, val_category_count = train_val_split2(train_json, val_percentage=0.1,
                                                                                     seed=seed)

    train_loader = DataLoader(
        # 创建训练数据加载器
        AudioVisualDataset(train_data, args.labelcount, args.personalized_features_file, max_len,
                           batch_size=args.batch_size,
                           audio_path=audio_path, video_path=video_path), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(
        # 创建验证数据加载器
        AudioVisualDataset(val_data, args.labelcount, args.personalized_features_file, max_len,
                           batch_size=args.batch_size,
                           audio_path=audio_path, video_path=video_path), batch_size=args.batch_size, shuffle=False)

    logger.info('The number of training samples = %d' % len(train_loader.dataset))  # 记录训练样本数量
    logger.info('The number of val samples = %d' % len(val_loader.dataset))  # 记录验证样本数量

    best_emo_acc = 0.0  # 用于存储最佳的情感准确率
    best_emo_f1 = 0.0  # 用于存储最佳的情感F1值
    best_emo_epoch = 1  # 用于存储最佳模型的训练轮数
    best_emo_cm = []  # 用于存储最佳模型的混淆矩阵

    for epoch in range(num_epochs):  # 遍历训练轮数
        model.train(True)  # 将模型设置为训练模式
        total_loss = 0  # 用于累加训练损失

        for i, data in enumerate(train_loader):  # 遍历训练数据加载器中的数据
            for k, v in data.items():
                data[k] = v.to(device)  # 将数据移动到指定设备上
            model.set_input(data)  # 将输入数据设置到模型中
            model.optimize_parameters(epoch)  # 优化模型参数

            losses = model.get_current_losses()  # 获取当前的损失
            total_loss += losses['emo_CE']  # 累加情感分类的交叉熵损失

        avg_loss = total_loss / len(train_loader)  # 计算平均损失

        # evaluation
        label, pred, emo_acc_weighted, emo_acc_unweighted, emo_f1_weighted, emo_f1_unweighted, emo_cm = eval(model, val_loader,
                                                                                                device)  # 在验证集上评估模型

        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.10f}, "
                    f"Weighted F1: {emo_f1_weighted:.10f}, Unweighted F1: {emo_f1_unweighted:.10f}, "
                    f"Weighted Acc: {emo_acc_weighted:.10f}, Unweighted Acc: {emo_acc_unweighted:.10f}")  # 记录训练信息
        logger.info('Confusion Matrix:\n{}'.format(emo_cm))  # 记录混淆矩阵

        if emo_f1_weighted > best_emo_f1:  # 如果当前的加权F1值大于之前的最佳值
            cur_time = time.strftime('%Y-%m-%d-%H.%M.%S', time.localtime(time.time()))  # 获取当前时间
            best_emo_f1 = emo_f1_weighted  # 更新最佳的加权F1值
            best_emo_f1_unweighted = emo_f1_unweighted  # 更新最佳的未加权F1值
            best_emo_acc = emo_acc_weighted  # 更新最佳的加权准确率
            best_emo_acc_unweighted = emo_acc_unweighted  # 更新最佳的未加权准确率
            best_emo_cm = emo_cm  # 更新最佳的混淆矩阵
            best_emo_epoch = epoch + 1  # 更新最佳的训练轮数
            best_model = model  # 更新最佳模型
            save_path = os.path.join(os.path.join(opt.checkpoints_dir, opt.name), best_model_name)  # 构建模型保存路径
            torch.save(model.state_dict(), save_path)  # 保存最佳模型的参数
            print("Saved best model.")  # 打印保存模型的信息

    logger.info(f"Training complete. Random seed: {seed}. Best epoch: {best_emo_epoch}.")  # 记录训练完成信息
    logger.info(f"Best Weighted F1: {best_emo_f1:.4f}, Best Unweighted F1: {best_emo_f1_unweighted:.4f}, "
                f"Best Weighted Acc: {best_emo_acc:.4f}, Best Unweighted Acc: {best_emo_acc_unweighted:.4f}.")  # 记录最佳评估指标
    logger.info('Confusion Matrix:\n{}'.format(best_emo_cm))  # 记录最佳模型的混淆矩阵

    # output results to CSV
    csv_file = f'{opt.log_dir}/{opt.name}.csv'  # 构建CSV文件路径
    formatted_best_emo_cm = ' '.join([f"[{' '.join(map(str, row))}]" for row in best_emo_cm])  # 格式化混淆矩阵
    header = f"Time,random seed,splitwindow_time,labelcount,audiofeature_method,videofeature_method," \
             f"batch_size,num_epochs,feature_max_len,lr," \
             f"Weighted_F1,Unweighted_F1,Weighted_Acc,Unweighted_Acc,Confusion_Matrix"  # CSV文件头
    result_value = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{seed},{args.splitwindow_time},{args.labelcount},{args.audiofeature_method},{args.videofeature_method}," \
                   f"{args.batch_size},{args.num_epochs},{opt.feature_max_len},{opt.lr:.6f}," \
                   f"{best_emo_f1:.4f},{best_emo_f1_unweighted:.4f},{best_emo_acc:.4f},{best_emo_acc_unweighted:.4f},{formatted_best_emo_cm}"  # 构建结果字符串
    file_exists = os.path.exists(csv_file)  # 检查CSV文件是否存在
    # Open file (append if file exists, create if it doesn't)
    with open(csv_file, mode='a') as file:  # 以追加模式打开CSV文件
        if not file_exists:
            file.write(header + '\n')  # 如果文件不存在，写入文件头
        file.write(result_value + '\n')  # 写入结果字符串

    return best_emo_f1, best_emo_f1_unweighted, best_emo_acc, best_emo_acc_unweighted, best_emo_cm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MDPP Model")  # 创建命令行参数解析器
    parser.add_argument('--labelcount', type=int, default=3,
                        help="Number of data categories (2, 3, or 5).")  # 添加标签数量参数
    parser.add_argument('--track_option', type=str, required=True,
                        help="Track1 or Track2")  # 添加任务选项参数
    parser.add_argument('--feature_max_len', type=int, required=True,
                        help="Max length of feature.")  # 添加特征最大长度参数
    parser.add_argument('--data_rootpath', type=str, required=True,
                        help="Root path to the program dataset")  # 添加数据根路径参数
    parser.add_argument('--train_json', type=str, required=False,
                        help="File name of the training JSON file")  # 添加训练JSON文件参数
    parser.add_argument('--personalized_features_file', type=str,
                        help="File name of the personalized features file")  # 添加个性化特征文件参数
    parser.add_argument('--audiofeature_method', type=str, default='mfccs',
                        choices=['mfccs', 'opensmile', 'wav2vec'],
                        help="Method for extracting audio features.")  # 添加音频特征提取方法参数
    parser.add_argument('--videofeature_method', type=str, default='densenet',
                        choices=['openface','resnet', 'densenet'],
                        help="Method for extracting video features.")  # 添加视频特征提取方法参数
    parser.add_argument('--splitwindow_time', type=str, default='1s',
                        help="Time window for splitted features. e.g. '1s' or '5s'")  # 添加特征分割时间窗口参数

    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for training")  # 添加训练批次大小参数
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Learning rate")  # 添加学习率参数
    parser.add_argument('--num_epochs', type=int, default=10,
                        help="Number of epochs to train the model")  # 添加训练轮数参数
    parser.add_argument('--device', type=str, default='cpu',
                        help="Device to train the model on, e.g. 'cuda' or 'cpu'")  # 添加训练设备参数

    args = parser.parse_args()  # 解析命令行参数

    args.train_json = os.path.join(args.data_rootpath, 'Training', 'labels', 'Training_Validation_files.json')  # 构建训练JSON文件路径
    args.personalized_features_file = os.path.join(args.data_rootpath, 'Training', 'individualEmbedding', 'descriptions_embeddings_with_ids.npy')  # 构建个性化特征文件路径

    config = load_config('config.json')  # 加载配置文件
    opt = Opt(config)  # 创建配置对象

    # Modify individual dynamic parameters in opt according to task category
    opt.emo_output_dim = args.labelcount  # 根据命令行参数更新模型的情感输出维度
    opt.feature_max_len = args.feature_max_len  # 根据命令行参数更新特征最大长度
    opt.lr = args.lr  # 根据命令行参数更新学习率

    # Splice out feature folder paths according to incoming audio and video feature types
    audio_path = os.path.join(args.data_rootpath, 'Training', f"{args.splitwindow_time}", 'Audio', f"{args.audiofeature_method}") + '/'  # 构建音频特征文件夹路径
    video_path = os.path.join(args.data_rootpath, 'Training', f"{args.splitwindow_time}", 'Visual', f"{args.videofeature_method}") + '/'  # 构建视频特征文件夹路径

    # Obtain input_dim_a, input_dim_v
    for filename in os.listdir(audio_path):
        if filename.endswith('.npy'):
            opt.input_dim_a = np.load(audio_path + filename).shape[1]  # 获取音频特征的输入维度
            break

    for filename in os.listdir(video_path):
        if filename.endswith('.npy'):
            opt.input_dim_v = np.load(video_path + filename).shape[1]  # 获取视频特征的输入维度
            break

    opt.name = f'{args.splitwindow_time}_{args.labelcount}labels_{args.audiofeature_method}+{args.videofeature_method}'  # 构建模型名称
    logger_path = os.path.join(opt.log_dir, opt.name)  # 构建日志记录器路径
    if not os.path.exists(opt.log_dir):
        os.mkdir(opt.log_dir)  # 创建日志目录
    if not os.path.exists(logger_path):
        os.mkdir(logger_path)  # 创建日志记录器路径下的目录
    logger = get_logger(logger_path,'result')  # 获取日志记录器

    model = ourModel(opt)  # 创建模型实例

    cur_time = time.strftime('%Y-%m-%d-%H.%M.%S', time.localtime(time.time()))  # 获取当前时间
    best_model_name = f"best_model_{cur_time}.pth"  # 构建最佳模型文件名

    logger.info(f"splitwindow_time={args.splitwindow_time}, audiofeature_method={args.audiofeature_method}, "
                f"videofeature_method={args.videofeature_method}")  # 记录特征分割时间窗口、音频和视频特征提取方法
    logger.info(f"batch_size={args.batch_size}, num_epochs={args.num_epochs}, "
                f"labels={opt.emo_output_dim}, feature_max_len={opt.feature_max_len}, lr={opt.lr}")  # 记录训练参数

    # set random seed
    # seed = np.random.randint(0, 10000) 
    seed = 3407  # 设置随机种子
    np.random.seed(seed)  # 设置NumPy的随机种子
    torch.manual_seed(seed)  
# 设置CPU上的随机种子，确保每次运行代码时，CPU上的随机数生成器产生相同的随机数序列，
# 从而使得模型初始化、数据采样等涉及随机数的操作具有可重复性。

torch.cuda.manual_seed_all(seed)  
# 如果使用GPU进行计算，该函数设置所有GPU上的随机种子。
# 确保在多GPU环境下，每次运行代码时，GPU上的随机数生成器也产生相同的随机数序列，
# 保证了模型训练过程中涉及GPU的随机操作的可重复性。

logger.info(f"Using random seed: {seed}")  
# 使用日志记录器记录当前使用的随机种子值，方便查看和记录实验设置。

# training
train_model(
    train_json=args.train_json,  # 传入训练数据的JSON文件路径
    model=model,  # 传入要训练的模型实例
    max_len=opt.feature_max_len,  # 传入特征的最大长度
    best_model_name=best_model_name,  # 传入最佳模型的保存文件名
    audio_path=audio_path,  # 传入音频特征文件的路径
    video_path=video_path,  # 传入视频特征文件的路径
    seed=seed  # 传入随机种子，可能在训练函数中用于数据划分等涉及随机数的操作
)
from datetime import datetime
import os
import json
import time
import argparse
import torch
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader
from train_val_split import train_val_split1, train_val_split2
from models.our.our_model import ourModel
from dataset import *
from utils.logger import get_logger
import numpy as np

class Opt:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

def load_config(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)

def eval(model, val_loader, device):
    model.eval()
    total_emo_pred = []
    total_emo_label = []

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            for k, v in data.items():
                data[k] = v.to(device)
            model.set_input(data)
            model.test()
            
            emo_pred_time_avg = model.emo_pred.mean(dim=1)
            emo_pred = emo_pred_time_avg.argmax(dim=1).cpu().numpy()
            emo_label = data['emo_label'].cpu().numpy()
            
            total_emo_pred.append(emo_pred)
            total_emo_label.append(emo_label)

    total_emo_pred = np.concatenate(total_emo_pred)
    total_emo_label = np.concatenate(total_emo_label)

    emo_acc_unweighted = accuracy_score(total_emo_label, total_emo_pred, sample_weight=None)
    class_counts = np.bincount(total_emo_label)
    sample_weights = 1 / (class_counts[total_emo_label] + 1e-6)
    emo_acc_weighted = accuracy_score(total_emo_label, total_emo_pred, sample_weight=sample_weights)

    emo_f1_weighted = f1_score(total_emo_label, total_emo_pred, average='weighted')
    emo_f1_unweighted = f1_score(total_emo_label, total_emo_pred, average='macro')
    emo_cm = confusion_matrix(total_emo_label, total_emo_pred)

    return total_emo_label, total_emo_pred, emo_acc_weighted, emo_acc_unweighted, emo_f1_weighted, emo_f1_unweighted, emo_cm

def train_model(train_json, model, audio_path='', video_path='', max_len=5,
                best_model_name='best_model.pth', seed=None, args=None, opt=None):
    """训练函数，现在需要传入args和opt参数"""
    if args is None or opt is None:
        raise ValueError("必须提供args和opt参数")
    
    logger.info(f'personalized features used：{args.personalized_features_file}')
    num_epochs = args.num_epochs
    device = args.device
    print(f"device: {device}")
    model.to(device)

    if args.track_option == 'Track1':
        train_data, val_data, train_category_count, val_category_count = train_val_split1(train_json, val_ratio=0.1, random_seed=seed)
    elif args.track_option == 'Track2':
        train_data, val_data, train_category_count, val_category_count = train_val_split2(train_json, val_percentage=0.1, seed=seed)

    train_loader = DataLoader(
        AudioVisualDataset(train_data, args.labelcount, args.personalized_features_file, max_len,
                           batch_size=args.batch_size,
                           audio_path=audio_path, video_path=video_path), 
        batch_size=args.batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        AudioVisualDataset(val_data, args.labelcount, args.personalized_features_file, max_len,
                           batch_size=args.batch_size,
                           audio_path=audio_path, video_path=video_path), 
        batch_size=args.batch_size, 
        shuffle=False
    )

    logger.info('The number of training samples = %d' % len(train_loader.dataset))
    logger.info('The number of val samples = %d' % len(val_loader.dataset))

    best_emo_acc = 0.0
    best_emo_f1 = 0.0
    best_emo_epoch = 1
    best_emo_cm = []
    # 新增：记录训练过程数据
    history = {
        "epoch": [],
        "train_loss": [],
        "val_weighted_acc": [],
        "val_unweighted_acc": [],
        "val_weighted_f1": [],
        "val_unweighted_f1": []
    }
    for epoch in range(num_epochs):
        model.train(True)
        total_loss = 0

        for i, data in enumerate(train_loader):
            for k, v in data.items():
                data[k] = v.to(device)
            model.set_input(data)
            model.optimize_parameters(epoch)

            losses = model.get_current_losses()
            total_loss += losses['emo_CE']

        avg_loss = total_loss / len(train_loader)
        # 新增：记录训练损失
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_loss)

        label, pred, emo_acc_weighted, emo_acc_unweighted, emo_f1_weighted, emo_f1_unweighted, emo_cm = eval(model, val_loader, device)
        # 验证并记录指标
       
        history["val_weighted_acc"].append(emo_acc_weighted)
        history["val_unweighted_acc"].append(emo_acc_unweighted)
        history["val_weighted_f1"].append(emo_f1_weighted)
        history["val_unweighted_f1"].append(emo_f1_unweighted)

        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.10f}, "
                    f"Weighted F1: {emo_f1_weighted:.10f}, Unweighted F1: {emo_f1_unweighted:.10f}, "
                    f"Weighted Acc: {emo_acc_weighted:.10f}, Unweighted Acc: {emo_acc_unweighted:.10f}")
        logger.info('Confusion Matrix:\n{}'.format(emo_cm))

        if emo_f1_weighted > best_emo_f1:
            cur_time = time.strftime('%Y-%m-%d-%H.%M.%S', time.localtime(time.time()))
            best_emo_f1 = emo_f1_weighted
            best_emo_f1_unweighted = emo_f1_unweighted
            best_emo_acc = emo_acc_weighted
            best_emo_acc_unweighted = emo_acc_unweighted
            best_emo_cm = emo_cm
            best_emo_epoch = epoch + 1
            best_model = model
            save_path = os.path.join(os.path.join(opt.checkpoints_dir, opt.name), best_model_name)
            torch.save(model.state_dict(), save_path)
            print("Saved best model.")

    logger.info(f"Training complete. Random seed: {seed}. Best epoch: {best_emo_epoch}.")
    logger.info(f"Best Weighted F1: {best_emo_f1:.4f}, Best Unweighted F1: {best_emo_f1_unweighted:.4f}, "
                f"Best Weighted Acc: {best_emo_acc:.4f}, Best Unweighted Acc: {best_emo_acc_unweighted:.4f}.")
    logger.info('Confusion Matrix:\n{}'.format(best_emo_cm))

    csv_file = f'{opt.log_dir}/{opt.name}.csv'
    formatted_best_emo_cm = ' '.join([f"[{' '.join(map(str, row))}]" for row in best_emo_cm])
    header = f"Time,random seed,splitwindow_time,labelcount,audiofeature_method,videofeature_method," \
             f"batch_size,num_epochs,feature_max_len,lr," \
             f"Weighted_F1,Unweighted_F1,Weighted_Acc,Unweighted_Acc,Confusion_Matrix"
    result_value = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{seed},{args.splitwindow_time},{args.labelcount},{args.audiofeature_method},{args.videofeature_method}," \
                   f"{args.batch_size},{args.num_epochs},{opt.feature_max_len},{opt.lr:.6f}," \
                   f"{best_emo_f1:.4f},{best_emo_f1_unweighted:.4f},{best_emo_acc:.4f},{best_emo_acc_unweighted:.4f},{formatted_best_emo_cm}"
    file_exists = os.path.exists(csv_file)
    with open(csv_file, mode='a') as file:
        if not file_exists:
            file.write(header + '\n')
        file.write(result_value + '\n')
     # 新增：保存历史数据为 JSON 文件
    history_path = os.path.join(opt.log_dir, opt.name, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    
    print(f"训练历史数据已保存至: {history_path}")
    
    return best_emo_f1, best_emo_f1_unweighted, best_emo_acc, best_emo_acc_unweighted, best_emo_cm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MDPP Model")
    parser.add_argument('--labelcount', type=int, default=3,
                        help="Number of data categories (2, 3, or 5).")
    parser.add_argument('--track_option', type=str, required=True,
                        help="Track1 or Track2")
    parser.add_argument('--feature_max_len', type=int, required=True,
                        help="Max length of feature.")
    parser.add_argument('--data_rootpath', type=str, required=True,
                        help="Root path to the program dataset")
    parser.add_argument('--train_json', type=str, required=False,
                        help="File name of the training JSON file")
    parser.add_argument('--personalized_features_file', type=str,
                        help="File name of the personalized features file")
    parser.add_argument('--audiofeature_method', type=str, default='mfccs',
                        choices=['mfccs', 'opensmile', 'wav2vec'],
                        help="Method for extracting audio features.")
    parser.add_argument('--videofeature_method', type=str, default='densenet',
                        choices=['openface','resnet', 'densenet'],
                        help="Method for extracting video features.")
    parser.add_argument('--splitwindow_time', type=str, default='1s',
                        help="Time window for splitted features. e.g. '1s' or '5s'")

    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for training")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=10,
                        help="Number of epochs to train the model")
    parser.add_argument('--device', type=str, default='cpu',
                        help="Device to train the model on, e.g. 'cuda' or 'cpu'")

    args = parser.parse_args()

    args.train_json = os.path.join(args.data_rootpath, 'Training', 'labels', 'Training_Validation_files.json')
    args.personalized_features_file = os.path.join(args.data_rootpath, 'Training', 'individualEmbedding', 'descriptions_embeddings_with_ids.npy')

    config = load_config('config.json')
    opt = Opt(config)

    opt.emo_output_dim = args.labelcount
    opt.feature_max_len = args.feature_max_len
    opt.lr = args.lr

    audio_path = os.path.join(args.data_rootpath, 'Training', f"{args.splitwindow_time}", 'Audio', f"{args.audiofeature_method}") + '/'
    video_path = os.path.join(args.data_rootpath, 'Training', f"{args.splitwindow_time}", 'Visual', f"{args.videofeature_method}") + '/'

    for filename in os.listdir(audio_path):
        if filename.endswith('.npy'):
            opt.input_dim_a = np.load(audio_path + filename).shape[1]
            break

    for filename in os.listdir(video_path):
        if filename.endswith('.npy'):
            opt.input_dim_v = np.load(video_path + filename).shape[1]
            break

    opt.name = f'{args.splitwindow_time}_{args.labelcount}labels_{args.audiofeature_method}+{args.videofeature_method}'
    logger_path = os.path.join(opt.log_dir, opt.name)
    if not os.path.exists(opt.log_dir):
        os.mkdir(opt.log_dir)
    if not os.path.exists(logger_path):
        os.mkdir(logger_path)
    logger = get_logger(logger_path,'result')

    model = ourModel(opt)

    cur_time = time.strftime('%Y-%m-%d-%H.%M.%S', time.localtime(time.time()))
    best_model_name = f"best_model_{cur_time}.pth"

    logger.info(f"splitwindow_time={args.splitwindow_time}, audiofeature_method={args.audiofeature_method}, "
                f"videofeature_method={args.videofeature_method}")
    logger.info(f"batch_size={args.batch_size}, num_epochs={args.num_epochs}, "
                f"labels={opt.emo_output_dim}, feature_max_len={opt.feature_max_len}, lr={opt.lr}")

    seed = 3407
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    logger.info(f"Using random seed: {seed}")

    # 传递args和opt到train_model函数
    train_model(
        train_json=args.train_json,
        model=model,
        max_len=opt.feature_max_len,
        best_model_name=best_model_name,
        audio_path=audio_path,
        video_path=video_path,
        seed=seed,
        args=args,
        opt=opt
    )
#Base Line Code:
# from datetime import datetime  # 导入datetime模块，用于处理日期和时间
# import os  # 导入os模块，用于与操作系统进行交互，如文件和目录操作
# import json  # 导入json模块，用于处理JSON数据
# import time  # 导入time模块，用于处理时间相关的操作
# import argparse  # 导入argparse模块，用于解析命令行参数
# import torch  # 导入PyTorch库，用于深度学习相关的操作
# from sklearn.metrics import f1_score, confusion_matrix, accuracy_score  # 从sklearn库中导入评估指标计算函数
# from torch.utils.data import DataLoader  # 从PyTorch的data模块中导入DataLoader，用于数据加载和批处理
# from train_val_split import train_val_split1, train_val_split2  # 从自定义模块中导入训练集和验证集划分函数
# from models.our.our_model import ourModel  # 从自定义模块中导入自定义的模型类
# from dataset import *  # 从自定义的dataset模块中导入所有内容（具体功能需看该模块定义）
# from utils.logger import get_logger  # 从自定义的logger模块中导入获取日志记录器的函数
# import numpy as np  # 导入NumPy库，用于数值计算


# class Opt:
#     def __init__(self, config_dict):
#         # 将配置字典中的键值对更新到对象的属性中
#         self.__dict__.update(config_dict)


# def load_config(config_file):
#     with open(config_file, 'r') as f:
#         # 读取配置文件并将其解析为JSON格式返回
#         return json.load(f)


# def eval(model, val_loader, device):
#     model.eval()  # 将模型设置为评估模式
#     total_emo_pred = []  # 用于存储所有预测的情感标签
#     total_emo_label = []  # 用于存储所有真实的情感标签

#     with torch.no_grad():  # 不计算梯度，用于评估阶段
#         for data in val_loader:  # 遍历验证数据加载器中的数据
#             for k, v in data.items():
#                 data[k] = v.to(device)  # 将数据移动到指定设备上（如GPU或CPU）
#             model.set_input(data)  # 将输入数据设置到模型中
#             model.test()  # 对输入数据进行测试
#             emo_pred = model.emo_pred.argmax(dim=1).cpu().numpy()  # 获取预测的情感标签并转换为NumPy数组
#             emo_label = data['emo_label'].cpu().numpy()  # 获取真实的情感标签并转换为NumPy数组
#             total_emo_pred.append(emo_pred)  # 将预测标签添加到列表中
#             total_emo_label.append(emo_label)  # 将真实标签添加到列表中

#     total_emo_pred = np.concatenate(total_emo_pred)  # 将所有预测标签连接成一个数组
#     total_emo_label = np.concatenate(total_emo_label)  # 将所有真实标签连接成一个数组

#     emo_acc_unweighted = accuracy_score(total_emo_label, total_emo_pred, sample_weight=None)  # 计算未加权的准确率
#     class_counts = np.bincount(total_emo_label)  # 获取每个类别的样本数量
#     sample_weights = 1 / (class_counts[total_emo_label] + 1e-6)  # 计算每个样本的权重，避免除零错误
#     emo_acc_weighted = accuracy_score(total_emo_label, total_emo_pred, sample_weight=sample_weights)  # 计算加权的准确率

#     emo_f1_weighted = f1_score(total_emo_label, total_emo_pred, average='weighted')  # 计算加权的F1值
#     emo_f1_unweighted = f1_score(total_emo_label, total_emo_pred, average='macro')  # 计算未加权的F1值
#     emo_cm = confusion_matrix(total_emo_label, total_emo_pred)  # 计算混淆矩阵

#     return total_emo_label, total_emo_pred, emo_acc_weighted, emo_acc_unweighted, emo_f1_weighted, emo_f1_unweighted, emo_cm


# def train_model(train_json, model, audio_path='', video_path='', max_len=5,
#                 best_model_name='best_model.pth', seed=None):
#     """
#     This is the traing function
#     """
#     logger.info(f'personalized features used：{args.personalized_features_file}')  # 记录使用的个性化特征文件
#     num_epochs = args.num_epochs  # 获取训练的轮数
#     device = args.device  # 获取使用的设备（如GPU或CPU）
#     print(f"device: {device}")  # 打印使用的设备信息
#     model.to(device)  # 将模型移动到指定设备上

#     # split training and validation set
#     # data = json.load(open(train_json, 'r'))
#     if args.track_option == 'Track1':
#         # 根据Track1的方式划分训练集和验证集
#         train_data, val_data, train_category_count, val_category_count = train_val_split1(train_json, val_ratio=0.1, random_seed=seed)
#     elif args.track_option == 'Track2':
#         # 根据Track2的方式划分训练集和验证集
#         train_data, val_data, train_category_count, val_category_count = train_val_split2(train_json, val_percentage=0.1,
#                                                                                      seed=seed)

#     train_loader = DataLoader(
#         # 创建训练数据加载器
#         AudioVisualDataset(train_data, args.labelcount, args.personalized_features_file, max_len,
#                            batch_size=args.batch_size,
#                            audio_path=audio_path, video_path=video_path), batch_size=args.batch_size, shuffle=True)
#     val_loader = DataLoader(
#         # 创建验证数据加载器
#         AudioVisualDataset(val_data, args.labelcount, args.personalized_features_file, max_len,
#                            batch_size=args.batch_size,
#                            audio_path=audio_path, video_path=video_path), batch_size=args.batch_size, shuffle=False)

#     logger.info('The number of training samples = %d' % len(train_loader.dataset))  # 记录训练样本数量
#     logger.info('The number of val samples = %d' % len(val_loader.dataset))  # 记录验证样本数量

#     best_emo_acc = 0.0  # 用于存储最佳的情感准确率
#     best_emo_f1 = 0.0  # 用于存储最佳的情感F1值
#     best_emo_epoch = 1  # 用于存储最佳模型的训练轮数
#     best_emo_cm = []  # 用于存储最佳模型的混淆矩阵

#     history = {
#         "epoch": [],
#         "train_loss": [],
#         "val_weighted_acc": [],
#         "val_unweighted_acc": [],
#         "val_weighted_f1": [],
#         "val_unweighted_f1": []
#     }
#     for epoch in range(num_epochs):  # 遍历训练轮数
#         model.train(True)  # 将模型设置为训练模式
#         total_loss = 0  # 用于累加训练损失

#         for i, data in enumerate(train_loader):  # 遍历训练数据加载器中的数据
#             for k, v in data.items():
#                 data[k] = v.to(device)  # 将数据移动到指定设备上
#             model.set_input(data)  # 将输入数据设置到模型中
#             model.optimize_parameters(epoch)  # 优化模型参数

#             losses = model.get_current_losses()  # 获取当前的损失
#             total_loss += losses['emo_CE']  # 累加情感分类的交叉熵损失

#         avg_loss = total_loss / len(train_loader)  # 计算平均损失
#         # 记录训练损失（直接添加浮点数）
#         history["epoch"].append(epoch + 1)
#         history["train_loss"].append(avg_loss)
#         # evaluation
#         label, pred, emo_acc_weighted, emo_acc_unweighted, emo_f1_weighted, emo_f1_unweighted, emo_cm = eval(model, val_loader,
#                                                                                                 device)  # 在验证集上评估模型
#         history["val_weighted_acc"].append(emo_acc_weighted)
#         history["val_unweighted_acc"].append(emo_acc_unweighted)
#         history["val_weighted_f1"].append(emo_f1_weighted)
#         history["val_unweighted_f1"].append(emo_f1_unweighted)
#         logger.info(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.10f}, "
#                     f"Weighted F1: {emo_f1_weighted:.10f}, Unweighted F1: {emo_f1_unweighted:.10f}, "
#                     f"Weighted Acc: {emo_acc_weighted:.10f}, Unweighted Acc: {emo_acc_unweighted:.10f}")  # 记录训练信息
#         logger.info('Confusion Matrix:\n{}'.format(emo_cm))  # 记录混淆矩阵

#         if emo_f1_weighted > best_emo_f1:  # 如果当前的加权F1值大于之前的最佳值
#             cur_time = time.strftime('%Y-%m-%d-%H.%M.%S', time.localtime(time.time()))  # 获取当前时间
#             best_emo_f1 = emo_f1_weighted  # 更新最佳的加权F1值
#             best_emo_f1_unweighted = emo_f1_unweighted  # 更新最佳的未加权F1值
#             best_emo_acc = emo_acc_weighted  # 更新最佳的加权准确率
#             best_emo_acc_unweighted = emo_acc_unweighted  # 更新最佳的未加权准确率
#             best_emo_cm = emo_cm  # 更新最佳的混淆矩阵
#             best_emo_epoch = epoch + 1  # 更新最佳的训练轮数
#             best_model = model  # 更新最佳模型
#             save_path = os.path.join(os.path.join(opt.checkpoints_dir, opt.name), best_model_name)  # 构建模型保存路径
#             torch.save(model.state_dict(), save_path)  # 保存最佳模型的参数
#             print("Saved best model.")  # 打印保存模型的信息

#     logger.info(f"Training complete. Random seed: {seed}. Best epoch: {best_emo_epoch}.")  # 记录训练完成信息
#     logger.info(f"Best Weighted F1: {best_emo_f1:.4f}, Best Unweighted F1: {best_emo_f1_unweighted:.4f}, "
#                 f"Best Weighted Acc: {best_emo_acc:.4f}, Best Unweighted Acc: {best_emo_acc_unweighted:.4f}.")  # 记录最佳评估指标
#     logger.info('Confusion Matrix:\n{}'.format(best_emo_cm))  # 记录最佳模型的混淆矩阵

#     # output results to CSV
#     csv_file = f'{opt.log_dir}/{opt.name}.csv'  # 构建CSV文件路径
#     formatted_best_emo_cm = ' '.join([f"[{' '.join(map(str, row))}]" for row in best_emo_cm])  # 格式化混淆矩阵
#     header = f"Time,random seed,splitwindow_time,labelcount,audiofeature_method,videofeature_method," \
#              f"batch_size,num_epochs,feature_max_len,lr," \
#              f"Weighted_F1,Unweighted_F1,Weighted_Acc,Unweighted_Acc,Confusion_Matrix"  # CSV文件头
#     result_value = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{seed},{args.splitwindow_time},{args.labelcount},{args.audiofeature_method},{args.videofeature_method}," \
#                    f"{args.batch_size},{args.num_epochs},{opt.feature_max_len},{opt.lr:.6f}," \
#                    f"{best_emo_f1:.4f},{best_emo_f1_unweighted:.4f},{best_emo_acc:.4f},{best_emo_acc_unweighted:.4f},{formatted_best_emo_cm}"  # 构建结果字符串
#     file_exists = os.path.exists(csv_file)  # 检查CSV文件是否存在
#     # Open file (append if file exists, create if it doesn't)
#     with open(csv_file, mode='a') as file:  # 以追加模式打开CSV文件
#         if not file_exists:
#             file.write(header + '\n')  # 如果文件不存在，写入文件头
#         file.write(result_value + '\n')  # 写入结果字符串
#     # 保存历史数据
#     history_path = os.path.join(opt.log_dir, opt.name, "training_history.json")
#     with open(history_path, "w") as f:
#       json.dump(history, f, indent=4)
#     return best_emo_f1, best_emo_f1_unweighted, best_emo_acc, best_emo_acc_unweighted, best_emo_cm


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Train MDPP Model")  # 创建命令行参数解析器
#     parser.add_argument('--labelcount', type=int, default=3,
#                         help="Number of data categories (2, 3, or 5).")  # 添加标签数量参数
#     parser.add_argument('--track_option', type=str, required=True,
#                         help="Track1 or Track2")  # 添加任务选项参数
#     parser.add_argument('--feature_max_len', type=int, required=True,
#                         help="Max length of feature.")  # 添加特征最大长度参数
#     parser.add_argument('--data_rootpath', type=str, required=True,
#                         help="Root path to the program dataset")  # 添加数据根路径参数
#     parser.add_argument('--train_json', type=str, required=False,
#                         help="File name of the training JSON file")  # 添加训练JSON文件参数
#     parser.add_argument('--personalized_features_file', type=str,
#                         help="File name of the personalized features file")  # 添加个性化特征文件参数
#     parser.add_argument('--audiofeature_method', type=str, default='mfccs',
#                         choices=['mfccs', 'opensmile', 'wav2vec'],
#                         help="Method for extracting audio features.")  # 添加音频特征提取方法参数
#     parser.add_argument('--videofeature_method', type=str, default='densenet',
#                         choices=['openface','resnet', 'densenet'],
#                         help="Method for extracting video features.")  # 添加视频特征提取方法参数
#     parser.add_argument('--splitwindow_time', type=str, default='1s',
#                         help="Time window for splitted features. e.g. '1s' or '5s'")  # 添加特征分割时间窗口参数

#     parser.add_argument('--batch_size', type=int, default=32,
#                         help="Batch size for training")  # 添加训练批次大小参数
#     parser.add_argument('--lr', type=float, default=1e-4,
#                         help="Learning rate")  # 添加学习率参数
#     parser.add_argument('--num_epochs', type=int, default=10,
#                         help="Number of epochs to train the model")  # 添加训练轮数参数
#     parser.add_argument('--device', type=str, default='cpu',
#                         help="Device to train the model on, e.g. 'cuda' or 'cpu'")  # 添加训练设备参数

#     args = parser.parse_args()  # 解析命令行参数

#     args.train_json = os.path.join(args.data_rootpath, 'Training', 'labels', 'Training_Validation_files.json')  # 构建训练JSON文件路径
#     args.personalized_features_file = os.path.join(args.data_rootpath, 'Training', 'individualEmbedding', 'descriptions_embeddings_with_ids.npy')  # 构建个性化特征文件路径

#     config = load_config('config.json')  # 加载配置文件
#     opt = Opt(config)  # 创建配置对象

#     # Modify individual dynamic parameters in opt according to task category
#     opt.emo_output_dim = args.labelcount  # 根据命令行参数更新模型的情感输出维度
#     opt.feature_max_len = args.feature_max_len  # 根据命令行参数更新特征最大长度
#     opt.lr = args.lr  # 根据命令行参数更新学习率

#     # Splice out feature folder paths according to incoming audio and video feature types
#     audio_path = os.path.join(args.data_rootpath, 'Training', f"{args.splitwindow_time}", 'Audio', f"{args.audiofeature_method}") + '/'  # 构建音频特征文件夹路径
#     video_path = os.path.join(args.data_rootpath, 'Training', f"{args.splitwindow_time}", 'Visual', f"{args.videofeature_method}") + '/'  # 构建视频特征文件夹路径

#     # Obtain input_dim_a, input_dim_v
#     for filename in os.listdir(audio_path):
#         if filename.endswith('.npy'):
#             opt.input_dim_a = np.load(audio_path + filename).shape[1]  # 获取音频特征的输入维度
#             break

#     for filename in os.listdir(video_path):
#         if filename.endswith('.npy'):
#             opt.input_dim_v = np.load(video_path + filename).shape[1]  # 获取视频特征的输入维度
#             break

#     opt.name = f'{args.splitwindow_time}_{args.labelcount}labels_{args.audiofeature_method}+{args.videofeature_method}'  # 构建模型名称
#     logger_path = os.path.join(opt.log_dir, opt.name)  # 构建日志记录器路径
#     if not os.path.exists(opt.log_dir):
#         os.mkdir(opt.log_dir)  # 创建日志目录
#     if not os.path.exists(logger_path):
#         os.mkdir(logger_path)  # 创建日志记录器路径下的目录
#     logger = get_logger(logger_path,'result')  # 获取日志记录器

#     model = ourModel(opt)  # 创建模型实例

#     cur_time = time.strftime('%Y-%m-%d-%H.%M.%S', time.localtime(time.time()))  # 获取当前时间
#     best_model_name = f"best_model_{cur_time}.pth"  # 构建最佳模型文件名

#     logger.info(f"splitwindow_time={args.splitwindow_time}, audiofeature_method={args.audiofeature_method}, "
#                 f"videofeature_method={args.videofeature_method}")  # 记录特征分割时间窗口、音频和视频特征提取方法
#     logger.info(f"batch_size={args.batch_size}, num_epochs={args.num_epochs}, "
#                 f"labels={opt.emo_output_dim}, feature_max_len={opt.feature_max_len}, lr={opt.lr}")  # 记录训练参数

#     # set random seed
#     # seed = np.random.randint(0, 10000) 
#     seed = 3407  # 设置随机种子
#     np.random.seed(seed)  # 设置NumPy的随机种子
#     torch.manual_seed(seed)  
# # 设置CPU上的随机种子，确保每次运行代码时，CPU上的随机数生成器产生相同的随机数序列，
# # 从而使得模型初始化、数据采样等涉及随机数的操作具有可重复性。

# torch.cuda.manual_seed_all(seed)  
# # 如果使用GPU进行计算，该函数设置所有GPU上的随机种子。
# # 确保在多GPU环境下，每次运行代码时，GPU上的随机数生成器也产生相同的随机数序列，
# # 保证了模型训练过程中涉及GPU的随机操作的可重复性。

# logger.info(f"Using random seed: {seed}")  
# # 使用日志记录器记录当前使用的随机种子值，方便查看和记录实验设置。

# # training
# train_model(
#     train_json=args.train_json,  # 传入训练数据的JSON文件路径
#     model=model,  # 传入要训练的模型实例
#     max_len=opt.feature_max_len,  # 传入特征的最大长度
#     best_model_name=best_model_name,  # 传入最佳模型的保存文件名
#     audio_path=audio_path,  # 传入音频特征文件的路径
#     video_path=video_path,  # 传入视频特征文件的路径
#     seed=seed  # 传入随机种子，可能在训练函数中用于数据划分等涉及随机数的操作
# )