# -*- coding:utf-8 -*-  # 指定文件的编码格式为UTF-8
"""
wav2vec: https://arxiv.org/abs/1904.05862  # 引用wav2vec模型的论文链接
official github repo: https://github.com/pytorch/fairseq/tree/master/examples/wav2vec  # 引用wav2vec模型的官方GitHub仓库链接
"""
import os  # 导入操作系统相关的模块，用于文件和目录操作
import time  # 导入时间模块，用于计时
import glob  # 导入glob模块，用于查找符合特定规则的文件路径名
import torch  # 导入PyTorch深度学习框架
import numpy as np  # 导入NumPy库，用于数值计算
import soundfile as sf  # 导入soundfile库，用于读取和写入音频文件
from fairseq.models.wav2vec import Wav2VecModel  # 从fairseq库中导入Wav2VecModel，注意要使用fairseq版本0.10.1 (pip install fairseq==0.10.1)

def write_feature_to_npy(feature, csv_file, feature_level):
    """
    将提取的特征保存为.npy文件
    :param feature: 提取的特征
    :param csv_file: 保存特征的.npy文件路径
    :param feature_level: 特征级别，如'UTTERANCE'
    """
    if feature_level == 'UTTERANCE':
        # 如果特征级别是'UTTERANCE'，将特征数组进行压缩
        feature = np.array(feature).squeeze()  # 压缩数组，去除维度为1的维度，得到形状为 [C,] 的数组
        if len(feature.shape) != 1:  # 如果特征形状不是一维的，即 [T, C] 形状
            feature = np.mean(feature, axis=0)  # 对特征在时间维度上求均值，将 [T, C] 转换为 [C,]
        np.save(csv_file, feature)  # 将处理后的特征保存为.npy文件
    else:
        np.save(csv_file, feature)  # 如果特征级别不是'UTTERANCE'，直接保存特征

def extract(audio_files, feature_level, model, save_dir, overwrite=False, gpu=None):
    """
    从音频文件中提取wav2vec特征
    :param audio_files: 音频文件列表
    :param feature_level: 特征级别
    :param model: wav2vec模型
    :param save_dir: 保存特征的目录
    :param overwrite: 是否覆盖已存在的目录
    :param gpu: 使用的GPU编号
    """
    start_time = time.time()  # 记录开始时间
    # 根据是否有可用的GPU，选择使用GPU或CPU设备
    device = torch.device(f'cuda:{gpu}' if gpu is not None and torch.cuda.is_available() else 'cpu')

    dir_name = 'wav2vec-large'  # 定义目录名称
    # 定义特征编码器输出特征的保存目录
    out_dir_z = os.path.join(save_dir, f'{dir_name}-z-{feature_level[:3]}') 
    if not os.path.exists(out_dir_z):  # 如果目录不存在
        os.makedirs(out_dir_z)  # 创建目录
    elif overwrite or len(os.listdir(save_dir)) == 0:  # 如果允许覆盖或者保存目录为空
        print(f'==> Warning: overwrite save_dir "{save_dir}"!')  # 打印警告信息
    else:
        # 如果目录已存在且不允许覆盖，抛出异常
        raise Exception(f'==> Error: save_dir "{save_dir}" already exists, set overwrite=True if needed!')

    # 定义上下文网络输出特征的保存目录
    out_dir_c = os.path.join(save_dir, f'{dir_name}-c-{feature_level[:3]}') 
    if not os.path.exists(out_dir_c):  # 如果目录不存在
        os.makedirs(out_dir_c)  # 创建目录
    elif overwrite or len(os.listdir(save_dir)) == 0:  # 如果允许覆盖或者保存目录为空
        print(f'==> Warning: overwrite save_dir "{save_dir}"!')  # 打印警告信息
    else:
        # 如果目录已存在且不允许覆盖，抛出异常
        raise Exception(f'==> Error: save_dir "{save_dir}" already exists, set overwrite=True if needed!')

    for idx, wav_file in enumerate(audio_files, 1):  # 遍历音频文件列表
        file_name = os.path.basename(wav_file)  # 获取音频文件的文件名
        vid = file_name[:-4]  # 去掉文件名的扩展名，得到视频ID
        print(f'Processing "{file_name}" ({idx}/{len(audio_files)})...')  # 打印正在处理的文件名和进度
        # 读取音频文件
        audio, sampling_rate = sf.read(wav_file)
        audio = audio.astype('float32')[np.newaxis, :]  # 将音频数据转换为float32类型，并添加一个维度
        audio = torch.from_numpy(audio)  # 将NumPy数组转换为PyTorch张量
        audio = audio.to(device)  # 将音频张量移动到指定设备
        # 检查音频采样率是否为16000Hz
        assert sampling_rate == 16000, f'Error: sampling rate ({sampling_rate}) != 16k!'
        with torch.no_grad():  # 不计算梯度，减少内存消耗
            # 通过特征编码器提取特征，输出形状为 (1, C, T)，步长为10ms (100Hz)，感受野为30ms
            z = model.feature_extractor(audio)
            # 通过上下文网络对特征进行聚合，输出形状为 (1, C, T)，步长为10ms (100Hz)，感受野为801ms (大版本)
            c = model.feature_aggregator(z)

        z_feature = z.detach().squeeze().t().cpu().numpy()  # 将z特征从计算图中分离，压缩维度，转置并转换为NumPy数组
        c_feature = c.detach().squeeze().t().cpu().numpy()  # 将c特征从计算图中分离，压缩维度，转置并转换为NumPy数组
        z_csv_file = os.path.join(out_dir_z, f'{vid}.npy')  # 定义z特征的保存文件路径
        c_csv_file = os.path.join(out_dir_c, f'{vid}.npy')  # 定义c特征的保存文件路径
        write_feature_to_npy(z_feature, z_csv_file, feature_level)  # 保存z特征
        write_feature_to_npy(c_feature, c_csv_file, feature_level)  # 保存c特征

    end_time = time.time()  # 记录结束时间
    print(f'Total time used: {end_time - start_time:.1f}s.')  # 打印总用时

if __name__ == '__main__':
    gpu = 0  # 指定使用的GPU编号
    feature_level = 'UTTERANCE'  # 指定特征级别为'UTTERANCE'
    overwrite = True  # 是否覆盖已存在的目录
    audio_dir = '/path/to/audio'  # 音频文件所在的目录，需要替换为实际路径
    save_dir = '/path/to/save'    # 保存特征的目录，需要替换为实际路径
    model_path = '/path/to/model/wav2vec_large.pt'  # 预训练模型的路径，需要替换为实际路径

    # 获取音频文件列表，只获取扩展名为.wav的文件
    audio_files = glob.glob(os.path.join(audio_dir, '*.wav'))
    print(f'Find total "{len(audio_files)}" audio files.')  # 打印找到的音频文件数量

    # 根据是否有可用的GPU，选择使用GPU或CPU设备
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    cp = torch.load(model_path, map_location=device)  # 加载预训练模型
    model = Wav2VecModel.build_model(cp['args'], task=None)  # 构建wav2vec模型
    model.load_state_dict(cp['model'])  # 加载模型的参数
    model.to(device)  # 将模型移动到指定设备
    model.eval()  # 将模型设置为评估模式

    # 调用extract函数提取特征
    extract(audio_files, feature_level=feature_level, model=model, save_dir=save_dir, overwrite=overwrite, gpu=gpu)