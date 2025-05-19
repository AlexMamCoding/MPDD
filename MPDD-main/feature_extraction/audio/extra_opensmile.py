# 导入os模块，用于进行操作系统相关的操作，如文件和目录的创建、查找等
import os
# 导入NumPy库，它是Python中用于科学计算的基础库，常用于处理数组和矩阵
import numpy as np
# 导入opensmile库，该库用于提取音频的声学特征
import opensmile

# 定义输入音频文件所在的目录路径
input_audio_dir = r"D:\HACI\MMchallenge\Audio_split1\Audio_split_16k"  # Directory containing audio files
# 定义输出特征文件要保存的目录路径，这些特征文件将以.npy格式存储
output_feature_dir = r"D:\HACI\MMchallenge\Audio_split1\features\opensmile"  # Directory to save .npy feature files

# 创建输出特征目录，如果目录已经存在则不会报错（exist_ok=True）
os.makedirs(output_feature_dir, exist_ok=True)

# 创建一个opensmile的Smile对象，用于提取音频特征
smile = opensmile.Smile(
    # 指定特征集为ComParE_2016，这是一个常用的音频特征集
    feature_set=opensmile.FeatureSet.ComParE_2016,
    # 指定特征级别为Functionals，即提取的是函数级别的特征，通常是经过统计聚合后的特征
    feature_level=opensmile.FeatureLevel.Functionals,
)

# 遍历输入音频目录中的所有文件
for audio_file in os.listdir(input_audio_dir):
    # 检查文件是否为.wav或.mp3格式的音频文件
    if audio_file.endswith((".wav", ".mp3")):
        # 构建完整的音频文件路径
        audio_path = os.path.join(input_audio_dir, audio_file)
        # 获取音频文件的文件名（不包含扩展名），并将扩展名替换为.npy
        feature_file = os.path.splitext(audio_file)[0] + ".npy"
        # 构建输出特征文件的完整路径
        output_path = os.path.join(output_feature_dir, feature_file)

        try:
            # 使用opensmile对象处理音频文件，提取特征
            features = smile.process_file(audio_path)

            # 将提取的特征转换为NumPy数组，并将其展平为一维数组
            feature_array = features.to_numpy().flatten()

            # 打印当前音频文件提取的特征数组的形状
            print(f"Shape of features for {audio_file}: {feature_array.shape}")

            # 将特征数组保存为.npy文件
            np.save(output_path, feature_array)
            # 打印保存成功的信息
            print(f"Features saved for {audio_file} as {output_path}")
        except Exception as e:
            # 如果在处理音频文件时出现异常，打印错误信息
            print(f"Error processing file {audio_file}: {e}")

# 打印特征提取完成的信息
print("Feature extraction completed.")