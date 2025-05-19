import os
import glob
import shutil
import argparse
import numpy as np
from util import read_hog, read_csv
# 将上级目录的上级目录添加到系统路径中，以便能够导入相应模块
import sys
sys.path.append('../../')


def generate_face_faceDir(input_root, save_root):
    # 遍历input_root目录下所有以"_aligned"结尾的子目录
    for dir_path in glob.glob(input_root + '/*_aligned'):  # 'xx/xx/000100_guest_aligned'
        # 获取当前目录下的所有文件名
        frame_names = os.listdir(dir_path)  # ['xxx.bmp']
        # 断言目录中的文件数量应小于等于1（这里可能是预期每个目录只有一个文件）
        # assert len(frame_names) <= 1
        # 如果目录中只有一个文件
        if len(frame_names) == 1:  
            # 拼接文件的完整路径
            frame_path = os.path.join(dir_path, frame_names[0])  # 'xx/xx/000100_guest_aligned/xxx.bmp'
            # 从目录名中提取文件名（去掉"_aligned"后缀）
            name = os.path.basename(dir_path)[:-len('_aligned')]  # '000100_guest'
            # 构建保存文件的路径
            save_path = os.path.join(save_root, name + '.bmp')
            # 将文件从原路径复制到保存路径
            shutil.copy(frame_path, save_path)


def generate_face_videoOne(input_root, save_root):
    # 遍历input_root目录下所有以"_aligned"结尾的子目录
    for dir_path in glob.glob(input_root + '/*_aligned'):  # 'xx/xx/000100_guest_aligned'
        # 获取当前目录下的所有文件名
        frame_names = os.listdir(dir_path)  # ['xxx.bmp']
        # 遍历目录中的每个文件
        for ii in range(len(frame_names)):
            # 拼接文件的完整路径
            frame_path = os.path.join(dir_path, frame_names[ii])  # 'xx/xx/000100_guest_aligned/xxx.bmp'
            # 获取文件名
            frame_name = os.path.basename(frame_path)
            # 构建保存文件的路径
            save_path = os.path.join(save_root, frame_name)
            # 将文件从原路径复制到保存路径
            shutil.copy(frame_path, save_path)


def generate_hog(input_root, save_root):
    # 遍历input_root目录下所有以".hog"结尾的文件
    for hog_path in glob.glob(input_root + '/*.hog'):
        # 根据".hog"文件路径生成对应的".csv"文件路径
        csv_path = hog_path[:-4] + '.csv'
        # 如果对应的".csv"文件存在
        if os.path.exists(csv_path):
            # 提取".hog"文件名（去掉".hog"后缀）
            hog_name = os.path.basename(hog_path)[:-4]
            # 从".hog"文件中读取数据（这里假设read_hog函数返回两个值，第一个值未使用，用下划线占位）
            _, feature = read_hog(hog_path)
            # 构建保存特征数据的路径
            save_path = os.path.join(save_root, hog_name + '.npy')
            # 将特征数据保存为".npy"文件
            np.save(save_path, feature)


def generate_csv(input_root, save_root, startIdx):
    # 遍历input_root目录下所有以".csv"结尾的文件
    for csv_path in glob.glob(input_root + '/*.csv'):
        # 提取".csv"文件名（去掉".csv"后缀）
        csv_name = os.path.basename(csv_path)[:-4]
        # 从".csv"文件中读取数据（传入startIdx参数，具体作用由read_csv函数定义）
        feature = read_csv(csv_path, startIdx)
        # 构建保存特征数据的路径
        save_path = os.path.join(save_root, csv_name + '.npy')
        # 将特征数据保存为".npy"文件
        np.save(save_path, feature)


def extract(input_dir, process_type, save_dir, face_dir, hog_dir, pose_dir):
    # 获取输入目录中的所有视频文件或文件夹
    vids = os.listdir(input_dir)
    # 打印找到的视频数量
    print(f'Find total "{len(vids)}" videos.')
    # 遍历每个视频文件或文件夹
    for i, vid in enumerate(vids, 1):
        print(vid)
        # if vid > '011_003_088': continue
        # 用于保存处理结果的视频名（如果是".mp4"或".avi"文件，则去掉文件后缀）
        saveVid = vid  
        if vid.endswith('.mp4') or vid.endswith('.avi'): saveVid = vid[:-4]  

        # 打印当前正在处理的视频信息
        print(f"Processing video '{vid}' ({i}/{len(vids)})...")
        # 构建当前视频的输入路径
        input_root = os.path.join(input_dir, vid)  
        # 构建当前视频的保存路径
        save_root = os.path.join(save_dir, saveVid)
        # 构建当前视频的人脸保存路径
        face_root = os.path.join(face_dir, saveVid)
        # 构建当前视频的HOG特征保存路径
        hog_root = os.path.join(hog_dir, saveVid)
        # 构建当前视频的姿态保存路径
        pose_root = os.path.join(pose_dir, saveVid)
        # 如果人脸保存路径已存在，则跳过当前视频（这里可能是为了避免重复处理）
        # if os.path.exists(face_root): continue
        # 如果保存路径不存在，则创建保存路径
        if not os.path.exists(save_root): os.makedirs(save_root)
        # 如果人脸保存路径不存在，则创建人脸保存路径
        if not os.path.exists(face_root): os.makedirs(face_root)
        # 如果HOG特征保存路径不存在，则创建HOG特征保存路径
        if not os.path.exists(hog_root):  os.makedirs(hog_root)
        # 如果姿态保存路径不存在，则创建姿态保存路径
        if not os.path.exists(pose_root): os.makedirs(pose_root)
        # 如果处理类型为'faceDir'
        if process_type == 'faceDir':
            # 构建OpenFace工具中FaceLandmarkImg.exe的路径
            exe_path = os.path.join(r'.\tools\OpenFace_2.2.0_win_x64', 
                                    'FaceLandmarkImg.exe')
            # 构建命令行命令，用于调用OpenFace工具处理视频帧
            commond = '%s -fdir \"%s\" -out_dir \"%s\"' % (exe_path, input_root, save_root)
            # 执行命令行命令
            os.system(commond)
            # 调用函数处理人脸数据并保存
            generate_face_faceDir(save_root, face_root)
            # 调用函数处理HOG数据并保存
            generate_hog(save_root, hog_root)
            # 调用函数处理CSV数据并保存（指定从第2列开始读取数据）
            generate_csv(save_root, pose_root, startIdx=2)
        # 如果处理类型为'videoOne'
        elif process_type == 'videoOne':
            # 构建OpenFace工具中FeatureExtraction.exe的路径
            exe_path = os.path.join(r'.\tools\OpenFace_2.2.0_win_x64',
                                    'FeatureExtraction.exe')
            # 构建命令行命令，用于调用OpenFace工具处理视频
            commond = '%s -f \"%s\" -out_dir \"%s\"' % (exe_path, input_root, save_root)
            # 执行命令行命令
            os.system(commond)
            # 调用函数处理人脸数据并保存
            generate_face_videoOne(save_root, face_root)
            # 调用函数处理HOG数据并保存
            generate_hog(save_root, hog_root)
            # 调用函数处理CSV数据并保存（指定从第5列开始读取数据）
            generate_csv(save_root, pose_root, startIdx=5)


if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Run.')
    # 添加一个布尔类型的命令行参数--overwrite，用于指定是否覆盖已存在的特征文件夹，默认值为True
    parser.add_argument('--overwrite', action='store_true', default=True,
                        help='whether overwrite existed feature folder.')
    # 添加一个字符串类型的命令行参数--dataset，用于指定输入数据集，默认值为'BoxOfLies'
    parser.add_argument('--dataset', type=str, default='BoxOfLies', help='input dataset')
    # 添加一个字符串类型的命令行参数--type，用于指定处理类型，只能从['faceDir', 'videoOne']中选择，默认值为'faceDir'
    parser.add_argument('--type', type=str, default='faceDir', choices=['faceDir', 'videoOne'],
                        help='faceDir: process on facedirs; videoOne: process on one video')
    # 解析命令行参数
    params = parser.parse_args()

    # 打印开始提取OpenFace特征的信息
    print(f'==> Extracting openface features...')

    # 输入目录：人脸目录
    dataset = params.dataset
    process_type = params.type
    input_dir = r"E:\MEIJU_data20241229\frame_5s"

    # 输出目录：特征CSV目录
    save_dir = os.path.join(r"\features\openface\frame_5s", 'openface_all')
    hog_dir = os.path.join(r"\features\openface\frame_5s", 'openface_hog')
    pose_dir = os.path.join(r"\features\openface\frame_5s", 'openface_pose')
    face_dir = os.path.join(r"\features\openface\frame_5s", 'openface_face')

    # 如果保存目录不存在，则创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 如果保存目录已存在且--overwrite参数为True，则打印警告信息
    elif params.overwrite:
        print(f'==> Warning: overwrite save_dir "{save_dir}"!')
    # 如果保存目录已存在且--overwrite参数为False，则抛出异常
    else:
        raise Exception(f'==> Error: save_dir "{save_dir}" already exists, set overwrite=TRUE if needed!')

    # 如果HOG特征保存目录不存在，则创建HOG特征保存目录
    if not os.path.exists(hog_dir):
        os.makedirs(hog_dir)
    # 如果HOG特征保存目录已存在且--overwrite参数为True，则打印警告信息
    elif params.overwrite:
        print(f'==> Warning: overwrite save_dir "{hog_dir}"!')
    # 如果HOG特征保存目录已存在且--overwrite参数为False，则抛出异常
    else:
        raise Exception(f'==> Error: save_dir "{hog_dir}" already exists, set overwrite=TRUE if needed!')

    # 如果姿态保存目录不存在，则创建姿态保存目录
    if not os.path.exists(pose_dir):
        os.makedirs(pose_dir)
    # 如果姿态保存目录已存在且--overwrite参数为True，则打印警告信息
    elif params.overwrite:
        print(f'==> Warning: overwrite save_dir "{pose_dir}"!')
    # 如果姿态保存目录已存在且--overwrite参数为False，则抛出异常
    else:
        raise Exception(f'==> Error: save_dir "{pose_dir}" already exists, set overwrite=TRUE if needed!')

    # 如果人脸保存目录不存在，则创建人脸保存目录
    if not os.path.exists(face_dir):
        os.makedirs(face_dir)
    # 如果人脸保存目录已存在且--overwrite参数为True，则打印警告信息
    elif params.overwrite:
        print(f'==> Warning: overwrite save_dir "{face_dir}"!')
    # 如果人脸保存目录已存在且--overwrite参数为False，则抛出异常
    else:
        raise Exception(f'==> Error: save_dir "{face_dir}" already exists, set overwrite=TRUE if needed!')

    # 调用extract函数进行处理
    extract(input_dir, process_type, save_dir, face_dir, hog_dir, pose_dir)

    # 打印处理完成的信息
    print(f'==> Finish')