import cv2
import torch
# 从torchvision.models模块中导入resnet50和densenet121模型
from torchvision.models import resnet50, densenet121
# 从torchvision.transforms模块中导入transforms，用于图像预处理
from torchvision.transforms import transforms
import os
import tqdm
# 导入torch.utils.data模块中的data，用于构建数据集和数据加载器
import torch.utils.data as data
import glob
import argparse
import numpy as np
# 从PIL库中导入Image，用于处理图像
from PIL import Image


# 自定义数据集类，继承自torch.utils.data.Dataset
class FrameDataset(data.Dataset):
    def __init__(self, vid, face_dir, transform=None):
        # 调用父类的初始化方法
        super(FrameDataset, self).__init__()
        # 视频ID
        self.vid = vid
        # 视频帧所在目录路径
        self.path = os.path.join(face_dir, vid)
        # 图像预处理变换
        self.transform = transform
        # 获取视频帧列表
        self.frames = self.get_frames()

    def get_frames(self):
        # 使用glob模块获取指定目录下的所有文件路径，即视频帧路径
        frames = glob.glob(os.path.join(self.path, '*'))
        return frames

    def __len__(self):
        # 返回视频帧的数量
        return len(self.frames)

    def __getitem__(self, index):
        # 获取指定索引的视频帧路径
        path = self.frames[index]
        # 使用PIL的Image.open方法打开图像
        img = Image.open(path)
        # 如果有图像预处理变换，则对图像进行变换
        if self.transform is not None:
            img = self.transform(img)
        # 获取图像文件名（去掉文件扩展名）
        name = os.path.basename(path)[:-4]
        # 返回预处理后的图像和文件名
        return img, name


# 从视频中提取帧的函数
def frame_extract(video_path, root_save_path, sample_rate=2):
    # 获取视频文件名（去掉文件扩展名）
    video_name = os.path.basename(video_path)[:-4]
    # 构建保存视频帧的目录路径
    save_dir = os.path.join(root_save_path, video_name)
    # 如果保存目录不存在，则创建目录
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # 使用cv2.VideoCapture打开视频文件
    video = cv2.VideoCapture(video_path)

    # 帧计数器
    count = 0
    # 循环读取视频帧，直到视频结束
    while video.isOpened():
        # 读取一帧视频，ret为是否成功读取，frame为读取到的帧
        ret, frame = video.read()
        # 如果读取失败，退出循环
        if not ret:
            break

        # 如果当前帧的序号是采样率的倍数，则保存该帧
        if count % sample_rate == 0:
            # 构建保存帧的文件路径
            save_path = os.path.join(root_save_path, video_name, f'frame{count:04d}.jpg')
            # 使用cv2.imwrite保存帧为图像文件
            cv2.imwrite(save_path, frame)
            # break
        # 帧计数器加1
        count += 1

    # 释放视频资源
    video.release()
    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()


# 从数据加载器中提取特征的函数
def extract(data_loader, model):
    # 将模型设置为评估模式
    model.eval()
    # 禁用梯度计算，以减少内存消耗和提高推理速度
    with torch.no_grad():
        # 初始化特征列表和时间戳列表
        features, timestamps = [], []
        # 遍历数据加载器中的图像和文件名
        for images, names in data_loader:
            # images = images.cuda()
            # 使用模型对图像进行前向传播，得到特征嵌入
            embedding = model(images)
            # 将特征嵌入转换为numpy数组并添加到特征列表中
            features.append(embedding.cpu().detach().numpy())
            # 将文件名添加到时间戳列表中
            timestamps.extend(names)
        # 将特征列表转换为二维numpy数组，将时间戳列表转换为numpy数组
        features, timestamps = np.row_stack(features), np.array(timestamps)
        # 返回特征和时间戳
        return features, timestamps


# 提取视频帧特征的函数
def feature_extract(frame_dir, save_dir, feature_level='UTT'):
    # 如果保存特征的目录不存在，则创建目录
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # 加载预训练的resnet50模型
    model = resnet50(pretrained=True)#.cuda()
    # 定义图像预处理变换
    transform = transforms.Compose([
        # transforms.ToPILImage(),
        # 将图像调整为224x224大小
        transforms.Resize((224, 224)),
        # 将图像转换为张量
        transforms.ToTensor(),
        # 对图像进行归一化处理
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # 获取视频帧目录下的所有视频ID
    vids = os.listdir(frame_dir)
    # 初始化嵌入维度
    EMBEDDING_DIM = -1
    # 打印找到的视频数量
    print(f'Find total "{len(vids)}" videos.')
    # 遍历每个视频ID
    for i, vid in enumerate(vids, 1):
        # 打印当前正在处理的视频信息
        print(f"Processing video '{vid}' ({i}/{len(vids)})...")
        # 构建保存特征的文件路径
        csv_file = os.path.join(save_dir, f'{vid}.npy')
        # 如果特征文件已存在，则跳过当前视频
        if os.path.exists(csv_file):
            continue

        # 前向传播
        # 创建FrameDataset数据集对象
        dataset = FrameDataset(vid, frame_dir, transform=transform)
        # 如果数据集中没有视频帧，则打印警告信息
        if len(dataset) == 0:
            print("Warning: number of frames of video {} should not be zero.".format(vid))
            embeddings, framenames = [], []
        else:
            # 创建数据加载器
            data_loader = torch.utils.data.DataLoader(dataset,
                                                      batch_size=32,
                                                      num_workers=4,
                                                      pin_memory=True)
            # 提取特征和帧名
            embeddings, framenames = extract(data_loader, model)

        # 保存结果
        # 根据帧名对特征进行排序
        indexes = np.argsort(framenames)
        embeddings = embeddings[indexes]
        # 更新最大嵌入维度
        EMBEDDING_DIM = max(EMBEDDING_DIM, np.shape(embeddings)[-1])

        # 如果特征级别为'FRAME'，则按帧保存特征
        if feature_level == 'FRAME':
            embeddings = np.array(embeddings).squeeze()
            # 如果没有特征，则创建一个全零的特征向量
            if len(embeddings) == 0:
                embeddings = np.zeros((1, EMBEDDING_DIM))
            # 如果特征是一维的，则增加一个维度
            elif len(embeddings.shape) == 1:
                embeddings = embeddings[np.newaxis, :]
            # 保存特征为npy文件，形状为(frame_num, 1000)
            np.save(csv_file, embeddings)
        else:
            embeddings = np.array(embeddings).squeeze()
            # 如果没有特征，则创建一个全零的特征向量
            if len(embeddings) == 0:
                embeddings = np.zeros((EMBEDDING_DIM, ))
            # 如果特征是二维的，则计算均值作为视频级别的特征
            elif len(embeddings.shape) == 2:
                embeddings = np.mean(embeddings, axis=0)
            # 保存特征为npy文件
            np.save(csv_file, embeddings)


# 视觉特征提取的主函数
def visual_extraction():
    # 采样率，每10帧取一帧
    sample_rate = 10
    # 视频文件所在目录路径
    video_path = 'D:/HACI/MMchallenge/Video_split1/Video_split1'
    # 获取视频目录下的所有视频文件名
    video_name = os.listdir(video_path)
    # 使用tqdm显示进度条，遍历每个视频
    for video in tqdm.tqdm(video_name):
        # 如果文件名包含'mp4'，则处理该视频
        if 'mp4' in video:
            # 构建视频文件的完整路径
            video_path = os.path.join(video_path, video)
            # 如果保存视频帧的目录不存在，则创建目录
            if not os.path.exists('D:/HACI/MMchallenge/Video_split1/frame'):
                os.mkdir('D:/HACI/MMchallenge/Video_split1/frame')
            # 从视频中提取帧
            frame_extract(video_path, r'D:/HACI/MMchallenge/Video_split1/frame', sample_rate=sample_rate)

    # 打印完成帧提取的信息
    print('Finished extracting frame!')

    # 视频帧所在目录路径
    video_frame_dir = 'D:/HACI/MMchallenge/Video_split1/frame'
    # 保存特征的目录路径
    save_dir = 'D:/HACI/MMchallenge/Video_split1/features'
    # 提取视频帧特征
    feature_extract(video_frame_dir, save_dir, feature_level='UTT')


if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Run.')
    # 添加一个布尔类型的命令行参数--overwrite，用于指定是否覆盖已存在的特征文件夹，默认值为True
    parser.add_argument('--overwrite', action='store_true', default=True, help='whether overwrite existed feature folder.')
    # 添加一个字符串类型的命令行参数--dataset，用于指定输入数据集，默认值为'BoxOfLies'
    parser.add_argument('--dataset', type=str, default='BoxOfLies', help='input dataset')
    # 解析命令行参数
    params = parser.parse_args()

    # 打印开始提取resnet特征的信息
    print(f'==> Extracting resnet features...')

    # 获取数据集名称
    dataset = params.dataset
    # 输入目录：视频帧目录
    input_dir = 'D:/HACI/MMchallenge/Video_split1/frame'

    # 输出目录：特征保存目录
    save_dir = 'D:/HACI/MMchallenge/Video_split1/features'

    # 调用视觉特征提取主函数
    visual_extraction()

    pass