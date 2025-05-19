import torch
import os
import json
from collections import OrderedDict
import torch.nn.functional as F
from models.base_model import BaseModel
from models.networks.lstm import LSTMEncoder
from models.networks.classifier import FcClassifier
from models.utils.config import OptConfig
import math
import torch.nn as nn
class VilTModule(nn.Module):
    """Vision-Language Transformer模块，用于跨模态交互"""
    def __init__(self, feature_dim, num_heads=8, num_layers=2, dropout=0.1):
        super(VilTModule, self).__init__()
        
        # 自注意力层 - 处理视觉特征
        self.visual_self_attn = nn.ModuleList([
            nn.MultiheadAttention(feature_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # 自注意力层 - 处理声学特征
        self.acoustic_self_attn = nn.ModuleList([
            nn.MultiheadAttention(feature_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # 交叉注意力层 - 视觉关注声学
        self.v2a_cross_attn = nn.ModuleList([
            nn.MultiheadAttention(feature_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # 交叉注意力层 - 声学关注视觉
        self.a2v_cross_attn = nn.ModuleList([
            nn.MultiheadAttention(feature_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # 前馈网络
        self.v_ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, 4 * feature_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * feature_dim, feature_dim)
            )
            for _ in range(num_layers)
        ])

        self.a_ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, 4 * feature_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * feature_dim, feature_dim)
            )
            for _ in range(num_layers)
        ])
        # 层归一化
        self.v_norm1 = nn.ModuleList([nn.LayerNorm(feature_dim) for _ in range(num_layers)])
        self.v_norm2 = nn.ModuleList([nn.LayerNorm(feature_dim) for _ in range(num_layers)])
        self.a_norm1 = nn.ModuleList([nn.LayerNorm(feature_dim) for _ in range(num_layers)])
        self.a_norm2 = nn.ModuleList([nn.LayerNorm(feature_dim) for _ in range(num_layers)])
        
        # 最终融合层
        self.fusion_projection = nn.Linear(2 * feature_dim, feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, visual_feat, acoustic_feat):
        """
        visual_feat: [batch_size, seq_len, feature_dim]
        acoustic_feat: [batch_size, seq_len, feature_dim]
        """
        # 转换为 [seq_len, batch_size, feature_dim] 以匹配PyTorch的MultiheadAttention输入要求
        visual_feat = visual_feat.permute(1, 0, 2)
        acoustic_feat = acoustic_feat.permute(1, 0, 2)
        
        v = visual_feat
        a = acoustic_feat
        
        # 多层Transformer块
        for i in range(len(self.visual_self_attn)):
            # 视觉自注意力
            v2 = self.v_norm1[i](v)
            v2, _ = self.visual_self_attn[i](v2, v2, v2)
            v = v + self.dropout(v2)
            
            # 声学自注意力
            a2 = self.a_norm1[i](a)
            a2, _ = self.acoustic_self_attn[i](a2, a2, a2)
            a = a + self.dropout(a2)
            
            # 视觉关注声学
            v3 = self.v_norm2[i](v)
            v3, _ = self.v2a_cross_attn[i](v3, a, a)
            v = v + self.dropout(v3)
            
            # 声学关注视觉
            a3 = self.a_norm2[i](a)
            a3, _ = self.a2v_cross_attn[i](a3, v, v)
            a = a + self.dropout(a3)
            
            # 前馈网络
            v4 = self.v_ffn[i](self.v_norm2[i](v))
            v = v + self.dropout(v4)
            
            a4 = self.a_ffn[i](self.a_norm2[i](a))
            a = a + self.dropout(a4)
        
        # 转换回 [batch_size, seq_len, feature_dim]
        v = v.permute(1, 0, 2)
        a = a.permute(1, 0, 2)
        
        # 融合两种模态的特征
        concat_feat = torch.cat([v, a], dim=-1)
        fused_feat = self.fusion_projection(concat_feat)
        fused_feat = self.layer_norm(fused_feat)
        
        return fused_feat
        
class ourModel(BaseModel, nn.Module):

    def __init__(self, opt):
        """
        初始化 LSTM 自编码器类
        参数:
            opt (Option 类) -- 存储所有实验标志; 需为 BaseOptions 的子类
        """
        # 调用 nn.Module 的构造函数
        nn.Module.__init__(self)
        # 调用父类 BaseModel 的构造函数
        super().__init__(opt)

        # 初始化损失名称列表，用于记录使用的损失函数名称
        self.loss_names = []
        # 初始化模型名称列表，用于记录使用的模型名称
        self.model_names = []

        # 声学模型
        # 创建一个 LSTMEncoder 实例，用于处理声学特征
        self.netEmoA = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, embd_method=opt.embd_method_a)
        # 将声学模型名称添加到模型名称列表中
        self.model_names.append('EmoA')

        # 视觉模型
        # 创建一个 LSTMEncoder 实例，用于处理视觉特征
        self.netEmoV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, opt.embd_method_v)



        # 将视觉模型名称添加到模型名称列表中
        self.model_names.append('EmoV')

        # # Transformer 融合模型
        # # 创建一个 Transformer 编码器层实例
        # emo_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=opt.hidden_size, nhead=int(opt.Transformer_head), batch_first=True)
        # # 创建一个 Transformer 编码器实例，用于融合声学和视觉特征
        # self.netEmoFusion = torch.nn.TransformerEncoder(emo_encoder_layer, num_layers=opt.Transformer_layers)
        # # 将融合模型名称添加到模型名称列表中
        # self.model_names.append('EmoFusion')
        feature_dim = opt.embd_size_a  # 或 opt.embd_size_v
        self.netVilT = VilTModule(
            feature_dim=feature_dim,
            num_heads=int(opt.Transformer_head),
            num_layers=opt.Transformer_layers,
            dropout=opt.dropout_rate
        )
        self.model_names.append('VilT')  # 添加到模型名称列表

        # 分类器
        # 将分类器层的配置字符串转换为整数列表
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))

        # cls_input_size = 5*opt.hidden_size，与最大长度相同
        # 计算分类器的输入大小，考虑了特征的最大长度和个性化特征
       
        cls_input_size = opt.feature_max_len * opt.hidden_size + 1024  # 包含个性化特征

        # 创建一个全连接分类器实例，用于情感分类
        self.netEmoC = FcClassifier(cls_input_size, cls_layers, output_dim=opt.emo_output_dim, dropout=opt.dropout_rate)
        # 将分类器模型名称添加到模型名称列表中
        self.model_names.append('EmoC')
        # 将情感分类交叉熵损失名称添加到损失名称列表中
        self.loss_names.append('emo_CE')

        # 创建另一个全连接分类器实例，用于融合特征的情感分类
        self.netEmoCF = FcClassifier(cls_input_size, cls_layers, output_dim=opt.emo_output_dim, dropout=opt.dropout_rate)
        # 将融合特征分类器模型名称添加到模型名称列表中
        self.model_names.append('EmoCF')
        # 将融合特征分类交叉熵损失名称添加到损失名称列表中
        self.loss_names.append('EmoF_CE')
        # 新增：在 __init__ 中定义 dim_mapping 层
        self.dim_mapping = nn.Linear(1152, 7424).to(self.device)  # 确保设备与模型一致
        # 温度参数，用于 softmax 操作
        self.temperature = opt.temperature
        # 确保设备设置为CUDA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert torch.cuda.is_available(), "CUDA is not available! Please check your GPU setup."
        # self.device = 'cpu'
        # 将声学模型移动到指定设备
        # self.netEmoA = self.netEmoA.to(self.device)
        # 将视觉模型移动到指定设备
        # self.netEmoV = self.netEmoV.to(self.device)
        # 将融合模型移动到指定设备
        # self.netEmoFusion = self.netEmoFusion.to(self.device)
        # 将分类器模型移动到指定设备
        # self.netEmoC = self.netEmoC.to(self.device)
        # 将融合特征分类器模型移动到指定设备
        # self.netEmoCF = self.netEmoCF.to(self.device)

        # 初始化交叉熵损失函数
        self.criterion_ce = torch.nn.CrossEntropyLoss()

        if self.isTrain:
            if not opt.use_ICL:
                # 不使用 ICL 时，初始化交叉熵损失函数
                self.criterion_ce = torch.nn.CrossEntropyLoss()
                # 不使用 ICL 时，初始化另一个交叉熵损失函数
                self.criterion_focal = torch.nn.CrossEntropyLoss() 
            else:
                # 使用 ICL 时，初始化交叉熵损失函数
                self.criterion_ce = torch.nn.CrossEntropyLoss()
                # 使用 ICL 时，初始化 Focal 损失函数
                self.criterion_focal = Focal_Loss()
            # 初始化优化器；调度器将由 <BaseModel.setup> 函数自动创建
            # 获取所有模型的参数
            paremeters = [{'params': getattr(self, 'net' + net).parameters()} for net in self.model_names]
            # 创建 Adam 优化器
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            # 将优化器添加到优化器列表中
            self.optimizers.append(self.optimizer)
            # 交叉熵损失的权重
            self.ce_weight = opt.ce_weight
            # Focal 损失的权重
            self.focal_weight = opt.focal_weight

        # 修改保存目录
        # 构建保存模型的目录路径
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name) 
        # 如果保存目录不存在，则创建该目录
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)  


    def post_process(self):
        # 在 model.setup() 之后调用
        def transform_key_for_parallel(state_dict):
            # 为并行训练修改状态字典的键名
            return OrderedDict([('module.' + key, value) for key, value in state_dict.items()])

        if self.isTrain:
            print('[ Init ] Load parameters from pretrained encoder network')
            # 定义一个函数，用于修改状态字典的键名
            f = lambda x: transform_key_for_parallel(x)
            # 从预训练的编码器网络加载声学模型的参数
            self.netEmoA.load_state_dict(f(self.pretrained_encoder.netEmoA.state_dict()))
            # 从预训练的编码器网络加载视觉模型的参数
            self.netEmoV.load_state_dict(f(self.pretrained_encoder.netEmoV.state_dict()))
            # 从预训练的编码器网络加载融合模型的参数
            self.netEmoFusion.load_state_dict(f(self.pretrained_encoder.netEmoFusion.state_dict()))

    def load_from_opt_record(self, file_path):
        # 从 JSON 文件中加载配置信息
        opt_content = json.load(open(file_path, 'r'))
        # 创建一个 OptConfig 实例
        opt = OptConfig()
        # 加载配置信息
        opt.load(opt_content)
        return opt

    def set_input(self, input):
        # 将声学特征转换为浮点型并移动到指定设备
        self.acoustic = input['A_feat'].float().to(self.device)
        # 将视觉特征转换为浮点型并移动到指定设备
        self.visual = input['V_feat'].float().to(self.device)

        # 将情感标签移动到指定设备
        self.emo_label = input['emo_label'].to(self.device)

        if 'personalized_feat' in input:
            # 如果输入中包含个性化特征，则将其转换为浮点型并移动到指定设备
            self.personalized = input['personalized_feat'].float().to(self.device)
        else:
            # 如果输入中不包含个性化特征，则将其设为 None
            self.personalized = None  # 如果没有给出个性化特征
            

    def forward(self, acoustic_feat=None, visual_feat=None):
        if acoustic_feat is not None:
            # 如果传入了声学特征，则将其转换为浮点型并移动到指定设备
            self.acoustic = acoustic_feat.float().to(self.device)
            # 如果传入了视觉特征，则将其转换为浮点型并移动到指定设备
            self.visual = visual_feat.float().to(self.device)
        
        """
        前向传播；由 <optimize_parameters> 和 <test> 函数调用
        """
        # 通过声学模型处理声学特征
        emo_feat_A = self.netEmoA(self.acoustic)
        # 通过视觉模型处理视觉特征
        emo_feat_V = self.netEmoV(self.visual)
        # print("emo feat a shape is ") 
        # print(emo_feat_A.shape)
        # print("emo feat v shape is ") 
        # print(emo_feat_V.shape)
        '''确保时间维度修改'''
        # 拼接声学和视觉特征
        # emo_fusion_feat = torch.cat((emo_feat_V, emo_feat_A), dim=-1) # (batch_size, seq_len, 2 * embd_size)
        # 可学习的权重参数
        # 可学习的权重参数
        # alpha = torch.nn.Parameter(torch.tensor(0.5))
        # emo_fusion_feat = alpha * emo_feat_V + (1-alpha) * emo_feat_A
        # emo_fusion_feat = torch.cat((emo_fusion_feat, emo_fusion_feat), dim=-1)
        
        # # 通过融合模型处理拼接后的特征
        # emo_fusion_feat = self.netEmoV(emo_fusion_feat)
        
   

        # 调整特征的维度
        emo_fusion_feat = self.netVilT(emo_feat_V, emo_feat_A)  # 调用VilT模块
        '''动态获取批量大小'''
        # 获取批量大小
        batch_size = emo_fusion_feat.size(0)
        
        if self.personalized is not None:
            personalized_expanded = self.personalized.unsqueeze(1)
            # 复制 personalized 特征，使其与 seq_len 维度匹配
            batch_size, seq_len, _ = emo_fusion_feat.shape
            personalized_expanded = personalized_expanded.expand(batch_size, seq_len, -1)
    
    # 现在两个张量都是 [batch_size, seq_len, feature_dim] 格式，可以在最后一维拼接
        emo_fusion_feat = torch.cat((emo_fusion_feat, personalized_expanded), dim=-1)
        #     # 如果存在个性化特征，则将其与融合特征拼接
        # emo_fusion_feat = torch.cat((emo_fusion_feat, self.personalized), dim=-1)  # [batch_size, seq_len * feature_dim + 1024]
        
        # 通过融合特征分类器得到融合特征的情感预测分数
        # print("emo_fusion_feat:")
        # print(emo_fusion_feat.shape)
        # self.dim_mapping = nn.Linear(1152, 7424).to(self.device)
        emo_fusion_feat = self.dim_mapping(emo_fusion_feat)
        self.emo_logits_fusion, _ = self.netEmoCF(emo_fusion_feat)
        """-----------"""

        # 通过分类器得到情感预测分数
        self.emo_logits, _ = self.netEmoC(emo_fusion_feat)
        # 对情感预测分数进行 softmax 操作，得到情感预测概率
        self.emo_pred = F.softmax(self.emo_logits, dim=-1)

    def backward(self):
        """
        计算反向传播的损失
        """
        # 计算情感分类的交叉熵损失
        # print("emo_logits:")
        # print(self.emo_logits.shape)
        # print("emo.label")
        # print(self.emo_label.shape)# 扩展标签维度以匹配模型输出
        batch_size, seq_len, num_classes = self.emo_logits.shape
        expanded_labels = self.emo_label.unsqueeze(1).expand(-1, seq_len)  # [batch_size, seq_len]
    
    # 重塑模型输出为 [batch_size*seq_len, num_classes]
        logits_reshaped = self.emo_logits.reshape(-1, num_classes)  # [batch_size*seq_len, num_classes]
        labels_reshaped = expanded_labels.reshape(-1)  # [batch_size*seq_len]
        
        self.loss_emo_CE = self.criterion_ce(logits_reshaped, labels_reshaped) 
        # 计算融合特征分类的 Focal 损失
        self.loss_EmoF_CE = self.focal_weight * self.criterion_focal(logits_reshaped, labels_reshaped)
        # 总损失为情感分类交叉熵损失和融合特征分类 Focal 损失之和
        loss = self.loss_emo_CE + self.loss_EmoF_CE

        # 反向传播计算梯度
        loss.backward()

        # 对每个模型的参数梯度进行裁剪，防止梯度爆炸
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net' + model).parameters(), 1.0)

    def optimize_parameters(self, epoch):
        """
        计算损失、梯度并更新网络权重；在每个训练迭代中调用
        """
        # 前向传播
        self.forward()
        # 清空优化器的梯度
        self.optimizer.zero_grad()
        # 反向传播计算梯度
        self.backward()

        # 更新网络权重
        self.optimizer.step()


class ActivateFun(torch.nn.Module):
    def __init__(self, opt):
        # 调用父类的构造函数
        super(ActivateFun, self).__init__()
        # 激活函数的名称
        self.activate_fun = opt.activate_fun

    def _gelu(self, x):
        # GELU 激活函数的实现
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x):
        if self.activate_fun == 'relu':
            # 如果激活函数为 ReLU，则使用 ReLU 激活函数
            return torch.relu(x)
        elif self.activate_fun == 'gelu':
            # 如果激活函数为 GELU，则使用 GELU 激活函数
            return self._gelu(x)


class Focal_Loss(torch.nn.Module):
    def __init__(self, weight=0.5, gamma=3, reduction='mean'):
        # 调用父类的构造函数
        super(Focal_Loss, self).__init__()
        # 聚焦参数
        self.gamma = gamma
        # 权重参数
        self.alpha = weight
        # 损失的缩减方式
        self.reduction = reduction

    def forward(self, preds, targets):
        """
        preds: softmax 输出
        labels: 真实值
        """
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(preds, targets, reduction='mean')
        # 计算预测概率
        pt = torch.exp(-ce_loss)
        # 计算 Focal 损失
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'none':
            # 如果缩减方式为 none，则返回未缩减的损失
            return focal_loss
        elif self.reduction == 'mean':
            # 如果缩减方式为 mean，则返回平均损失
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            # 如果缩减方式为 sum，则返回损失之和
            return torch.sum(focal_loss)
        else:
            # 如果缩减方式无效，则抛出异常
            raise NotImplementedError("Invalid reduction mode. Please choose 'none', 'mean', or 'sum'.")

#Baseline Code:
# import torch
# import os
# import json
# from collections import OrderedDict
# import torch.nn.functional as F
# from models.base_model import BaseModel
# from models.networks.lstm import LSTMEncoder
# from models.networks.classifier import FcClassifier
# from models.utils.config import OptConfig
# import math
# import torch.nn as nn


# class ourModel(BaseModel, nn.Module):

#     def __init__(self, opt):
#         """
#         初始化 LSTM 自编码器类
#         参数:
#             opt (Option 类) -- 存储所有实验标志; 需为 BaseOptions 的子类
#         """
#         # 调用 nn.Module 的构造函数
#         nn.Module.__init__(self)
#         # 调用父类 BaseModel 的构造函数
#         super().__init__(opt)

#         # 初始化损失名称列表，用于记录使用的损失函数名称
#         self.loss_names = []
#         # 初始化模型名称列表，用于记录使用的模型名称
#         self.model_names = []

#         # 声学模型
#         # 创建一个 LSTMEncoder 实例，用于处理声学特征
#         self.netEmoA = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, embd_method=opt.embd_method_a)
#         # 将声学模型名称添加到模型名称列表中
#         self.model_names.append('EmoA')

#         # 视觉模型
#         # 创建一个 LSTMEncoder 实例，用于处理视觉特征
#         self.netEmoV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, opt.embd_method_v)
#         # 将视觉模型名称添加到模型名称列表中
#         self.model_names.append('EmoV')

#         # Transformer 融合模型
#         # 创建一个 Transformer 编码器层实例
#         emo_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=opt.hidden_size, nhead=int(opt.Transformer_head), batch_first=True)
#         # 创建一个 Transformer 编码器实例，用于融合声学和视觉特征
#         self.netEmoFusion = torch.nn.TransformerEncoder(emo_encoder_layer, num_layers=opt.Transformer_layers)
#         # 将融合模型名称添加到模型名称列表中
#         self.model_names.append('EmoFusion')

#         # 分类器
#         # 将分类器层的配置字符串转换为整数列表
#         cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))

#         # cls_input_size = 5*opt.hidden_size，与最大长度相同
#         # 计算分类器的输入大小，考虑了特征的最大长度和个性化特征
#         cls_input_size = opt.feature_max_len * opt.hidden_size + 1024  # 包含个性化特征

#         # 创建一个全连接分类器实例，用于情感分类
#         self.netEmoC = FcClassifier(cls_input_size, cls_layers, output_dim=opt.emo_output_dim, dropout=opt.dropout_rate)
#         # 将分类器模型名称添加到模型名称列表中
#         self.model_names.append('EmoC')
#         # 将情感分类交叉熵损失名称添加到损失名称列表中
#         self.loss_names.append('emo_CE')

#         # 创建另一个全连接分类器实例，用于融合特征的情感分类
#         self.netEmoCF = FcClassifier(cls_input_size, cls_layers, output_dim=opt.emo_output_dim, dropout=opt.dropout_rate)
#         # 将融合特征分类器模型名称添加到模型名称列表中
#         self.model_names.append('EmoCF')
#         # 将融合特征分类交叉熵损失名称添加到损失名称列表中
#         self.loss_names.append('EmoF_CE')

#         # 温度参数，用于 softmax 操作
#         self.temperature = opt.temperature

#         # self.device = 'cpu'
#         # 将声学模型移动到指定设备
#         # self.netEmoA = self.netEmoA.to(self.device)
#         # 将视觉模型移动到指定设备
#         # self.netEmoV = self.netEmoV.to(self.device)
#         # 将融合模型移动到指定设备
#         # self.netEmoFusion = self.netEmoFusion.to(self.device)
#         # 将分类器模型移动到指定设备
#         # self.netEmoC = self.netEmoC.to(self.device)
#         # 将融合特征分类器模型移动到指定设备
#         # self.netEmoCF = self.netEmoCF.to(self.device)

#         # 初始化交叉熵损失函数
#         self.criterion_ce = torch.nn.CrossEntropyLoss()

#         if self.isTrain:
#             if not opt.use_ICL:
#                 # 不使用 ICL 时，初始化交叉熵损失函数
#                 self.criterion_ce = torch.nn.CrossEntropyLoss()
#                 # 不使用 ICL 时，初始化另一个交叉熵损失函数
#                 self.criterion_focal = torch.nn.CrossEntropyLoss() 
#             else:
#                 # 使用 ICL 时，初始化交叉熵损失函数
#                 self.criterion_ce = torch.nn.CrossEntropyLoss()
#                 # 使用 ICL 时，初始化 Focal 损失函数
#                 self.criterion_focal = Focal_Loss()
#             # 初始化优化器；调度器将由 <BaseModel.setup> 函数自动创建
#             # 获取所有模型的参数
#             paremeters = [{'params': getattr(self, 'net' + net).parameters()} for net in self.model_names]
#             # 创建 Adam 优化器
#             self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
#             # 将优化器添加到优化器列表中
#             self.optimizers.append(self.optimizer)
#             # 交叉熵损失的权重
#             self.ce_weight = opt.ce_weight
#             # Focal 损失的权重
#             self.focal_weight = opt.focal_weight

#         # 修改保存目录
#         # 构建保存模型的目录路径
#         self.save_dir = os.path.join(opt.checkpoints_dir, opt.name) 
#         # 如果保存目录不存在，则创建该目录
#         if not os.path.exists(self.save_dir):
#             os.makedirs(self.save_dir)  


#     def post_process(self):
#         # 在 model.setup() 之后调用
#         def transform_key_for_parallel(state_dict):
#             # 为并行训练修改状态字典的键名
#             return OrderedDict([('module.' + key, value) for key, value in state_dict.items()])

#         if self.isTrain:
#             print('[ Init ] Load parameters from pretrained encoder network')
#             # 定义一个函数，用于修改状态字典的键名
#             f = lambda x: transform_key_for_parallel(x)
#             # 从预训练的编码器网络加载声学模型的参数
#             self.netEmoA.load_state_dict(f(self.pretrained_encoder.netEmoA.state_dict()))
#             # 从预训练的编码器网络加载视觉模型的参数
#             self.netEmoV.load_state_dict(f(self.pretrained_encoder.netEmoV.state_dict()))
#             # 从预训练的编码器网络加载融合模型的参数
#             self.netEmoFusion.load_state_dict(f(self.pretrained_encoder.netEmoFusion.state_dict()))

#     def load_from_opt_record(self, file_path):
#         # 从 JSON 文件中加载配置信息
#         opt_content = json.load(open(file_path, 'r'))
#         # 创建一个 OptConfig 实例
#         opt = OptConfig()
#         # 加载配置信息
#         opt.load(opt_content)
#         return opt

#     def set_input(self, input):
#         # 将声学特征转换为浮点型并移动到指定设备
#         self.acoustic = input['A_feat'].float().to(self.device)
#         # 将视觉特征转换为浮点型并移动到指定设备
#         self.visual = input['V_feat'].float().to(self.device)

#         # 将情感标签移动到指定设备
#         self.emo_label = input['emo_label'].to(self.device)

#         if 'personalized_feat' in input:
#             # 如果输入中包含个性化特征，则将其转换为浮点型并移动到指定设备
#             self.personalized = input['personalized_feat'].float().to(self.device)
#         else:
#             # 如果输入中不包含个性化特征，则将其设为 None
#             self.personalized = None  # 如果没有给出个性化特征
            

#     def forward(self, acoustic_feat=None, visual_feat=None):
#         if acoustic_feat is not None:
#             # 如果传入了声学特征，则将其转换为浮点型并移动到指定设备
#             self.acoustic = acoustic_feat.float().to(self.device)
#             # 如果传入了视觉特征，则将其转换为浮点型并移动到指定设备
#             self.visual = visual_feat.float().to(self.device)
        
#         """
#         前向传播；由 <optimize_parameters> 和 <test> 函数调用
#         """
#         # 通过声学模型处理声学特征
#         emo_feat_A = self.netEmoA(self.acoustic)
#         # 通过视觉模型处理视觉特征
#         emo_feat_V = self.netEmoV(self.visual)

#         '''确保时间维度修改'''
#         # 拼接声学和视觉特征
#         emo_fusion_feat = torch.cat((emo_feat_V, emo_feat_A), dim=-1) # (batch_size, seq_len, 2 * embd_size)
        
#         # 通过融合模型处理拼接后的特征
#         emo_fusion_feat = self.netEmoFusion(emo_fusion_feat)
        
#         '''动态获取批量大小'''
#         # 获取批量大小
#         batch_size = emo_fusion_feat.size(0)

#         # 调整特征的维度
#         emo_fusion_feat = emo_fusion_feat.permute(1, 0, 2).reshape(batch_size, -1)  # 转换为 [batch_size, feature_dim] 1028

#         if self.personalized is not None:
#             # 如果存在个性化特征，则将其与融合特征拼接
#             emo_fusion_feat = torch.cat((emo_fusion_feat, self.personalized), dim=-1)  # [batch_size, seq_len * feature_dim + 1024]

#         '''用于反向传播'''
#         # 通过融合特征分类器得到融合特征的情感预测分数
#         self.emo_logits_fusion, _ = self.netEmoCF(emo_fusion_feat)
#         """-----------"""

#         # 通过分类器得到情感预测分数
#         self.emo_logits, _ = self.netEmoC(emo_fusion_feat)
#         # 对情感预测分数进行 softmax 操作，得到情感预测概率
#         self.emo_pred = F.softmax(self.emo_logits, dim=-1)

#     def backward(self):
#         """
#         计算反向传播的损失
#         """
#         # 计算情感分类的交叉熵损失
#         self.loss_emo_CE = self.criterion_ce(self.emo_logits, self.emo_label) 
#         # 计算融合特征分类的 Focal 损失
#         self.loss_EmoF_CE = self.focal_weight * self.criterion_focal(self.emo_logits_fusion, self.emo_label)
#         # 总损失为情感分类交叉熵损失和融合特征分类 Focal 损失之和
#         loss = self.loss_emo_CE + self.loss_EmoF_CE

#         # 反向传播计算梯度
#         loss.backward()

#         # 对每个模型的参数梯度进行裁剪，防止梯度爆炸
#         for model in self.model_names:
#             torch.nn.utils.clip_grad_norm_(getattr(self, 'net' + model).parameters(), 1.0)

#     def optimize_parameters(self, epoch):
#         """
#         计算损失、梯度并更新网络权重；在每个训练迭代中调用
#         """
#         # 前向传播
#         self.forward()
#         # 清空优化器的梯度
#         self.optimizer.zero_grad()
#         # 反向传播计算梯度
#         self.backward()

#         # 更新网络权重
#         self.optimizer.step()


# class ActivateFun(torch.nn.Module):
#     def __init__(self, opt):
#         # 调用父类的构造函数
#         super(ActivateFun, self).__init__()
#         # 激活函数的名称
#         self.activate_fun = opt.activate_fun

#     def _gelu(self, x):
#         # GELU 激活函数的实现
#         return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

#     def forward(self, x):
#         if self.activate_fun == 'relu':
#             # 如果激活函数为 ReLU，则使用 ReLU 激活函数
#             return torch.relu(x)
#         elif self.activate_fun == 'gelu':
#             # 如果激活函数为 GELU，则使用 GELU 激活函数
#             return self._gelu(x)


# class Focal_Loss(torch.nn.Module):
#     def __init__(self, weight=0.5, gamma=3, reduction='mean'):
#         # 调用父类的构造函数
#         super(Focal_Loss, self).__init__()
#         # 聚焦参数
#         self.gamma = gamma
#         # 权重参数
#         self.alpha = weight
#         # 损失的缩减方式
#         self.reduction = reduction

#     def forward(self, preds, targets):
#         """
#         preds: softmax 输出
#         labels: 真实值
#         """
#         # 计算交叉熵损失
#         ce_loss = F.cross_entropy(preds, targets, reduction='mean')
#         # 计算预测概率
#         pt = torch.exp(-ce_loss)
#         # 计算 Focal 损失
#         focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

#         if self.reduction == 'none':
#             # 如果缩减方式为 none，则返回未缩减的损失
#             return focal_loss
#         elif self.reduction == 'mean':
#             # 如果缩减方式为 mean，则返回平均损失
#             return torch.mean(focal_loss)
#         elif self.reduction == 'sum':
#             # 如果缩减方式为 sum，则返回损失之和
#             return torch.sum(focal_loss)
#         else:
#             # 如果缩减方式无效，则抛出异常
#             raise NotImplementedError("Invalid reduction mode. Please choose 'none', 'mean', or 'sum'.")