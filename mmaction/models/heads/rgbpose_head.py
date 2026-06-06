# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model.weight_init import normal_init, constant_init
from mmengine.structures import LabelData
from mmaction.registry import MODELS
from mmaction.utils import SampleList
from .base import BaseHead
import numpy as np

# 辅助函数：动作 -> 身体部位映射 (保留这部分，这是针对数据集的基础适配)
def action2body(x):
    if x <= 4:
        return 0
    elif 5 <= x <= 10:
        return 1
    elif 11 <= x <= 23:
        return 2
    elif 24 <= x <= 31:
        return 3
    elif 32 <= x <= 37:
        return 4
    elif 38 <= x <= 47:
        return 5
    else:
        return 6

def map_action_to_body_tensor(labels):
    """
    将 52 类的细粒度 label 批量映射为 7 类的 coarse body part label.
    支持 PyTorch Tensor.
    """
    body_labels = torch.zeros_like(labels)
    body_labels[labels <= 4] = 0
    body_labels[(labels >= 5) & (labels <= 10)] = 1
    body_labels[(labels >= 11) & (labels <= 23)] = 2
    body_labels[(labels >= 24) & (labels <= 31)] = 3
    body_labels[(labels >= 32) & (labels <= 37)] = 4
    body_labels[(labels >= 38) & (labels <= 47)] = 5
    body_labels[labels >= 48] = 6
    return body_labels

@MODELS.register_module()
class RGBPoseHead(BaseHead):
    def __init__(self,
                 num_classes: int,
                 in_channels: Tuple[int],
                 loss_cls: Dict = dict(type='CrossEntropyLoss'),
                 loss_components: List[str] = ['rgb', 'pose'],
                 loss_weights: Union[float, Tuple[float]] = 1.,
                 dropout: float = 0.5,
                 init_std: float = 0.01,
                 **kwargs) -> None:
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        # 1. 初始化 Loss 组件
        if loss_components is not None:
            self.loss_components = loss_components
            if isinstance(loss_weights, float):
                loss_weights = [loss_weights] * len(loss_components)
            self.loss_weights = loss_weights
        
        if isinstance(dropout, float):
            dropout = {'rgb': dropout, 'pose': dropout}
        
        self.dropout = dropout
        self.init_std = init_std

        # 2. 定义 Dropout 和 分类头
        self.dropout_rgb = nn.Dropout(p=self.dropout['rgb'])
        self.dropout_pose = nn.Dropout(p=self.dropout['pose'])

        self.fc_rgb = nn.Linear(self.in_channels[0], num_classes)
        self.fc_pose = nn.Linear(self.in_channels[1], num_classes)
        
        # [Fix] 恢复 Body Coarse Heads (解决 Body Acc 2% 的问题)
        self.fc_rgb_coarse = nn.Linear(self.in_channels[0], 7)
        self.fc_pose_coarse = nn.Linear(self.in_channels[1], 7)
        
        # [CLF] Class-wise Learnable Fusion Parameter
        # 定义一个长度为 num_classes (52) 的可学习参数
        # 初始值为 0 -> 经过 tanh 后为 0 -> 最终权重为 1.0 (Baseline)
        self.pose_fusion_weight = nn.Parameter(torch.zeros(1, num_classes))

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def init_weights(self) -> None:
        normal_init(self.fc_rgb, std=self.init_std)
        normal_init(self.fc_pose, std=self.init_std)
        normal_init(self.fc_rgb_coarse, std=self.init_std)
        normal_init(self.fc_pose_coarse, std=self.init_std)

        # 显式初始化融合权重为 0 (对应 Baseline 策略)
        nn.init.constant_(self.pose_fusion_weight, 0)

    def forward(self, x: List[torch.Tensor]) -> Dict:
        x_rgb, x_pose = self.avg_pool(x[0]), self.avg_pool(x[1])
        x_rgb = x_rgb.view(x_rgb.size(0), -1)
        x_pose = x_pose.view(x_pose.size(0), -1)

        x_rgb_drop = self.dropout_rgb(x_rgb)
        x_pose_drop = self.dropout_pose(x_pose)

        # 1. 计算 Action Logits (52类)
        logits_rgb = self.fc_rgb(x_rgb_drop)
        logits_pose = self.fc_pose(x_pose_drop)

        # 2. 计算 Body Logits (7类) - [Fix] 恢复计算
        logits_rgb_coarse = self.fc_rgb_coarse(x_rgb_drop)
        logits_pose_coarse = self.fc_pose_coarse(x_pose_drop)

        # 3. [CLF] 类别级动态融合
        # weight = 1.0 + tanh(p) -> 范围 (0, 2)，初始值 1.0
        w_pose = 1.0 + torch.tanh(self.pose_fusion_weight)

        # 广播机制: [1, 52] * [N, 52] -> [N, 52]
        score_fusion = logits_rgb + w_pose * logits_pose

        # 4. 统一装入字典
        cls_scores = dict()
        cls_scores['rgb'] = logits_rgb
        cls_scores['pose'] = logits_pose
        cls_scores['rgb_coarse'] = logits_rgb_coarse 
        cls_scores['pose_coarse'] = logits_pose_coarse 

        if self.training:
            cls_scores['fusion_train'] = score_fusion
        else:
            cls_scores['fusion'] = score_fusion

        return cls_scores
    
    def loss(self, feats, data_samples, **kwargs):
        cls_scores = self(feats, **kwargs)
        return self.loss_by_feat(cls_scores, data_samples)

    def loss_by_feat(self, cls_scores, data_samples):
        labels = torch.stack([x.gt_labels.item for x in data_samples])
        labels = labels.squeeze()
        if labels.shape == torch.Size([]): labels = labels.unsqueeze(0)

        losses = dict()
        
        # 1. Action Loss (RGB & Pose)
        for loss_name in self.loss_components:
            cls_score = cls_scores[loss_name]
            loss_cls = self.loss_by_scores(cls_score, labels)
            losses[f'{loss_name}_loss_cls'] = loss_cls['loss_cls']
            
            # 2. Body Loss (Auxiliary) - [Fix] 恢复辅助 Loss
            # 生成 Body Labels
            labels_body = labels.cpu().numpy()
            labels_body = np.array([action2body(i) for i in labels_body])
            labels_body = torch.tensor(labels_body).to(labels.device)
            
            cls_score_coarse = cls_scores[loss_name+'_coarse']
            loss_cls_coarse = self.loss_by_scores(cls_score_coarse, labels_body)
            losses[f'{loss_name}_coarse_loss_cls'] = loss_cls_coarse['loss_cls']

        # 3. [CLF] Fusion Loss
        # 必须监督融合分数，否则 pose_fusion_weight 没有梯度
        if 'fusion_train' in cls_scores:
            loss_fusion = self.loss_by_scores(cls_scores['fusion_train'], labels)
            losses['loss_fusion'] = loss_fusion['loss_cls']

        return losses

    def loss_by_scores(self, cls_scores, labels):
        losses = dict()
        loss_cls = self.loss_cls(cls_scores, labels)
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls
        return losses

    def predict_by_feat(self, cls_scores, data_samples):
        pred_scores = [LabelData() for _ in range(len(data_samples))]
        pred_labels = [LabelData() for _ in range(len(data_samples))]
        num_segs = cls_scores['fusion'].shape[0] // len(data_samples)

        keys_to_save = ['rgb', 'pose', 'rgb_coarse', 'pose_coarse', 'fusion']

        
        for key in keys_to_save:
            if key in cls_scores:
                logits = cls_scores[key]
                avg_score = self.average_clip(logits, num_segs=num_segs)
                pred_label = avg_score.argmax(dim=-1, keepdim=True).detach()
                for i, (score_v, label_v) in enumerate(zip(avg_score, pred_label)):
                    pred_scores[i].set_data({key: score_v})
                    pred_labels[i].set_data({key: label_v})

        for data_sample, pred_score, pred_label in zip(data_samples, pred_scores, pred_labels):
            data_sample.pred_scores = pred_score
            data_sample.pred_labels = pred_label

        return data_samples

# 将这段代码添加到你的 rgbpose_head.py 中，然后修改 config 文件调用这个新的 Head 即可跑对比实验
@MODELS.register_module()
class RGBPoseHeadInstance(BaseHead):
    """
    用于对比实验的 Instance-wise Attention Fusion Head
    """
    def __init__(self,
                 num_classes: int,
                 in_channels: Tuple[int],
                 loss_cls: Dict = dict(type='CrossEntropyLoss'),
                 loss_components: List[str] = ['rgb', 'pose'],
                 loss_weights: Union[float, Tuple[float]] = 1.,
                 dropout: float = 0.5,
                 init_std: float = 0.01,
                 **kwargs) -> None:
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        
        if loss_components is not None:
            self.loss_components = loss_components
            if isinstance(loss_weights, float):
                loss_weights = [loss_weights] * len(loss_components)
            self.loss_weights = loss_weights
        
        if isinstance(dropout, float):
            dropout = {'rgb': dropout, 'pose': dropout}
        
        self.dropout = dropout
        self.init_std = init_std

        # Dropout 和 分类头
        self.dropout_rgb = nn.Dropout(p=self.dropout['rgb'])
        self.dropout_pose = nn.Dropout(p=self.dropout['pose'])

        self.fc_rgb = nn.Linear(self.in_channels[0], num_classes)
        self.fc_pose = nn.Linear(self.in_channels[1], num_classes)
        
        self.fc_rgb_coarse = nn.Linear(self.in_channels[0], 7)
        self.fc_pose_coarse = nn.Linear(self.in_channels[1], 7)
        
        # [Instance-wise Attention] 
        # 将 RGB (2048) 和 Pose (512) 拼接，经过一个小 MLP 预测当前样本的类别权重
        concat_dim = self.in_channels[0] + self.in_channels[1]
        reduction_dim = concat_dim // 4 # 降维比例可调
        self.instance_attention = nn.Sequential(
            nn.Linear(concat_dim, reduction_dim),
            nn.ReLU(inplace=True),
            nn.Linear(reduction_dim, num_classes) # 输出每个样本针对 52 个类别的权重偏移量
        )

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def init_weights(self) -> None:
        normal_init(self.fc_rgb, std=self.init_std)
        normal_init(self.fc_pose, std=self.init_std)
        normal_init(self.fc_rgb_coarse, std=self.init_std)
        normal_init(self.fc_pose_coarse, std=self.init_std)
        
        # 初始化 Attention 模块，使得初始阶段倾向于输出 0
        for m in self.instance_attention.modules():
            if isinstance(m, nn.Linear):
                normal_init(m, std=0.001)
        constant_init(self.instance_attention[-1], val=0, bias=0)

    def forward(self, x: List[torch.Tensor]) -> Dict:
        x_rgb, x_pose = self.avg_pool(x[0]), self.avg_pool(x[1])
        x_rgb = x_rgb.view(x_rgb.size(0), -1)
        x_pose = x_pose.view(x_pose.size(0), -1)

        x_rgb_drop = self.dropout_rgb(x_rgb)
        x_pose_drop = self.dropout_pose(x_pose)

        logits_rgb = self.fc_rgb(x_rgb_drop)
        logits_pose = self.fc_pose(x_pose_drop)
        
        logits_rgb_coarse = self.fc_rgb_coarse(x_rgb_drop)
        logits_pose_coarse = self.fc_pose_coarse(x_pose_drop)
        
        # ---------------------------------------------------------
        # [Instance-wise] 动态计算权重
        # 1. 拼接当前样本特征: [N, 2048+512]
        concat_feat = torch.cat([x_rgb_drop, x_pose_drop], dim=1)
        
        # 2. 生成基于当前样本的权重偏移: [N, 52]
        # 注意：这里的 p 针对每个样本都是不同的
        p_instance = self.instance_attention(concat_feat)
        
        # 3. 限制权重范围在 (0, 2) 之间，初始接近 1.0
        w_pose = 1.0 + torch.tanh(p_instance)
        # ---------------------------------------------------------
        
        # 融合: [N, 52] + [N, 52] * [N, 52] (element-wise 乘法)
        score_fusion = logits_rgb + w_pose * logits_pose

        cls_scores = dict()
        cls_scores['rgb'] = logits_rgb
        cls_scores['pose'] = logits_pose
        cls_scores['rgb_coarse'] = logits_rgb_coarse 
        cls_scores['pose_coarse'] = logits_pose_coarse 

        if self.training:
            cls_scores['fusion_train'] = score_fusion
        else:
            cls_scores['fusion'] = score_fusion

        return cls_scores

    def loss(self, feats, data_samples, **kwargs):
        cls_scores = self(feats, **kwargs)
        return self.loss_by_feat(cls_scores, data_samples)

    def loss_by_feat(self, cls_scores, data_samples):
        labels = torch.stack([x.gt_labels.item for x in data_samples])
        labels = labels.squeeze()
        if labels.shape == torch.Size([]): labels = labels.unsqueeze(0)

        losses = dict()
        
        # 1. Action Loss (RGB & Pose)
        for loss_name in self.loss_components:
            cls_score = cls_scores[loss_name]
            loss_cls = self.loss_by_scores(cls_score, labels)
            losses[f'{loss_name}_loss_cls'] = loss_cls['loss_cls']
            
            # 2. Body Loss (Auxiliary)
            labels_body = labels.cpu().numpy()
            labels_body = np.array([action2body(i) for i in labels_body])
            labels_body = torch.tensor(labels_body).to(labels.device)
            
            cls_score_coarse = cls_scores[loss_name+'_coarse']
            loss_cls_coarse = self.loss_by_scores(cls_score_coarse, labels_body)
            losses[f'{loss_name}_coarse_loss_cls'] = loss_cls_coarse['loss_cls']

        # 3. Fusion Loss (Optional for Baseline)
        # 对于 Baseline，通常也会监督融合后的分数，这有助于网络在融合层面学习
        if 'fusion_train' in cls_scores:
            loss_fusion = self.loss_by_scores(cls_scores['fusion_train'], labels)
            losses['loss_fusion'] = loss_fusion['loss_cls']

        return losses

    def loss_by_scores(self, cls_scores, labels):
        losses = dict()
        loss_cls = self.loss_cls(cls_scores, labels)
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls
        return losses
    
    def predict_by_feat(self, cls_scores, data_samples):
        pred_scores = [LabelData() for _ in range(len(data_samples))]
        pred_labels = [LabelData() for _ in range(len(data_samples))]
        
        # 兼容 batch size 可能变化的情况
        if 'fusion' in cls_scores:
            num_segs = cls_scores['fusion'].shape[0] // len(data_samples)
        else:
            num_segs = 1
        
        keys_to_save = ['rgb', 'pose', 'rgb_coarse', 'pose_coarse', 'fusion']
        
        for key in keys_to_save:
            if key in cls_scores:
                logits = cls_scores[key]
                avg_score = self.average_clip(logits, num_segs=num_segs)
                pred_label = avg_score.argmax(dim=-1, keepdim=True).detach()
                for i, (score_v, label_v) in enumerate(zip(avg_score, pred_label)):
                    pred_scores[i].set_data({key: score_v})
                    pred_labels[i].set_data({key: label_v})

        for data_sample, pred_score, pred_label in zip(data_samples, pred_scores, pred_labels):
            data_sample.pred_scores = pred_score
            data_sample.pred_labels = pred_label

        return data_samples

# baseline head
@MODELS.register_module()
class RGBPoseHeadBaseline(BaseHead):
    def __init__(self,
                 num_classes: int,
                 in_channels: Tuple[int],
                 loss_cls: Dict = dict(type='CrossEntropyLoss'),
                 loss_components: List[str] = ['rgb', 'pose'],
                 loss_weights: Union[float, Tuple[float]] = 1.,
                 dropout: float = 0.5,
                 init_std: float = 0.01,
                 **kwargs) -> None:
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        
        # 1. 初始化 Loss 组件
        if loss_components is not None:
            self.loss_components = loss_components
            if isinstance(loss_weights, float):
                loss_weights = [loss_weights] * len(loss_components)
            self.loss_weights = loss_weights
        
        if isinstance(dropout, float):
            dropout = {'rgb': dropout, 'pose': dropout}
        
        self.dropout = dropout
        self.init_std = init_std

        # 2. 定义 Dropout 和 分类头
        self.dropout_rgb = nn.Dropout(p=self.dropout['rgb'])
        self.dropout_pose = nn.Dropout(p=self.dropout['pose'])

        self.fc_rgb = nn.Linear(self.in_channels[0], num_classes)
        self.fc_pose = nn.Linear(self.in_channels[1], num_classes)
        
        # [Fix] 恢复 Body Coarse Heads
        self.fc_rgb_coarse = nn.Linear(self.in_channels[0], 7)
        self.fc_pose_coarse = nn.Linear(self.in_channels[1], 7)
        
        # ===> 删除了 CLF 的 pose_fusion_weight 参数 <===

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def init_weights(self) -> None:
        normal_init(self.fc_rgb, std=self.init_std)
        normal_init(self.fc_pose, std=self.init_std)
        normal_init(self.fc_rgb_coarse, std=self.init_std)
        normal_init(self.fc_pose_coarse, std=self.init_std)
        # ===> 删除了 CLF 的初始化代码 <===

    def forward(self, x: List[torch.Tensor]) -> Dict:
        x_rgb, x_pose = self.avg_pool(x[0]), self.avg_pool(x[1])
        x_rgb = x_rgb.view(x_rgb.size(0), -1)
        x_pose = x_pose.view(x_pose.size(0), -1)

        x_rgb_drop = self.dropout_rgb(x_rgb)
        x_pose_drop = self.dropout_pose(x_pose)

        # 1. 计算 Action Logits (52类)
        logits_rgb = self.fc_rgb(x_rgb_drop)
        logits_pose = self.fc_pose(x_pose_drop)
        
        # 2. 计算 Body Logits (7类)
        logits_rgb_coarse = self.fc_rgb_coarse(x_rgb_drop)
        logits_pose_coarse = self.fc_pose_coarse(x_pose_drop)
        
        # ===> 删除了 w_pose 动态权重的计算 <===
        
        # 3. Static Fusion: 直接 1:1 相加，没有任何偏好
        score_fusion = logits_rgb + logits_pose

        cls_scores = dict()
        cls_scores['rgb'] = logits_rgb
        cls_scores['pose'] = logits_pose
        cls_scores['rgb_coarse'] = logits_rgb_coarse
        cls_scores['pose_coarse'] = logits_pose_coarse

        if self.training:
            cls_scores['fusion_train'] = score_fusion
        else:
            cls_scores['fusion'] = score_fusion

        return cls_scores

    def loss(self, feats, data_samples, **kwargs):
        cls_scores = self(feats, **kwargs)
        return self.loss_by_feat(cls_scores, data_samples)

    def loss_by_feat(self, cls_scores, data_samples):
        labels = torch.stack([x.gt_labels.item for x in data_samples])
        labels = labels.squeeze()
        if labels.shape == torch.Size([]): labels = labels.unsqueeze(0)

        losses = dict()
        
        # 1. Action Loss (RGB & Pose)
        for loss_name in self.loss_components:
            cls_score = cls_scores[loss_name]
            loss_cls = self.loss_by_scores(cls_score, labels)
            losses[f'{loss_name}_loss_cls'] = loss_cls['loss_cls']
            
            # 2. Body Loss (Auxiliary)
            labels_body = labels.cpu().numpy()
            labels_body = np.array([action2body(i) for i in labels_body])
            labels_body = torch.tensor(labels_body).to(labels.device)
            
            cls_score_coarse = cls_scores[loss_name+'_coarse']
            loss_cls_coarse = self.loss_by_scores(cls_score_coarse, labels_body)
            losses[f'{loss_name}_coarse_loss_cls'] = loss_cls_coarse['loss_cls']

        # ===> 删除了 Fusion 相关的监督 Loss，因为现在没有参数需要训练了 <===

        return losses

    def loss_by_scores(self, cls_scores, labels):
        losses = dict()
        loss_cls = self.loss_cls(cls_scores, labels)
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls
        return losses
    
    def predict_by_feat(self, cls_scores, data_samples):
        pred_scores = [LabelData() for _ in range(len(data_samples))]
        pred_labels = [LabelData() for _ in range(len(data_samples))]
        num_segs = cls_scores['fusion'].shape[0] // len(data_samples)
        
        # 保存所有 Key
        keys_to_save = ['rgb', 'pose', 'rgb_coarse', 'pose_coarse', 'fusion']
        
        for key in keys_to_save:
            if key in cls_scores:
                logits = cls_scores[key]
                avg_score = self.average_clip(logits, num_segs=num_segs)
                pred_label = avg_score.argmax(dim=-1, keepdim=True).detach()
                for i, (score_v, label_v) in enumerate(zip(avg_score, pred_label)):
                    pred_scores[i].set_data({key: score_v})
                    pred_labels[i].set_data({key: label_v})

        for data_sample, pred_score, pred_label in zip(data_samples, pred_scores, pred_labels):
            data_sample.pred_scores = pred_score
            data_sample.pred_labels = pred_label

        return data_samples