import torch
import torch.nn as nn
from einops.einops import rearrange

from .backbone import build_backbone
from .utils.position_encoding import PositionEncodingSine
from .loftr_module import LocalFeatureTransformer, FinePreprocess
from .utils.coarse_matching import CoarseMatching
from .utils.fine_matching import FineMatching

#就是loftr匹配器
class Matcher(nn.Module):#定义了一个名为 Matcher 的类，继承自 nn.Module
    def __init__(self, config):#定义了该类的初始化方法，接受一个参数 config，用于配置模型。
        super().__init__()#调用父类 nn.Module 的初始化方法。有时你需要在子类构造函数中执行一些父类构造函数中的操作。 这时，你可以使用 super().
        # Misc
        self.config = config

        # Modules
        self.backbone = build_backbone(config)#使用 build_backbone 方法构建骨干网络，并将结果保存在 backbone 属性中。

        self.pos_encoding = PositionEncodingSine(
            config['coarse']['d_model'],
            temp_bug_fix=config['coarse']['temp_bug_fix'])
        #使用 Sine 位置编码构建一个 PositionEncodingSine 实例，并将结果保存在 pos_encoding 属性中。

        self.loftr_coarse = LocalFeatureTransformer(config['coarse'])#使用传入的粗特征配置信息构建一个 LocalFeatureTransformer 实例，并将结果保存在 loftr_coarse 属性中。（LoFTR论文里那个1/8粗特征）

        self.coarse_matching = CoarseMatching(config['match_coarse'])#使用传入的粗匹配配置信息构建一个 CoarseMatching 实例，并将结果保存在 coarse_matching 属性中。

        self.fine_preprocess = FinePreprocess(config)#使用传入的精细预处理配置信息构建一个 FinePreprocess 实例，并将结果保存在 fine_preprocess 属性中。（LoFTR论文里那个1/2细特征）

        self.loftr_fine = LocalFeatureTransformer(config["fine"])#使用传入的精细特征配置信息构建一个 LocalFeatureTransformer 实例，并将结果保存在 loftr_fine 属性中。

        self.fine_matching = FineMatching()
    
    def forward(self, data, only_att_fea=False ):
        #定义了 forward 方法，用于执行前向传播
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """#给 forward 方法添加了注释，说明了参数 data 的格式和含义。

        # 1. Local Feature CNN
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })
        #将输入数据字典 data 更新，添加了额外的键值对，包括批量大小（'bs'）、图像0和1的空间维度（'hw0_i'和'hw1_i'）。

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence#检查图像0和图像1的空间维度是否相同
            feats_c, feats_f = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))
            #将图像0和图像1在批次维度上拼接，并传入骨干网络中，得到粗特征（feats_c）和精细特征（feats_f）
            (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(data['bs']), feats_f.split(data['bs'])#将得到的特征按批次大小分割，得到图像0和图像1对应的粗特征和精细特征。
        else:  # handle different input shapes
            (feat_c0, feat_f0), (feat_c1, feat_f1) = self.backbone(data['image0']), self.backbone(data['image1'])
            #如果图像0和图像1的空间维度不同，则分别传入骨干网络中得到粗特征和精细特征。

        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]
        })#将特征的空间维度信息更新到数据字典 data 中。

        # 2. 粗级别的LOFTR模块
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')
        #对粗特征进行局部特征变换和位置编码，并将结果展平成序列。

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
            #初始化两个掩码变量，用于训练中的可选位置掩码。

        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)
        #使用粗级别的LOFTR模块处理粗特征。
        
        if only_att_fea:
            return  feat_c0, feat_c1
        #如果设置了 only_att_fea 标志，则直接返回注意力特征


        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)
        #调用了粗匹配的方法 coarse_matching，将粗特征 feat_c0 和 feat_c1 以及一些数据传递给该方法进行匹配。这个方法的目的是在粗糙级别上匹配两个输入图像的特征。

        # 4. fine-level refinement
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, data)
        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold = self.loftr_fine(feat_f0_unfold, feat_f1_unfold)
        #这部分代码进行了精细级别的特征细化。首先，调用了 fine_preprocess 方法对精细特征进行预处理，该方法可能会根据粗特征的预测情况来调整特征。
        # 然后，检查是否至少有一个粗特征被预测出来，如果是，则调用 loftr_fine 方法对精细特征进行细化。

        # 5. match fine-level
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)
        #这行代码调用了精细匹配的方法 fine_matching，将细化后的特征 feat_f0_unfold 和 feat_f1_unfold 以及一些数据传递给该方法进行匹配。
        # 这个方法的目的是在精细级别上匹配两个输入图像的特征。

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
#这个方法用于加载模型的状态字典，它先将状态字典中的键名中以 "matcher." 开头的部分去掉前缀 "matcher."，
# 然后再调用父类的 load_state_dict 方法加载状态字典。这个方法的目的是确保加载预训练模型时，与模型结构中的命名一致