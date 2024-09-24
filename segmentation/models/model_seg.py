"""
@Author: Tiange Xiang
@Contact: txia7609@uni.sydney.edu.au
@File: curvenet_seg.py
@Time: 2021/01/21 3:10 PM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import Block, sample_and_group_former, Local_op
from functools import partial
from .pointnet2_utils import PointNetFeaturePropagation, AttentionPointNetFeaturePropagation


class Model(nn.Module):
    def __init__(self, samplepoint=256, embed_dim=256, depth=2, num_heads=4, mlp_ratio=2.0, qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 masked_im_modeling=False, num_classes=50, category=16, k=32, **kwargs):
        super(Model, self).__init__()

        self.samplepoint = samplepoint
        # Input Embedding
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        self.gather_local_0 = Local_op(in_channels=128, out_channels=embed_dim)

        # self.cls_token = nn.Parameter(torch.zeros(1, embed_dim, 1))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # segmentation head
        self.reduce_dims = nn.ModuleList([nn.Sequential(nn.Conv1d(embed_dim, 256, kernel_size=1, bias=False),
                                                        nn.BatchNorm1d(256),
                                                        nn.LeakyReLU(negative_slope=0.2))
                                          for i in range(3)])

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(0.2))

        # self.propagation_0 = PointNetFeaturePropagation(in_channel=512+64, mlp=[512, 512], att=[512, 64, 256])
        self.propagation_0 = PointNetFeaturePropagation(in_channel=768 + 64, mlp=[embed_dim * 4, 1024])

        # self.up_cic1 = CIC(npoint=2048, radius=0.2, k=k, in_channels=512*3+category+3, output_channels=256, bottleneck_ratio=4)

        self.h_conv1 = nn.Conv1d(2624, 512, 1)
        self.h_bn1 = nn.BatchNorm1d(512)
        self.h_drop1 = nn.Dropout(0.5)
        self.h_conv2 = nn.Conv1d(512, 256, 1)
        self.h_bn2 = nn.BatchNorm1d(256)
        self.h_conv3 = nn.Conv1d(256, num_classes, 1)
        self.se = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                nn.Conv1d(512, 512//8, 1, bias=False),
                                nn.BatchNorm1d(512//8),
                                nn.LeakyReLU(negative_slope=0.2),
                                nn.Conv1d(512//8, 512, 1, bias=False),
                                nn.Sigmoid())
                                
    def forward(self, x, l=None):
        xyz = x.detach().permute(0, 2, 1).contiguous()
        bs, N, _ = xyz.size()
        # x_clone = x.clone()
        # B, D, N
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        # B, D, N
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)

        x = x.permute(0, 2, 1).contiguous()
        x_local = x.clone()
        new_xyz, new_feature = sample_and_group_former(npoint=256, radius=None,
                                                       nsample=self.samplepoint, xyz=xyz, points=x)
        x = self.gather_local_0(new_feature)

        # cls_tokens = self.cls_token.expand(bs, -1, -1)
        # x = torch.cat((cls_tokens, x), dim=2)

        x_list = []
        fetch_idx = [3, 7]
        x = x.permute(0,2,1)
        x_list.append(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in fetch_idx:
                x_list.append(x)

        # reduce feature dims
        x_list = [self.norm(x).transpose(-1, -2).contiguous() for x in x_list]
        for i, r_blk in enumerate(self.reduce_dims):
            x_list[i] = r_blk(x_list[i])
        x = torch.cat((x_list[0], x_list[1], x_list[2]), dim=-2)  # 256*3

        x_max = torch.max(x,dim=-1, keepdim=True)[0]
        x_avg = torch.mean(x,dim=-1, keepdim=True)
        x_max_feature = x_max.repeat(1, 1, N)
        x_avg_feature = x_avg.repeat(1, 1, N)
        cls_label_one_hot = l.view(bs, -1, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)

        x_global_feature = torch.cat((x_max_feature, x_avg_feature, cls_label_feature), 1)  # 768*2 + 64
        x_fine_feature = self.propagation_0(xyz.transpose(-1, -2), new_xyz.transpose(-1, -2), x_local.transpose(-1, -2), x)

        x = torch.cat((x_fine_feature, x_global_feature), 1)

        x =  F.leaky_relu(self.h_bn1(self.h_conv1(x)), 0.2, inplace=True)
        se = self.se(x)
        x = x * se
        x = self.h_drop1(x)
        x = F.leaky_relu(self.h_bn2(self.h_conv2(x)), 0.2, inplace=True)
        x = self.h_conv3(x)

        return x


def POS_BERT(samplepoint=32, **kwargs):
    model = Model(samplepoint=samplepoint, embed_dim=512, depth=8, num_heads=8, mlp_ratio=2.0,
                  qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


class cal_loss(nn.Module):
    def __init__(self, smoothing=True):
        super(cal_loss, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, gold):
        gold = gold.contiguous().view(-1)

        if self.smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, gold, reduction='mean')

        return loss

# def cal_loss(pred, gold, smoothing=True):
#     ''' Calculate cross entropy loss, apply label smoothing if needed. '''
#
#     gold = gold.contiguous().view(-1)
#
#     if smoothing:
#         eps = 0.2
#         n_class = pred.size(1)
#
#         one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
#         one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
#         log_prb = F.log_softmax(pred, dim=1)
#
#         loss = -(one_hot * log_prb).sum(dim=1).mean()
#     else:
#         loss = F.cross_entropy(pred, gold, reduction='mean')
#
#     return loss
