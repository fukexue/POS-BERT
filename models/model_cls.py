import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .transformer import Block, sample_and_group_former, Local_op
from functools import partial


curve_config = {
        'default': [[100, 5], [100, 5], None, None],
        'long':  [[10, 30], None,  None,  None]
    }

class Model(nn.Module):
    def __init__(self, samplepoint=256, embed_dim=256, num_classes=40, setting='default', depth=2, num_heads=4, mlp_ratio=2.0,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, masked_im_modeling=False, **kwargs):
        super(Model, self).__init__()

        assert setting in curve_config

        # additional_channel = 64
        # k = samplepoint
        self.samplepoint = samplepoint
        # Input Embedding
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        # add feature
        # self.lpfa = LPFA(9, 64, k=self.samplepoint, mlp_num=2, initial=True)

        # encoder
        # self.cic11 = CIC(npoint=1024, radius=0.05, k=k, in_channels=additional_channel, output_channels=64, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][0])
        # self.cic12 = CIC(npoint=1024, radius=0.05, k=k, in_channels=64, output_channels=64, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][0])
        #
        # self.cic21 = CIC(npoint=1024, radius=0.05, k=k, in_channels=64, output_channels=128, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][1])
        # self.cic22 = CIC(npoint=1024, radius=0.1, k=k, in_channels=128, output_channels=128, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][1])

        # self.cic31 = CIC(npoint=None, radius=0.1, k=k, in_channels=128, output_channels=embed_dim, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][2])
        # self.cic32 = CIC(npoint=None, radius=0.2, k=k, in_channels=embed_dim, output_channels=embed_dim, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][2])

        # self.cic41 = CIC(npoint=64, radius=0.2, k=k, in_channels=256, output_channels=512, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][3])
        # self.cic42 = CIC(npoint=64, radius=0.4, k=k, in_channels=512, output_channels=512, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][3])

        # self.reduce_dim = nn.Sequential(nn.Conv1d(128, embed_dim, kernel_size=1, bias=False),
        #                                 nn.BatchNorm1d(embed_dim),
        #                                 nn.ReLU(inplace=True))

        self.gather_local_0 = Local_op(in_channels=128, out_channels=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, embed_dim, 1))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.head_conv0 = nn.Sequential(
            nn.Conv1d(embed_dim, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True))
        self.head_conv1 = nn.Linear(1024 * 2, 512, bias=False)
        self.head_conv2 = nn.Linear(512, num_classes)
        self.head_bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)

        self.se = nn.Sequential(nn.Linear(512, 512//8, bias=False),
                                nn.BatchNorm1d(512//8),
                                nn.LeakyReLU(negative_slope=0.2),
                                nn.Linear(512//8, 512, bias=False),
                                nn.Sigmoid())

    def forward(self, x):
        xyz = x.detach().permute(0, 2, 1).contiguous()
        bs, N, _ = xyz.size()
        # x_clone = x.clone()
        # B, D, N
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        # B, D, N
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        # x = x + self.lpfa(x_clone, x_clone)

        # l1_xyz, l1_points = self.cic11(xyz, l0_points)
        # l1_xyz, l1_points = self.cic12(l1_xyz, l1_points)
        #
        # l2_xyz, l2_points = self.cic21(l1_xyz, l1_points)
        # l2_xyz, l2_points = self.cic22(l2_xyz, l2_points)

        x = x.permute(0, 2, 1).contiguous()
        new_xyz, new_feature = sample_and_group_former(npoint=int(np.ceil(x.size(1) / self.samplepoint)), radius=None,
                                                       nsample=self.samplepoint, xyz=xyz, points=x)
        x = self.gather_local_0(new_feature)

        cls_tokens = self.cls_token.expand(bs, -1, -1)
        x = torch.cat((cls_tokens, x), dim=2)

        x = x.permute(0,2,1)
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # l3_xyz, l3_points = self.cic31(l2_xyz, l2_points)
        # l3_xyz, l3_points = self.cic32(l3_xyz, l3_points)
        #
        # l4_xyz, l4_points = self.cic41(l3_xyz, l3_points)
        # l4_xyz, l4_points = self.cic42(l4_xyz, l4_points)

        x = self.head_conv0(x.transpose(1,2))
        x_max = F.adaptive_max_pool1d(x, 1)
        x_avg = F.adaptive_avg_pool1d(x, 1)
        
        x = torch.cat((x_max, x_avg), dim=1).squeeze(-1)
        x = F.relu(self.head_bn1(self.head_conv1(x).unsqueeze(-1)), inplace=True).squeeze(-1)
        x = x * self.se(x)
        x = self.dp1(x)
        x = self.head_conv2(x)
        return x


def POS_BERT(samplepoint=32, **kwargs):
    model = Model(samplepoint=samplepoint, embed_dim=512, depth=8, num_heads=8, mlp_ratio=2.0,
                  qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
