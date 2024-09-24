import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_util import sample_and_group_former, get_fps_nn_idx
import sys, math
from functools import partial
import numpy as np
import random

class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.reshape(-1, d, s)
        batch_size, _, N = x.size()
        x = F.relu(self.bn1(self.conv1(x))) # B, D, N
        x = F.relu(self.bn2(self.conv2(x))) # B, D, N
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1).contiguous()
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PCT(nn.Module):
    def __init__(self, samplepoint=256, embed_dim=256, depth=2, num_heads=4, mlp_ratio=2.0,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, masked_im_modeling=False, **kwargs):
        super(PCT, self).__init__()
        self.samplepoint = samplepoint
        self.embed_dim = embed_dim
        # Input Embedding
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, embed_dim, 1))
        # self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.samplepoint + 1))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

        # masked image modeling
        self.masked_im_modeling = masked_im_modeling
        if masked_im_modeling:
            self.mask_ratio = [0.2,0.4]
            self.mask_token = nn.Parameter(torch.zeros(1, embed_dim, 1))
            self.pos_embedding = nn.Sequential(
                nn.Linear(3, 128),
                nn.GELU(),
                nn.Linear(128, embed_dim)
            )

        self.conv0 = nn.Sequential(
            nn.Conv1d(embed_dim, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True))
        self.ln1 = nn.Linear(1024 * 2, 512, bias=False)
        self.ln2 = nn.Linear(512, 40)
        self.bn3 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def prepare_tokens(self, x):
        xyz = x.detach().permute(0, 2, 1).contiguous()
        batch_size, _, _ = x.size()
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        # B, D, N
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1).contiguous()
        new_xyz, new_feature = sample_and_group_former(npoint=int(np.ceil(x.size(1)/self.samplepoint)), radius=None, nsample=self.samplepoint, xyz=xyz, points=x)
        x = self.gather_local_0(new_feature) #(B, N, k, C1) -> (B, C2, N)

        bool_masked_pos = self._mask_center_rand(x.transpose(1,2), self.masked_im_modeling)
        if self.masked_im_modeling:
            mask_token = self.mask_token.expand(batch_size, -1, x.size(-1))
            pos_emb = self.pos_embedding(new_xyz).transpose(1,2)
            mask_token = mask_token+pos_emb
            # mask the input tokens
            w = bool_masked_pos.unsqueeze(-2).type_as(mask_token)
            x = x * (1 - w) + mask_token * w

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=2)

        # add positional encoding to each token
        # x = x + self.pos_embed

        return self.pos_drop(x), bool_masked_pos

    def _mask_center_rand(self, center, aug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if not aug or self.mask_ratio[1] == 0:
            return torch.zeros(center.shape[:2]).bool().to(center.device)

        ratio = random.random() * (self.mask_ratio[1] - self.mask_ratio[0]) + self.mask_ratio[0]
        bool_masked_pos = (torch.rand(center.shape[:2]) < ratio).bool().to(center.device)

        return bool_masked_pos

    def forward(self, x):
        # Shape (B, 3, N)
        ## Input Embedding
        batch_size, _, _ = x.size()

        if not math.isfinite(torch.sum(x)):
            print("pct input is {}, stopping training".format(torch.sum(x).item()), force=True)
            sys.exit(1)

        x, bool_masked_pos = self.prepare_tokens(x)

        if not math.isfinite(torch.sum(x)):
            print("pct token output is {}, stopping training".format(torch.sum(x).item()), force=True)
            sys.exit(1)

        ## transformer
        x = x.permute(0,2,1)
        for blk in self.blocks:
            x = blk(x)

        if not math.isfinite(torch.sum(x)):
            print("pct tsfmer output is {}, stopping training".format(torch.sum(x).item()), force=True)
            sys.exit(1)

        x = self.conv0(x.transpose(1, 2))
        x_max = F.adaptive_max_pool1d(x, 1)
        x_avg = F.adaptive_avg_pool1d(x, 1)

        x = torch.cat((x_max, x_avg), dim=1).squeeze(-1)
        x = F.relu(self.bn3(self.ln1(x).unsqueeze(-1)), inplace=True).squeeze(-1)
        x = self.dp1(x)
        x = self.ln2(x)

        return x

    def prepare_tokens_lastattention(self, x, return_newxyz=False):
        xyz = x.detach().permute(0, 2, 1).contiguous()
        batch_size, _, _ = x.size()
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        # B, D, N
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1).contiguous()
        new_xyz, new_feature = sample_and_group_former(npoint=int(np.ceil(x.size(1)/self.samplepoint)), radius=None, nsample=self.samplepoint, xyz=xyz, points=x)
        x = self.gather_local_0(new_feature) #(B, N, k, C1) -> (B, C2, N)

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=2)

        # add positional encoding to each token
        # x = x + self.pos_embed

        return self.pos_drop(x), new_xyz if return_newxyz else self.pos_drop(x)

    def get_last_selfattention(self, x):
        x, new_xyz = self.prepare_tokens_lastattention(x, return_newxyz=True)
        x = x.permute(0, 2, 1)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True), new_xyz

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        x = x.permute(0, 2, 1)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

def pct_base(samplepoint=256, **kwargs):
    model = PCT(samplepoint=samplepoint, embed_dim=192, depth=4, num_heads=4, mlp_ratio=2,
                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def pct_large(samplepoint=512, **kwargs):
    model = PCT(samplepoint=samplepoint, embed_dim=512, depth=8, num_heads=8, mlp_ratio=2.0,
                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model