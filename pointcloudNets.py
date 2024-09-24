import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetSAModuleMSGGlr, PointnetSAModuleGlr

# Part of the code is referred from: https://github.com/WangYueFt/dcp/blob/master/model.py

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn

def nearest_neighbor(src, dst):
    inner = -2 * torch.matmul(src.transpose(1, 0).contiguous(), dst)  # src, dst (num_dims, num_points)
    distances = -torch.sum(src ** 2, dim=0, keepdim=True).transpose(1, 0).contiguous() - inner - torch.sum(dst ** 2,
                                                                                                           dim=0,
                                                                                                           keepdim=True)
    distances, indices = distances.topk(k=1, dim=-1)
    return distances, indices

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

# def get_graph_feature(x, k=20):
#     # x = x.squeeze()
#     idx = knn(x, k=k)  # (batch_size, num_points, k)
#     batch_size, num_points, _ = idx.size()
#     device = torch.device('cuda')
#
#     idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
#
#     idx = idx + idx_base
#
#     idx = idx.view(-1)
#
#     _, num_dims, _ = x.size()
#
#     x = x.transpose(2,
#                     1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
#     feature = x.view(batch_size * num_points, -1)[idx, :]
#     feature = feature.view(batch_size, num_points, k, num_dims)
#     x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
#
#     feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)
#
#     return feature


def get_graph_feature(x, k=20):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (batch_size, 2*num_dims, num_points, k)


class PointNet(nn.Module):
    def __init__(self, embed_dim=256):
        super(PointNet, self).__init__()
        self.embed_dim = embed_dim
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, embed_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        batch_size, _, N = x.size()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        return x


class PointNet2(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        self._build_model()

    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.2,
                nsample=64,
                mlp=[0, 64, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=64,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 512, 1024], use_xyz=True
            )
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, self.embed_dim))

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud.transpose(2,1).contiguous())

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        return self.fc_layer(features.squeeze(-1))

class Normalize(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        norm = torch.norm(x, p=2, dim=self.dim, keepdim=True)
        return x / norm

class PointNet2_glr(nn.Module):
    def __init__(self, embed_dim=512, use_xyz=True, point_wise_out=False, multi=1.0):
        super().__init__()
        print('Using', multi, 'times PointNet SSG model')
        self.SA_modules = nn.ModuleList()
        self.embed_dim = embed_dim

        self.point_wise_out = point_wise_out

        self.SA_modules.append(
            PointnetSAModuleMSGGlr(
                npoint=512,
                radii=[0.23],
                nsamples=[48],
                mlps=[[0, int(multi * 64), int(multi * 128)]],
                use_xyz=use_xyz,
                use_global_xyz=True
            )
        )
        self.SA_modules.append(
            PointnetSAModuleMSGGlr(
                npoint=128,
                radii=[0.32],
                nsamples=[64],
                mlps=[[int(multi * 128), int(multi * 128), int(multi * 512)]],
                use_xyz=use_xyz,
                use_global_xyz=True
            )
        )
        self.SA_modules.append(
            PointnetSAModuleGlr(
                nsample=128,
                mlp=[int(multi * 512), int(embed_dim)],
                use_xyz=use_xyz
            )
        )

        # self.prediction_modules = nn.ModuleList()
        #
        # mid_channel = min(int(multi * 128), embed_dim)
        # self.prediction_modules.append(
        #     nn.Sequential(
        #         nn.Conv1d(int(multi * 128), mid_channel, 1),
        #         nn.BatchNorm1d(mid_channel),
        #         nn.ReLU(inplace=True),
        #         nn.Conv1d(mid_channel, embed_dim, 1),
        #         Normalize(dim=1)
        #     )
        # )
        #
        # mid_channel = min(int(multi * 512), embed_dim)
        # self.prediction_modules.append(
        #     nn.Sequential(
        #         nn.Conv1d(int(multi * 512), mid_channel, 1),
        #         nn.BatchNorm1d(mid_channel),
        #         nn.ReLU(inplace=True),
        #         nn.Conv1d(mid_channel, embed_dim, 1),
        #         Normalize(dim=1)
        #     )
        # )
        #
        # mid_channel = min(int(multi * 1024), embed_dim)
        # self.prediction_modules.append(
        #     nn.Sequential(
        #         nn.Conv1d(int(multi * 1024), mid_channel, 1),
        #         nn.BatchNorm1d(mid_channel),
        #         nn.ReLU(inplace=True),
        #         nn.Conv1d(mid_channel, embed_dim, 1),
        #         Normalize(dim=1)
        #     )
        # )

        self.adaptive_maxpool = nn.AdaptiveMaxPool1d(1)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )
        return xyz, features

    def forward(self, pointcloud, get_feature=False):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud.transpose(2,1).contiguous())
        # out = []
        for module in self.SA_modules:
            xyz, features = module(xyz, features)
            # last_feature = prediction_modules(features)
            # out.append(last_feature)

        # global_feature = torch.cat([self.adaptive_maxpool(now_out).squeeze(2) for now_out in out], dim=1)
        return features.squeeze(-1)


# class DGCNN(nn.Module):
#     def __init__(self, embed_dim=256):
#         super(DGCNN, self).__init__()
#         self.embed_dim = embed_dim
#         self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
#         self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
#         self.conv5 = nn.Conv2d(512, embed_dim, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.bn4 = nn.BatchNorm2d(256)
#         self.bn5 = nn.BatchNorm2d(embed_dim)
#
#     def forward(self, x):
#         batch_size, num_dims, num_points = x.size()
#         x = get_graph_feature(x)
#         x = F.relu(self.bn1(self.conv1(x)))
#         x1 = x.max(dim=-1, keepdim=True)[0]
#
#         x = F.relu(self.bn2(self.conv2(x)))
#         x2 = x.max(dim=-1, keepdim=True)[0]
#
#         x = F.relu(self.bn3(self.conv3(x)))
#         x3 = x.max(dim=-1, keepdim=True)[0]
#
#         x = F.relu(self.bn4(self.conv4(x)))
#         x4 = x.max(dim=-1, keepdim=True)[0]
#
#         x = torch.cat((x1, x2, x3, x4), dim=1)
#
#         x = F.relu(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)
#
#         x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
#         return x


# class DGCNN(nn.Module):
#     def __init__(self, embed_dim=256):
#         super(DGCNN, self).__init__()
#         self.embed_dim = embed_dim
#         self.k = 20
#
#         self.bn1 = nn.BatchNorm2d(64)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.bn4 = nn.BatchNorm2d(256)
#         self.bn5 = nn.BatchNorm1d(1024)
#
#         self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
#                                    self.bn1,
#                                    nn.ReLU())
#         self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
#                                    self.bn2,
#                                    nn.ReLU())
#         self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
#                                    self.bn3,
#                                    nn.ReLU())
#         self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
#                                    self.bn4,
#                                    nn.ReLU())
#         self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
#                                    self.bn5,
#                                    nn.ReLU())
#         self.linear1 = nn.Linear(1024*2, embed_dim, bias=False)
#         self.ln1 = nn.BatchNorm1d(embed_dim)
#
#     def forward(self, x):
#         batch_size = x.size(0)
#         x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
#         x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
#         x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
#
#         x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
#         x = self.conv2(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
#         x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
#
#         x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
#         x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
#         x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)
#
#         x = get_graph_feature(x3, k=self.k)  # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
#         x = self.conv4(x)  # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
#         x4 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)
#
#         x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)
#
#         x = self.conv5(x)  # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
#         x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
#         x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
#         x = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*2)
#
#         x = self.ln1(self.linear1(x)) # (batch_size, emb_dims*2) -> (batch_size, 512)
#
#         return x


class DGCNN(nn.Module):
    def __init__(self, embed_dim=256, k=20):
        super(DGCNN, self).__init__()
        self.embed_dim = embed_dim
        self.k = k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024*2, embed_dim, bias=False)
        self.bn6 = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        # if x.size(2)==2048:
        #     self.k = 40
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)                       # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        x = self.conv5(x)                       # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*2)

        x = self.bn6(self.linear1(x)) # (batch_size, emb_dims*2) -> (batch_size, 512)

        return x


class DGCNN_semseg(nn.Module):
    def __init__(self, embed_dim=512, k=20):
        super(DGCNN_semseg, self).__init__()
        self.embed_dim = embed_dim
        self.k = k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, 1024, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.addlinear1 = nn.Linear(1024, embed_dim, bias=False)
        self.addbn6 = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        x = x[:, :3, :]

        x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)

        x = self.conv6(x)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = self.addbn6(self.addlinear1(x))  # (batch_size, emb_dims*2) -> (batch_size, 512)

        return x

# from pointnet2_vote.pointnet2_modules import PointnetFPModule, PointnetSAModuleVotes

class VoteNet(nn.Module):
    def __init__(self, embed_dim=512, input_feature_dim=0):
        super(VoteNet, self).__init__()
        self.embed_dim = embed_dim

        self.sa1 = PointnetSAModuleVotes(
                npoint=2048,
                radius=0.2,
                nsample=64,
                mlp=[input_feature_dim, 64, 64, 128],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa2 = PointnetSAModuleVotes(
                npoint=1024,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa3 = PointnetSAModuleVotes(
                npoint=512,
                radius=0.8,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa4 = PointnetSAModuleVotes(
                npoint=256,
                radius=1.2,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.fp1 = PointnetFPModule(mlp=[256+256,256,256])
        self.fp2 = PointnetFPModule(mlp=[256+256,256,256])

        self.addlinear1 = nn.Linear(1024, embed_dim, bias=False)
        self.addbn6 = nn.BatchNorm1d(embed_dim)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud, end_points=None):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            end_points: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """
        if not end_points:
            end_points = {}
        batch_size = pointcloud.shape[0]

        pointcloud = pointcloud.transpose(2,1).contiguous()

        xyz, features = self._break_up_pc(pointcloud)

        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz, features, fps_inds = self.sa1(xyz, features)
        end_points['sa1_inds'] = fps_inds
        end_points['sa1_xyz'] = xyz
        end_points['sa1_features'] = features

        xyz, features, fps_inds = self.sa2(xyz, features) # this fps_inds is just 0,1,...,1023
        end_points['sa2_inds'] = fps_inds
        end_points['sa2_xyz'] = xyz
        end_points['sa2_features'] = features

        xyz, features, fps_inds = self.sa3(xyz, features) # this fps_inds is just 0,1,...,511
        end_points['sa3_xyz'] = xyz
        end_points['sa3_features'] = features

        xyz, features, fps_inds = self.sa4(xyz, features) # this fps_inds is just 0,1,...,255
        end_points['sa4_xyz'] = xyz
        end_points['sa4_features'] = features

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        features = self.fp1(end_points['sa3_xyz'], end_points['sa4_xyz'], end_points['sa3_features'], end_points['sa4_features'])
        features = self.fp2(end_points['sa2_xyz'], end_points['sa3_xyz'], end_points['sa2_features'], features)

        features_cls = features.max(dim=-2)[0]
        features_cls = self.addbn6(self.addlinear1(features_cls))  # (batch_size, emb_dims*2) -> (batch_size, 512)

        return features_cls


def pointnet(**kwargs):
    model = PointNet(embed_dim=192,**kwargs)
    return model

def pointnet_large(**kwargs):
    model = PointNet(embed_dim=512,**kwargs)
    return model

def pointnet2(**kwargs):
    model = PointNet2(embed_dim=192,**kwargs)
    return model

def pointnet2_large(**kwargs):
    model = PointNet2(embed_dim=512,**kwargs)
    return model

def pointnet2_glr(**kwargs):
    model = PointNet2_glr(embed_dim=192,**kwargs)
    return model

def pointnet2_glr_large(**kwargs):
    model = PointNet2_glr(embed_dim=512,**kwargs)
    return model

def dgcnn(**kwargs):
    model = DGCNN(embed_dim=192,**kwargs)
    return model

def dgcnn_large(**kwargs):
    model = DGCNN(embed_dim=512,**kwargs)
    return model

def dgcnn_large_k40(**kwargs):
    model = DGCNN(embed_dim=512, k=40, **kwargs)
    return model

def dgcnn_large_semseg(**kwargs):
    model = DGCNN_semseg(embed_dim=512, k=20, **kwargs)
    return model

def votenet_backbone(**kwargs):
    model = VoteNet(embed_dim=512, **kwargs)
    return model