import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet_util import (
    PointNetSetAbstraction,
    PointNetFeaturePropagation,
    square_distance,
)


class get_model(nn.Module):
    def __init__(self, num_classes, inchannels=64):
        super(get_model, self).__init__()
        # self.img_size = img_size
        self.sa1 = PointNetSetAbstraction(
            0.06, 32, 6 + 3, [inchannels, inchannels, inchannels * 2], False
        )
        self.sa2 = PointNetSetAbstraction(
            0.1,
            32,
            inchannels * 2 + 3,
            [inchannels * 2, inchannels * 2, inchannels * 4],
            False,
        )
        self.sa3 = PointNetSetAbstraction(
            0.14,
            32,
            inchannels * 4 + 3,
            [inchannels * 4, inchannels * 4, inchannels * 8],
            False,
        )
        self.sa4 = PointNetSetAbstraction(
            0.18,
            32,
            inchannels * 8 + 3,
            [inchannels * 8, inchannels * 8, inchannels * 16],
            False,
        )

        # for classify
        # self.sa5 = PointNetSetAbstraction(radius=None, nsample=None, in_channel=inchannels*16 + 3,
        #                                   mlp=[inchannels*16, inchannels*32, inchannels*64], group_all=True)
        # self.fc1 = nn.Linear(inchannels*64, inchannels*32)
        # self.bnc1 = nn.LayerNorm(inchannels*32)
        # self.drop1 = nn.Dropout(0.4)
        # self.fc2 = nn.Linear(inchannels*32, inchannels*16)
        # self.bnc2 = nn.LayerNorm(inchannels*16)
        # self.drop2 = nn.Dropout(0.4)
        # self.fc3 = nn.Linear(inchannels*16, num_classes)

        self.fp4 = PointNetFeaturePropagation(
            inchannels * 24, [inchannels * 12, inchannels * 8]
        )
        # self.fp3 = PointNetFeaturePropagation(inchannels*12, [inchannels*8, inchannels*8])
        # self.fp2 = PointNetFeaturePropagation(inchannels*10, [inchannels*8, inchannels*4])
        # self.fp1 = PointNetFeaturePropagation(inchannels*4, [inchannels*4, inchannels*4, inchannels*4])
        # self.conv1 = nn.Conv1d(inchannels*4, inchannels, 1)
        self.fp3 = PointNetFeaturePropagation(
            inchannels * 12, [inchannels * 8, inchannels * 4]
        )
        self.fp2 = PointNetFeaturePropagation(
            inchannels * 6, [inchannels * 4, inchannels * 2]
        )
        self.fp1 = PointNetFeaturePropagation(
            inchannels * 2, [inchannels * 2, inchannels * 2, inchannels]
        )
        self.conv1 = nn.Conv1d(inchannels, inchannels, 1)
        # self.bn1 = nn.BatchNorm1d(inchannels)
        # self.bn1 = nn.LayerNorm(inchannels)
        self.bn1 = nn.InstanceNorm1d(inchannels)
        self.drop1 = nn.Dropout(0.5)
        # self.conv2 = nn.Conv1d(128, num_classes, 1)

    # def mesh2xyz(self, meshes):
    #     xyz = []
    #     for mesh in meshes:
    #         vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
    #         vertices[..., 0] = vertices[..., 0] / (self.img_size[-3] - 1)
    #         vertices[..., 1] = vertices[..., 1] / (self.img_size[-2] - 1)
    #         vertices[..., 2] = vertices[..., 2] / (self.img_size[-1] - 1)

    #         normals = torch.tensor(mesh.vertex_normals, dtype=torch.float32)
    #         vertices = torch.cat([vertices, normals], dim=-1)
    #         xyz.extend(vertices)
    #     return torch.stack(xyz, dim=0)[None].cuda()
    def get_dist(self, vertices, len_nums):
        # sorted_dists_list = []
        # sorted_idx_list = []
        init_num = 0
        # max_nums = max([vertices[i].shape[0] for i in range(vertices.shape[0])])
        dists = torch.ones((len_nums, len_nums), device=vertices[0].device)
        # idxs = -1 * torch.ones((len_nums, len_nums))
        for i in range(len(vertices)):
            sqrdists = square_distance(
                vertices[i][None], vertices[i][None]
            )  # (b, n, n)
            dists[
                init_num : init_num + vertices[i].shape[0],
                init_num : init_num + vertices[i].shape[0],
            ] = sqrdists[0]
            # _, idx = sqrdists.sort(dim=-1)
            # idx += init_num
            # idxs[
            #     init_num : init_num + vertices[i].shape[0],
            #     init_num : init_num + vertices[i].shape[0],
            # ] = idx
            # # 创建一个全1的张量，大小为 (n, k-m)
            # ones = -1 * torch.ones(dists.shape[0], max_nums - dists.shape[1])
            # dists = torch.cat([dists, ones], dim=-1)
            # idx = torch.cat([idx, ones], dim=-1)
            # sorted_dists_list.append(dists)
            # sorted_idx_list.append(idx + init_num)
            init_num += vertices[i].shape[0]

        # sorted_dists = torch.cat(sorted_dists_list, dim=0)
        # sorted_idx = torch.cat(sorted_idx_list, dim=0)
        return dists

    def forward(self, vertices, normals, istrain=False):
        dists = self.get_dist(vertices, len(normals))

        vertices = torch.cat(vertices, dim=0)
        xyz = torch.cat([vertices, normals], dim=-1)[None]
        xyz = xyz.permute(0, 2, 1)  # (b, 6, n)
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        l1_xyz, l1_points, fps_idx = self.sa1(l0_xyz, l0_points, dists)
        dists2 = dists[fps_idx[0]][:, fps_idx[0]]
        l2_xyz, l2_points, fps_idx2 = self.sa2(l1_xyz, l1_points, dists2)
        dists3 = dists2[fps_idx2[0]][:, fps_idx2[0]]
        l3_xyz, l3_points, fps_idx3 = self.sa3(l2_xyz, l2_points, dists3)
        dists4 = dists3[fps_idx3[0]][:, fps_idx3[0]]
        l4_xyz, l4_points, fps_idx4 = self.sa4(l3_xyz, l3_points, dists4)

        # cls_logit = None
        # if istrain:
        #     _, l5_points = self.sa5(l4_xyz, l4_points)
        #     l5 = l5_points.view(1, 1024)
        #     l5 = self.drop1(F.relu(self.bnc1(self.fc1(l5))))
        #     l5 = self.drop2(F.relu(self.bnc2(self.fc2(l5))))
        #     l5 = self.fc3(l5)
        #     cls_logit = F.log_softmax(l5, -1)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points, fps_idx4, dists4)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points, fps_idx3, dists3)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points, fps_idx2, dists2)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points, fps_idx, dists)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        # x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, weight=None):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss


if __name__ == "__main__":

    model = get_model(13)
    xyz = torch.rand(6, 9, 2048)
    (model(xyz))
