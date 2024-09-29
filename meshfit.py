import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pymesh
import trimesh
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.inferers import sliding_window_inference
from pytorch3d.structures import Meshes
from rasterize.rasterize import Rasterize
from monai.networks.blocks.convolutions import Convolution

from img_model import get_model
from pointnet import get_model as get_point_model
from pointnet_util import square_distance, index_points
from utils import get_each_mesh


def fix_mesh(mesh, target_len=(1.0, 1.0, 1.0), detail="normal"):
    count = 0
    mesh, _ = pymesh.remove_degenerated_triangles(mesh, 5)
    mesh, _ = pymesh.split_long_edges(mesh, max(target_len) * 2.3)
    num_vertices = mesh.num_vertices
    while count < 3:
        mesh, __ = pymesh.collapse_short_edges(
            mesh, min(target_len) * 0.5, preserve_feature=True
        )
        if mesh.num_vertices == num_vertices:
            break

        num_vertices = mesh.num_vertices
        print("#v: {}".format(num_vertices))
        count += 1

    return mesh


class GraphConv(nn.Module):
    __constants__ = ["bias", "in_features", "out_features"]

    def __init__(self, in_features, out_features):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.neighbours_fc = nn.Linear(in_features * 6, out_features)

    def forward(self, inputs, neighbors_index):
        neighbors_feat = torch.index_select(
            inputs.flatten(start_dim=0, end_dim=1), 0, neighbors_index.view(-1)
        )
        neighbors_feat = neighbors_feat.view(
            inputs.shape[0], -1, neighbors_index.shape[-1] * inputs.shape[-1]
        )  # (b, n, k*c)
        neighbors_feat = torch.cat([inputs, neighbors_feat], dim=-1)
        neighbors_feat = self.neighbours_fc(neighbors_feat)

        return neighbors_feat

    def extra_repr(self):
        return "in_features={}, out_features={}".format(
            self.in_features, self.out_features is not None
        )


class Feature2DeltaLayer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = GraphConv(in_channels, in_channels // 2)
        self.lrelu = get_act_layer(
            name=("leakyrelu", {"inplace": True, "negative_slope": 0.01})
        )
        self.norm1 = get_norm_layer(
            name="instance", spatial_dims=1, channels=in_channels // 2
        )
        self.conv2 = GraphConv(in_channels // 2, in_channels // 2)
        self.norm2 = get_norm_layer(
            name="instance", spatial_dims=1, channels=in_channels // 2
        )
        self.conv3 = nn.Linear(in_channels, in_channels // 2)
        self.norm3 = get_norm_layer(
            name="instance", spatial_dims=1, channels=in_channels // 2
        )
        self.conv4 = nn.Linear(in_channels // 2, 1, bias=False)
        self.conv4.weight.data.zero_()

    def forward(self, features, neighbors_index):
        residual = features
        x = self.conv1(features, neighbors_index)
        x = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        x = self.lrelu(x)
        x = self.conv2(x, neighbors_index)
        x = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        residual = self.conv3(residual)
        residual = self.norm3(residual.transpose(1, 2)).transpose(1, 2)
        x += residual
        x = self.lrelu(x)
        out = self.conv4(x)
        return out


# class CenterNet(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super().__init__()
#         self.num_classes = num_classes
#         self.conv1 = Convolution(
#             3,
#             in_channels,
#             64,
#             strides=1,
#             kernel_size=1,
#             padding=0,
#             act="RELU",
#             norm="INSTANCE",
#         )
#         self.conv2 = Convolution(
#             3,
#             64,
#             128,
#             strides=1,
#             kernel_size=1,
#             padding=0,
#             act="RELU",
#             norm="INSTANCE",
#         )
#         self.conv3 = Convolution(
#             3,
#             128,
#             num_classes,
#             strides=1,
#             kernel_size=1,
#             padding=0,
#             act="RELU",
#             norm="INSTANCE",
#         )
#
#     def forward(self, feat):
#         # Predict center point probability for each organ
#         center_prob = self.conv1(feat)
#         center_prob = self.conv2(center_prob)
#         center_prob = self.conv3(center_prob)
#
#         return center_prob


class MeshFit(nn.Module):
    def __init__(
            self,
            model_name,
            in_channels,
            base_channels,
            num_classes,
            roi_size,
    ):
        super().__init__()

        self.roi_size = roi_size
        self.num_classes = num_classes

        self.init_meshes = None
        # Image branching
        self.img_module = get_model(
            model_name,
            in_channels=in_channels,
            out_channels=num_classes,
            roi_size=roi_size,
        )
        # point cloud branching
        self.point_module = get_point_model(
            num_classes=num_classes, inchannels=base_channels
        )
        # image and point cloud fusion
        self.img_point_fuse = nn.Sequential(
            nn.Linear(base_channels * 2, base_channels), nn.ReLU()
        )
        # offset
        self.f2v = Feature2DeltaLayer(base_channels)

    # 获取每个顶点的邻域顶点索引
    def get_neighbors(self, mesh):
        # step 1 calculate the average length
        # lengths = [len(x) for x in mesh.vertex_neighbors]
        # average_length = int(sum(lengths) / len(lengths))
        average_length = 5
        # step 2 adjust the length of each list
        adjusted_data = []
        for neighbor in mesh.vertex_neighbors:
            if len(neighbor) >= average_length:
                # If the list length is greater than the average length, the first n elements are truncated
                adjusted_data.append(neighbor[:average_length])
            else:
                # If the list length is less than the average length, it is extended to n elements by repeated sampling
                repeats = average_length // len(neighbor) + 1
                repeated_data = (neighbor * repeats)[:average_length]
                adjusted_data.append(repeated_data)
        # step 3 convert to tensors
        adjusted_neighbors = torch.tensor(adjusted_data)
        return adjusted_neighbors

    # get the coordinates and normals of each vertex
    @staticmethod
    def get_vertices_normals(mesh):
        normals = torch.tensor(mesh.vertex_normals, dtype=torch.float32)  # 法向量
        # vertices = torch.cat([vertices, normals], dim=-1)
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
        return vertices, normals

    # predicted volume to mesh
    def volume2mesh(self, volume, device):
        all_cls = np.unique(volume)
        meshes = {}
        neighbors = []
        vertices = []
        normals = []
        init_index = 0
        for cls in all_cls[1:]:  # background 0 is not considered
            mesh = get_each_mesh(volume.cpu(), cls)
            ver, nor = self.get_vertices_normals(mesh)  # mesh vertex coordinates normalized to 0 1
            meshes[int(cls)] = mesh
            vertices.append(ver.to(device))  # [0, 1]
            normals.append(nor)
            neighbors.append(self.get_neighbors(mesh) + init_index)  # pay attention to updating the neighbor index

            init_index += len(mesh.vertices)

        neighbors = torch.cat(neighbors, dim=0).to(device)
        # vertices = torch.cat(vertices, dim=0).cuda()
        normals = torch.cat(normals, dim=0).to(device)
        self.init_meshes = meshes
        return meshes, neighbors, vertices, normals

    # calculate the distance between two points
    def get_dist(self, vertices, new_vertices):
        # sorted_dists_list = []
        # sorted_idx_list = []
        init_num = 0
        init_num_new = 0
        length = sum(len(sublist) for sublist in vertices)
        new_length = sum(len(sublist) for sublist in new_vertices)
        # max_nums = max([vertices[i].shape[0] for i in range(vertices.shape[0])])
        dists = torch.ones((new_length, length), device=vertices[0].device)
        # idxs = -1 * torch.ones((len_nums, len_nums))
        for i in range(len(vertices)):
            sqrdists = square_distance(
                new_vertices[i][None], vertices[i][None]
            )  # (b, m, n)
            dists[
            init_num_new: init_num_new + new_vertices[i].shape[0],
            init_num: init_num + vertices[i].shape[0],
            ] = sqrdists[0]

            init_num += vertices[i].shape[0]
            init_num_new += new_vertices[i].shape[0]

        return dists

    # The point cloud features are updated after each iteration
    def update_points_feat(self, points_feat, vertices, new_vertices):
        dists = self.get_dist(new_vertices, vertices)

        dists, idx = torch.topk(dists, 3, largest=False, sorted=False)

        weight = torch.softmax(-dists, dim=-1)
        interpolated_points_feat = torch.sum(
            index_points(points_feat, idx[None]) * weight.view(1, dists.shape[0], 3, 1),
            dim=2,
        )

        return interpolated_points_feat

    # Vertex coordinates are updated after each iteration
    def iter_move_module(
            self, img_feat, points_feat, vertices, new_vertices, neighbors, i
    ):
        # update points_feat
        if i != 0:
            points_feat = self.update_points_feat(points_feat, vertices, new_vertices)
            vertices = new_vertices

        vertices = torch.cat(vertices, dim=0).to(img_feat.device)
        # vertices = 2 * vertices - 1
        vertices = vertices[None][:, :, None, None]
        img_feat = F.grid_sample(
            img_feat,
            2 * vertices - 1,  # Corrected to [-1, 1] range
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[:, :, :, 0, 0]  # (b, c, n)
        img_feat = img_feat.permute(0, 2, 1)  # (b, n, c)

        feat = torch.cat([img_feat, points_feat], dim=-1)  # (b, n, 2c)
        feat = self.img_point_fuse(feat)  # (b, n, c)
        # distance prediction
        d_delta = self.f2v(feat, neighbors)  # (b, n, 1)
        return d_delta

    # adaptive topology optimization module
    def ATMO(self, vertices, faces):
        new_mesh = pymesh.form_mesh(vertices.cpu().detach().numpy(), faces)
        new_mesh = fix_mesh(
            new_mesh,
            target_len=(
                1.0 / self.roi_size[0],
                1.0 / self.roi_size[1],
                1.0 / self.roi_size[2],
            ),
        )
        new_mesh = trimesh.Trimesh(vertices=new_mesh.vertices, faces=new_mesh.faces)
        # if new_mesh.vertices.shape[0] < 10:
        #     return None
        new_mesh = trimesh.smoothing.filter_taubin(
            new_mesh, nu=0.5, lamb=0.5, iterations=5
        )
        return new_mesh

    # Update meshes
    def update_meshes(self, logit_map, d_delta, meshes, i, need_amto=True):
        init_num = 0
        init_num_new = 0
        train_meshes = {}
        new_vertices = []
        new_neighbors = []
        for cls, mesh in meshes.items():
            vertices = torch.tensor(mesh.vertices, dtype=torch.float32).to(logit_map.device)
            num_vertices = len(vertices)
            normals = torch.tensor(mesh.vertex_normals, dtype=torch.float32).to(logit_map.device)
            logit_vertices = F.grid_sample(
                logit_map,
                2 * vertices[None, :, None, None] - 1,  # -1~1
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )[:, :, :, 0, 0]
            logit_vertices = logit_vertices.permute(0, 2, 1)  # (b, n, cls)
            logit_vertices = torch.argmax(logit_vertices, dim=-1, keepdim=True)
            # if logit_vertices == cls, then it is foreground, set to 1, otherwise it is background, set to -1
            fore_back = torch.where(logit_vertices == cls, 1, -1)
            update_vertices = (
                    vertices
                    + d_delta[0, init_num: init_num + num_vertices, :]
                    * fore_back[0]
                    * normals
            )  # (n, 3)
            init_num += num_vertices

            update_vertices = torch.clip(update_vertices, 0, 1)
            train_mesh = Meshes(
                verts=list(update_vertices[None]),
                faces=list(torch.tensor(mesh.faces, dtype=torch.float32).to(logit_map.device)[None]),
            )
            train_meshes[cls] = train_mesh
            if i == 1:
                self.init_meshes[cls].vertices = 0.9 * self.init_meshes[cls].vertices + 0.1 * update_vertices

            # 自适应拓扑优化
            if i > 1 and need_amto:
                new_mesh = self.ATMO(update_vertices, mesh.faces)
                new_vertices.append(
                    torch.tensor(new_mesh.vertices, dtype=torch.float32).to(logit_map.device)
                )
                new_neighbors.append(self.get_neighbors(new_mesh) + init_num_new)

                num_new_vertices = len(new_vertices)
                init_num_new += num_new_vertices
            else:
                new_vertices.append(update_vertices)
                new_neighbors.append(self.get_neighbors(mesh) + init_num_new)
                num_new_vertices = len(new_vertices)
                init_num_new += num_new_vertices

        # new_vertices = torch.cat(new_vertices, dim=0).cuda()
        new_neighbors = torch.cat(new_neighbors, dim=0).to(logit_map.device)

        return train_meshes, new_vertices, new_neighbors

    def get_mesh_from_init_mesh(self, init_meshes, centers, shape, transforms, device):
        neighbors = []
        vertices = []
        normals = []
        init_index = 0
        for cls, mesh in init_meshes.items():
            mesh = self.transform_meshes(shape, mesh, transforms)
            center = centers[cls]
            # move the center point of the mesh to the center
            mesh.vertices = mesh.vertices - mesh.vertices.mean(0) + center
            ver, nor = self.get_vertices_normals(mesh)
            vertices.append(ver.to(device))  # [0, 1]
            normals.append(nor)
            neighbors.append(self.get_neighbors(mesh) + init_index)  # 注意更新邻居索引

            init_index += len(mesh.vertices)

        neighbors = torch.cat(neighbors, dim=0).to(device)
        normals = torch.cat(normals, dim=0).to(device)

        return init_meshes, neighbors, vertices, normals

    @staticmethod
    def get_centers_from_heatmap(logit_map):
        # logit_map (b, cls, h, w, d)
        centers = {}
        pred = torch.argmax(logit_map, dim=1, keepdim=True)  # (b, 1, h, w, d)
        for i in range(1, logit_map.shape[1]):
            # get all coords of the class
            coords = torch.where(pred[0, 0] == i)
            if coords[0].numel() == 0:
                continue
            else:
                coords = torch.stack(coords, dim=1).float()  # (n, 3)
                # get the center of the class
                center = torch.mean(coords, dim=0)  # (3,)
                # center /shape
                center = center / torch.tensor(logit_map.shape[-3:], device=logit_map.device)
                centers[i] = center
        return centers

    @staticmethod
    def mesh_flipd(vertices, axis):
        vertices[..., axis] = - vertices[..., axis]
        # if axis == 0:
        #     # flip it up and down
        #     vertices[..., 0] = shape[0] - vertices[..., 0]
        # elif axis == 1:
        #     # flip left and right
        #     vertices[..., 1] = shape[1] - vertices[..., 1]
        # else:
        #     # flip back and forth
        #     vertices[..., 2] = shape[2] - vertices[..., 2]
        return vertices

    @staticmethod
    def mesh_rotate90d(vertices, k):
        # Rotate 90 degrees counterclockwise, k is the number of rotations
        # h, w = shape[:2]
        for _ in range(k):
            vertices[..., 1], vertices[..., 0] = vertices[..., 0], - vertices[..., 1]
            # h, w, = w, h
        return vertices

    def transform_meshes(self, shape, mesh, transforms):
        if transforms[2]['do_transforms']:  # RandFlipd
            mesh.vertices = self.mesh_flipd(mesh.vertices, 0)
        if transforms[3]['do_transforms']:  # RandFlipd
            mesh.vertices = self.mesh_flipd(mesh.vertices, 1)
        if transforms[4]['do_transforms']:  # RandFlipd
            mesh.vertices = self.mesh_flipd(mesh.vertices, 2)
        if transforms[5]['do_transforms']:  # RandRotate90d
            # mesh.vertices = self.mesh_rotate90d(mesh.vertices, transforms[5]['k'])
            mesh.vertices = self.mesh_rotate90d(mesh.vertices, transforms[5]['extra_info']['extra_info']['k'])
        return mesh

    def get_meshes(self, label, logit_map, transforms):
        # centers: (b, cls, h, w, d)
        if self.init_meshes is None and label is not None:  # Initial training: Construct an initial mesh based on the label
            meshes, neighbors, vertices, normals = self.volume2mesh(label[0][0], label.device)
        else:
            centers = self.get_centers_from_heatmap(logit_map)
            meshes, neighbors, vertices, normals = self.get_mesh_from_init_mesh(
                self.init_meshes, centers, logit_map.shape[-3:], transforms, logit_map.device
            )

        return meshes, neighbors, vertices, normals

    def forward(self, x, label=None):
        img_feat, logit_map = sliding_window_inference(
            x, self.roi_size, 4, self.img_module, overlap=0.0
        )
        meshes, neighbors, vertices, normals = self.get_meshes(label, logit_map, x.applied_operations[0])

        points_feat = self.point_module(vertices, normals)  # (b, n, c)

        iter_meshes = []
        iter_vertices = []
        new_vertices = vertices

        for i in range(10):
            # distance prediction
            d_delta = self.iter_move_module(
                img_feat, points_feat, vertices, new_vertices, neighbors, i
            )
            vertices = new_vertices

            train_meshes, new_vertices, neighbors = self.update_meshes(
                logit_map, d_delta, meshes, i, need_amto=False
            )
            iter_meshes.append(train_meshes)
            iter_vertices.append(torch.cat(new_vertices, dim=0))

            if torch.max(d_delta) > 1 and label is None:  # for validation
                continue
            if label is not None and i == 2:  # for train
                break

        vertices = torch.cat(iter_vertices, dim=0)

        if label is not None:
            # for train
            return logit_map, iter_meshes, vertices
        else:
            # for inference
            # Rasterize directly based on vertex position to obtain voxel output
            pred_voxels = torch.zeros(
                (
                    1,
                    self.num_classes,
                    x.shape[-3],
                    x.shape[-2],
                    x.shape[-1],
                ),
                dtype=torch.long,
            )
            for i, (cls, mesh) in enumerate(train_meshes.items()):
                cls_vertices = new_vertices[i]
                cls_vertices[..., 0] *= (x[-3] - 1)
                cls_vertices[..., 1] *= (x[-2] - 1)
                cls_vertices[..., 2] *= (x[-1] - 1)
                cls_faces = mesh._faces_list[0]

                rasterizer = Rasterize(self.roi_size)
                pred_cls_voxels = rasterizer(
                    cls_vertices[None][..., [2, 1, 0]], cls_faces[None]
                ).long()  # (b, h, w, d)
                pred_voxels[0, cls][pred_cls_voxels[0] == 1] = 1

            pred_voxels[:, 0, ...] = (pred_voxels.sum(dim=1) == 0).int()

            return pred_voxels
