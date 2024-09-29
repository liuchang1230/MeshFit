import torch.nn as nn
from utils import get_each_mesh
from monai.losses import FocalLoss
import torch.nn.functional as F
import torch
from monai.losses import DiceCELoss
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
    mesh_edge_loss,
)


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.inout_loss_func = FocalLoss(to_onehot_y=True, gamma=2.0, weight=None)
        self.logit_loss_func = DiceCELoss(to_onehot_y=True, softmax=True)

    def labelmesh(self, label):
        all_cls = torch.unique(label)
        meshes = {}
        for cls in all_cls[1:]:
            mesh = get_each_mesh(label[0][0], cls)
            # mesh, _, _ = self.get_vertices_normals(mesh)
            # meshes[cls] = Meshes(verts=[mesh.vertices], faces=[mesh.faces])
            meshes[cls.item()] = mesh

        return meshes

    def mesh_loss(self, pmesh, lmesh):
        # pmesh, lmesh is dict
        loss = 0
        for cls in pmesh.keys():
            if cls not in lmesh.keys():
                label_vertices = torch.mean(pmesh[cls].vertices)
                chamfer_loss = self.chamfer_loss(pmesh[cls].vertices, label_vertices)
            else:
                pvertices = sample_points_from_meshes(
                    pmesh[cls], lmesh[cls].vertices.shape[0]
                )
                # pvertices = pmesh[cls].vertices
                chamfer_loss = (
                        chamfer_distance(
                            pvertices,
                            torch.tensor(
                                lmesh[cls].vertices,
                                dtype=torch.float32,
                                device=pvertices.device,
                            )[None],
                        )[0]
                        / pvertices.shape[1]
                )

            laplacian_loss = mesh_laplacian_smoothing(pmesh[cls], method="uniform")

            loss += chamfer_loss + 0.05 * laplacian_loss

        loss /= len(pmesh)

        return loss

    def forward(self, pred, label):
        logit_map = pred[0]
        iter_meshes = pred[1]
        vertices = pred[2]

        # for logit loss
        logit_loss = self.logit_loss_func(logit_map, label)
        # for in_out loss, logit_map and vertices loss
        vertices = 2 * vertices[None][:, :, None, None] - 1
        logit_vertices = F.grid_sample(logit_map, vertices, align_corners=True)[
                         :, :, :, 0, 0
                         ]  # (b, c, n)
        logit_target = F.grid_sample(label, vertices, align_corners=True)[:, :, :, 0, 0]  # (b, 1, n)
        inout_loss = self.inout_loss_func(logit_vertices, logit_target)

        # for mesh loss
        label_meshes = self.labelmesh(label)
        mesh_loss = 0
        for meshes in iter_meshes:
            mesh_loss += self.mesh_loss(meshes, label_meshes)

        loss = logit_loss + 10 * inout_loss + mesh_loss / len(iter_meshes)
        return loss
