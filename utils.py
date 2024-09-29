import numpy as np
import nibabel as nib
import trimesh
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.ndimage import grey_erosion, grey_dilation

import torch

# from PIL import Image
# import io
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_pretrained(model, pretrained_model, logger, strict=False):

    logger.info("=> loading pretrain...")
    ckpt = torch.load(pretrained_model)
    if "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    if strict:
        model.load_state_dict(state_dict, strict=True)
    else:
        for k in list(state_dict.keys()):
            if (model.state_dict().get(k) is None) or model.state_dict()[
                k
            ].shape != state_dict[k].shape:
                state_dict.pop(k)
        model.load_state_dict(state_dict, strict=False)
    logger.info(f"==>  loaded pretrained weights '{pretrained_model}'  successfully")


def get_each_mesh(image, cls: int, need_normalize=True):
    """
    Returns a mesh for each class.
    image: npy image
    """
    cls_image = image == cls
    cls_mesh = trimesh.voxel.ops.matrix_to_marching_cubes(cls_image.cpu(), pitch=1.0)
    # Get the largest mesh and avoid multiple discrete orphaned meshes
    cls_meshes = cls_mesh.split(only_watertight=False)
    cls_mesh = max(cls_meshes, key=lambda m: m.volume)

    if need_normalize:
        # vertices = torch.tensor(cls_mesh.vertices, dtype=torch.float32)
        vertices = cls_mesh.vertices
        vertices[..., 0] = vertices[..., 0] / (image.shape[-3] - 1)
        vertices[..., 1] = vertices[..., 1] / (image.shape[-2] - 1)
        vertices[..., 2] = vertices[..., 2] / (image.shape[-1] - 1)
        cls_mesh.vertices = vertices

    # show mesh
    # cls_mesh.show()
    return cls_mesh


def show_all_meshes(meshes, colors=None):
    o3d_meshes = []
    if colors is None:
        num_meshes = len(meshes)
        colors = np.random.uniform(0, 1, size=(num_meshes, 3))
    for i, mesh in enumerate(meshes):
        if mesh is None:
            continue
        cls_mesh = o3d.geometry.TriangleMesh()
        cls_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        cls_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        cls_mesh.compute_vertex_normals()

        color = colors[i]
        cls_mesh.paint_uniform_color(color)  # Red

        o3d_meshes.append(cls_mesh)

    o3d.visualization.draw_geometries(
        o3d_meshes,
        mesh_show_back_face=True,
        point_show_normal=True,
        mesh_show_wireframe=True,
    )


