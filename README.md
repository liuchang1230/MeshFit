# MeshFit: A Mesh-based Segmentation Model for Medical Image Analysis

This repository contains the implementation of MeshFit, a mesh-based segmentation model designed for medical image analysis. MeshFit addresses the limitations of pixel-based segmentation methods, especially for complex organs with low contrast boundaries. By iteratively deforming a universal initial mesh, MeshFit progressively fits the target organ shape, ensuring a closed, continuous segmentation.

## Key Features

- **Mesh-based segmentation**: Ensures closed, contiguous surfaces, reducing false positives in low-contrast regions.
- **Progressive mesh initialization**: Starts with a universal mesh and iteratively refines it to fit target organ surfaces.
- **Supports complex organ shapes**: Effective for handling complex geometries and challenging boundaries.
- **Center point prediction**: Supports two approaches—segmentation label-based prediction and Gaussian heatmap-based prediction.

## Requirements

To run this code, you need to install the following dependencies:

- `pytorch3d`: A library for 3D deep learning built on PyTorch.
- `pymesh`: A mesh processing library for geometric processing.
- `trimesh`: A library for loading and processing triangular meshes.
- `pytorch`: The PyTorch deep learning framework.
- `monai`: Medical Open Network for AI, a deep learning framework specialized for medical imaging.
