import torch
from pointnet2_ops import pointnet2_utils
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

def sample_and_group(npoint, radius, nsample, xyz, features):
    '''
    Parameters
    ----------
    xyz : torch.Tensor
        (B, N, 3) tensor of the xyz coordinates of the features
    features : torch.Tensor
        (B, N, C) tensor of the descriptors of the the features

    Returns
    -------
    new_xyz: torch.Tensor
        (B, 3, npoint, nsample)
    new_features: torch.Tensor
        (B, C+3, npoint, nsample)
    '''
    xyz_flipped = xyz.transpose(1, 2).contiguous()
    fps_idx = pointnet2_utils.furthest_point_sample(xyz_flipped, npoint) # [B, npoint]
    print("furthese point sample idx:", fps_idx.shape)
    new_xyz = pointnet2_utils.gather_operation(xyz_flipped, fps_idx).transpose(1, 2).contiguous() # [B, npoint, 3]
    print("new xyz: ", new_xyz.shape)
    idx = pointnet2_utils.ball_query(radius, nsample, xyz, new_xyz) # [B, npoint, nsamples]
    print("query ball idx: ", idx.shape)
    xyz_trans = xyz.transpose(1, 2).contiguous()
    grouped_xyz = pointnet2_utils.grouping_operation(xyz_trans, idx) # [B, 3, npoint, nsample]
    grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1) # normalization
    print("grouped xyz:", grouped_xyz.shape)

    if features is not None:
        features_flipped = features.transpose(1, 2).contiguous()
        grouped_features = pointnet2_utils.grouping_operation(features_flipped, idx) # [B, C, npoint, nsample]
        print("grouped features:", grouped_features.shape)

        new_features = torch.cat([grouped_xyz, grouped_features], dim=1) # [B, C+3, npoint, nsample]
        print("new points: ", new_features.shape)
    else:
        new_features = grouped_xyz

    return new_xyz, new_features

def sample_and_group_all(xyz, features):
    '''
    Parameters
    ----------
    xyz : torch.Tensor
        (B, N, 3) tensor of the xyz coordinates of the features
    features : torch.Tensor
        (B, N, C) tensor of the descriptors of the the features

    Returns
    -------
    new_xyz: torch.Tensor
    new_features: torch.Tensor
    '''
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B,1,C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if features is not None:
        new_features = torch.cat([grouped_xyz, features.view(B, 1, N, -1)], dim=-1)
    else:
        new_features = grouped_xyz
    return new_xyz, new_features



class PointNetSAModule(nn.Module):
    def __init__(self):
        super(PointNetSAModule, self).__init__()

    def forward(self):
        pass

class PointNetFPModule(nn.Module):
    def __init__(self):
        super(PointNetFPModule, self).__init__()

    def forward(self):
        pass

if __name__ == '__main__':
    xyz, features = torch.rand(8, 1024, 3), torch.rand(8, 1024, 5)
    xyz, features = xyz.cuda(), features.cuda()
    npoint, radius, nsample = 256, 0.1, 32
    sample_and_group(npoint, radius, nsample, xyz, features)
    sample_and_group_all(xyz, features)