import torch
from pointnet2_ops import pointnet2_utils
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

def sample_and_group(npoint, radius, nsample, xyz, features, returnfps=False):
    '''
    Parameters
    ----------
    xyz : torch.Tensor
        (B, N, 3) xyz coordinates of the features
    features : torch.Tensor
        (B, N, C) descriptors of the the features

    Returns
    -------
    new_xyz: torch.Tensor
        (B, npoint, 3) sampled points position data
    new_features: torch.Tensor
        (B, npoint, nsample, 3+C) sampled points data
    '''
    xyz_flipped = xyz.transpose(1, 2).contiguous()
    fps_idx = pointnet2_utils.furthest_point_sample(xyz_flipped, npoint) # [B, npoint]
    #print("furthese point sample idx:", fps_idx.shape)
    new_xyz = pointnet2_utils.gather_operation(xyz_flipped, fps_idx).transpose(1, 2).contiguous() # [B, npoint, 3]
    #print("new xyz: ", new_xyz.shape)
    idx = pointnet2_utils.ball_query(radius, nsample, xyz, new_xyz) # [B, npoint, nsamples]
    #print("query ball idx: ", idx.shape)
    xyz_trans = xyz.transpose(1, 2).contiguous()
    grouped_xyz = pointnet2_utils.grouping_operation(xyz_trans, idx) # [B, 3, npoint, nsample]
    grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1) # normalization
    #print("grouped xyz:", grouped_xyz.shape)

    if features is not None:
        features_flipped = features.transpose(1, 2).contiguous()
        grouped_features = pointnet2_utils.grouping_operation(features_flipped, idx) # [B, C, npoint, nsample]
        #print("grouped features:", grouped_features.shape)

        fps_features = pointnet2_utils.gather_operation(features_flipped, fps_idx).transpose(1, 2).contiguous()
        #print("fps features: ", fps_features.shape)
        fps_features = torch.cat([new_xyz, fps_features], dim=-1)
        #print("fps features: ", fps_features.shape)

        new_features = torch.cat([grouped_xyz, grouped_features], dim=1) # [B, C+3, npoint, nsample]
        #print("new features: ", new_features.shape)
    else:
        new_features = grouped_xyz
        fps_features = new_xyz

    new_features = new_features.permute(0, 2, 3, 1)
    grouped_xyz = grouped_xyz.permute(0, 2, 3, 1)
    if returnfps:
        # print("new xyz: ", new_xyz.shape)
        # print("new features: ", new_features.shape)
        # print("grouped xyz:", grouped_xyz.shape)
        # print("fps features: ", fps_features.shape)
        return new_xyz, new_features, grouped_xyz, fps_features
    else:
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
        (B, 1, 3) sampled points position data
    new_features: torch.Tensor
        (B, 1, N, 3+C) sampled points data
    '''
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B,1,C).to(device)
    #print("new xyz: ", new_xyz.shape)
    grouped_xyz = xyz.view(B, 1, N, C)
    #print("grouped xyz:", grouped_xyz.shape)
    if features is not None:
        new_features = torch.cat([grouped_xyz, features.view(B, 1, N, -1)], dim=-1)
        #print("new features: ", new_features.shape)
    else:
        new_features = grouped_xyz
    #print("new xyz: ", new_xyz.shape)
    #print("new features: ", new_features.shape)
    return new_xyz, new_features

class GraphAttention(nn.Module):
    def __init__(self, all_channel, feature_dim, dropout, alpha):
        super(GraphAttention, self).__init__()
        self.alpha = alpha
        self.a = nn.Parameter(torch.zeros(size=(all_channel, feature_dim)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, center_xyz, center_feature, grouped_xyz, grouped_feature):
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
            (B, 1, 3) sampled points position data
        new_features: torch.Tensor
            (B, 1, N, 3+C) sampled points data
        '''
        B, npoint, C = center_xyz.size()
        _, _, nsample, D = grouped_feature.size()
        delta_p = center_xyz.view(B, npoint, 1, C).expand(B, npoint, nsample, C) - grouped_xyz # [B, npoint, nsample, C]
        delta_h = center_feature.view(B, npoint, 1, D).expand(B, npoint, nsample, D) - grouped_feature # [B, npoint, nsample, D]
        delta_p_concat_h = torch.cat([delta_p,delta_h],dim = -1) # [B, npoint, nsample, C+D]
        e = self.leakyrelu(torch.matmul(delta_p_concat_h, self.a)) # [B, npoint, nsample,D]
        attention = F.softmax(e, dim=2) # [B, npoint, nsample,D]
        attention = F.dropout(attention, self.dropout, training=self.training)
        graph_pooling = torch.sum(torch.mul(attention, grouped_feature),dim = 2) # [B, npoint, D]
        return graph_pooling

class GraphAttentionConvLayer(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, droupout=0.6, alpha=0.2):
        super(GraphAttentionConvLayer, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.droupout = droupout
        self.alpha = alpha
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all
        self.GAT = GraphAttention(3+last_channel, last_channel, self.droupout, self.alpha)

    def forward(self, xyz, points):
        '''
        Parameters
        ----------
        xyz : torch.Tensor
            (B, 3, N) tensor of the xyz coordinates of the features
        points : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz: torch.Tensor
            sampled points position data
        new_features: torch.Tensor
            sampled points feature data
        '''
        xyz = xyz.permute(0, 2, 1).contiguous()
        if points is not None:
            points = points.permute(0, 2, 1).contiguous()
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points, grouped_xyz, fps_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, True)
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        fps_points = fps_points.unsqueeze(3).permute(0, 2, 3, 1) # [B, C+D, 1,npoint]
        #print("new points: ", new_points.shape)
        #print("fps points: ", fps_points.shape)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            fps_points = F.relu(bn(conv(fps_points)))
            new_points =  F.relu(bn(conv(new_points)))
        #print("new points: ", new_points.shape)
        #print("fps points: ", fps_points.shape)
        new_points = self.GAT(center_xyz=new_xyz,
                              center_feature=fps_points.squeeze().permute(0,2,1),
                              grouped_xyz=grouped_xyz,
                              grouped_feature=new_points.permute(0,3,2,1))
        #print("new points: ", new_points.shape)
        new_xyz = new_xyz.permute(0, 2, 1)
        new_points = new_points.permute(0, 2, 1)
        #print("new xyz: ", new_xyz.shape)
        #print("new features: ", new_points.shape)
        return new_xyz, new_points



class GACNet(nn.Module):
    def __init__(self, num_classes, dropout=0.2, alpha=0.2, normalize=True):
        super(GACNet, self).__init__()
        if normalize:
            self.sa1 = GraphAttentionConvLayer(1024, 0.1, 32, 3 + 5, [32, 32, 64], False, dropout, alpha)
        else:
            self.sa1 = GraphAttentionConvLayer(1024, 0.1, 32, 3, [32, 32, 64], False, dropout, alpha)
        self.sa2 = GraphAttentionConvLayer(256, 0.2, 32, 64 + 3, [64, 64, 128], False, dropout, alpha)
        self.sa3 = GraphAttentionConvLayer(64, 0.4, 32, 128 + 3, [128, 128, 256], False, dropout, alpha)
        self.sa4 = GraphAttentionConvLayer(16, 0.8, 32, 256 + 3, [256, 256, 512], False, dropout, alpha)

        self.fc1 = nn.Linear(960, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(256, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.drop3 = nn.Dropout(0.4)

        self.fc4 = nn.Linear(64, num_classes)



    def forward(self, xyz, point):
        l1_xyz, l1_points = self.sa1(xyz, point)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # print("l1: ", l1_xyz.shape, l1_points.shape)
        # print("l2: ", l2_xyz.shape, l2_points.shape)
        # print("l3: ", l3_xyz.shape, l3_points.shape)
        # print("l4: ", l4_xyz.shape, l4_points.shape)

        net = torch.cat([torch.max(l1_points, dim=2)[0],
                         torch.max(l2_points, dim=2)[0],
                         torch.max(l3_points, dim=2)[0],
                         torch.max(l4_points, dim=2)[0]], axis=-1)
        #print("net: ", net.shape)

        net = self.drop1(F.relu(self.bn1(self.fc1(net))))
        #print("first dense:", net.shape)
        net = self.drop2(F.relu(self.bn2(self.fc2(net))))
        net = self.drop3(F.relu(self.bn3(self.fc3(net))))
        net = self.fc4(net)
        x = F.log_softmax(net, -1)
#         x = net
        return x
        #net = self.drop1(F.relu(self.bn1(self.fc1(net))))


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)

        return total_loss





if __name__ == '__main__':
    # #test sample and group
    # xyz, features = torch.rand(8, 1024, 3), torch.rand(8, 1024, 5)
    # xyz, features = xyz.cuda(), features.cuda()
    # npoint, radius, nsample = 256, 0.1, 32
    # print('------sample and group------')
    # sample_and_group(npoint, radius, nsample, xyz, features, True)
    #
    # print('------sample and group all------')
    # sample_and_group_all(xyz, features)

    # #test Graph Attn Conv
    # xyz, features = torch.rand(8, 3, 1024), torch.rand(8, 5, 1024)
    # xyz, features = xyz.cuda(), features.cuda()
    # layer = GraphAttentionConvLayer(256, 0.1, 32, 5 + 3, [32, 32, 64], False, 0.2, 0.2).cuda()
    # layer(xyz, features)

    # test GACNet
    xyz, features = torch.rand(8, 3, 2048), torch.rand(8, 5, 2048)
    xyz, features = xyz.cuda(), features.cuda()
    model = GACNet(5).cuda()
    model(xyz, features)