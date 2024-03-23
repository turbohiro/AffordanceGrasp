import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from .modules import MultiHeadAttention,PositionwiseFeedForward,SelfAttention
from .ExternalAttention import ExternalAttention
import pdb


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    # (batch_size, num_points, k)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(
        0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature      # (batch_size, 2*num_dims, num_points, k)


class Transform_Net(nn.Module):
    def __init__(self, args):
        super(Transform_Net, self).__init__()
        self.args = args
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv1(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = self.conv2(x)
        # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)
        x = x.max(dim=-1, keepdim=False)[0]

        # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = self.conv3(x)
        # (batch_size, 1024, num_points) -> (batch_size, 1024)
        x = x.max(dim=-1, keepdim=False)[0]

        # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)
        # (batch_size, 512) -> (batch_size, 256)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)

        # (batch_size, 256) -> (batch_size, 3*3)
        x = self.transform(x)
        # (batch_size, 3*3) -> (batch_size, 3, 3)
        x = x.view(batch_size, 3, 3)

        return x

class PointGenCon(nn.Module):
    def __init__(self,  bottleneck_size = 8192):
        super(PointGenCon, self).__init__()
        
        self.bottleneck_size = bottleneck_size
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size//2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size//4)
        #self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv1 = nn.Sequential(nn.Conv1d(768, 256, kernel_size=1, bias=False),self.bn1, nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(64*3, 128, kernel_size=1, bias=False),self.bn2, nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1, bias=False),self.bn3)
        #self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size//2, 1)
        #self.conv3 = torch.nn.Conv1d(self.bottleneck_size//2, self.bottleneck_size//4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size//4, 1, 1)

        self.sh = torch.nn.Sigmoid()
       

        
        self.dropout  = nn.Dropout(p=0.5)
        self.mha_1 = MultiHeadAttention(n_head = 4,d_model =64)
        self.ea = ExternalAttention(d_model=64,S=8)
        self.ffn_1 = PositionwiseFeedForward(64, 64, use_residual=False) #无residual v1,有 v2

    def forward(self, x):
        batchsize = x.size()[0]
        #x = get_graph_feature(x, k=40)
        #x = self.conv1(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
       # x = x.max(dim=-1, keepdim=False)[0]
        #x = F.relu(self.bn1(self.conv1(x)))
        #x = F.relu(self.bn2(self.conv2(x)))
        #x = F.relu(self.bn3(self.conv3(x)))
        #x = F.leaky_relu(self.bn2(self.conv2(x)),negative_slope = 0.2)
        #x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        #x =  self.bn2(self.conv2(x))
       # x = x.transpose(2, 1).contiguous()
        #x = self.mha_1(x)
        #x = self.dropout(x)
        #x = self.ffn_1(x)
        #x = self.dropout(x)
        #x = x.transpose(2, 1).contiguous()
        x = self.sh(self.conv4(x))
        
        return x

class TransGCN_Estimation(nn.Module):
    def __init__(self, args, num_classes,bottleneck_size = 256):
        super(TransGCN_Estimation, self).__init__()
        self.args = args
        self.k = args.k
        self.transform_net = Transform_Net(args)
        self.num_classes = num_classes
        self.bottleneck_size = bottleneck_size

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(128)
        self.bn8 = nn.BatchNorm1d(64)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(64*3, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.dropout  = nn.Dropout(p=0.1)
        self.mha_1 = MultiHeadAttention(n_head = 4,d_model =64)
        self.ea = ExternalAttention(d_model=512,S=8)
        self.ffn_1 = PositionwiseFeedForward(64, d_out = 64, use_residual=True) #无residual v1,有 v2
        self.sa1 = SelfAttention(256,64)

        self.classifier = nn.ModuleList([PointGenCon(bottleneck_size = self.bottleneck_size ) for i in range(0,self.num_classes)])

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        #x0 = get_graph_feature(x, k=self.k)
        #t = self.transform_net(x0)              # (batch_size, 3, 3)
        # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        #x = x.transpose(2, 1)
        # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        #x = torch.bmm(x, t)
        # (batch_size, num_points, 3) -> (batch_size, 3, num_points)
        #x = x.transpose(2, 1)

        #EdgeConv1
        # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = get_graph_feature(x, k=self.k)
        # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv1(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x1 = x.max(dim=-1, keepdim=False)[0]
        
        #EdgeConv2        
        # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = get_graph_feature(x1, k=self.k)
        # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv3(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x2 = x.max(dim=-1, keepdim=False)[0]
        
        #EdgeConv3
        # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = get_graph_feature(x2, k=self.k)
        # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv3(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x3 = x.max(dim=-1, keepdim=False)[0]


        #EdgeConv4
        # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = get_graph_feature(x3, k=self.k)
        # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv5(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x4 = x.max(dim=-1, keepdim=False)[0]
     

        # (batch_size, 64*3, num_points)
        x = torch.cat((x2, x3, x4), dim=1)

        # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)#64*3---256
        x5 = self.conv6(x)
        x_att = self.sa1(x5) 
        x = self.conv7(x5)
        x =  self.conv8(x)
        # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        x = x.max(dim=-1, keepdim=True)[0]
        #x_att = x_att.max(dim=-1, keepdim=True)[0]
        # (batch_size, 1088, num_points)
        x6 = x.repeat(1, 1, num_points)
        #x_att = x_att.repeat(1, 1, num_points)

        # (batch_size, 1088+64*3, num_points)
        x = torch.cat((x1, x_att, x6), dim=1)

        # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
        out = self.classifier[0](x)
        for index, classifier in enumerate(self.classifier):
            if index == 0:
                continue
            #rand_value = Variable(torch.cuda.FloatTensor(x.size(0),1,num_points))
            #rand_value.data.uniform_(0,1)
            #y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_value.size(2)).contiguous()
            #y = torch.cat((rand_value, xt), 1).contiguous()
            #pdb.set_trace()
            score_ = classifier(x)
            out = torch.cat((out, score_), dim=1)
        
        return out