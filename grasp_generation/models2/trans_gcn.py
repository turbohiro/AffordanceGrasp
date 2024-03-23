import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from .modules import MultiHeadAttention,PositionwiseFeedForward,SelfAttention
from .ExternalAttention import ExternalAttention
import random
import pdb 


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    # (batch_size, num_points, k)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    
    #sam_num = 40
    #data_length = range(idx.size()[2])
    #samp_list = [i for i in data_length] # [0, 1, 2, 3]
    #samp_list = random.sample(samp_list, sam_num) # [1, 2]
    #samp_list = np.random.choice(data_length, sam_num, replace=False)
    #idx = idx[:,:,samp_list]               # 最后进行采样，得到采样后曲面
    #print(idx.shape)
 
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False,sam_num = 40):
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
    feature = feature.view(batch_size, num_points, sam_num, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, sam_num, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature      # (batch_size, 2*num_dims, num_points, k)

class PointNetfeat_transformer(nn.Module):
    def __init__(self, num_points = 8192, global_feat = True):
        super(PointNetfeat_transformer, self).__init__()
        self.stn = STN3d(num_points = num_points)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.dropout  = nn.Dropout(p=0.1)
        self.mha_1 = MultiHeadAttention(n_head = 8,d_model =128)
        self.ffn_1 = PositionwiseFeedForward(128, 128, use_residual=True) #无residual v1,有 v2
        self.maxpool = torch.nn.MaxPool1d(2048, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)

        self.num_points = num_points
        self.global_feat = global_feat
    def forward(self, x):
        #print("uuuu",x.shape)
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        #print("kk",x.shape)
        x = x.transpose(2, 1).contiguous()
        x = self.mha_1(x)
        #print(h2.shape)
        x = self.dropout(x)
        x = self.ffn_1(x)
        #print(h3.shape)
        x = self.dropout(x)
        x = x.transpose(2, 1).contiguous()
        x = torch.squeeze(self.maxpool(x),1)  #
        x = self.bn3(self.conv3(x))

        #print("#####",x.shape)
        #[8,1024,2048]

        x,_ = torch.max(x, 2)
        #print("$$$",x.shape)  ##[8,1024]
        x = x.view(-1, 1024)
        print("SSS",x.shape)  #[8,1024]
        return x

class PointGenCon(nn.Module):
    def __init__(self,  bottleneck_size = 8192):
        super(PointGenCon, self).__init__()
        
        self.bottleneck_size = bottleneck_size
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size//2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size//4)
        #self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv1 = nn.Sequential(nn.Conv1d(1024, 256, kernel_size=1, bias=False),self.bn1, nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),self.bn2, nn.LeakyReLU(negative_slope=0.2))
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
        x = self.conv1(x)
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

class PointGenCon_transformer(nn.Module):
    def __init__(self, bottleneck_size = 8192):
        
        super(PointGenCon_transformer, self).__init__()
        self.k = args.k
        self.bottleneck_size = bottleneck_size
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size//2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size//4)

        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size//2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size//2, self.bottleneck_size//4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size//4, 3, 1)

        self.dropout  = nn.Dropout(p=0.1)
        self.mha_1 = MultiHeadAttention(n_head = 4,d_model =1026)
        self.ffn_1 = PositionwiseFeedForward(1026, 1026, use_residual=True) #无residual v1,有 v2
        self.maxpool = torch.nn.MaxPool1d(128, 1)
        self.avg_pool =  GlobalPooling(dim = 2)

        self.th = nn.Tanh()
        

    def forward(self, x):
        batchsize = x.size()[0]
        x = x.transpose(2, 1).contiguous()
        x = self.mha_1(x)
        #print(h2.shape)
        x = self.dropout(x)
        x = self.ffn_1(x)
        #print("test",x.shape)
        #x = self.dropout(x)
        x = x.transpose(2, 1).contiguous()

        #x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        #print("test",x.shape)
        return x

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

class PointNetRes(nn.Module):
    def __init__(self):
        super(PointNetRes, self).__init__()
        self.conv1 = torch.nn.Conv1d(4, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.conv4 = torch.nn.Conv1d(1088, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 256, 1)
        self.conv6 = torch.nn.Conv1d(256, 128, 1)
        self.conv7 = torch.nn.Conv1d(128, 3, 1)


        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.bn5 = torch.nn.BatchNorm1d(256)
        self.bn6 = torch.nn.BatchNorm1d(128)
        self.bn7 = torch.nn.BatchNorm1d(3)
        self.th = nn.Tanh()

    def forward(self, x):
        batchsize = x.size()[0]
        npoints = x.size()[2]
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, 1024)
        x = x.view(-1, 1024, 1).repeat(1, 1, npoints)
        x = torch.cat([x, pointfeat], 1)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.th(self.conv7(x))
        return x

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class TransGCN_Estimation(nn.Module):
    def __init__(self, args, num_classes, bottleneck_size = 256, use_attention = True, global_feat = True, feature_transform = False):
        super(TransGCN_Estimation, self).__init__()
        self.args = args
        self.k = args.k
        self.transform_net = Transform_Net(args)
        self.num_classes = num_classes
        self.bottleneck_size = bottleneck_size
        self.use_attention = use_attention

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64*2)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)
        self.bnpn1 = nn.BatchNorm1d(512)
        self.bnpn2 = nn.BatchNorm1d(256)

        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.convpn1 = nn.Sequential(nn.Conv1d(1088, 512, kernel_size=1, bias=False),
                                   self.bnpn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.convpn2 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bnpn2,
                                   nn.LeakyReLU(negative_slope=0.2))


        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64*2, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(128*2, 128*2, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(64*7, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(1025, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))

        #self.conv8 = nn.Sequential(nn.Conv1d(1216, 256, kernel_size=1, bias=False),
        #                          self.bn8)
        #self.nn =  nn.Sequential(nn.Linear(256,256),nn.BatchNorm1d(256),nn.LeakyReLU(negative_slope=0.2))
        
        self.dropout  = nn.Dropout(p=0.1)
        self.mha_1 = MultiHeadAttention(n_head = 6,d_model =64)
        self.ea = ExternalAttention(d_model=512,S=8)
        self.ffn_2 = PositionwiseFeedForward(64, d_out = 128, use_residual=False) #无residual v1,有 v2
        self.ffn_1 = PositionwiseFeedForward(64, d_out = 64, use_residual=True) #无residual v1,有 v2
        if self.use_attention:
            self.sa1 = SelfAttention(448,128)
            self.sa2 = SelfAttention(256,128)
        
        self.classifier2 = nn.ModuleList()
        for i in range(num_classes):
            classifier2 = nn.Sequential(
                nn.Conv1d(256, 128, 1),
                nn.BatchNorm1d(128),
                # nn.Dropout(0.5),
                nn.Conv1d(128, 1, 1)
            )
            self.classifier2.append(classifier2)

        
        self.classifier = nn.ModuleList([PointGenCon(bottleneck_size = self.bottleneck_size ) for i in range(0,self.num_classes)])

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)
        x_pn, trans, trans_feat = self.feat(x)
        
        x_pn = self.convpn1(x_pn)
        x_pn = self.convpn2(x_pn)
        
        #pdb.set_trace()
        ##Spatial transorm: Similiar to the alignment network of Pointnet
        # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x0 = get_graph_feature(x, k=self.k)
        t = self.transform_net(x0)              # (batch_size, 3, 3)
        # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)
        # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)
        # (batch_size, num_points, 3) -> (batch_size, 3, num_points)
        x = x.transpose(2, 1)

        #Each Edgeconv 1
        # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = get_graph_feature(x, k=self.k)
        # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv1(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        #x = self.conv2(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x0 = x.max(dim=-1, keepdim=False)[0]
        
        #Each Edgeconv 2
        # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = get_graph_feature(x0, k=self.k)
        # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x) #64
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        #x = self.conv2(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x1 = x.max(dim=-1, keepdim=False)[0]
       
        #Add Transformer Module
        #x = x1.transpose(2, 1).contiguous()
        #x = self.ea(x)
        #x = self.dropout(x)
        #x = self.ffn_1(x)
       # x = self.dropout(x)
        #x1_att = x.transpose(2, 1).contiguous()
        
        #x1 = torch.cat((x1, x1_att), dim=1)

        
        #Each Edgeconv 3
        # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        #x = get_graph_feature(x1, k=self.k)
        # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        #x = self.conv3(x)  #128
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        #x = self.conv4(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        #x2 = x.max(dim=-1, keepdim=False)[0]    #max pooling

        #Add Transformer Module
        x = x1.transpose(2, 1).contiguous()
        #x = self.mha_1(x)
        #x = self.dropout(x)
        #x = self.ffn_1(x)

        x = self.mha_1(x)
        x = self.dropout(x)
        x = self.ffn_2(x)
        #x = self.dropout(x)
        x2_tran = x.transpose(2, 1).contiguous()
        #x2 = torch.cat((x2, x2_att), dim=1)
        #pdb.set_trace()
        #Each Edgeconv 4
        # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = get_graph_feature(x2_tran, k=self.k)
        # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv5(x)  #256
        #x = self.conv3(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x3 = x.max(dim=-1, keepdim=False)[0]

        #Add Transformer Module
        #x = x3.transpose(2, 1).contiguous()
        #x = self.mha_1(x)
        #x = self.dropout(x)
        #x = self.ffn_1(x)
        #x = self.dropout(x)
        #x3_att = x.transpose(2, 1).contiguous()
        #x3 = torch.cat((x3, x3_att), dim=1)

        # (batch_size, 64*3, num_points)
        x4 = torch.cat((x1, x2_tran, x3), dim=1)
        #x4 = torch.cat((x1, x2, x3, x1_att, x2_att, x3_att), dim=1) #64*5 = 320

        if self.use_attention:
            x_att =  self.sa1(x4) 
            x_att = x_att.max(dim=-1, keepdim=True)[0]
            x_att = x_att.repeat(1, 1, num_points)

        # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x4 = self.conv6(x4) 

        # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        x = x4.max(dim=-1, keepdim=True)[0]

        # (batch_size, num_categoties, 1)
        # l = l.view(batch_size, -1, 1)
        # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)
        # l = self.conv7(l)

        # x = torch.cat((x, l), dim=1)            # (batch_size, 1088, 1)
        # (batch_size, 1088, num_points)
        x = x.repeat(1, 1, num_points)
                    
        #x5 = x.transpose(2, 1).contiguous()
       # x5 = self.mha_1(x5)
        #x = self.dropout(x)
        #x = self.ffn_1(x)
        #x = self.dropout(x)
        #x5_att = x5.transpose(2, 1).contiguous()
        
        if self.use_attention:
        # (batch_size, 1088+64*3, num_points)
            xt = torch.cat((x, x0, x1,x2_tran,x_pn), dim=1) #512 256 64

        else:
            xt = torch.cat((x, x0, x1, x2, x3), dim=1)
        #x5 = xt.transpose(2, 1).contiguous()
        #x5 = self.mha_1(x5)
        #x5_att = x5.transpose(2, 1).contiguous()
        #xt = torch.cat((xt, x_att), dim=1)       

        # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
        #x = self.conv8(x)

        x,_ = torch.max(xt, 2)
        #print("$$$",x.shape)  ##[8,1024]
        x = x.view(-1, 1024) #704/1024
        #x = self.nn(x)
        #pdb.set_trace()
        #rand_value = Variable(torch.cuda.FloatTensor(x.size(0),1,num_points))
        #rand_value.data.uniform_(0,1)
        #y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_value.size(2)).contiguous()
        #y = torch.cat((rand_value, xt), 1).contiguous()
        y = xt
        #pdb.set_trace()
       
        
        out = self.classifier[0](y)
        for index, classifier in enumerate(self.classifier):
            if index == 0:
                continue
            #rand_value = Variable(torch.cuda.FloatTensor(x.size(0),1,num_points))
            #rand_value.data.uniform_(0,1)
            #y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_value.size(2)).contiguous()
            #y = torch.cat((rand_value, xt), 1).contiguous()
            #pdb.set_trace()
            score_ = classifier(y)
            out = torch.cat((out, score_), dim=1)

       # score_ = []
        
        #for i in range(self.num_classes):
        #    score_.append(self.classifier[i](x))
        
       # out = torch.cat(score_,1).contiguous() 
        
        
        return out  #torch.Size([8, 18, 2048]) --> (batch_size, label_num, num_points)
