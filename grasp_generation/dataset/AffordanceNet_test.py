# Copyright (c) Gorilla-Lab. All rights reserved.
import os
import open3d as o3d
from os.path import join as opj
import numpy as np
from torch.utils.data import Dataset
import h5py
import json
#import sys
#sys.path.append("./utils/")
#from utils import *
from utils.provider import rotate_point_cloud_SO3, rotate_point_cloud_y

import pickle as pkl
import torch
from tqdm import tqdm
#import pdb

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, centroid, m


def semi_points_transform(points):
    spatialExtent = np.max(points, axis=0) - np.min(points, axis=0)
    eps = 2e-3*spatialExtent[np.newaxis, :]
    jitter = eps*np.random.randn(points.shape[0], points.shape[1])
    points_ = points + jitter
    return points_


class AffordNetDataset(Dataset):
    def __init__(self, data_dir, split, partial=False, rotate='None', semi=False):
        super().__init__()
        self.data_dir = data_dir
        self.split = split

        self.partial = partial
        self.rotate = rotate
        self.semi = semi

        self.load_data()

        self.affordance = self.all_data[0]["affordance"]

        return

    def load_data(self):
        self.all_data = []
        if self.semi:
            with open(opj(self.data_dir, 'semi_label_1.pkl'), 'rb') as f:
                temp_data = pkl.load(f)
        else:
            if self.partial and self.split == 'Train':
                with open(opj(self.data_dir, 'partial_%s_data.pkl' % self.split), 'rb') as f:
                    temp_data = pkl.load(f)
             
            elif self.rotate != "None" and self.split != 'train' and self.partial == True:
                with open(opj(self.data_dir, 'rotate_%s_data.pkl' % self.split), 'rb') as f:
                    temp_data_rotate = pkl.load(f)
                with open(opj(self.data_dir, 'full_shape_%s_data.pkl' % self.split), 'rb') as f:
                    temp_data = pkl.load(f)

            elif self.rotate != "None" and self.split != 'train' and self.partial != True:
                with open(opj(self.data_dir, 'rotate_%s_data.pkl' % self.split), 'rb') as f:
                    temp_data_rotate = pkl.load(f)
                with open(opj(self.data_dir, 'full_shape_%s_data.pkl' % self.split), 'rb') as f:
                    temp_data = pkl.load(f)                
            else:
                with open(opj(self.data_dir, 'full_shape_%s_data.pkl' % self.split), 'rb') as f:
                    temp_data = pkl.load(f)
        for index, info in enumerate(temp_data):
            if self.partial and self.split == 'Train':
                partial_info = info["partial"]
                for view, data_info in partial_info.items():
                    temp_info = {}
                    temp_info["shape_id"] = info["shape_id"]
                    temp_info["semantic class"] = info["semantic class"]
                    temp_info["affordance"] = info["affordance"]
                    temp_info["view_id"] = view
                    temp_info["data_info"] = data_info
                    self.all_data.append(temp_info)
            elif self.split != 'train' and self.rotate != 'None':
                rotate_info = temp_data_rotate[index]["rotate"][self.rotate]
                full_shape_info = info["full_shape"]
                for r, r_data in rotate_info.items():
                    temp_info = {}
                    temp_info["shape_id"] = info["shape_id"]
                    temp_info["semantic class"] = info["semantic class"]
                    temp_info["affordance"] = info["affordance"]
                    temp_info["data_info"] = full_shape_info
                    temp_info["rotate_matrix"] = r_data.astype(np.float32)
                    self.all_data.append(temp_info)
            else:
                temp_info = {}
                temp_info["shape_id"] = info["shape_id"]  
                temp_info["semantic class"] = info["semantic class"]
                temp_info["affordance"] = info["affordance"]
                temp_info["data_info"] = info["full_shape"]
                self.all_data.append(temp_info)

    def __getitem__(self, index):

        data_dict = self.all_data[index]
        modelid = data_dict["shape_id"]         #certain object id/file name: '9ebba4182b7ddad284432ce2f42f498'
        modelcat = data_dict["semantic class"]

        data_info = data_dict["data_info"]
        model_data = data_info["coordinate"].astype(np.float32)
        labels = data_info["label"]
        for aff in self.affordance:
            temp = labels[aff].astype(np.float32).reshape(-1, 1)
            model_data = np.concatenate((model_data, temp), axis=1)

        datas = model_data[:, :3]    #coordinate of each point
        targets = model_data[:, 3:]  #affordance label of each point

        if self.rotate != 'None':
            if self.split == 'train':
                if self.rotate == 'so3':
                    datas = rotate_point_cloud_SO3(
                        datas[np.newaxis, :, :]).squeeze()
                elif self.rotate == 'z':
                    datas = rotate_point_cloud_y(
                        datas[np.newaxis, :, :]).squeeze()
            else:
                r_matrix = data_dict["rotate_matrix"]
                datas = (np.matmul(r_matrix, datas.T)).T

        datas, _, _ = pc_normalize(datas)

        return datas, datas, targets, modelid, modelcat

    def __len__(self):
        return len(self.all_data)


class AffordNetDataset_Unlabel(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.load_data()
        self.affordance = self.all_data[0]["affordance"]
        return

    def load_data(self):
        self.all_data = []
        with open(opj(self.data_dir, 'semi_unlabel_1.pkl'), 'rb') as f:
            temp_data = pkl.load(f)
        for info in temp_data:
            temp_info = {}
            temp_info["shape_id"] = info["shape_id"]
            temp_info["semantic class"] = info["semantic class"]
            temp_info["affordance"] = info["affordance"]
            temp_info["data_info"] = info["full_shape"]
            self.all_data.append(temp_info)

    def __getitem__(self, index):
        data_dict = self.all_data[index]
        modelid = data_dict["shape_id"]
        modelcat = data_dict["semantic class"]

        data_info = data_dict["data_info"]
        datas = data_info["coordinate"].astype(np.float32)

        datas, _, _ = pc_normalize(datas)

        return datas, datas, modelid, modelcat

    def __len__(self):
        return len(self.all_data)



def get_AffordNetDataset(data_dir='/data/wchen/3D_affordance/full-shape',
                       split = 'train', partial=False, rotate='None', semi=False,batch_size = 6):
    """Get PyTorch dataloader for processed 3DAffordance data.

    :param data_dir: str: path to 3d affordance dataset
    :param split: str: path to directory containing datatset of 'train','valid','test'
    :param partial: bool: path to directory containing different point cloud shape 'full-shape','partial'
    :param rotate: string if the full-shape object is rotating.
    :param semi: bool: whether it is a semi-supervised learning
    """
    train_set = AffordNetDataset(
            data_dir, 'train', partial=False, rotate='None', semi=False)
    val_set = AffordNetDataset(
            data_dir, 'val', partial=False, rotate='None', semi=False)
    dataset_dict = dict(
            train_set=train_set,
            val_set=val_set
        )
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, drop_last=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1,
                            shuffle=False, num_workers=8, drop_last=False)
    loader_dict = dict(
        train_loader=train_loader,
        val_loader=val_loader
    )
    return loader_dict


if True:
    loader_dict = get_AffordNetDataset(batch_size=20)
    train_loader = loader_dict.get("train_loader", None)
    val_loader = loader_dict.get("val_loader", None)
    num_batches = len(train_loader)
    print(num_batches,len(val_loader))
    #print(val_loader[0]['affordace'])
    for data, data1, label, object_id, object_class in tqdm(train_loader, total=len(train_loader), smoothing=0.9):
        data, label = data.float().cuda(), label.float().cuda()  #torch.Size([6, 2048, 3])  torch.Size([6, 2048, 18]) 
        print(data.shape,label.shape)
        for i in range(20):
            print(object_id[i],object_class[i])                           #object file name: 'ff5a2e340869e9c45981503fc6dfccb2', object category: 'Table'
            
            #visualize point cloud
            if object_class[i] == 'Scissors':
                pc= data[i].data.cpu().numpy()
                pc_label =label[i].data.cpu().numpy()
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc)
                o3d.io.write_point_cloud('points1.xyz', pcd)
                np.savetxt('points1_label.txt',pc_label)
                print('##################################################')
        #pdb.set_trace()
        
        
       

        
  #      data = data.permute(0, 2, 1)   #data --- input into different network models
  #      batch_size = data.size()[0]
  #      num_point = data.size()[2]