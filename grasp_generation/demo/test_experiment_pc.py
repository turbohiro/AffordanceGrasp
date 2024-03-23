# Copyright (c) Gorilla-Lab. All rights reserved.
import os
from os.path import join as opj
import torch
import torch.nn as nn
import numpy as np
from gorilla.config import Config
from models2 import *
import loss
from utils2 import *
import argparse
import open3d as o3d
import random
import colorsys
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import trimesh.transformations as tra
import pdb

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, centroid, m

def _calc_distances(p0, points):
    return ((p0 - points) ** 2).sum(axis=1)

#viridis,Oranges_r,Blues,plasma,rainbow
viridis = cm.get_cmap('plasma', 4)
def farthest_points_downsampling(points, num_points=2048):
    """
    Farthest point downsampling function for NumPy arrays.
    @Angelie => Look into the function below if you have a GPU available 🚀
    :param points: Numpy array of Point Cloud points.
    :param num_points: Number of points the final downsampled PC should have.
    :return: farthest_points: NumPy array containing the downsampled PC.
    """
    points_length = len(points)
    if points_length < num_points:
        print(f"Warning: point cloud to subsample has less than {num_points} points!")
    farthest_points = np.zeros((num_points, 3))
    farthest_points[0] = points[np.random.randint(points_length)]
    distances = _calc_distances(farthest_points[0], points)
    for i in range(1, num_points):
        farthest_points[i] = points[np.argmax(distances)]
        distances = np.minimum(distances, _calc_distances(farthest_points[i], points))
    return farthest_points
    
def visualize_special_point_cloud(points, label):
    '''
    points: (N, 3)
    cid: (N), color id.
    '''
    colors = np.zeros(points.shape)
    for i in range(len(label)):
        #in_color = random_colors(5,2)
        #print(in_color[2])
        color = viridis(label[i])
        #color =[0.5,0,0.5]
        color = np.array(color)[0:3]
        colors[i, :] = color
        #pdb.set_trace()

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud



def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--config", type = str, default = 'config/dgcnn/rotate_cfg.py',help="train config file path")
    parser.add_argument("--checkpoint", type=str, default = 'logs_seg_dcnn_rotate/model_95.t7',
                        help="the path to checkpoints")
    parser.add_argument(
        "--gpu",
        default = '0,1',
        type=str,
        help="Number of gpus to use"
    )
    parser.add_argument(
        "--with_loss", help="show the test loss", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.gpu != None:
        cfg.training_cfg.gpu = args.gpu
    print(cfg)
 
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.training_cfg.gpu
    num_gpu = len(cfg.training_cfg.gpu.split(','))

    if cfg.get('seed', None) != None:
        set_random_seed(cfg.seed)

    if cfg.get('with_loss', None) == None:
        cfg.update({"with_loss": args.with_loss})
    model = build_model(cfg).cuda()

    if num_gpu > 1:
        model = nn.DataParallel(model)
    
    if args.checkpoint != None:
        print("Loading checkpoint....")
        _, exten = os.path.splitext(args.checkpoint)
        if exten == '.t7':
            #pdb.set_trace()
            model.load_state_dict(torch.load(args.checkpoint))
        elif exten == '.pth':
            check = torch.load(args.checkpoint)
            model.load_state_dict(check['model_state_dict'])
    else:
        raise ValueError("Must specify a checkpoint path!")

    model.eval()
    print('model is loaded and begins to eval')
    
    #data = np.loadtxt(cfg.work_dir) #point cloud file--.xyz
    for home, dirs, files in os.walk('test_experiment/mechmind2/'):
        print('hhhh', files)

        for filename in files:
          
            fullname = os.path.join(home, filename)
            grasp_object=  o3d.io.read_point_cloud(fullname)  ##point cloud file--.pcd/ply
            #grasp_object = pc_normalize(grasp_object)
            #grasp_object.points = o3d.utility.Vector3dVector(
            #farthest_points_downsampling(grasp_object.points, num_points=2048))
            data = np.asarray(grasp_object.points,dtype='float32')

            pc_mean = np.mean(data, 0, keepdims=True)
            data[:, :3] -= pc_mean[:, :3]

            #get the 3d affordance value
            pose = tra.euler_matrix(0,0 , 0)  #-np.pi/2 
            data_aff = data.dot(pose[:3,:3].T)  
            pose2 = tra.euler_matrix(0 ,-np.pi/2 , 0)  #-np.pi/2 
            data2 = data_aff.dot(pose2[:3,:3].T)  

            pc_data_norm,_,_ = pc_normalize(data2.squeeze()) 
            ind = np.where(pc_data_norm[:,0:1]<0.05)[0]
            pc_data_norm = pc_data_norm[ind]
            pc_data_norm = farthest_points_downsampling(pc_data_norm, num_points=2048)
            #pc_data = data["pc"][0].squeeze()  ##unnorm
            pc_data = torch.from_numpy(pc_data_norm).unsqueeze(0)
            pc_data = pc_data.float().cuda()
            pc_data = pc_data.transpose(2,1).contiguous()

            pc_data_norm2,_,_ = pc_normalize(data.squeeze()) 
            #pc_data = data["pc"][0].squeeze()  ##unnorm
            pc_data2 = torch.from_numpy(pc_data_norm2).unsqueeze(0)
            pc_data2 = pc_data2.float().cuda()
            pc_data2 = pc_data2.transpose(2,1).contiguous()
            #pdb.set_trace()
            #print(pc_data.shape)
            afford_pred = torch.sigmoid(model(pc_data))
            afford_pred = afford_pred.permute(0, 2, 1).contiguous()
            #print(afford_pred[0].shape)
            target_dir1 = ('predict_results/mechmind/knife_%s_predict.xyz')%(filename)
            target_dir2 = ('predict_results/mechmind/knife_ori_%s_original.xyz')%(filename)
            np.savetxt(target_dir1, afford_pred[0].data.cpu().numpy())
            np.savetxt(target_dir2, pc_data[0].transpose(0,1).data.cpu().numpy())
    #np.savetxt('mug_label.xyz', afford_predict.data.cpu().numpy())

  

    