from __future__ import print_function

import numpy as np
import argparse
import grasp_estimator
import sys
import os
import glob
import mayavi.mlab as mlab
from utils.visualization_utils_mug import *
from utils import utils
from data import DataLoader
from options.test_options import TestOptions

from os.path import join as opj
import torch
import torch.nn as nn
from gorilla.config import Config
from models2 import *
import loss
from utils2 import *
import open3d as o3d
import random
import colorsys
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from models import create_model
from utils.writer import Writer
import trimesh.transformations as tra
import pdb


def affordance_parse_args():
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

def make_parser():
    parser = argparse.ArgumentParser(
        description='6-DoF GraspNet Demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--grasp_sampler_folder',
                        type=str,
                        default='checkpoints_knife/wrap/gan_lr_0002_bs_64_scale_1_npoints_128_radius_02_latent_size_2')  #checkpoints_grasp/gan_lr_0002_bs_64_scale_1_npoints_128_radius_02_latent_size_2
    parser.add_argument('--grasp_evaluator_folder',
                        type=str,
                        default='checkpoints_all/evaluator_lr_0002_bs_64_scale_1_npoints_128_radius_02/')#checkpoints_all/evaluator_lr_0002_bs_64_scale_1_npoints_128_radius_02/
    parser.add_argument('--refinement_method',
                        choices={"gradient", "sampling"},
                        default='sampling')
    parser.add_argument('--refine_steps', type=int, default=25)

    parser.add_argument('--npy_folder', type=str, default='demo/data/')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.8,  #0.8
        help=
        "When choose_fn is something else than all, all grasps with a score given by the evaluator notwork less than the threshold are removed"
    )
    parser.add_argument(
        '--choose_fn',
        choices={
            "all", "better_than_threshold", "better_than_threshold_in_sequence"
        },
        default='better_than_threshold',
        help=
        "If all, no grasps are removed. If better than threshold, only the last refined grasps are considered while better_than_threshold_in_sequence consideres all refined grasps"
    )

    parser.add_argument('--target_pc_size', type=int, default=2048)  #1024
    parser.add_argument('--num_grasp_samples', type=int, default=200)
    parser.add_argument(
        '--generate_dense_grasps',
        action='store_true',
        help=
        "If enabled, it will create a [num_grasp_samples x num_grasp_samples] dense grid of latent space values and generate grasps from these."
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=60,
        help=
        "Set the batch size of the number of grasps we want to process and can fit into the GPU memory at each forward pass. The batch_size can be increased for a GPU with more memory."
    )

    parser.add_argument('--dataset_root_folder',
                            type=str,
                            default = 'grasp_data2/', #datatset_mug
                            help='path to root directory of the dataset.')
    return parser

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

#visulization of object point cloud
def get_color_for_pc(pc, K, color_image):
    proj = pc.dot(K.T)
    proj[:, 0] /= proj[:, 2]
    proj[:, 1] /= proj[:, 2]

    pc_colors = np.zeros((pc.shape[0], 3), dtype=np.uint8)
    for i, p in enumerate(proj):
        x = int(p[0])
        y = int(p[1])
        pc_colors[i, :] = color_image[y, x, :]

    return pc_colors


#obtain the object point cloud from depth image
def backproject(depth_cv,
                intrinsic_matrix,
                return_finite_depth=True,
                return_selection=False):

    depth = depth_cv.astype(np.float32, copy=True)

    # get intrinsic matrix
    K = intrinsic_matrix
    Kinv = np.linalg.inv(K)

    # compute the 3D points
    width = depth.shape[1]
    height = depth.shape[0]

    # construct the 2D points matrix
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    x2d = np.stack((x, y, ones), axis=2).reshape(width * height, 3)

    # backprojection
    R = np.dot(Kinv, x2d.transpose())

    # compute the 3D points
    X = np.multiply(np.tile(depth.reshape(1, width * height), (3, 1)), R)
    X = np.array(X).transpose()
    if return_finite_depth:
        selection = np.isfinite(X[:, 0])
        X = X[selection, :]

    if return_selection:
        return X, selection

    return X

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, centroid, m

def main(args):

    ##3d affordcanc network
    args_affordance = affordance_parse_args()
    cfg = Config.fromfile(args_affordance.config)
    if args_affordance.gpu != None:
        cfg.training_cfg.gpu = args_affordance.gpu
    print(cfg)
 
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.training_cfg.gpu
    num_gpu = len(cfg.training_cfg.gpu.split(','))

    if cfg.get('seed', None) != None:
        set_random_seed(cfg.seed)

    if cfg.get('with_loss', None) == None:
        cfg.update({"with_loss": args_affordance.with_loss})
    model = build_model(cfg).cuda()

    if num_gpu > 1:
        model = nn.DataParallel(model)
    
    if args_affordance.checkpoint != None:
        print("Loading checkpoint....")
        _, exten = os.path.splitext(args_affordance.checkpoint)
        if exten == '.t7':
            #pdb.set_trace()
            model.load_state_dict(torch.load(args_affordance.checkpoint))
        elif exten == '.pth':
            check = torch.load(args_affordance.checkpoint)
            model.load_state_dict(check['model_state_dict'])
    else:
        raise ValueError("Must specify a checkpoint path!")

    model.eval()
    print('affordance segmentation model is loaded and begins to eval')


    #grasp sampling and evaluating network
    parser = make_parser()
    args = parser.parse_args()
    grasp_sampler_args = utils.read_checkpoint_args(args.grasp_sampler_folder)
    grasp_sampler_args.is_train = False
    grasp_evaluator_args = utils.read_checkpoint_args(
        args.grasp_evaluator_folder)
    grasp_evaluator_args.continue_train = True
    estimator = grasp_estimator.GraspEstimator(grasp_sampler_args,
                                               grasp_evaluator_args, args)
    train_data = True  
    affordance = False

    if train_data:
        grasp_sampler_args.dataset_root_folder = args.dataset_root_folder
        grasp_sampler_args.num_grasps_per_object = 1
        grasp_sampler_args.num_objects_per_batch = 1
        opt = TestOptions().parse()
        opt.serial_batches = True  # no shuffle
        #opt.name = name
        dataset = DataLoader(opt)
        #print('&&&', args.dataset_root_folder)
        #dataset = DataLoader(grasp_sampler_args)
        #writer.reset_counter() 
        
        for home, dirs, files in os.walk('test_experiment/mechmind2/'):

            for filename in files:
          
                fullname = os.path.join(home, filename)
                grasp_object=  o3d.io.read_point_cloud(fullname)  ##point cloud file--.pcd/ply
                #grasp_object = pc_normalize(grasp_object)
                grasp_object,_ = grasp_object.remove_statistical_outlier(800,0.6)
                grasp_object.points = o3d.utility.Vector3dVector(
                farthest_points_downsampling(grasp_object.points, num_points=2048))
                data = np.asarray(grasp_object.points,dtype='float32')

                pc_mean = np.mean(data, 0, keepdims=True)
                data[:, :3] -= pc_mean[:, :3]

            
                #get the 3d affordance value
                pose = tra.euler_matrix(0,0 , 0)  #-np.pi/2 
                data_aff = data.dot(pose[:3,:3].T)  
                pose2 = tra.euler_matrix(0 ,-np.pi/2 , 0)  #-np.pi/2 
                data2 = data_aff.dot(pose2[:3,:3].T)  

                pc_data_norm,_,_ = pc_normalize(data2.squeeze()) 

                #ind = np.where(pc_data_norm[:,0:1]<0.05)[0]
                #pc_data_norm = pc_data_norm[ind]
                #pc_data_norm = farthest_points_downsampling(pc_data_norm, num_points=2048)
               
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
                pdb.set_trace()
                ##get the affordance score
                label = afford_pred[0].data.cpu().numpy()
                points =  pc_data2[0].transpose(0,1).data.cpu().numpy()
                grasp = label[:,8]   #the label of grasp
                grasp_index = np.argwhere(grasp>0.005)
                grasp_limit = min(grasp[np.argpartition(grasp,-20)[-20:]])  #pick top 20 affordance points
                grasp_index = np.argwhere(grasp>grasp_limit)
                a = grasp_index.T

                points_grasp = points[grasp_index.T[0]]
                #print(points_grasp)
                #pdb.set_trace()
                #generate and refine grasp
                #pdb.set_trace()
                #data = pc_data_norm
                #pc_data_norm,_,_ = pc_normalize(data)
                
                generated_grasps, generated_scores = estimator.generate_and_refine_grasps(data.squeeze())
                
                #x_arr = []
                #y_arr = []
                #z_arr = []

                #for i in range(len(points_grasp)):
                #    x_arr.append(points_grasp[i][0])
                #    y_arr.append(points_grasp[i][1])
                #    z_arr.append(points_grasp[i][2])
                #pc_mean = np.array([np.mean(x_arr),np.mean(y_arr),np.mean(z_arr)])
                #print(np.mean(x_arr),np.mean(y_arr),np.mean(z_arr)) 
                afford_score_array =[] 
                for i in range(len(generated_grasps)):
                    g = generated_grasps[i]               
                    score_arr = []
                    for j in range(len(points_grasp)):
                        score = np.linalg.norm(points_grasp[j] - g[:3, 3]) 
                        score_arr.append(score)                  
                    affordance_scores = -min(score_arr) 
                    afford_score_array.append(affordance_scores)
                
                #pdb.set_trace()
                mlab.figure(bgcolor=(1, 1, 1))
                #pdb.set_trace()
                mid_points = draw_scene(data,
                            pc_color=None, #None,pc_colors
                            gripper_color=(1, 0, 0),
                            grasps=generated_grasps,
                            grasp_scores=generated_scores,
                            affordance_score = afford_score_array,
                            visualize_diverse_grasps=False,
                            show_gripper_mesh=False)
                print('close the window to continue to next object . . .')
                mlab.show()

                pdb.set_trace()
    


if __name__ == '__main__':
    main(sys.argv[1:])
