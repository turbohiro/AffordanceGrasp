# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from __future__ import print_function
from __future__ import absolute_import

import argparse
import h5py
import numpy as np
import copy
import os
import math
import time
import trimesh.transformations as tra
import json
import sample
import utils
import pdb
import trimesh
try:
    from Queue import Queue
except:
    from queue import Queue

from online_object_renderer import OnlineObjectRenderer
import random
import glob

class NoPositiveGraspsException(Exception):
   """raised when there's no positive grasps for an object."""
   pass


def inverse_transform(trans):
    """
      Computes the inverse of 4x4 transform.
    """
    rot = trans[:3, :3]
    t = trans[:3, 3]
    rot = np.transpose(rot)
    t = -np.matmul(rot, t)
    output = np.zeros((4, 4), dtype=np.float32)
    output[3][3] = 1
    output[:3, :3] = rot
    output[:3, 3] = t

    return output


def distance_by_translation_grasp(p1, p2):
    """
      Gets two nx4x4 numpy arrays and computes the translation of all the
      grasps.
    """
    t1 = p1[:, :3, 3]
    t2 = p2[:, :3, 3]
    return np.sqrt(np.sum(np.square(t1 - t2), axis=-1))


def distance_by_translation_point(p1, p2):
    """
      Gets two nx3 points and computes the disntace between point p1 and p2.
    """
    return np.sqrt(np.sum(np.square(p1 - p2), axis=-1))


def farthest_points(data, nclusters, dist_func, return_center_indexes=False, return_distances=False, verbose=False):
    """
      Performs farthest point sampling on data points.
      Args:
        data: numpy array of the data points.
        nclusters: int, number of clusters.
        dist_dunc: distance function that is used to compare two data points.
        return_center_indexes: bool, If True, returns the indexes of the center of 
          clusters.
        return_distances: bool, If True, return distances of each point from centers.
      
      Returns clusters, [centers, distances]:
        clusters: numpy array containing the cluster index for each element in 
          data.
        centers: numpy array containing the integer index of each center.
        distances: numpy array of [npoints] that contains the closest distance of 
          each point to any of the cluster centers.
    """
    if nclusters >= data.shape[0]:
        if return_center_indexes:
            return np.arange(data.shape[0], dtype=np.int32), np.arange(data.shape[0], dtype=np.int32)

        return np.arange(data.shape[0], dtype=np.int32)

    clusters = np.ones((data.shape[0],), dtype=np.int32) * -1
    distances = np.ones((data.shape[0],), dtype=np.float32) * 1e7
    centers = []
    for iter in range(nclusters):
        index = np.argmax(distances)
        centers.append(index)
        shape = list(data.shape)
        for i in range(1, len(shape)):
            shape[i] = 1

        broadcasted_data = np.tile(np.expand_dims(data[index], 0), shape)
        new_distances = dist_func(broadcasted_data, data)
        distances = np.minimum(distances, new_distances)
        clusters[distances == new_distances] = iter
        if verbose:
            print('farthest points max distance : {}'.format(np.max(distances)))

    if return_center_indexes:
        if return_distances:
            return clusters, np.asarray(centers, dtype=np.int32), distances
        return clusters, np.asarray(centers, dtype=np.int32)

    return clusters


def regularize_pc_point_count(pc, npoints, use_farthest_point=False):
    """
      If point cloud pc has less points than npoints, it oversamples.
      Otherwise, it downsample the input pc to have npoint points.
      use_farthest_point: indicates whether to use farthest point sampling
      to downsample the points. Farthest point sampling version runs slower.
    """
    if pc.shape[0] > npoints:
        if use_farthest_point:
            _, center_indexes = farthest_points(pc, npoints, distance_by_translation_point, return_center_indexes=True)
        else:
            center_indexes = np.random.choice(range(pc.shape[0]), size=npoints, replace=False)
        pc = pc[center_indexes, :]
    else:
        required = npoints - pc.shape[0]
        if required > 0:
            index = np.random.choice(range(pc.shape[0]), size=required)
            pc = np.concatenate((pc, pc[index, :]), axis=0)
    return pc



def perturb_grasp(grasp, num, min_translation, max_translation, min_rotation, max_rotation):
    """
      Self explanatory.
    """
    output_grasps = []
    for _ in range(num):
        sampled_translation = [np.random.uniform(lb, ub) for lb, ub in zip(min_translation, max_translation)]
        sampled_rotation = [np.random.uniform(lb, ub) for lb, ub in zip(min_rotation, max_rotation)]
        grasp_transformation = tra.euler_matrix(*sampled_rotation)
        grasp_transformation[:3, 3] = sampled_translation
        output_grasps.append(np.matmul(grasp, grasp_transformation))
    
    return output_grasps


def evaluate_grasps(grasp_tfs, obj_mesh):
    """
        Check the collision of the grasps and also heuristic quality for each
        grasp.
    """
    collisions, _ = sample.in_collision_with_gripper(
        obj_mesh,
        grasp_tfs,
        gripper_name='panda',
        silent=True,
    )
    qualities = sample.grasp_quality_point_contacts(
        grasp_tfs,
        collisions,
        object_mesh=obj_mesh,
        gripper_name='panda',
        silent=True,
    )

    return np.asarray(collisions), np.asarray(qualities)

def create_gripper_marker(color=[0, 0, 255], tube_radius=0.002, sections=6):
    """Create a 3D mesh visualizing a parallel yaw gripper. It consists of four cylinders.

    Args:
        color (list, optional): RGB values of marker. Defaults to [0, 0, 255].
        tube_radius (float, optional): Radius of cylinders. Defaults to 0.002.
        sections (int, optional): Number of sections of each cylinder. Defaults to 6.

    Returns:
        trimesh.Trimesh: A mesh that represents a simple parallel yaw gripper.
    """
    cfl = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[
            [4.10000000e-02, -7.27595772e-12, 6.59999996e-02],
            [4.10000000e-02, -7.27595772e-12, 1.12169998e-01],
        ],
    )
    cfr = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[
            [-4.100000e-02, -7.27595772e-12, 6.59999996e-02],
            [-4.100000e-02, -7.27595772e-12, 1.12169998e-01],
        ],
    )
    cb1 = trimesh.creation.cylinder(
        radius=tube_radius, sections=sections, segment=[[0, 0, 0], [0, 0, 6.59999996e-02]]
    )
    cb2 = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[[-4.100000e-02, 0, 6.59999996e-02], [4.100000e-02, 0, 6.59999996e-02]],
    )

    tmp = trimesh.util.concatenate([cb1, cb2, cfr, cfl])
    tmp.visual.face_colors = color

    return tmp

class PointCloudReader:
    def __init__(
        self,
        root_folder,
        batch_size,
        num_grasp_clusters,
        npoints,
        min_difference_allowed=(0, 0, 0),
        max_difference_allowed=(3, 3, 0),
        occlusion_nclusters=0,
        occlusion_dropout_rate=0.,
        caching=True,
        run_in_another_process=True,
        collision_hard_neg_min_translation=(-0.03,-0.03,-0.03),
        collision_hard_neg_max_translation=(0.03,0.03,0.03),
        collision_hard_neg_min_rotation=(-0.6,-0.2,-0.6),
        collision_hard_neg_max_rotation=(+0.6,+0.2,+0.6),
        collision_hard_neg_num_perturbations=10,
        use_uniform_quaternions=False,
        ratio_of_grasps_used=1.0,
        ratio_positive=0.3,
        ratio_hardnegative=0.4,
        balanced_data=True,
    ):
        self._root_folder = root_folder
        self._batch_size = batch_size
        self._num_grasp_clusters = num_grasp_clusters
        self._max_difference_allowed = max_difference_allowed
        self._min_difference_allowed = min_difference_allowed
        self._npoints = npoints
        self._occlusion_nclusters = occlusion_nclusters
        self._occlusion_dropout_rate = occlusion_dropout_rate
        self._caching = caching
        self._collision_hard_neg_min_translation = collision_hard_neg_min_translation
        self._collision_hard_neg_max_translation = collision_hard_neg_max_translation
        self._collision_hard_neg_min_rotation = collision_hard_neg_min_rotation
        self._collision_hard_neg_max_rotation = collision_hard_neg_max_rotation
        self._collision_hard_neg_num_perturbations = collision_hard_neg_num_perturbations
        self._collision_hard_neg_queue = {}
        self._ratio_of_grasps_used = ratio_of_grasps_used
        self._ratio_positive = ratio_positive
        self._ratio_hardnegative = ratio_hardnegative
        self._balanced_data = balanced_data

        for i in range(3):
            assert(collision_hard_neg_min_rotation[i] <= collision_hard_neg_max_rotation[i])
            assert(collision_hard_neg_min_translation[i] <= collision_hard_neg_max_translation[i])

        self._current_pc = None
        self._cache = {}
      
        self._renderer = OnlineObjectRenderer(caching=True)

        self._renderer.start()

        if use_uniform_quaternions:
            quaternions = [l[:-1].split('\t') for l in open('uniform_quaternions/data2_4608.qua', 'r').readlines()]

            quaternions = [[float(t[0]),
                            float(t[1]),
                            float(t[2]),
                            float(t[3])] for t in quaternions]
            quaternions = np.asarray(quaternions)
            quaternions = np.roll(quaternions, 1, axis=1)
            self._all_poses = [tra.quaternion_matrix(q) for q in quaternions]
        else:
            self._all_poses = utils.nonuniform_quaternions()
            
        
        self._eval_files = [json.load(open(f)) for f in glob.glob(os.path.join(self._root_folder, 'splits', '*.json'))]
        


    def apply_dropout(self, pc):
        if self._occlusion_nclusters == 0 or self._occlusion_dropout_rate == 0.:
            return np.copy(pc)

        labels = farthest_points(pc, self._occlusion_nclusters, distance_by_translation_point)

        removed_labels = np.unique(labels)
        removed_labels = removed_labels[np.random.rand(removed_labels.shape[0]) < self._occlusion_dropout_rate]
        if removed_labels.shape[0] == 0:
            return np.copy(pc)
        mask = np.ones(labels.shape, labels.dtype)
        for l in removed_labels:
            mask = np.logical_and(mask, labels != l)
        return pc[mask]
    
    def change_object_and_render(self,cad_path,cad_scale,camera_pose=None,thread_id=0):
        """
          Renders a random view and return (pc, camera_pose, object_pose). 
          object_pose is None for single object per scene.
        """
        if camera_pose is None:
            viewing_index = np.random.randint(0, high=len(self._all_poses))
            camera_pose = self._all_poses[viewing_index]
            
        in_camera_pose = copy.deepcopy(camera_pose)
        _, _, pc, camera_pose = self._renderer.change_and_render(
            cad_path, cad_scale, in_camera_pose, 0)
        pc = self.apply_dropout(pc)
        pc = regularize_pc_point_count(pc, 2048)
        pc_mean = np.mean(pc, 0, keepdims=True)
        pc[:, :3] -= pc_mean[:, :3]
        camera_pose[:3, 3] -= pc_mean[0, :3]

        return pc, camera_pose, in_camera_pose


    def change_object(self, cad_path, cad_scale):
        self._renderer.change_object(cad_path, cad_scale)

    def get_evaluator_data(self, grasp_path, verify_grasps=False):
        if self._balanced_data:
            return self._get_uniform_evaluator_data(grasp_path, verify_grasps)

        pos_grasps, pos_qualities, neg_grasps, neg_qualities, obj_mesh, cad_path, cad_scale = self.read_grasp_file(grasp_path)
        
        output_pcs = []
        output_grasps = []
        output_qualities = []
        output_labels = []
        output_pc_poses = []
        output_cad_paths = [cad_path] * self._batch_size
        output_cad_scales = np.asarray([cad_scale] * self._batch_size, np.float32)
        
        num_positive = int(self._batch_size * self._ratio_positive)
        positive_clusters = self.sample_grasp_indexes(num_positive, pos_grasps, pos_qualities)
        num_negative = self._batch_size - num_positive
        negative_clusters = self.sample_grasp_indexes(self._batch_size - num_positive, neg_grasps, neg_qualities)
        
        hard_neg_candidates = []
        # Fill in Positive Examples.
        for positive_cluster in positive_clusters:
            #print(positive_cluster)
            selected_grasp = pos_grasps[positive_cluster[0]][positive_cluster[1]]
            selected_quality = pos_qualities[positive_cluster[0]][positive_cluster[1]]
            output_grasps.append(selected_grasp)
            output_qualities.append(selected_quality)
            output_labels.append(1)
            hard_neg_candidates += perturb_grasp(
                selected_grasp,
                self._collision_hard_neg_num_perturbations,
                self._collision_hard_neg_min_translation,
                self._collision_hard_neg_max_translation,
                self._collision_hard_neg_min_rotation,
                self._collision_hard_neg_max_rotation,
            )

        if verify_grasps:
            collisions, heuristic_qualities = evaluate_grasps(
                output_grasps, obj_mesh
            )
            for computed_quality, expected_quality, g in zip(heuristic_qualities, output_qualities, output_grasps):
                err = abs(computed_quality - expected_quality)
                if err > 1e-3:
                    raise ValueError(
                        'Heuristic does not match with the values from data generation {}!={}'.format(
                            computed_quality, expected_quality
                        ))


        
        # If queue does not have enough data, fill it up with hard negative examples from the positives.
        if grasp_path not in self._collision_hard_neg_queue or self._collision_hard_neg_queue[grasp_path].qsize() < num_negative:
            if grasp_path not in self._collision_hard_neg_queue:
                self._collision_hard_neg_queue[grasp_path] = Queue()
            #hard negatives are perturbations of correct grasps.
            random_selector = np.random.rand()
            if random_selector < self._ratio_hardnegative:
                print('add hard neg')
                collisions, heuristic_qualities = evaluate_grasps(
                    hard_neg_candidates, obj_mesh
                )

                hard_neg_mask = collisions | (heuristic_qualities < 0.001)
                hard_neg_indexes = np.where(hard_neg_mask)[0].tolist()
                np.random.shuffle(hard_neg_indexes)
                for index in hard_neg_indexes:
                    self._collision_hard_neg_queue[grasp_path].put(
                        (hard_neg_candidates[index], -1.0)
                    )
            if random_selector >= self._ratio_hardnegative or self._collision_hard_neg_queue[grasp_path].qsize() < num_negative:
                for negative_cluster in negative_clusters:
                    selected_grasp = neg_grasps[negative_cluster[0]][negative_cluster[1]]
                    selected_quality = neg_qualities[negative_cluster[0]][negative_cluster[1]]
                    self._collision_hard_neg_queue[grasp_path].put(
                        (selected_grasp, selected_quality)
                    )

        # Use negative examples from queue.
        for _ in range(num_negative):
            #print('qsize = ', self._collision_hard_neg_queue[file_path].qsize())
            grasp, quality = self._collision_hard_neg_queue[grasp_path].get()
            output_grasps.append(grasp)
            output_qualities.append(quality)
            output_labels.append(0)

        self.change_object(cad_path, cad_scale)
        for iter in range(self._batch_size):
            if iter > 0:
                output_pcs.append(np.copy(output_pcs[0]))
                output_pc_poses.append(np.copy(output_pc_poses[0]))
            else:
                pc, camera_pose, _ = self.render_random_scene()
                output_pcs.append(pc)
                output_pc_poses.append(inverse_transform(camera_pose))

            output_grasps[iter] = camera_pose.dot(output_grasps[iter])

        output_pcs = np.asarray(output_pcs, dtype=np.float32)
        output_grasps = np.asarray(output_grasps, dtype=np.float32)
        output_labels = np.asarray(output_labels, dtype=np.int32)
        output_qualities = np.asarray(output_qualities, dtype=np.float32)
        output_pc_poses = np.asarray(output_pc_poses, dtype=np.float32)
        

        return output_pcs, output_grasps, output_labels, output_qualities, output_pc_poses, output_cad_paths, output_cad_scales


    def _get_uniform_evaluator_data(self, grasp_path, verify_grasps=False):
        pos_grasps, pos_qualities, neg_grasps, neg_qualities, obj_mesh, cad_path, cad_scale = self.read_grasp_file(grasp_path)
        
        output_pcs = []
        output_grasps = []
        output_qualities = []
        output_labels = []
        output_pc_poses = []
        output_cad_paths = [cad_path] * self._batch_size
        output_cad_scales = np.asarray([cad_scale] * self._batch_size, np.float32)
        
        num_positive = int(self._batch_size * self._ratio_positive)
        positive_clusters = self.sample_grasp_indexes(num_positive, pos_grasps, pos_qualities)
        num_hard_negative = int(self._batch_size * self._ratio_hardnegative)
        num_flex_negative = self._batch_size - num_positive - num_hard_negative
        negative_clusters = self.sample_grasp_indexes(num_flex_negative, neg_grasps, neg_qualities)
        #print(
        #    'positive = {}, hard_neg = {}, flex_neg = {}'.format(
        #        num_positive, num_hard_negative, num_flex_negative)
        #)
        
        
        hard_neg_candidates = []
        # Fill in Positive Examples.

        for clusters, grasps, qualities in zip([positive_clusters, negative_clusters], [pos_grasps, neg_grasps], [pos_qualities, neg_qualities]):
            for cluster in clusters:
                selected_grasp = grasps[cluster[0]][cluster[1]]
                selected_quality = qualities[cluster[0]][cluster[1]]
                hard_neg_candidates += perturb_grasp(
                    selected_grasp,
                    self._collision_hard_neg_num_perturbations,
                    self._collision_hard_neg_min_translation,
                    self._collision_hard_neg_max_translation,
                    self._collision_hard_neg_min_rotation,
                    self._collision_hard_neg_max_rotation,
                )

        
            

        if verify_grasps:
            collisions, heuristic_qualities = evaluate_grasps(
                output_grasps, obj_mesh
            )
            for computed_quality, expected_quality, g in zip(heuristic_qualities, output_qualities, output_grasps):
                err = abs(computed_quality - expected_quality)
                if err > 1e-3:
                    raise ValueError(
                        'Heuristic does not match with the values from data generation {}!={}'.format(
                            computed_quality, expected_quality
                        ))

        
        # If queue does not have enough data, fill it up with hard negative examples from the positives.
        if grasp_path not in self._collision_hard_neg_queue or len(self._collision_hard_neg_queue[grasp_path]) < num_hard_negative:
            if grasp_path not in self._collision_hard_neg_queue:
                self._collision_hard_neg_queue[grasp_path] = []
            #hard negatives are perturbations of correct grasps.
            collisions, heuristic_qualities = evaluate_grasps(
                hard_neg_candidates, obj_mesh
            )

            hard_neg_mask = collisions | (heuristic_qualities < 0.001)
            hard_neg_indexes = np.where(hard_neg_mask)[0].tolist()
            np.random.shuffle(hard_neg_indexes)
            for index in hard_neg_indexes:
                self._collision_hard_neg_queue[grasp_path].append(
                    (hard_neg_candidates[index], -1.0)
                )
            random.shuffle(self._collision_hard_neg_queue[grasp_path])
                
        # Adding positive grasps
        for positive_cluster in positive_clusters:
            #print(positive_cluster)
            selected_grasp = pos_grasps[positive_cluster[0]][positive_cluster[1]]
            selected_quality = pos_qualities[positive_cluster[0]][positive_cluster[1]]
            output_grasps.append(selected_grasp)
            output_qualities.append(selected_quality)
            output_labels.append(1)

        # Adding hard neg
        for i in range(num_hard_negative):
            #print('qsize = ', self._collision_hard_neg_queue[file_path].qsize())
            grasp, quality = self._collision_hard_neg_queue[grasp_path][i]
            output_grasps.append(grasp)
            output_qualities.append(quality)
            output_labels.append(0)

        self._collision_hard_neg_queue[grasp_path] = self._collision_hard_neg_queue[grasp_path][num_hard_negative:]
        
        # Adding flex neg
        if len(negative_clusters) != num_flex_negative:
            raise ValueError(
                'negative clusters should have the same length as num_flex_negative {} != {}'.format(len(negative_clusters), num_flex_negative)
            )
        
        for negative_cluster in negative_clusters:
            selected_grasp = neg_grasps[negative_cluster[0]][negative_cluster[1]]
            selected_quality = neg_qualities[negative_cluster[0]][negative_cluster[1]]
            output_grasps.append(selected_grasp)
            output_qualities.append(selected_quality)
            output_labels.append(0)


        self.change_object(cad_path, cad_scale)
        for iter in range(self._batch_size):
            if iter > 0:
                output_pcs.append(np.copy(output_pcs[0]))
                output_pc_poses.append(np.copy(output_pc_poses[0]))
            else:
                pc, camera_pose, _ = self.render_random_scene()
                output_pcs.append(pc)
                output_pc_poses.append(inverse_transform(camera_pose))

            output_grasps[iter] = camera_pose.dot(output_grasps[iter])

        output_pcs = np.asarray(output_pcs, dtype=np.float32)
        output_grasps = np.asarray(output_grasps, dtype=np.float32)
        output_labels = np.asarray(output_labels, dtype=np.int32)
        output_qualities = np.asarray(output_qualities, dtype=np.float32)
        output_pc_poses = np.asarray(output_pc_poses, dtype=np.float32)
        

        return output_pcs, output_grasps, output_labels, output_qualities, output_pc_poses, output_cad_paths, output_cad_scales



    def get_vae_data(self, grasp_path):
        pos_grasps, pos_qualities, _,  _, _, cad_path, cad_scale = self.read_grasp_file(grasp_path)
        
        #num = 0
        #good_pose = []
        # create visual markers for grasps
        #for i in range(len(pos_grasps[0])):
        #    if pos_grasps[0][i][2][3]<-0.02:
        #        num +=1
        #        good_pose.append(pos_grasps[0][i])
#
        #T = np.array(good_pose)
        #successful_grasps = [
        #    create_gripper_marker(color=[0, 255, 0]).apply_transform(t) for t in T]
        #json_dict = json.load(open(grasp_path))
        #obj_mesh = trimesh.load(cad_path)
        #obj_mesh = obj_mesh.apply_scale(cad_scale)
       # obj_mesh.vertices -= np.expand_dims(np.mean(obj_mesh.vertices, 0), 0)
        
        #trimesh.Scene([obj_mesh]+successful_grasps).show()   ##visualize successful grasp

        
        output_pcs = []
        output_grasps = []
        output_pc_poses = []
        output_cad_files = [cad_path] * self._batch_size
        output_cad_scales = np.asarray([cad_scale] * self._batch_size, dtype=np.float32)
        output_qualities = []

        
        all_clusters = self.sample_grasp_indexes(self._batch_size, pos_grasps, pos_qualities)


        pc, camera_pose, _ = self.change_object_and_render(
            cad_path,
            cad_scale,
            thread_id= 0)



        #self.change_object(cad_path, cad_scale)
        for iter in range(self._batch_size):
            selected_grasp_index = all_clusters[iter]

            selected_grasp = pos_grasps[selected_grasp_index[0]][selected_grasp_index[1]]
            selected_quality = pos_qualities[selected_grasp_index[0]][selected_grasp_index[1]]
            output_qualities.append(selected_quality)
            output_grasps.append(camera_pose.dot(selected_grasp))

            
            #if iter == 0:
           #     pc, camera_pose, _ = self.render_random_scene()
            #    output_pcs.append(pc)
            #    output_pc_poses.append(inverse_transform(camera_pose))
            #else:
            #    output_pcs.append(output_pcs[0].copy())
            #    output_pc_poses.append(output_pc_poses[0].copy())

            #output_grasps.append(
            #    camera_pose.dot(selected_grasp)
            #)
        def pc_normalize(pc):
            centroid = np.mean(pc, axis=0)
            pc = pc - centroid
            m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
            pc = pc / m
            return pc, centroid, m
        meta = {}
        meta['pc'] = np.array([pc] * self._batch_size)[:, :, :3]   #(640, 2048, 3)
        pc2,_,_ = pc_normalize(meta['pc'][0])

        meta['grasp_rt'] = np.array(output_grasps).reshape(        #(640, 16)
            len(output_grasps), -1)

        meta['pc_pose'] = np.array([inverse_transform(camera_pose)] *  #(640, 4, 4)
                                   self._batch_size)
        meta['cad_path'] = np.array([cad_path] *
                                    self._batch_size)
        meta['cad_scale'] = np.array([cad_scale] *
                                     self._batch_size)
        meta['quality'] = np.array(output_qualities)

        pdb.set_trace()
        #meta['target_cps'] = np.array(gt_control_points[:, :, :3])

        output_pcs = np.asarray(output_pcs, dtype=np.float32)    #64,(1024, 4)
        output_qualities = np.asarray(output_qualities, dtype=np.float32)
        output_grasps = np.asarray(output_grasps, dtype=np.float32)   #(4, 4)
        output_pc_poses = np.asarray(output_pc_poses, dtype=np.float32)  #(4, 4)
        return output_pcs, output_grasps, output_pc_poses, output_cad_files, output_cad_scales, output_qualities
    

    def sample_grasp_indexes(self, n, grasps, qualities):
        """
          Stratified sampling of the graps.
        """
        nonzero_rows = [i for i in range(len(grasps)) if len(grasps[i]) > 0]
        num_clusters = len(nonzero_rows)
        replace = n > num_clusters
        if num_clusters == 0:
            raise NoPositiveGraspsException
            
        grasp_rows = np.random.choice(range(num_clusters), size=n, replace=replace).astype(np.int32)
        grasp_rows = [nonzero_rows[i] for i in grasp_rows]
        grasp_cols = []
        for grasp_row in grasp_rows:
            if len(grasps[grasp_rows]) == 0:
                raise ValueError('grasps cannot be empty')

            grasp_cols.append(np.random.randint(len(grasps[grasp_row])))

        grasp_cols = np.asarray(grasp_cols, dtype=np.int32)

        
        return np.vstack((grasp_rows, grasp_cols)).T


    def read_grasp_file(self, path, return_all_grasps=False):
        file_name = path

        if self._caching and file_name in self._cache:
            pos_grasps, pos_qualities, neg_grasps, neg_qualities, cad, cad_path, cad_scale = copy.deepcopy(self._cache[file_name])
            return pos_grasps, pos_qualities, neg_grasps, neg_qualities, cad, cad_path, cad_scale

        
        pos_grasps, pos_qualities, neg_grasps, neg_qualities, cad, cad_path, cad_scale = self.read_object_grasp_data(
            path, ratio_of_grasps_to_be_used=self._ratio_of_grasps_used, return_all_grasps=return_all_grasps
        )
        
        if self._caching:
            self._cache[file_name] = (pos_grasps, pos_qualities, neg_grasps, neg_qualities,
                                      cad, cad_path, cad_scale)
            return copy.deepcopy(self._cache[file_name])

        return pos_grasps, pos_qualities, neg_grasps, neg_qualities, cad, cad_path, cad_scale
    

    def read_object_grasp_data(
        self, json_path, quality='quality_flex_object_in_gripper', 
        ratio_of_grasps_to_be_used=1., return_all_grasps=False):
        """
        Reads the grasps from the json path and loads the mesh and all the 
        grasps.
        """
        num_clusters = self._num_grasp_clusters
        root_folder = self._root_folder

        if num_clusters <= 0:
            raise NoPositiveGraspsException

        json_dict = json.load(open(json_path))
        
        object_model = sample.Object(os.path.join(root_folder, json_dict['object']))
        object_model.rescale(json_dict['object_scale'])
        object_model = object_model.mesh
        object_mean = np.mean(object_model.vertices, 0, keepdims=1)

        object_model.vertices -= object_mean
        grasps = np.asarray(json_dict['transforms'])
        grasps[:, :3, 3] -= object_mean

        flex_qualities = np.asarray(json_dict[quality])
        try:
            heuristic_qualities = np.asarray(json_dict['quality_number_of_contacts'])
        except KeyError:
            heuristic_qualities = np.ones(flex_qualities.shape)

        successful_mask = np.logical_and(flex_qualities > 0.01, heuristic_qualities > 0.01)
        
        positive_grasp_indexes = np.where(successful_mask)[0]
        negative_grasp_indexes = np.where(~successful_mask)[0]

        positive_grasps = grasps[positive_grasp_indexes, :, :]
        negative_grasps = grasps[negative_grasp_indexes, :, :]
        positive_qualities = heuristic_qualities[positive_grasp_indexes]
        negative_qualities = heuristic_qualities[negative_grasp_indexes]
        # print('positive grasps: {} negative grasps: {}'.format(positive_grasps.shape, negative_grasps.shape))

        def cluster_grasps(grasps, qualities):
            cluster_indexes = np.asarray(farthest_points(grasps, num_clusters, distance_by_translation_grasp))
            output_grasps = []
            output_qualities = []

            for i in range(num_clusters):
                indexes = np.where(cluster_indexes == i)[0]
                if ratio_of_grasps_to_be_used < 1:
                    num_grasps_to_choose = max(1, int(ratio_of_grasps_to_be_used * float(len(indexes))))
                    if len(indexes) == 0:
                        raise NoPositiveGraspsException
                    indexes = np.random.choice(indexes, size=num_grasps_to_choose, replace=False)
                    
                output_grasps.append(grasps[indexes, :, :])
                output_qualities.append(qualities[indexes])
            
            output_grasps = np.asarray(output_grasps)
            output_qualities = np.asarray(output_qualities)
            
            return output_grasps, output_qualities
        
        if not return_all_grasps:
            positive_grasps, positive_qualities = cluster_grasps(
                positive_grasps, positive_qualities
            )
            negative_grasps, negative_qualities = cluster_grasps(
                negative_grasps, negative_qualities
            )
            num_positive_grasps = np.sum([p.shape[0] for p in positive_grasps])
            num_negative_grasps = np.sum([p.shape[0] for p in negative_grasps])
        else:
            num_positive_grasps = positive_grasps.shape[0]
            num_negative_grasps = negative_grasps.shape[0]


        return positive_grasps, positive_qualities, negative_grasps, negative_qualities, object_model, os.path.join(root_folder, json_dict['object']), json_dict['object_scale']

    
    def generate_object_set(self, split_name):
        obj_files = self._eval_files[np.random.randint(0, len(self._eval_files))][split_name]
        return os.path.join('grasps', obj_files[np.random.randint(0, len(obj_files))])

    
    def arrange_objects(self, meshes):
        return np.eye(4)

    def __del__(self):
        print('********** terminating renderer **************')
        self._renderer.terminate()
        self._renderer.join()
        print('done')
    

if __name__ == '__main__':
   
    parser = argparse.ArgumentParser(description="Grasp data reader")
    parser.add_argument(
            '--root-folder',
            help='Root dir for data',
            type=str,
            default='../unified_grasp_data')
    parser.add_argument(
            '--vae-mode',
            help='True for vae mode',
            action='store_true',
            default=False)
    parser.add_argument(
            '--grasps-ratio',
            help='ratio of grasps to be used from each cluster. At least one grasp is chosen from each cluster.',
            type=float,
            default=1.0
    )
    parser.add_argument(
        '--balanced_data',
        action='store_true',
        default=False,
    )
    parser.add_argument('--allowed_category', default='', type=str)

    args = parser.parse_args()
    args.root_folder = os.path.abspath(args.root_folder)
    print('Root folder', args.root_folder)

    #import glob
    #from visualization_utils import draw_scene
    #import mayavi.mlab as mlab

    pcreader = PointCloudReader(
        root_folder=args.root_folder,
        batch_size=640,
        num_grasp_clusters=32,
        npoints=1024,
        ratio_of_grasps_used=args.grasps_ratio,
        balanced_data=args.balanced_data
    )


    #grasp_paths = glob.glob(os.path.join(args.root_folder, 'grasps') + '/*.json')

    #grasp_paths = os.path.join(args.root_folder, 'grasps') + '/box_box013_1.0.json'  #read_box_date   validated success!!
    grasp_paths = os.path.join(args.root_folder, 'grasps') + '/mug_10f6e09036350e92b3f21f1137c3c347_0.191847137181.json' #success--shapenetCore
    #grasp_paths = os.path.join(args.root_folder, 'grasps') + '/mug_fad118b32085f3f2c2c72e575af174cd_0.187040581981.json' 


    #cad_file = '/data/pytorch_6dof-graspnet/unified_grasp_data/meshes/mug/fad118b32085f3f2c2c72e575af174cd.stl'
    #cad_scale = 0.18704058
    #obj_mesh = trimesh.load(cad_file)
    #obj_mesh = obj_mesh.apply_scale(cad_scale)
    #obj_mesh.vertices -= np.expand_dims(np.mean(obj_mesh.vertices, 0), 0)
    #grasps = np.load('grasps.npy')
    #successful_grasps = [
    #        create_gripper_marker(color=[0, 255, 0]).apply_transform(t) for t in grasps]
    #trimesh.Scene([obj_mesh]+successful_grasps).show()   ##visualize successful grasp
    if args.vae_mode:
        output_pcs, output_grasps, output_pc_poses, output_cad_files, output_cad_scales, output_qualities = pcreader.get_vae_data(grasp_paths)
        output_labels = None
    else:
        output_pcs, output_grasps, output_labels, output_qualities, output_pc_poses, output_cad_files, output_cad_scales = pcreader.get_evaluator_data(grasp_paths, verify_grasps=False)
    
    
    print(output_grasps.shape)
    
    for pc, pose in zip(output_pcs, output_pc_poses):
        assert(np.all(pc == output_pcs[0]))
        assert(np.all(pose == output_pc_poses[0]))

    
    pc = output_pcs[0]
    pose = output_pc_poses[0]
    cad_file = output_cad_files[0]
    cad_scale = output_cad_scales[0]
    #obj = sample.Object(cad_file)
   # obj.rescale(cad_scale)
   # obj = obj.mesh
    #obj.vertices -= np.expand_dims(np.mean(obj.vertices, 0), 0)
    

    
    print('mean_pc', np.mean(pc, 0))
    print('pose', pose)

    grasps=np.array([pose.dot(g) for g in output_grasps])
    # num = 0
       # good_pose = []
        # create visual markers for grasps
      #  for i in range(len(pos_grasps[0])):
      #      if pos_grasps[0][i][2][3]<-0.02:
      #          num +=1
    #            good_pose.append(pos_grasps[0][i])
#
      #  T = np.array(good_pose)
    np.save('grasps.npy', grasps)
    pdb.set_trace()
    
      #  json_dict = json.load(open(grasp_path))

    draw_scene(
        pc,
        grasps=output_grasps,
        grasp_scores=None if args.vae_mode else output_labels,
    )
    mlab.figure()
    draw_scene(
        pc.dot(pose.T),
        grasps=[pose.dot(g) for g in output_grasps],
        mesh=obj,
        grasp_scores=None if args.vae_mode else output_labels,
    )
    mlab.show()