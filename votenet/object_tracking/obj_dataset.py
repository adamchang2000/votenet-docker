# coding: utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Dataset for 3D object detection on SUN RGB-D (with support of vote supervision).

A sunrgbd oriented bounding box is parameterized by (cx,cy,cz), (l,w,h) -- (dx,dy,dz) in upright depth coord
(Z is up, Y is forward, X is right ward), heading angle (from +X rotating to -Y) and semantic class

Point clouds are in **upright_depth coordinate (X right, Y forward, Z upward)**
Return heading class, heading residual, size class and size residual for 3D bounding boxes.
Oriented bounding box is parameterized by (cx,cy,cz), (l,w,h), heading_angle and semantic class label.
(cx,cy,cz) is in upright depth coordinate
(l,h,w) are *half length* of the object sizes
The heading angle is a rotation rad from +X rotating towards -Y. (+X is 0, -Y is pi/2)

Author: Charles R. Qi
Date: 2019

"""
import os
import sys
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
import scipy.io as sio # to load .mat files for depth points
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util
from obj_to_pointcloud_util import *
from model_util_obj import OBJDatasetConfig
from scipy.spatial.transform import Rotation as R

DC = OBJDatasetConfig() # dataset specific config
MAX_NUM_OBJ = 1 # maximum number of objects allowed per scene
MEAN_COLOR_RGB = np.array([0.,0.,0.]) # sunrgbd color is in 0~1
MEAN_EXTRA_CHANNELS = np.array([0.])

class OBJDetectionVotesDataset(Dataset):
    def __init__(self, model_path, split_set='train', num_points=25000, use_color=False, extra_channels=0, augment=True):

        #assert(num_points<=100000)

        assert(os.path.isdir(model_path))
        assert(os.path.exists(os.path.join(model_path, split_set + '_samples.npz')))

        self.model_path = model_path
        self.samples = np.load(os.path.join(model_path, split_set + '_samples.npz'))['samples']
        self.num_points = num_points
        self.use_color = use_color
        self.extra_channels = extra_channels

        #make sure the mean is the same size
        assert(extra_channels == MEAN_EXTRA_CHANNELS.shape[0])
        
        self.augment = augment

        if split_set == 'train' and not augment:
            print("WARNING, AUGMENTATION OFF FOR TRAINING")

        print(split_set, augment)
       
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns a dict with following keys:
            point_clouds: (N,3+C)
            center_label: (MAX_NUM_OBJ,3) for GT box center XYZ
            heading_class_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
            heading_residual_label: (MAX_NUM_OBJ,)
            size_classe_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
            size_residual_label: (MAX_NUM_OBJ,3)
            sem_cls_label: (MAX_NUM_OBJ,) semantic class index
            box_label_mask: (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
            vote_label: (N,9) with votes XYZ (3 votes: X1Y1Z1, X2Y2Z2, X3Y3Z3)
                if there is only one vote than X1==X2==X3 etc.
            vote_label_mask: (N,) with 0/1 with 1 indicating the point
                is in one of the object's OBB.
            scan_idx: int scan index in scan_names list
            max_gt_bboxes: unused
        """
        sample_num = self.samples[idx]
        #print('getting sample ', sample_num)
        #pcld is in o3d pointcloud format pcld.points, pcld.colors
        #bb needs to be converted as well 
        #votes are numpy array

        data = np.load(os.path.join(self.model_path, str(sample_num) + '_data.npz'))
        scene_point_cloud = data['scene_point_cloud']
        model_point_cloud = data['model_point_cloud']
        box3d_centers = data['box3d_centers']
        axis_angles = data['axis_angles']
        theta = data['theta']

        if self.augment:

            rot = R.from_rotvec(axis_angles * theta)

            #new_axis_angles = axis_angles + np.random.uniform(-0.2, 0.2, size=3)
            #new_axis_angles /= np.linalg.norm(new_axis_angles)

            #testing only 1dof rotation
            new_axis_angles = axis_angles

            #print('dist from new to old ', np.linalg.norm(new_axis_angles - axis_angles))

            new_theta = np.random.uniform(0, np.pi * 2)

            model_point_cloud_centered = model_point_cloud[:,:3] - box3d_centers
            model_point_cloud[:,:3] = (axisAnglesToRotationMatrix(new_axis_angles, new_theta) @ model_point_cloud_centered.T).T + box3d_centers

            rot2 = R.from_rotvec(new_axis_angles * new_theta)

            axis_angles = (rot2 * rot).as_rotvec()
            theta = np.linalg.norm(axis_angles)
            axis_angles /= np.linalg.norm(axis_angles)

        votes = box3d_centers - model_point_cloud[:,:3]

        votes = np.vstack((votes, np.zeros((scene_point_cloud.shape[0], 3))))

        if len(scene_point_cloud) > 0:
            point_cloud = np.vstack((model_point_cloud, scene_point_cloud))
        else:
            point_cloud = model_point_cloud

        vote_mask = np.hstack((np.ones(model_point_cloud.shape[0]), np.zeros(scene_point_cloud.shape[0])))

        if self.use_color and self.extra_channels > 0:
            point_cloud = point_cloud[:,0:6 + self.extra_channels]
            point_cloud[:,3:] = (point_cloud[:,3:]-MEAN_COLOR_RGB)
            #point_cloud[:,6:] = 1. - point_cloud[:,6:] #INVERT SINGLE CHANNEL BITS
            point_cloud[:,6:] = (point_cloud[:,6:]-MEAN_EXTRA_CHANNELS)
        elif self.use_color:
            point_cloud = point_cloud[:,0:6]
            point_cloud[:,3:] = (point_cloud[:,3:]-MEAN_COLOR_RGB)
        elif self.extra_channels > 0:
            #point_cloud[:,3:] = 1. - point_cloud[:,3:] #INVERT SINGLE CHANNEL BITS
            point_cloud = point_cloud[:,0:3 + self.extra_channels]
            point_cloud[:,3:] = (point_cloud[:,3:]-MEAN_EXTRA_CHANNELS)
        else:
            point_cloud = point_cloud[:,0:3]

        #random sample points
        n = self.num_points
        if point_cloud.shape[0] > n:
            index = np.random.choice(point_cloud.shape[0], n, replace=False)
            point_cloud = point_cloud[index]
            votes = votes[index]
            vote_mask = vote_mask[index]

        if self.augment:
            #randomly perturb each point by normal dist, std = 0.1cm
            perturb = np.random.normal(0, 0.001, size=(point_cloud.shape[0], 3))
            point_cloud[:,0:3] += perturb

            #vote = center - point, vote = center - (point + perturb)
            votes -= perturb

            #adding noise to color channels
            color_noise_std = 0.1
            color_channel_noise_std = 0.02

            #add noise to binary channels, extra_channels
            noise_ratio = 0.1 #10% point of points, set to no activation
            index = np.random.choice(point_cloud.shape[0], int(n * noise_ratio), replace=False)

            if self.use_color and self.extra_channels > 0:
                point_cloud[:,3:] += np.random.normal(0, color_noise_std) #global illumination change
                point_cloud[:,3:] = np.random.normal(point_cloud[:,3:], color_channel_noise_std) #changing every value, smaller std
                point_cloud[:,6:][index] = 0. #turn off binary bits
            elif self.use_color:
                point_cloud[:,3:] += np.random.normal(0, color_noise_std) #global illumination change
                point_cloud[:,3:] = np.random.normal(point_cloud[:,3:], color_channel_noise_std) #changing every value, smaller std
            elif self.extra_channels > 0:
                point_cloud[:,3:][index] = 0. #turn off binary bits

        # ------------------------------- LABELS ------------------------------
        """
        Returns a dict with following keys:
            point_clouds: (N,3+C)
            center_label: (MAX_NUM_OBJ,3) for GT box center XYZ
            heading_class_label: (MAX_NUM_OBJ,3) with int values in 0,...,NUM_HEADING_BIN-1
            heading_residual_label: (MAX_NUM_OBJ,3)
            size_classe_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
            size_residual_label: (MAX_NUM_OBJ,3)
            sem_cls_label: (MAX_NUM_OBJ,) semantic class index
            box_label_mask: (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
            vote_label: (N,9) with votes XYZ (3 votes: X1Y1Z1, X2Y2Z2, X3Y3Z3)
                if there is only one vote than X1==X2==X3 etc.
            vote_label_mask: (N,) with 0/1 with 1 indicating the point
                is in one of the object's OBB.
            scan_idx: int scan index in scan_names list
            max_gt_bboxes: unused
        """

        
        #angle_class_and_residuals = [DC.angle2class(euler_angles[2])]
        angle_class_and_residuals = [DC.angle2class(theta)]

        rotation_vector = np.asarray([axis_angles])

        angle_classes = np.asarray([x[0] for x in angle_class_and_residuals])
        angle_residuals = np.asarray([x[1] for x in angle_class_and_residuals])

        angle_classes1 = np.asarray([angle_classes[0]])
        angle_residuals1 = np.asarray([angle_residuals[0]])

        label_mask = np.asarray([1])

        vote_labels = np.asarray([[a[0], a[1], a[2], a[0], a[1], a[2], a[0], a[1], a[2]] for a in votes])

        ret_dict = {}
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ret_dict['center_label'] = box3d_centers.astype(np.float32)
        ret_dict['heading_class_label'] = angle_classes1.astype(np.int64)
        ret_dict['heading_residual_label'] = angle_residuals1.astype(np.float32)
        ret_dict['rotation_vector_label'] = rotation_vector.astype(np.float32)
        ret_dict['box_label_mask'] = label_mask.astype(np.float32)
        ret_dict['vote_label'] = vote_labels.astype(np.float32)
        ret_dict['vote_label_mask'] = vote_mask.astype(np.int64)
        ret_dict['scan_idx'] = np.array(sample_num).astype(np.int64)
        ret_dict['max_gt_bboxes'] = np.asarray([])
        return ret_dict

def viz_votes(pc, point_votes, point_votes_mask):
    """ Visualize point votes and point votes mask labels
    pc: (N,3 or 6), point_votes: (N,9), point_votes_mask: (N,)
    """
    inds = (point_votes_mask==1)
    pc_obj = pc[inds,0:3]
    pc_obj_voted1 = pc_obj + point_votes[inds,0:3]
    pc_obj_voted2 = pc_obj + point_votes[inds,3:6]
    pc_obj_voted3 = pc_obj + point_votes[inds,6:9]
    pc_util.write_ply(pc_obj, 'pc_obj.ply')
    pc_util.write_ply(pc_obj_voted1, 'pc_obj_voted1.ply')
    pc_util.write_ply(pc_obj_voted2, 'pc_obj_voted2.ply')
    pc_util.write_ply(pc_obj_voted3, 'pc_obj_voted3.ply')

def viz_obb(pc, label, mask, angle_class_and_residual, rotation_vector):
    """ Visualize oriented bounding box ground truth
    pc: (N,3)
    label: (K,3)  K == MAX_NUM_OBJ
    mask: (K,)
    angle_classes: (K,3)
    angle_residuals: (K,3)
    size_classes: (K,)
    size_residuals: (K,3)
    """
    oriented_boxes = []
    K = label.shape[0]
    for i in range(K):
        if mask[i] == 0: continue
        obb = DC.param2obb(label[i, 0:3], angle_class_and_residual, rotation_vector, 0)
        print(obb)
        oriented_boxes.append(obb)
    #pc_util.write_oriented_bbox(oriented_boxes, 'gt_obbs.ply')
    pc_util.write_oriented_bbox_6dof(oriented_boxes, 'gt_obbs.ply')
    pc_util.write_ply(label[mask==1,:], 'gt_centroids.ply')

def get_sem_cls_statistics():
    """ Compute number of objects for each semantic class """
    d = SunrgbdDetectionVotesDataset(use_height=True, use_color=True, use_v1=True, augment=True)
    sem_cls_cnt = {}
    for i in range(len(d)):
        if i%10==0: print(i)
        sample = d[i]
        pc = sample['point_clouds']
        sem_cls = sample['sem_cls_label']
        mask = sample['box_label_mask']
        for j in sem_cls:
            if mask[j] == 0: continue
            if sem_cls[j] not in sem_cls_cnt:
                sem_cls_cnt[sem_cls[j]] = 0
            sem_cls_cnt[sem_cls[j]] += 1
    print(sem_cls_cnt)

if __name__=='__main__':
    assert (len(sys.argv) == 2)
    d = OBJDetectionVotesDataset(sys.argv[1], num_points=75000, extra_channels=1, augment=True, split_set='train')
    sample = d[0]
    #print(sample['vote_label'].shape, sample['vote_label_mask'].shape, np.sum(sample['vote_label']))
    pc_util.write_ply(sample['point_clouds'], 'pc.ply')
    viz_votes(sample['point_clouds'], sample['vote_label'], sample['vote_label_mask'])
    viz_obb(sample['point_clouds'], sample['center_label'], sample['box_label_mask'],
        [sample['heading_class_label'], sample['heading_residual_label']], sample['rotation_vector_label'])
