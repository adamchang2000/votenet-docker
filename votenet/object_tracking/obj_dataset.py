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
from torch.utils.data import Dataset
import scipy.io as sio # to load .mat files for depth points
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util
from obj_to_pointcloud_util import *
from model_util_obj import OBJDatasetConfig

DC = OBJDatasetConfig() # dataset specific config
MAX_NUM_OBJ = 1 # maximum number of objects allowed per scene
MEAN_COLOR_RGB = np.array([0.5,0.5,0.5]) # sunrgbd color is in 0~1

class OBJDetectionVotesDataset(Dataset):
    def __init__(self, model_path, split_set='train', num_points=10000, use_color=False, augment=True, dropout_rate=0.2):

        #assert(num_points<=100000)

        assert(os.path.isdir(model_path))
        assert(os.path.exists(os.path.join(model_path, 'train_samples.npz')))
        assert(os.path.exists(os.path.join(model_path, 'test_samples.npz')))

        self.model_path = model_path
        self.samples = np.load(os.path.join(model_path, split_set + '_samples.npz'))['samples']
        self.num_points = num_points
        self.use_color = use_color
        self.augment = augment
        self.dropout_rate = dropout_rate
       
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
        point_cloud = data['point_cloud']
        box3d_centers = data['box3d_centers']
        box3d_sizes = data['box3d_sizes']
        euler_angles = data['euler_angles']
        #axis_angles = data['axis_angles']

        #print('xd ', axis_angles)

        votes = data['votes']
        vote_mask = data['vote_mask']
        
        #points = np.asarray(pcld.points)
        #colors = np.asarray(pcld.colors)

        #assert(len(point_cloud.points) == len(point_cloud.colors))
        #assert(len(point_cloud) == self.num_points)


        if not self.use_color:
            point_cloud = point_cloud[:,0:3]
        else:
            point_cloud = point_cloud[:,0:6]
            point_cloud[:,3:] = (point_cloud[:,3:]-MEAN_COLOR_RGB)

        #random sample points
        n = self.num_points
        if point_cloud.shape[0] > n:
            index = np.random.choice(point_cloud.shape[0], n, replace=False)
            point_cloud = point_cloud[index]
            votes = votes[index]
            vote_mask = vote_mask[index]

        #if self.augment:
            #randomly scale by +/-15%
            #scale_ratio = np.random.random()*0.3+0.85
            #scale_ratio = np.expand_dims(np.tile(scale_ratio,3),0)
            #point_cloud[:,0:3] *= scale_ratio
            #box3d_centers *= scale_ratio
            #box3d_sizes *= scale_ratio
            #votes *= scale_ratio

            #random dropout, drop self.dropout_rate
            

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
        angle_class_and_residuals = [DC.angle2class(x) for x in euler_angles]

        #angle = np.linalg.norm(axis_angles)
        #axis_angles /= angle
        #axis_angles = np.asarray([axis_angles])
        #angle_class_and_residuals = [DC.angle2class(angle)]

        size_class, size_residual = DC.size2class(box3d_sizes[0], DC.class2type[0])

        #print(angle_class_and_residuals)

        angle_classes = np.asarray([x[0] for x in angle_class_and_residuals])
        angle_residuals = np.asarray([x[1] for x in angle_class_and_residuals])

        angle_classes1 = np.asarray([angle_classes[0]])
        angle_residuals1 = np.asarray([angle_residuals[0]])

        angle_classes2 = np.asarray([angle_classes[1]])
        angle_residuals2 = np.asarray([angle_residuals[1]])

        angle_classes3 = np.asarray([angle_classes[2]])
        angle_residuals3 = np.asarray([angle_residuals[2]])

        #print(angle_classes.shape)
        #print(angle_residuals.shape)

        #print(angle_classes1, angle_residuals1)
        #print(angle_classes2, angle_residuals2)
        #print(angle_classes3, angle_residuals3)

        size_classes = np.asarray([size_class])
        size_residuals = np.asarray([size_residual])

        #print(size_classes.shape)
        #print(size_residuals.shape)

        semantic_class_index = np.asarray([0])
        label_mask = np.asarray([1])

        vote_labels = np.asarray([[a[0], a[1], a[2], a[0], a[1], a[2], a[0], a[1], a[2]] for a in votes])

        ret_dict = {}
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ret_dict['center_label'] = box3d_centers.astype(np.float32)
        ret_dict['heading_class_label'] = angle_classes1.astype(np.int64)
        ret_dict['heading_residual_label'] = angle_residuals1.astype(np.float32)
        ret_dict['heading_class_label2'] = angle_classes2.astype(np.int64)
        ret_dict['heading_residual_label2'] = angle_residuals2.astype(np.float32)
        ret_dict['heading_class_label3'] = angle_classes3.astype(np.int64)
        ret_dict['heading_residual_label3'] = angle_residuals3.astype(np.float32)
        #ret_dict['rotation_vector_label'] = axis_angles.astype(np.float32)
        ret_dict['size_class_label'] = size_classes.astype(np.int64)
        ret_dict['size_residual_label'] = size_residuals.astype(np.float32)
        ret_dict['sem_cls_label'] = semantic_class_index.astype(np.int64)
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

def viz_obb(pc, label, mask, angle_classes_and_residuals,
    size_classes, size_residuals):
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
        #obb = np.zeros(7)
        obb = np.zeros(9)
        obb[0:3] = label[i,0:3]
        heading_angles = [DC.class2angle(angle_classes_and_residuals[k][0][i], angle_classes_and_residuals[k][1][i]) for k in range(3)]
        box_size = DC.class2size(size_classes[i], size_residuals[i])
        obb[3:6] = box_size
        obb[6:9] = heading_angles
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
    d = OBJDetectionVotesDataset(sys.argv[1], num_points=50000, use_color=True, augment=True)
    sample = d[3]
    print(sample['vote_label'].shape, sample['vote_label_mask'].shape)
    pc_util.write_ply(sample['point_clouds'], 'pc.ply')
    viz_votes(sample['point_clouds'], sample['vote_label'], sample['vote_label_mask'])
    viz_obb(sample['point_clouds'], sample['center_label'], sample['box_label_mask'],
        [[sample['heading_class_label'], sample['heading_residual_label']], [sample['heading_class_label2'], sample['heading_residual_label2']], 
        [sample['heading_class_label3'], sample['heading_residual_label3']]], 
        sample['size_class_label'], sample['size_residual_label'])
