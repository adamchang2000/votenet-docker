# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Evaluation routine for 3D object detection with SUN RGB-D and ScanNet.
"""

import os
import sys
import numpy as np
from datetime import datetime
import argparse
import importlib

import open3d as o3d
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from ap_helper import parse_predictions
from ap_helper import softmax


# net = 0
# CONFIG_DICT = {}
# DATASET_CONFIG = {}
# MODEL = 0


def setup(checkpoint_path):
    global net
    global CONFIG_DICT
    global DATASET_CONFIG
    global MODEL

    #parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
    #parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
    #parser.add_argument('--num_target', type=int, default=256, help='Point Number [default: 256]')
    #parser.add_argument('--vote_factor', type=int, default=1, help='Number of votes generated from each seed [default: 1]')
    #parser.add_argument('--cluster_sampling', default='vote_fps', help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')
    #parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')
    #parser.add_argument('--use_3d_nms', action='store_true', help='Use 3D NMS instead of 2D NMS.')
    #parser.add_argument('--use_cls_nms', action='store_true', help='Use per class NMS.')
    #parser.add_argument('--use_old_type_nms', action='store_true', help='Use old type of NMS, IoBox2Area.')
    #parser.add_argument('--per_class_proposal', action='store_true', help='Duplicate each proposal num_class times.')
    #parser.add_argument('--nms_iou', type=float, default=0.25, help='NMS IoU threshold. [default: 0.25]')
    #parser.add_argument('--conf_thresh', type=float, default=0.05, help='Filter out predictions with obj prob less than it. [default: 0.05]')
    #parser.add_argument('--faster_eval', action='store_true', help='Faster evaluation by skippling empty bounding box removal.')

    num_point = 25000
    use_color = False
    channels = 1
    num_target = 256
    vote_factor = 1
    cluster_sampling = 'vote_fps'
    use_3d_nms = True
    use_cls_nms = False
    use_old_type_nms = False
    per_class_proposal = False
    nms_iou = 0.25
    conf_thresh = 0.05
    faster_eval = False


# ------------------------------------------------------------------------- GLOBAL CONFIG BEG

    sys.path.append(os.path.join(ROOT_DIR, 'object_tracking'))
    from obj_dataset import MAX_NUM_OBJ
    from model_util_obj import OBJDatasetConfig
    DATASET_CONFIG = OBJDatasetConfig()

    # Init the model and optimzier
    MODEL = importlib.import_module('votenet') # import network module
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_input_channel = int(use_color)*3 + int(channels)*1

    Detector = MODEL.VoteNet

    net = Detector(num_class=DATASET_CONFIG.num_class,
                   num_heading_bin=DATASET_CONFIG.num_heading_bin,
                   num_size_cluster=DATASET_CONFIG.num_size_cluster,
                   mean_size_arr=DATASET_CONFIG.mean_size_arr,
                   num_proposal=num_target,
                   input_feature_dim=num_input_channel,
                   vote_factor= vote_factor,
                   sampling=cluster_sampling)
    net.to(device)
    criterion = MODEL.get_loss

    # Load checkpoint if there is any
    if checkpoint_path is not None and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        print("Loaded checkpoint %s (epoch: %d)"%(checkpoint_path, epoch))

    # Used for AP calculation
    CONFIG_DICT = {'remove_empty_box': (not faster_eval), 'use_3d_nms': use_3d_nms, 'nms_iou': nms_iou,
        'use_old_type_nms': use_old_type_nms, 'cls_nms': use_cls_nms, 'per_class_proposal': per_class_proposal,
        'conf_thresh': conf_thresh, 'dataset_config':DATASET_CONFIG}
    # ------------------------------------------------------------------------- GLOBAL CONFIG END

def extract_best_pred(end_points, config):

    OUTPUT_THRESH = 0.8


    # NETWORK OUTPUTS
    seed_xyz = end_points['seed_xyz'].detach().cpu().numpy() # (B,num_seed,3)
    if 'vote_xyz' in end_points:
        aggregated_vote_xyz = end_points['aggregated_vote_xyz'].detach().cpu().numpy()
        vote_xyz = end_points['vote_xyz'].detach().cpu().numpy() # (B,num_seed,3)
        aggregated_vote_xyz = end_points['aggregated_vote_xyz'].detach().cpu().numpy()
    objectness_scores = end_points['objectness_scores'].detach().cpu().numpy() # (B,K,2)
    pred_center = end_points['center'].detach().cpu().numpy() # (B,K,3)
    pred_heading_class = torch.argmax(end_points['heading_scores'], -1) # B,num_proposal
    pred_heading_residual = torch.gather(end_points['heading_residuals'], 2, pred_heading_class.unsqueeze(-1)) # B,num_proposal,1
    pred_heading_class = pred_heading_class.detach().cpu().numpy() # B,num_proposal
    pred_heading_residual = pred_heading_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal

    pred_heading_class2 = torch.argmax(end_points['heading_scores2'], -1) # B,num_proposal
    pred_heading_residual2 = torch.gather(end_points['heading_residuals2'], 2, pred_heading_class2.unsqueeze(-1)) # B,num_proposal,1
    pred_heading_class2 = pred_heading_class2.detach().cpu().numpy() # B,num_proposal
    pred_heading_residual2 = pred_heading_residual2.squeeze(2).detach().cpu().numpy() # B,num_proposal

    pred_heading_class3 = torch.argmax(end_points['heading_scores3'], -1) # B,num_proposal
    pred_heading_residual3 = torch.gather(end_points['heading_residuals3'], 2, pred_heading_class3.unsqueeze(-1)) # B,num_proposal,1
    pred_heading_class3 = pred_heading_class3.detach().cpu().numpy() # B,num_proposal
    pred_heading_residual3 = pred_heading_residual3.squeeze(2).detach().cpu().numpy() # B,num_proposal

    # OTHERS
    pred_mask = end_points['pred_mask'] # B,num_proposal
    objectness_prob = softmax(objectness_scores[0,:,:])[:,1] # (K,)

    if np.sum(objectness_prob>OUTPUT_THRESH)>0:
        num_proposal = pred_center.shape[1]
        obbs = []
        for j in range(num_proposal):
            obb = (pred_center[0,j,0:3], [config.class2angle(pred_heading_class[0,j], pred_heading_residual[0,j]), config.class2angle(pred_heading_class2[0,j], pred_heading_residual2[0,j]), 
                config.class2angle(pred_heading_class3[0,j], pred_heading_residual3[0,j])],
                            0)
            obbs.append(obb)
        if len(obbs)>0:
            obbs = np.vstack(tuple(obbs)) # (num_proposal, 7)
            return obbs[np.logical_and(objectness_prob==np.max(objectness_prob), pred_mask[0,:]==1),:]
        else:
            print('warning, no obbs')

    return False

i = 0

#run the network on a pointcloud
#pointcloud needs to be of format n x (XYZ RGB)
def run_network(pointcloud):
    global i
    stat_dict = {}
    net.eval() # set model to eval mode (for bn and dp)
        

    pointcloud_tensor = torch.from_numpy(np.array([pointcloud])).cuda()
    # Forward pass
    inputs = {'point_clouds': pointcloud_tensor}
    with torch.no_grad():
        end_points = net(inputs)

    print('call time! ', datetime.now())

    end_points['point_clouds'] = pointcloud_tensor

    #this should return, [(class index, obb params, box confidence)]
    batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT) 
    #print(batch_pred_map_cls)

    max_conf = -1
    max_pred = 0
    for pred in batch_pred_map_cls[0]:
        if pred[2] > max_conf:
            max_conf = pred[2]
            max_pred = pred

        print('max conf ', max_conf)

    # try:
    #     MODEL.dump_results(end_points, 'test_dump', DATASET_CONFIG, idx_beg = i)
    # except:
    #     pass

    i += 10

    best_bbox = extract_best_pred(end_points, DATASET_CONFIG)
    return max_conf, best_bbox

if __name__=='__main__':

    run_network(np.array([]))
