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


net = 0
CONFIG_DICT = {}
DATASET_CONFIG = {}
MODEL = 0


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

    num_point = 8000
    use_color = True
    num_target = 256
    vote_factor = 1
    cluster_sampling = 'vote_fps'
    use_3d_nms = True
    use_cls_nms = False
    use_old_type_nms = False
    per_class_proposal = False
    nms_iou = 0.25
    conf_thresh = 0.05
    faster_eval = True


# ------------------------------------------------------------------------- GLOBAL CONFIG BEG

    sys.path.append(os.path.join(ROOT_DIR, 'object_tracking'))
    from obj_dataset import MAX_NUM_OBJ
    from model_util_obj import OBJDatasetConfig
    DATASET_CONFIG = OBJDatasetConfig()

    # Init the model and optimzier
    MODEL = importlib.import_module('votenet') # import network module
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_input_channel = int(use_color)*3

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

    max_conf = -1
    max_pred = 0
    for pred in batch_pred_map_cls[0]:
        if pred[2] > max_conf:
            max_conf = pred[2]
            max_pred = pred

        print('max conf ', max_conf)

    try:
        MODEL.dump_results(end_points, 'test_dump', DATASET_CONFIG, idx_beg = i)
    except:
        pass

    i += 10

    return batch_pred_map_cls

if __name__=='__main__':

    run_network(np.array([]))
