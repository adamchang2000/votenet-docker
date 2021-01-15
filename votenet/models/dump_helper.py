# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util

DUMP_CONF_THRESH = 0.1 # Dump boxes with obj prob larger than that.
GT_VOTE_FACTOR = 3 # number of GT votes per point

def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs

def dump_results(end_points, dump_dir, config, inference_switch=False, idx_beg = 0):
    ''' Dump results.

    Args:
        end_points: dict
            {..., pred_mask}
            pred_mask is a binary mask array of size (batch_size, num_proposal) computed by running NMS and empty box removal
    Returns:
        None
    '''

    print('dump results called with idx_beg %d' % idx_beg)

    if not os.path.exists(dump_dir):
        os.system('mkdir %s'%(dump_dir))

    # INPUT
    point_clouds = end_points['point_clouds'].cpu().numpy()
    batch_size = point_clouds.shape[0]

    # NETWORK OUTPUTS
    seed_xyz = end_points['seed_xyz'].detach().cpu().numpy() # (B,num_seed,3)
    if 'vote_xyz' in end_points:
        aggregated_vote_xyz = end_points['aggregated_vote_xyz'].detach().cpu().numpy()
        vote_xyz = end_points['vote_xyz'].detach().cpu().numpy() # (B,num_seed,3)
        aggregated_vote_xyz = end_points['aggregated_vote_xyz'].detach().cpu().numpy()
    objectness_scores = end_points['objectness_scores'].detach().cpu().numpy() # (B,K,2)
    pred_center = end_points['center'].detach().cpu().numpy() # (B,K,3)
    # pred_heading_class = torch.argmax(end_points['heading_scores'], -1) # B,num_proposal
    # pred_heading_residual = torch.gather(end_points['heading_residuals'], 2, pred_heading_class.unsqueeze(-1)) # B,num_proposal,1
    # pred_heading_class = pred_heading_class.detach().cpu().numpy() # B,num_proposal
    # pred_heading_residual = pred_heading_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal

    # pred_heading_class2 = torch.argmax(end_points['heading_scores2'], -1) # B,num_proposal
    # pred_heading_residual2 = torch.gather(end_points['heading_residuals2'], 2, pred_heading_class2.unsqueeze(-1)) # B,num_proposal,1
    # pred_heading_class2 = pred_heading_class2.detach().cpu().numpy() # B,num_proposal
    # pred_heading_residual2 = pred_heading_residual2.squeeze(2).detach().cpu().numpy() # B,num_proposal

    # pred_heading_class3 = torch.argmax(end_points['heading_scores3'], -1) # B,num_proposal
    # pred_heading_residual3 = torch.gather(end_points['heading_residuals3'], 2, pred_heading_class3.unsqueeze(-1)) # B,num_proposal,1
    # pred_heading_class3 = pred_heading_class3.detach().cpu().numpy() # B,num_proposal
    # pred_heading_residual3 = pred_heading_residual3.squeeze(2).detach().cpu().numpy() # B,num_proposal

    pred_rotation_vector = end_points['rotation_vector'].detach().cpu().numpy()
    pred_theta = end_points['theta'].detach().cpu().numpy()

    #pred_size_class = torch.argmax(end_points['size_scores'], -1) # B,num_proposal
    #pred_size_residual = torch.gather(end_points['size_residuals'], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # B,num_proposal,1,3
    #pred_size_residual = pred_size_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal,3

    # OTHERS
    pred_mask = end_points['pred_mask'] # B,num_proposal

    for i in range(batch_size):
        pc = point_clouds[i,:,:]
        objectness_prob = softmax(objectness_scores[i,:,:])[:,1] # (K,)

        # Dump various point clouds
        pc_util.write_ply(pc, os.path.join(dump_dir, '%06d_pc.ply'%(idx_beg+i)))
        pc_util.write_ply(seed_xyz[i,:,:], os.path.join(dump_dir, '%06d_seed_pc.ply'%(idx_beg+i)))
        if 'vote_xyz' in end_points:
            pc_util.write_ply(end_points['vote_xyz'][i,:,:], os.path.join(dump_dir, '%06d_vgen_pc.ply'%(idx_beg+i)))
            pc_util.write_ply(aggregated_vote_xyz[i,:,:], os.path.join(dump_dir, '%06d_aggregated_vote_pc.ply'%(idx_beg+i)))
            pc_util.write_ply(aggregated_vote_xyz[i,:,:], os.path.join(dump_dir, '%06d_aggregated_vote_pc.ply'%(idx_beg+i)))
        pc_util.write_ply(pred_center[i,:,0:3], os.path.join(dump_dir, '%06d_proposal_pc.ply'%(idx_beg+i)))
        if np.sum(objectness_prob>DUMP_CONF_THRESH)>0:
            pc_util.write_ply(pred_center[i,objectness_prob>DUMP_CONF_THRESH,0:3], os.path.join(dump_dir, '%06d_confident_proposal_pc.ply'%(idx_beg+i)))

        # Dump predicted bounding boxes
        if np.sum(objectness_prob>DUMP_CONF_THRESH)>0:
            num_proposal = pred_center.shape[1]
            obbs = []
            for j in range(num_proposal):
                obb = config.param2obb(pred_center[i,j,0:3], pred_rotation_vector[i,j], pred_theta[i,j], 0)
                obbs.append(obb)
            if len(obbs)>0:
                obbs = np.vstack(tuple(obbs)) # (num_proposal, 7)
                try:
                    pc_util.write_oriented_bbox_6dof(obbs[objectness_prob>DUMP_CONF_THRESH,:], os.path.join(dump_dir, '%06d_pred_confident_bbox.ply'%(idx_beg+i)))
                    pc_util.write_oriented_bbox_6dof(obbs[np.logical_and(objectness_prob>DUMP_CONF_THRESH, pred_mask[i,:]==1),:], os.path.join(dump_dir, '%06d_pred_confident_nms_bbox.ply'%(idx_beg+i)))
                    pc_util.write_oriented_bbox_6dof(obbs[np.logical_and(objectness_prob==np.max(objectness_prob), pred_mask[i,:]==1),:], os.path.join(dump_dir, '%06d_pred_SUPER_%f_confident_nms_bbox.ply'%(idx_beg+i, round(np.max(objectness_prob), 4))))
                except:
                    print("write bbox output failed")
                pc_util.write_oriented_bbox_6dof(obbs[pred_mask[i,:]==1,:], os.path.join(dump_dir, '%06d_pred_nms_bbox.ply'%(idx_beg+i)))
                pc_util.write_oriented_bbox_6dof(obbs, os.path.join(dump_dir, '%06d_pred_bbox.ply'%(idx_beg+i)))
            else:
                print('warning, no obbs')

    # Return if it is at inference time. No dumping of groundtruths
    if inference_switch:
        return

    # LABELS

    num_seed = end_points['seed_xyz'].shape[1] # B,num_seed,3
    seed_inds = end_points['seed_inds'].long() # B,num_seed in [0,num_points-1]
    seed_gt_votes_mask = torch.gather(end_points['vote_label_mask'], 1, seed_inds)
    seed_inds_expand = seed_inds.view(batch_size,num_seed,1).repeat(1,1,3*GT_VOTE_FACTOR)
    seed_gt_votes = torch.gather(end_points['vote_label'], 1, seed_inds_expand)
    seed_gt_votes += end_points['seed_xyz'].repeat(1,1,3)
    seed_gt_votes = seed_gt_votes.cpu().numpy()
    seed_gt_votes_mask = seed_gt_votes_mask.cpu().numpy()
    seed_gt_votes = seed_gt_votes[:,seed_gt_votes_mask[0] == 1]

    gt_center = end_points['center_label'].cpu().numpy() # (B,MAX_NUM_OBJ,3)
    gt_mask = end_points['box_label_mask'].cpu().numpy() # B,K2
    #gt_heading_class = end_points['heading_class_label'].cpu().numpy() # B,K2
    #gt_heading_residual = end_points['heading_residual_label'].cpu().numpy() # B,K2

    gt_rotation_vector = end_points['rotation_vector_label'].cpu().numpy()
    gt_theta = end_points['theta_label'].cpu().numpy()

    #gt_size_class = end_points['size_class_label'].cpu().numpy() # B,K2
    #gt_size_residual = end_points['size_residual_label'].cpu().numpy() # B,K2,3
    objectness_label = end_points['objectness_label'].detach().cpu().numpy() # (B,K,)
    objectness_mask = end_points['objectness_mask'].detach().cpu().numpy() # (B,K,)

    for i in range(batch_size):
        pc_util.write_ply(seed_gt_votes[i,:,0:3], os.path.join(dump_dir, '%06d_gt_votes_pc.ply'%(idx_beg+i)))
        if np.sum(objectness_label[i,:])>0:
            pc_util.write_ply(pred_center[i,objectness_label[i,:]>0,0:3], os.path.join(dump_dir, '%06d_gt_positive_proposal_pc.ply'%(idx_beg+i)))
        if np.sum(objectness_mask[i,:])>0:
            pc_util.write_ply(pred_center[i,objectness_mask[i,:]>0,0:3], os.path.join(dump_dir, '%06d_gt_mask_proposal_pc.ply'%(idx_beg+i)))
        pc_util.write_ply(gt_center[i,:,0:3], os.path.join(dump_dir, '%06d_gt_centroid_pc.ply'%(idx_beg+i)))
        pc_util.write_ply_color(pred_center[i,:,0:3], objectness_label[i,:], os.path.join(dump_dir, '%06d_proposal_pc_objectness_label.obj'%(idx_beg+i)))

        # Dump GT bounding boxes
        obbs = []
        for j in range(gt_center.shape[1]):
            if gt_mask[i,j] == 0: continue
            obb = config.param2obb(gt_center[i,j,0:3], gt_rotation_vector[i,j], gt_theta[i,j], 0)
            obbs.append(obb)
        if len(obbs)>0:
            obbs = np.vstack(tuple(obbs)) # (num_gt_objects, 7)
            pc_util.write_oriented_bbox_6dof(obbs, os.path.join(dump_dir, '%06d_gt_bbox.ply'%(idx_beg+i)))

    # OPTIONALL, also dump prediction and gt details
    if 'batch_pred_map_cls' in end_points:
        for ii in range(batch_size):
            fout = open(os.path.join(dump_dir, '%06d_pred_map_cls.txt'%(ii)), 'w')
            for t in end_points['batch_pred_map_cls'][ii]:
                fout.write(str(t[0])+' ')
                fout.write(",".join([str(x) for x in list(t[1].flatten())]))
                fout.write(' '+str(t[2]))
                fout.write('\n')
            fout.close()
    if 'batch_gt_map_cls' in end_points:
        for ii in range(batch_size):
            fout = open(os.path.join(dump_dir, '%06d_gt_map_cls.txt'%(ii)), 'w')
            for t in end_points['batch_gt_map_cls'][ii]:
                fout.write(str(t[0])+' ')
                fout.write(",".join([str(x) for x in list(t[1].flatten())]))
                fout.write('\n')
            fout.close()
