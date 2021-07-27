import os
import sys
sys.path.append(os.path.dirname(__file__))

import random
import argparse
import numpy as np
import json
import time
from datetime import datetime
from lapsolver import solve_dense

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.data import Dataset, DataLoader
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

from tqdm import tqdm

from nuscenes.utils.splits import create_splits_scenes

from utils.nuScenes_dataset import nuScenes_detection_dataset, collate
from utils.sequential_bp import sequential_bp
from utils.gnn import DA_GNN
from utils.save_estimations_by_scene import save_estimations_by_scene

from CenterPoint.tools.create_model import create_model

TRACKING_NAMES = ['bicycle', 'bus', 'car', 'motorcycle', 'pedestrian', 'trailer', 'truck']

def get_ground_truth_affinity(estimations_existing,
                            targets_id_existing,
                            ground_truths_id_existing,
                            estimations_new,
                            targets_id_new,
                            ground_truths):

    batch_size, num_max_gt, dim_gt = ground_truths.shape
    _, num_max_est, _ = estimations_existing.shape
    _, num_max_est_new, _ = estimations_new.shape

    affinity_gt = torch.zeros(batch_size, num_max_est, num_max_est_new, device = estimations_existing.device)

    for batch in range(batch_size):

        dist_threshold = torch.max(ground_truths[batch, :, 3 : 5], dim = -1)[0]
        dist_threshold[dist_threshold > 2] = 2

        dist_matrix = torch.sum((estimations_new[batch, :, : 2].unsqueeze(1) - ground_truths[batch, :, : 2].unsqueeze(0) ) ** 2, dim = -1) ** 0.5
        dist_matrix[dist_matrix - dist_threshold[None, :] > 0] = float('nan')
        # dist_matrix[dist_matrix > 2] = float('nan')
        row_inds, col_inds = solve_dense(dist_matrix.detach().cpu().numpy())
        ground_truths_id_meas = -torch.ones(num_max_est_new, dtype = torch.long, device = estimations_existing.device)
        ground_truths_id_meas[row_inds] = torch.tensor(col_inds, dtype = torch.long, device = estimations_existing.device)

        # dist_matrix = torch.sum((estimations_existing[batch, :, : 2].unsqueeze(1) - ground_truths[batch, :, : 2].unsqueeze(0) ) ** 2, dim = -1) ** 0.5
        # dist_matrix[:, ground_truths_id_existing[batch, ground_truths_id_existing[batch, :] != -1]] = float('nan')
        # dist_matrix[dist_matrix - dist_threshold[None, :] > 0] = float('nan')
        # row_inds, col_inds = solve_dense(dist_matrix.detach().cpu().numpy())
        # ground_truths_id_legacy = -torch.ones(num_max_est, dtype = torch.long, device = estimations_existing.device)
        ground_truths_id_legacy = ground_truths_id_existing[batch, :]
        # ground_truths_id_legacy[row_inds] = torch.tensor(col_inds, dtype = torch.long, device = estimations_existing.device)

        affinity_gt[batch, :, :] = torch.logical_and(ground_truths_id_legacy.unsqueeze(1) == ground_truths_id_meas.unsqueeze(0),
                                                     ground_truths_id_legacy.unsqueeze(1) != -1).float()

        # affinity_gt[batch, :, :] = torch.logical_and(ground_truths_id_existing[batch, :].unsqueeze(1) == ground_truths_id_meas.unsqueeze(0),
        #                                              ground_truths_id_existing[batch, :].unsqueeze(1) != -1).float()

        # dist_matrix2 = torch.sum((estimations_existing[batch, :, : 2].unsqueeze(1) - estimations_new[batch, :, : 2].unsqueeze(0) ) ** 2, dim = -1) ** 0.5
        # dist_matrix2[torch.any(affinity_gt[batch, :, :].bool(), dim = -1), :] = float('nan')
        # dist_matrix2[:, torch.any(affinity_gt[batch, :, :].bool(), dim = -2)] = float('nan')
        # dist_matrix2[dist_matrix2 > 2] = float('nan')
        # row_inds, col_inds = solve_dense(dist_matrix2.detach().cpu().numpy())
        # affinity_gt[batch, row_inds, col_inds] = 1.

        # dist_matrix = torch.sum((estimations_existing[batch, :, : 2].unsqueeze(1) - estimations_new[batch, :, : 2].unsqueeze(0) ) ** 2, dim = -1) ** 0.5
        # dist_matrix[dist_matrix > 2] = float('nan')

        # row_inds, col_inds = solve_dense(dist_matrix.detach().cpu().numpy())

        # affinity_gt[batch, row_inds, col_inds] = 1.

    affinity_gt_miss = ~torch.any(affinity_gt.bool(), dim = -1, keepdim = True)
    affinity_gt = torch.cat([affinity_gt_miss, affinity_gt], dim = -1)

    for batch in range(batch_size):
        dist_matrix = torch.sum((estimations_existing[batch, :, : 2].unsqueeze(1) - estimations_new[batch, :, : 2].unsqueeze(0) ) ** 2, dim = -1) ** 0.5
        dist_matrix[:, torch.any(affinity_gt[batch, :, 1:].bool(), dim = -2)] = float('nan')
        dist_matrix[dist_matrix > 2] = float('nan')
        false_targets_inds = torch.logical_and(affinity_gt[batch, :, 0] == 1, ~torch.all(torch.isnan(dist_matrix), dim = -1))
        affinity_gt[batch, false_targets_inds, 0] = 0.

    return affinity_gt

def assign_ground_truth_id(estimations_existing,
                           targets_id_existing,
                           ground_truths_id_existing,
                           estimations_new,
                           targets_id_new,
                           ground_truths):

    batch_size, num_max_gt, dim_gt = ground_truths.shape
    _, num_max_est, _ = estimations_existing.shape
    _, num_max_est_new, _ = estimations_new.shape

    ground_truths_id_existing_update = ground_truths_id_existing.clone()
    ground_truths_id_new = -torch.ones(batch_size, num_max_est_new, dtype = torch.long, device = estimations_new.device)
    ground_truths_label_existing = torch.zeros(batch_size, num_max_est, device = estimations_existing.device)
    ground_truths_label_new = torch.zeros(batch_size, num_max_est_new, device = estimations_new.device)
    ground_truths_existing_mask = torch.zeros(batch_size, num_max_gt, dtype = torch.bool)

    ground_truths_id_mask = ground_truths_id_existing != -1

    for batch in range(batch_size):
        ground_truths_id_existing_batch = ground_truths_id_existing[batch, :]
        ground_truths_id_mask_batch = ground_truths_id_mask[batch, :]
        ground_truths_batch = ground_truths[batch, :, :]
        ground_truths_existing_mask_batch = ground_truths_existing_mask[batch, :]

        dist_threshold = torch.max(ground_truths_batch[:, 3 : 5], dim = -1)[0]
        dist_threshold[dist_threshold > 2] = 2

        # ground_truths_label_existing[batch, ground_truths_id_mask_batch] = \
        #     (~torch.any(torch.isnan(ground_truths_batch[ground_truths_id_existing_batch[ground_truths_id_mask_batch], :]), dim = -1)).float()
        # ground_truths_existing_mask_batch[ground_truths_id_existing_batch[ground_truths_id_mask_batch]] = True

        ground_truths_label_existing[batch, ground_truths_id_mask_batch] = \
            (torch.logical_and(~torch.any(torch.isnan(ground_truths_batch[ground_truths_id_existing_batch[ground_truths_id_mask_batch], :]), dim = -1),
                               torch.sum((estimations_existing[batch, ground_truths_id_mask_batch, : 2] -
                                          ground_truths_batch[ground_truths_id_existing_batch[ground_truths_id_mask_batch], : 2]) ** 2, dim = -1) ** 0.5 -
                               dist_threshold[None, ground_truths_id_existing_batch[ground_truths_id_mask_batch]] < 0 )).float()
        # dist_threshold[None, ground_truths_id_existing_batch[ground_truths_id_mask_batch]]
        ground_truths_existing_mask_batch[ground_truths_id_existing_batch[ground_truths_label_existing[batch, :].bool()]] = True

        estimations_existing_batch = estimations_existing[batch, :, :]
        estimations_new_batch = estimations_new[batch, :, :]
        estimations_all_batch = torch.cat([estimations_existing_batch[:, : 2], estimations_new_batch[:, : 2]], dim = -2)
        targets_id_all = torch.cat([targets_id_existing, targets_id_new], dim = -1)
        # dist_matrix = torch.sum((estimations_new_batch[:, None, : 2] - ground_truths_batch[None, :, : 2]) ** 2, dim = -1) ** 0.5
        # dist_matrix[:, ground_truths_existing_mask_batch] = float('nan')
        # dist_matrix[targets_id_new[batch, :] == -1, :] = float('nan')
        # dist_matrix[dist_matrix > 2] = float('nan')

        dist_matrix = torch.sum((estimations_all_batch[:, None, : 2] - ground_truths_batch[None, :, : 2]) ** 2, dim = -1) ** 0.5
        dist_matrix[:, ground_truths_existing_mask_batch] = float('nan')
        dist_matrix[targets_id_all[batch, :] == -1, :] = float('nan')
        dist_matrix[torch.cat([ground_truths_label_existing[batch, :].bool(),
                               torch.zeros(num_max_est_new, dtype = torch.bool, device = estimations_new.device)], dim = 0), :] = float('nan')
        # dist_matrix[torch.nonzero(ground_truths_label_existing[batch, :]), :] = float('nan')
        dist_matrix[dist_matrix - dist_threshold[None, :] > 0] = float('nan')
        # dist_matrix[dist_matrix > 2] = float('nan')

        row_inds, col_inds = solve_dense(dist_matrix.detach().cpu().numpy())

        row_inds_existing = row_inds[row_inds < num_max_est]
        col_inds_existing = col_inds[row_inds < num_max_est]
        row_inds_new = row_inds[row_inds >= num_max_est] - num_max_est
        col_inds_new = col_inds[row_inds >= num_max_est]

        ground_truths_label_existing[batch, row_inds_existing] = 1
        ground_truths_id_existing_update[batch, ~(ground_truths_label_existing[batch, :].bool())] = -1
        ground_truths_id_existing_update[batch, row_inds_existing] = torch.tensor(col_inds_existing, dtype = torch.long, device = estimations_existing.device)
        ground_truths_id_new[batch, row_inds_new] = torch.tensor(col_inds_new, dtype = torch.long, device = estimations_new.device)

        # for new detections
        dist_matrix = torch.sum((estimations_new_batch[:, None, : 2] - ground_truths_batch[None, :, : 2]) ** 2, dim = -1) ** 0.5
        # ground_truths_label_new[batch, :] = torch.any(dist_matrix < 2, dim = -1)
        ground_truths_label_new[batch, :] = torch.any(dist_matrix - dist_threshold[None, :] < 0, dim = -1)
        # dist_matrix[dist_matrix > 2] = float('nan')
        # row_inds, col_inds = solve_dense(dist_matrix.detach().cpu().numpy())
        # ground_truths_id_new[batch, row_inds] = torch.tensor(col_inds, dtype = torch.long, device = estimations_new.device)

        # ground_truths_label_new[batch, :] = (ground_truths_id_new[batch, :] != -1).float()


    # print('ground_truths_id_new', ground_truths_id_new[1, :])
    # print('ground_truths_id_existing', ground_truths_id_existing[1, :])
    # print('estimations_new', estimations_new[1, ground_truths_id_new[1, :] != -1, :2])
    # print('estimations_existing', estimations_existing[1, ground_truths_id_existing[1, :] != -1, :2])
    # print('ground_truths_new', ground_truths[1, ground_truths_id_new[1, ground_truths_id_new[1, :] != -1], :2])
    # print('ground_truths_existing', ground_truths[1, ground_truths_id_existing[1, ground_truths_id_existing[1, :] != -1], :2])

    return ground_truths_label_existing, ground_truths_label_new, ground_truths_id_existing_update, ground_truths_id_new

def merge_and_prune(particles_kinematic,
                    prob_existence,
                    tracking_score,
                    targets_id,
                    ground_truths_id,
                    particles_kinematic_new,
                    prob_existence_new,
                    tracking_score_new,
                    targets_id_new,
                    ground_truths_id_new,
                    prob_existence_new_deep,
                    parameters):

    batch_size, num_max_targets, _, dim_state = particles_kinematic_prior.shape

    pruning_threshold = 1 / (1 + parameters['mean_clutter'] / parameters['mean_newborn']) * 0.01
    targets_id[prob_existence < pruning_threshold] = -1
    ground_truths_id[prob_existence < pruning_threshold] = -1

    pruning_threshold_new = 1 / (1 + parameters['mean_clutter'] / parameters['mean_newborn']) * 0.8
    mask_tmp = torch.logical_or(prob_existence_new < pruning_threshold_new, prob_existence_new_deep < 0.5)
    targets_id_new[mask_tmp] = -1
    ground_truths_id_new[mask_tmp] = -1
    # targets_id_new[prob_existence_new < pruning_threshold_new] = -1
    # ground_truths_id_new[prob_existence_new < pruning_threshold_new] = -1

    # -----------------------------
    # merge new and existing targets
    # -----------------------------
    particles_kinematic_all = torch.cat([particles_kinematic, particles_kinematic_new], dim = 1)
    prob_existence_all = torch.cat([prob_existence, prob_existence_new], dim = 1)
    targets_id_all = torch.cat([targets_id, targets_id_new], dim = 1)
    tracking_score_all = torch.cat([tracking_score, tracking_score_new], dim = 1)
    ground_truths_id_all = torch.cat([ground_truths_id, ground_truths_id_new], dim = 1)

    targets_mask_all = targets_id_all != -1

    # print('particles_kinematic_all.shape', particles_kinematic_all.shape)
    # print('prob_existence_all.shape', prob_existence_all.shape)
    # print('targets_id_all.shape', targets_id_all.shape)

    _, indices = torch.sort(targets_mask_all.float(), dim = 1, descending = True)


    particles_kinematic_all = particles_kinematic_all.scatter(dim = 1, index = indices[:, :, None, None].expand(-1, -1, parameters['num_particles'], dim_state), src = particles_kinematic_all)
    prob_existence_all = prob_existence_all.scatter(dim = 1, index = indices, src = prob_existence_all)
    targets_id_all = targets_id_all.scatter(dim = 1, index = indices, src = targets_id_all)
    tracking_score_all = tracking_score_all.scatter(dim = 1, index = indices, src = tracking_score_all)
    targets_mask_all = targets_mask_all.scatter(dim = 1, index = indices, src = targets_mask_all)
    ground_truths_id_all = ground_truths_id_all.scatter(dim = 1, index = indices, src = ground_truths_id_all)

    particles_kinematic_all = particles_kinematic_all[:, torch.any(targets_mask_all, dim = 0), :, :]
    prob_existence_all = prob_existence_all[:, torch.any(targets_mask_all, dim = 0)]
    targets_id_all = targets_id_all[:, torch.any(targets_mask_all, dim = 0)]
    tracking_score_all = tracking_score_all[:, torch.any(targets_mask_all, dim = 0)]
    ground_truths_id_all = ground_truths_id_all[:, torch.any(targets_mask_all, dim = 0)]

    # print('particles_kinematic_all.shape', particles_kinematic_all.shape)
    # print('prob_existence_all.shape', prob_existence_all.shape)
    # print('targets_id_all.shape', targets_id_all.shape)
    # print('\n')

    return particles_kinematic_all, prob_existence_all, tracking_score_all, targets_id_all, ground_truths_id_all


def mot_loss(estimations_existing,
             prob_existence_existing,
             targets_id_existing,
             ground_truths_label_existing,
             ground_truths_id_existing,
             prob_existence_new,
             targets_id_new,
             ground_truths_label_new,
             ground_truths_id_new,
             ground_truths,
             affinity_gt,
             affinity_pred,
             affinity_pred_single,
             parameters):

    batch_size, num_max_targets, dim_state = estimations_existing.shape
    _, num_max_targets_new = prob_existence_new.shape
    mask_existing = targets_id_existing != -1
    mask_new = targets_id_new != -1
    mask_matched_existing = torch.logical_and(mask_existing, ground_truths_label_existing.bool())
    mask_unmatched_existing = torch.logical_and(mask_existing, ~ground_truths_label_existing.bool())
    mask_matched_new = torch.logical_and(mask_new, ground_truths_label_new.bool())
    mask_unmatched_new = torch.logical_and(mask_new, ~ground_truths_label_new.bool())

    new_max_probability = 1 / (1 + parameters['mean_clutter'] / parameters['mean_newborn'])
    prob_threshold = 1 / (1 + parameters['mean_clutter'] / parameters['mean_newborn']) * 0.8

    # loss_matched_dist_existing = torch.zeros(1, requires_grad=True, device = estimations_existing.device)
    # loss_matched_log_existing = torch.zeros(1, requires_grad=True, device = estimations_existing.device)
    # loss_unmatched_log_existing = torch.zeros(1, requires_grad=True, device = estimations_existing.device)
    # loss_matched_log_new = torch.zeros(1, requires_grad=True, device = estimations_existing.device)
    # loss_unmatched_log_new = torch.zeros(1, requires_grad=True, device = estimations_existing.device)
    loss_matched_dist_existing_list = []
    loss_matched_log_existing_list = []
    loss_unmatched_log_existing_list = []
    loss_matched_log_new_list = []
    loss_unmatched_log_new_list = []
    loss_affinity_list = []
    for batch in range(batch_size):
        estimations_existing_batch_matched = estimations_existing[batch, ground_truths_label_existing[batch, :].bool(), :2]
        ground_truths_batch_matched = ground_truths[batch, ground_truths_id_existing[batch, ground_truths_label_existing[batch, :].bool()], :2]

        if torch.sum(mask_matched_existing[batch, :].float()) > 0:
            # loss_matched_dist_existing = loss_matched_dist_existing + torch.mean(torch.sum((estimations_existing_batch_matched - ground_truths_batch_matched) ** 2, dim = -1))
            # loss_matched_log_existing = loss_matched_log_existing + torch.mean(-torch.log(prob_existence_existing[batch, mask_matched_existing[batch, :]] + 1e-6))
            loss_matched_dist_existing_list.append(torch.sum(torch.sum((estimations_existing_batch_matched - ground_truths_batch_matched) ** 2, dim = -1)))
            loss_matched_log_existing_list.append(torch.sum(-torch.log(prob_existence_existing[batch, mask_matched_existing[batch, :]] + 1e-6) ))
        if torch.sum(mask_unmatched_existing[batch, :].float()) > 0:
            # loss_unmatched_log_existing = loss_unmatched_log_existing + torch.mean(-torch.log(1 - prob_existence_existing[batch, mask_unmatched_existing[batch, :]] + 1e-6))
            loss_unmatched_log_existing_list.append(torch.sum(-torch.log(1 - prob_existence_existing[batch, mask_unmatched_existing[batch, :]] + 1e-6)))
            # loss_unmatched_log_existing_list.append(torch.sum(-torch.log(1 - prob_existence_existing[batch, mask_unmatched_existing[batch, :]] + 1e-6) *
            #                                             (prob_existence_existing[batch, mask_unmatched_existing[batch, :]] > prob_threshold).float() ))

        if torch.sum(mask_matched_new[batch, :].float()) > 0:
            # loss_matched_log_new = loss_matched_log_new + torch.mean(-torch.log((prob_existence_new[batch, mask_matched_new[batch, :]] / (new_max_probability + 1e-6)) + 1e-6))
            # loss_matched_log_new_list.append(torch.sum(-torch.log((prob_existence_new[batch, mask_matched_new[batch, :]] / (new_max_probability + 1e-6)) + 1e-6)))
            loss_matched_log_new_list.append(torch.sum(-torch.log(prob_existence_new[batch, mask_matched_new[batch, :]] + 1e-6) *
                                                        (1 - prob_existence_new[batch, mask_matched_new[batch, :]]) ** 2 ))
            # print('matched new loss', -torch.log((prob_existence_new[batch, mask_matched_new[batch, :]] ) + 1e-6))
        if torch.sum(mask_unmatched_new[batch, :].float()) > 0:
            # loss_unmatched_log_new = loss_unmatched_log_new + torch.mean(-torch.log((1 - prob_existence_new[batch, mask_unmatched_new[batch, :]] / (new_max_probability + 1e-6)) + 1e-6))
            # loss_unmatched_log_new_list.append(torch.sum(-torch.log((1 - prob_existence_new[batch, mask_unmatched_new[batch, :]] / (new_max_probability + 1e-6)) + 1e-6)))
            loss_unmatched_log_new_list.append(torch.sum(-torch.log(1 - prob_existence_new[batch, mask_unmatched_new[batch, :]] + 1e-6) *
                                                         prob_existence_new[batch, mask_unmatched_new[batch, :]] ** 2 ))
            # print('unmatched new loss', -torch.log((1 - prob_existence_new[batch, mask_unmatched_new[batch, :]] ) + 1e-6))
            # loss_unmatched_log_new_list.append(torch.sum(-torch.log((1 - prob_existence_new[batch, mask_unmatched_new[batch, :]] / (new_max_probability + 1e-6)) + 1e-6) *
            #                                             (prob_existence_new[batch, mask_unmatched_new[batch, :]] > prob_threshold).float() ))

        if torch.numel(affinity_gt) > 0:
            # loss_affinity_ce = torch.sum((-torch.log(affinity_pred[batch, :, :] + 1e-6) * affinity_gt[batch, :, :]) *
            #                               (targets_id_existing[batch, :] != -1)[:, None].float() )
            loss_affinity_pos_ce = torch.sum((-torch.log(affinity_pred[batch, :, 1:] + 1e-6) * affinity_gt[batch, :, 1:]) )
            loss_affinity_neg_ce = torch.sum((-torch.log(affinity_pred[batch, :, 0] + 1e-6) * affinity_gt[batch, :, 0]) )
            loss_affinity_single_ce = torch.sum((-torch.log(affinity_pred_single[batch, :, :] + 1e-6) * affinity_gt[batch, :, :]) )
            loss_affinity_list.append(loss_affinity_pos_ce / num_max_targets +
                                      loss_affinity_neg_ce / num_max_targets * 0.01 +
                                      loss_affinity_single_ce / (num_max_targets) * 0)
            # (targets_id_existing[batch, :] != -1)[:, None].float()

    # print('targets_id', targets_id_existing)
    # print('ground_truths_id', ground_truths_id_existing)
    # print('positive loss', -torch.log(affinity_pred[batch, :, :] + 1e-6) * affinity_gt[batch, :, :])
    # print('negative loss', -torch.log(1 - affinity_pred[batch, :, :] + 1e-6) * (~(affinity_gt[batch, :, :].bool())).float())
    # print('affinity_gt', affinity_gt)
    # print('affinity_pred', affinity_pred)

    # loss_matched_dist_existing = torch.mean(torch.stack(loss_matched_dist_existing_list)) if len(loss_matched_dist_existing_list) else torch.zeros(1, device = estimations_existing.device)
    # loss_matched_log_existing = torch.mean(torch.stack(loss_matched_log_existing_list)) if len(loss_matched_log_existing_list) else torch.zeros(1, device = estimations_existing.device)
    # loss_unmatched_log_existing = torch.mean(torch.stack(loss_unmatched_log_existing_list)) if len(loss_unmatched_log_existing_list) else torch.zeros(1, device = estimations_existing.device)
    # loss_matched_log_new = torch.mean(torch.stack(loss_matched_log_new_list)) if len(loss_matched_log_new_list) else torch.zeros(1, device = estimations_existing.device)
    # loss_unmatched_log_new = torch.mean(torch.stack(loss_unmatched_log_new_list)) if len(loss_unmatched_log_new_list) else torch.zeros(1, device = estimations_existing.device)

    # return loss_matched_dist_existing / batch_size, \
    #     loss_matched_log_existing / batch_size, \
    #     loss_unmatched_log_existing / batch_size, \
    #     loss_matched_log_new / batch_size, \
    #     loss_unmatched_log_new / batch_size

    loss_matched_dist_existing_ = torch.sum(torch.stack(loss_matched_dist_existing_list)) / batch_size if len(loss_matched_dist_existing_list) else torch.zeros(1, device = estimations_existing.device)
    loss_matched_log_existing_ = torch.sum(torch.stack(loss_matched_log_existing_list)) / batch_size if len(loss_matched_log_existing_list) else torch.zeros(1, device = estimations_existing.device)
    loss_unmatched_log_existing_ = torch.sum(torch.stack(loss_unmatched_log_existing_list)) / batch_size if len(loss_unmatched_log_existing_list) else torch.zeros(1, device = estimations_existing.device)
    loss_matched_log_new_ = torch.sum(torch.stack(loss_matched_log_new_list)) / batch_size if len(loss_matched_log_new_list) else torch.zeros(1, device = estimations_existing.device)
    loss_unmatched_log_new_ = torch.sum(torch.stack(loss_unmatched_log_new_list)) / batch_size if len(loss_unmatched_log_new_list) else torch.zeros(1, device = estimations_existing.device)
    loss_affinity_ = torch.sum(torch.stack(loss_affinity_list)) / batch_size if len(loss_affinity_list) else torch.zeros(1, device = estimations_existing.device)

    factor = num_max_targets + num_max_targets_new if num_max_targets + num_max_targets_new != 0 else 1

    return loss_matched_dist_existing_ / factor, \
        loss_matched_log_existing_ / factor, \
        loss_unmatched_log_existing_ / factor, \
        loss_matched_log_new_ / factor, \
        loss_unmatched_log_new_ / factor, \
        loss_affinity_

if __name__ == '__main__':

    # torch.manual_seed(10)
    # torch.cuda.manual_seed(10)

    parser = argparse.ArgumentParser(description='Run graph-based MOT. Use notebook for data preprocessing first',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--detection_path',
                        type = str,
                        default = './data/centerPoint_detection_all_range',
                        help = 'Directory of the detections stored by scenes')
    parser.add_argument('--split',
                        type = str,
                        default = 'mini_val',
                        help = 'The split of the dataset to run')
    parser.add_argument('--config_path',
                        type = str,
                        default = './configs/centerPoint_config.json',
                        help = 'The json file that stores the filter parameters')
    parser.add_argument('--submission_path',
                        type = str,
                        default = './data/submission_train.json',
                        help = 'The json file that stores the submission')
    parser.add_argument('--checkpoint_path',
                        type = str,
                        default = None,
                        help = 'The path to load model')
    parser.add_argument('--model_save_dir',
                        type = str,
                        default = './model_saved',
                        help = 'The directory that stores the trained models')
    parser.add_argument('--estimation_dir',
                        type = str,
                        default = None,
                        help = 'The directory to save the estimation results')
    parser.add_argument('--backbone_path',
                        type = str,
                        default = './model_saved',
                        help = 'The directory that stores the backbone models')
    parser.add_argument('--centerpoint_config_path',
                        type = str,
                        default = './config',
                        help = 'The directory to save the centerpoint configs')
    parser.add_argument('--voxel_dir',
                        type = str,
                        default = '/content/python/data/voxels',
                        help = 'The directory to voxels')
    parser.add_argument('--lr',
                        type = float,
                        default = 1e-4,
                        help = 'The learning rate')
    parser.add_argument('--num_epochs',
                        type = int,
                        default = 1,
                        help = 'The training epochs')
    parser.add_argument('--batch_size',
                        type = int,
                        default = 1,
                        help = 'The size of batch')
    parser.add_argument('--no_model', action='store_true')

    hyperparameters = {}

    args = parser.parse_args()
    detection_path_ = os.path.expanduser(args.detection_path)
    split_ = args.split
    config_path_ = os.path.expanduser(args.config_path)
    submission_path_ = os.path.expanduser(args.submission_path)
    model_save_dir_ = os.path.expanduser(args.model_save_dir)
    checkpoint_path_ = args.checkpoint_path
    estimation_dir_ = args.estimation_dir
    no_model_ = args.no_model
    backbone_path_ = args.backbone_path
    centerpoint_config_path_ = args.centerpoint_config_path
    voxel_dir_ = args.voxel_dir
    hyperparameters['lr'] = args.lr
    hyperparameters['num_epochs'] = args.num_epochs
    hyperparameters['batch_size'] = args.batch_size

    if not os.path.exists(model_save_dir_):
        os.mkdir(model_save_dir_)
    if estimation_dir_ is not None:
        if not os.path.exists(estimation_dir_):
            os.mkdir(estimation_dir_)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
        extras = {"num_workers": 4, "pin_memory": True}
        print("CUDA is supported")
    else: # Otherwise, train on the CPU
        device = torch.device("cpu")
        extras = {"num_workers": 4, "pin_memory": False}
        print("CUDA NOT supported")
    # device = torch.device("cpu")
    extras['pin_memory'] = False

    # scene_list = None
    # scene_list = ['scene-0103']
    # scene_list = ['scene-0916']
    # scene_list = ['scene-0103', 'scene-0916']
    # scene_list = ['scene-0061', 'scene-0553', 'scene-0655', 'scene-0757',
    #               'scene-0796', 'scene-1077', 'scene-1094', 'scene-1100']
    # scene_list = ['scene-0061', 'scene-0553', 'scene-0655',
    #               'scene-0796', 'scene-1077', 'scene-1094', 'scene-1100']
    # scene_list = ['scene-0061', 'scene-0103', 'scene-0916',
    #               'scene-0796', 'scene-1077', 'scene-1094', 'scene-1100']
    scene_splits = create_splits_scenes()
    scene_list = scene_splits[split_][ : 62] + scene_splits['mini_train']
    dataset = nuScenes_detection_dataset(detection_path = detection_path_,
                                         split = split_,
                                         scene_list = scene_list)

    dataloader = DataLoader(dataset = dataset,
                            batch_size = hyperparameters['batch_size'],
                            shuffle = False,
                            collate_fn = collate,
                            num_workers = extras['num_workers'],
                            pin_memory = extras['pin_memory']
    )


    parameters = dict(zip(TRACKING_NAMES, [dict() for _ in TRACKING_NAMES]))
    with open(config_path_, 'r') as f:
        parameters = json.load(f)

    hyperparameters['dim_node'] = 32
    hyperparameters['dim_edge'] = 32
    hyperparameters['num_max_prior'] = 84
    nebp_da = DA_GNN(dim_node = hyperparameters['dim_node'],
                     dim_edge = hyperparameters['dim_edge'],
                     num_max_prior = hyperparameters['num_max_prior'])
    nebp_da = nebp_da.to(device)
    backbone = create_model(config_path = centerpoint_config_path_,
                            checkpoint_path = backbone_path_)
    backbone = backbone.to(device)
    optimizer = torch.optim.Adam(params = nebp_da.parameters(),
                                 lr = hyperparameters['lr'])

    if no_model_:
        print('Model is None! Not training!')
        nebp_da = None
    else:
        if checkpoint_path_ is not None and os.path.exists(checkpoint_path_):
            print('Model {} loaded'.format(checkpoint_path_))
            record = torch.load(checkpoint_path_)
            nebp_da.load_state_dict(record['nebp_da'])

    start_time = time.process_time()
    tracking_names = TRACKING_NAMES
    # tracking_names = ['car']
    # tracking_names = ['pedestrian']
    # tracking_names = ['truck']
    # tracking_names = ['bicycle']
    # tracking_names = ['motorcycle']
    submission = {}
    submission['results'] = {}

    for epoch in tqdm(range(hyperparameters['num_epochs'])):
        loss_epoch_list = []
        for minibatch_count, (measurements, measurements_mask, ground_truths, ego_poses, cs_poses, sample_tokens, scenes) in enumerate(dataloader):

            # batch_size, num_max_steps, num_max_meas, dim_meas = measurements.shape
            if not os.path.exists(os.path.join(voxel_dir_, sample_tokens[0][0] + '.pt')):
                continue

            batch_size = measurements[TRACKING_NAMES[0]][0].shape[0]
            num_max_steps = len(measurements[TRACKING_NAMES[0]])
            # num_max_steps = 20
            particles_kinematic = dict(zip(TRACKING_NAMES, [[torch.zeros(batch_size, 0, parameters[category]['num_particles'], 6).to(device)] for category in TRACKING_NAMES]))
            prob_existence = dict(zip(TRACKING_NAMES, [[torch.zeros(batch_size, 0).to(device)] for category in TRACKING_NAMES]))
            tracking_score = dict(zip(TRACKING_NAMES, [[] for category in TRACKING_NAMES]))
            targets_id = dict(zip(TRACKING_NAMES, [[torch.zeros(batch_size, 0, dtype = torch.long).to(device)] for category in TRACKING_NAMES]))
            ground_truths_id = dict(zip(TRACKING_NAMES, [[torch.zeros(batch_size, 0, dtype = torch.long).to(device)] for category in TRACKING_NAMES]))
            estimations = dict(zip(TRACKING_NAMES, [[] for _ in TRACKING_NAMES]))
            ego_poses = ego_poses.to(device)
            cs_poses = cs_poses.to(device)

            loss_list = []
            loss_matched_dist_existing_list = []
            loss_matched_log_existing_list = []
            loss_unmatched_log_existing_list = []
            loss_matched_log_new_list = []
            loss_unmatched_log_new_list = []
            loss_affinity_list = []

            count_id = 0 # For assigning ids to targets

            for step in range(num_max_steps):
                # print('step', step, '\n')
                loss = torch.zeros(1, requires_grad=True, device = device)

                voxel_data = torch.load(os.path.join(voxel_dir_, sample_tokens[0][step] + '.pt'))
                for k in voxel_data.keys():
                    if isinstance(voxel_data[k], torch.Tensor):
                        voxel_data[k] = voxel_data[k].to(device)
                st = time.time()
                with torch.no_grad():
                    backbone_feat, _ = backbone.extract_feat(voxel_data)
                    backbone_feat = backbone_feat.permute(0, 3, 2, 1)
                et = time.time()
                # print('backbone time:', et - st)

                TRACKING_NAMES_zipped = list(zip(range(len(TRACKING_NAMES)), TRACKING_NAMES))
                random.shuffle(TRACKING_NAMES_zipped)
                for category_ind, category in TRACKING_NAMES_zipped:
                    if category not in tracking_names:
                        continue

                    particles_kinematic_prior = particles_kinematic[category][-1]
                    prob_existence_prior = prob_existence[category][-1]
                    measurements_step = measurements[category][step].to(device)
                    measurements_mask_step = measurements_mask[category][step].to(device)
                    ground_truths_step = ground_truths[category][step].to(device)
                    ground_truths_id_existing = ground_truths_id[category][-1].to(device)

                    # data augmentation
                    batch_size, num_max_meas, dim_meas = measurements_step.shape
                    measurements_step_augmented = measurements_step[:, None, :, :] + \
                                                  torch.randn(batch_size, 10, num_max_meas, dim_meas, device = measurements_step.device) * 0.1
                    measurements_step_augmented = measurements_step_augmented.reshape(batch_size, 10 * num_max_meas, dim_meas)
                    measurements_step = torch.cat([measurements_step, measurements_step_augmented], dim = 1)  
                    measurements_mask_step = measurements_mask_step.repeat_interleave(11, dim = 1)  

                    # ego_poses[0, :, 0] = 685
                    # ego_poses[0, :, 1] = 1676

                    if nebp_da is not None:
                        nebp_da.set_category(ind = category_ind, device = device)

                    st = time.time()
                    particles_kinematic_posterior, estimations_posterior, prob_existence_posterior, tracking_score_posterior, targets_id_posterior, \
                    particles_kinematic_new, estimations_new, prob_existence_new, tracking_score_new, targets_id_new, \
                    asso_prob, asso_prob_single, asso_prob_single_deep, diff_loss, prob_existence_new_deep = \
                        sequential_bp(particles_kinematic_prior = particles_kinematic_prior,
                                      prob_existence_prior = prob_existence_prior,
                                      targets_id = targets_id[category][-1],
                                      measurements = measurements_step,
                                      measurements_mask = measurements_mask_step,
                                      backbone_feat = backbone_feat.to(device),
                                      ego_poses = ego_poses[:, step, :],
                                      cs_poses = cs_poses[:, step, :],
                                      scan_time = 0.5,
                                      count_id = count_id,
                                      parameters = parameters[category],
                                      nebp = nebp_da)

                    et = time.time()
                    # print('nebp time: ', et - st)

                    batch_size, num_max_targets, _, _ = particles_kinematic_posterior.shape
                    batch_size, num_max_meas, _, _ = particles_kinematic_new.shape

                    st = time.process_time()
                    ground_truths_label_existing, ground_truths_label_new, ground_truths_id_existing, ground_truths_id_new = \
                        assign_ground_truth_id(estimations_existing = estimations_posterior,
                                               targets_id_existing = targets_id_posterior,
                                               ground_truths_id_existing = ground_truths_id_existing,
                                               estimations_new = estimations_new,
                                               targets_id_new = targets_id_new,
                                               ground_truths = ground_truths_step)
                    et = time.process_time()
                    # print('ground_truths_id_new', ground_truths_id_new)
                    # print('ground_truths_label_new', ground_truths_label_new)
                    # print('prob_existence_new_deep', prob_existence_new_deep)
                    # pred_label_new = (prob_existence_new_deep > 0.5).float()
                    # acc_pos = torch.logical_and(pred_label_new == ground_truths_label_new, ground_truths_label_new == 1).sum()
                    # acc_neg = torch.logical_and(pred_label_new == ground_truths_label_new, ground_truths_label_new == 0).sum()
                    # print('acc_pos: {}/{}'.format(acc_pos.item(), (ground_truths_label_new == 1).sum()))
                    # print('acc_neg: {}/{}'.format(acc_neg.item(), (ground_truths_label_new == 0).sum()))

                    st = time.process_time()
                    # print('assign time 1: ', et - st)
                    affinity_gt = \
                        get_ground_truth_affinity(estimations_existing = estimations_posterior,
                                                targets_id_existing = targets_id_posterior,
                                                ground_truths_id_existing = ground_truths_id_existing,
                                                estimations_new = estimations_new,
                                                targets_id_new = targets_id_new,
                                                ground_truths = ground_truths_step)
                    et = time.process_time()
                    # print('assign time 2: ', et - st)

                    # print('affinity_gt', affinity_gt)
                    # print('dist_matrix', torch.sum((estimations_posterior[:, :, None, :2] - estimations_new[:, None, :, :2]) ** 2, dim = -1) ** 0.5)

                    # print('targets_id_posterior', targets_id_posterior)
                    # print('targets_id_new', targets_id_new)
                    # print('ground_truths_id_existing', ground_truths_id[category][-1])
                    # print('ground_truths_id_new', ground_truths_id_new)

                    st = time.process_time()
                    loss_matched_dist_existing, loss_matched_log_existing, loss_unmatched_log_existing, loss_matched_log_new, loss_unmatched_log_new, loss_affinity = \
                    mot_loss(estimations_existing = estimations_posterior,
                             prob_existence_existing = prob_existence_posterior,
                             targets_id_existing = targets_id_posterior,
                             ground_truths_label_existing = ground_truths_label_existing,
                             ground_truths_id_existing = ground_truths_id_existing,
                             prob_existence_new = prob_existence_new_deep,
                             targets_id_new = targets_id_new,
                             ground_truths_label_new = ground_truths_label_new,
                             ground_truths_id_new = ground_truths_id_new,
                             ground_truths = ground_truths_step,
                             affinity_gt = affinity_gt,
                             affinity_pred = asso_prob,
                             affinity_pred_single = asso_prob_single_deep,
                             parameters = parameters[category])
                    et = time.process_time()
                    # print('loss time: ', et - st)

                    loss_ = loss_matched_dist_existing * 0 + \
                            loss_matched_log_existing * 0 + \
                            loss_unmatched_log_existing * 0. + \
                            loss_matched_log_new * 1 + \
                            loss_unmatched_log_new * 1 + \
                            loss_affinity * 0 + \
                            diff_loss * 0.
                    # loss = loss + loss_
                    loss_list.append(loss_.item())
                    loss_epoch_list.append(loss_.item())
                    loss_matched_dist_existing_list.append(loss_matched_dist_existing.item())
                    loss_matched_log_existing_list.append(loss_matched_log_existing.item())
                    loss_unmatched_log_existing_list.append(loss_unmatched_log_existing.item())
                    loss_matched_log_new_list.append(loss_matched_log_new.item())
                    loss_unmatched_log_new_list.append(loss_unmatched_log_new.item())
                    loss_affinity_list.append(loss_affinity.item())
                    # print(category, 'loss: {:.4f}'.format(loss_.item()))
                    # print(category, 'loss_matched_dist_existing: ', loss_matched_dist_existing)
                    # print(category, 'loss_matched_log_existing: ', loss_matched_log_existing)
                    # print(category, 'loss_unmatched_log_existing: ', loss_unmatched_log_existing)
                    # print(category, 'loss_matched_log_new: ', loss_matched_log_new)
                    # print(category, 'loss_unmatched_log_new: ', loss_unmatched_log_new)

                    st = time.process_time()
                    particles_kinematic_all, prob_existence_all, tracking_score_all, targets_id_all, ground_truths_id_all = \
                        merge_and_prune(particles_kinematic = particles_kinematic_posterior,
                                        prob_existence = prob_existence_posterior,
                                        tracking_score = tracking_score_posterior,
                                        targets_id = targets_id_posterior,
                                        ground_truths_id = ground_truths_id_existing,
                                        particles_kinematic_new = particles_kinematic_new,
                                        prob_existence_new = prob_existence_new,
                                        tracking_score_new = tracking_score_new,
                                        targets_id_new = targets_id_new,
                                        ground_truths_id_new = ground_truths_id_new,
                                        prob_existence_new_deep = prob_existence_new_deep,
                                        parameters = parameters[category])
                    et = time.process_time()
                    # print('prune time: ', et - st)

                    # particles_kinematic[category].append(particles_kinematic_all)
                    # particles_kinematic[category][0] = particles_kinematic_all.detach()
                    # prob_existence[category].append(prob_existence_all.detach())
                    # tracking_score[category].append(tracking_score_all.detach())
                    # targets_id[category].append(targets_id_all.detach())
                    # ground_truths_id[category].append(ground_truths_id_all.detach())
                    # estimations[category].append(torch.mean(particles_kinematic_all, dim = -2).detach())

                    count_id = count_id + measurements_mask_step.shape[1]

                    st = time.process_time()
                    if loss_.requires_grad:
                        optimizer.zero_grad()
                        loss_.backward()
                        optimizer.step()
                    et = time.process_time()
                    # print('backprop time: ', et - st)

                # print('step {} loss: {:.4f}'.format(step, loss.item() / len(TRACKING_NAMES)))

                # optimizer.step()
                # optimizer.zero_grad()

            if estimation_dir_ is not None:
                estimations_tensor_list = []
                estimations_category_list = []
                for category in tracking_names:
                    estimation_tensor = \
                        save_estimations_by_scene(estimations = estimations[category],
                                                  prob_existence = prob_existence[category],
                                                  tracking_score = tracking_score[category],
                                                  targets_id = targets_id[category],
                                                  ground_truths_id = ground_truths_id[category])
                    estimations_tensor_list.append(estimation_tensor)
                    num_est, _, _ = estimation_tensor.shape
                    estimations_category_list = estimations_category_list + [category] * num_est

                estimations_list = torch.cat(estimations_tensor_list, dim = 0).permute(2, 1, 0).numpy().tolist()
                with open(os.path.join(estimation_dir_, scenes[0] + '.json'), 'w') as f:
                    json.dump({'estimated_tracks': estimations_list,
                               'estimated_categories': estimations_category_list}, f)



            print('Minibatch {} loss: {:.4f}'.format(minibatch_count, np.mean(loss_list)))
            print('Minibatch {} loss_matched_dist_existing: {:.4f}'.format(minibatch_count, np.mean(loss_matched_dist_existing_list)))
            print('Minibatch {} loss_matched_log_existing: {:.4f}'.format(minibatch_count, np.mean(loss_matched_log_existing_list)))
            print('Minibatch {} loss_unmatched_log_existing: {:.4f}'.format(minibatch_count, np.mean(loss_unmatched_log_existing_list)))
            print('Minibatch {} loss_matched_log_new: {:.4f}'.format(minibatch_count, np.mean(loss_matched_log_new_list)))
            print('Minibatch {} loss_unmatched_log_new: {:.4f}'.format(minibatch_count, np.mean(loss_unmatched_log_new_list)))
            print('Minibatch {} loss_affinity: {:.4f}'.format(minibatch_count, np.mean(loss_affinity_list)))
            torch.save({'nebp_da': nebp_da.state_dict(),
                        'parameters': parameters,
                        'hyperparameters': hyperparameters,
                        'epoch': (epoch, minibatch_count)},
                        os.path.join('/content/drive/MyDrive/nuScenes', 'model-tmp.pt'))
        print('Epoch loss: {:.4f}'.format(np.mean(loss_epoch_list)))
        # optimizer.step()
        # optimizer.zero_grad()

    end_time = time.process_time()
    print('elapsed time: {:.4f}'.format(end_time - start_time))

    torch.save({'nebp_da': nebp_da.state_dict(),
                'parameters': parameters,
                'hyperparameters': hyperparameters},
               os.path.join(model_save_dir_, 'model-{}.pt').format(datetime.now().strftime('%Y-%m-%d-%H:%M:%S')))

        # for batch in range(batch_size):
        #     scene = scenes[batch]
        #     sample_tokens_scene = sample_tokens[batch]
        #     num_steps = len(sample_tokens_scene)

        #     for step in range(num_steps):
        #         submission['results'][sample_tokens_scene[step]] = []
        #         for category in tracking_names:
        #             targets_mask = targets_id[category][step + 1][batch, :] != -1
        #             num_est = torch.sum(targets_mask.int())
        #             # print(type(num_est), num_est)
        #             estimations_step = estimations[category][step][batch, :, :][targets_mask, :]
        #             prob_existence_step = prob_existence[category][step + 1][batch, :][targets_mask]
        #             tracking_score_step = tracking_score[category][step][batch, :][targets_mask]
        #             targets_id_step = targets_id[category][step + 1][batch, :][targets_mask]
        #             for ind in range(num_est):
        #                 if prob_existence_step[ind].item() > 1 / (1 + parameters[category]['mean_clutter'] / parameters[category]['mean_newborn']) * 0.9:
        #                     sample_result = {}
        #                     sample_result['sample_token'] = sample_tokens_scene[step]
        #                     sample_result['translation'] = estimations_step[ind, [0, 1, 5]].detach().cpu().numpy().tolist()
        #                     sample_result['size'] = [2] * 3
        #                     sample_result['rotation'] = [1] + [0] * 3
        #                     sample_result['velocity'] = estimations_step[ind, [2, 3]].detach().cpu().numpy().tolist()
        #                     sample_result['tracking_id'] = targets_id_step[ind].item()
        #                     sample_result['tracking_name'] = category
        #                     sample_result['tracking_score'] = tracking_score_step[ind].item() + prob_existence_step[ind].item()
        #                     sample_result['prob_existence'] = prob_existence_step[ind].item()
        #                     submission['results'][sample_tokens_scene[step]].append(sample_result)


    # meta = {}
    # meta['use_camera'] = False
    # meta['use_lidar'] = True
    # meta['use_radar'] = False
    # meta['use_map'] = False
    # meta['use_external'] = False
    # submission['meta'] = meta

    # with open(submission_path_, 'w') as f:
    #     json.dump(submission, f)
