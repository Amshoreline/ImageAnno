import os
import random
from glob import glob
from PIL import Image
import cv2
from tqdm import tqdm, trange
import argparse
import warnings
warnings.filterwarnings('ignore')
#
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


def remove_redundant_cells(predictor, ori_masks, ori_cls_ids, new_masks, new_cls_ids):
    '''
    Params:
        predictor: SamPredictor, we assume that predictor.is_image_set == True
        ori_masks: (#ori_instances, H, W)
        ori_cls_list: (#ori_instances, )
        new_masks: (#new_instances, H, W)
        new_cls_list: (#new_instances, )
    Return:
        masks: np.ndarray(#instances, H, W)
        cls_list: np.ndarray(#instances, )
    '''
    assert predictor.is_image_set
    # feat_maps.shape = (1, 256, H, W)
    if 'postprocess_masks' in dir(predictor.model):
        feat_maps = predictor.model.postprocess_masks(predictor.features, predictor.input_size, predictor.original_size)
    else:
        feat_maps = predictor.postprocess_masks(predictor.features, predictor.input_size, predictor.original_size)
    feat_maps = feat_maps.squeeze(0)  # (256, H, W)
    device = feat_maps.device
    # add positional encoding to feature maps
    _, H, W = feat_maps.shape
    Y, X = torch.meshgrid(torch.arange(H), torch.arange(W))
    posenc_maps = torch.cat((Y[None], X[None]), dim=0).float().to(device)  # (2, H, W)
    # get neg feats
    new_union_mask = np.sum(new_masks, axis=0) > 0  # (H, W)
    cls_id2neg_feats = {}
    for mask, cls_id in zip(ori_masks, ori_cls_ids):
        if np.any(new_union_mask & mask):
            continue
        mask = torch.as_tensor(mask[None]).to(device)
        mask_sum = mask.sum()
        #
        feat = torch.sum(
            mask * feat_maps,
            dim=(1, 2)
        ) / mask_sum # (256, )
        feat = F.normalize(feat, dim=0, p=2)
        #
        posenc = torch.sum(
            mask * posenc_maps,
            dim=(1, 2)
        ) / mask_sum # (2, )
        #
        if not cls_id in cls_id2neg_feats:
            cls_id2neg_feats[cls_id] = []
        cls_id2neg_feats[cls_id].append(torch.cat([feat, posenc], dim=0)[:, None])
    print('Neg feats')
    for cls_id in cls_id2neg_feats.keys():
        cls_id2neg_feats[cls_id] = torch.cat(cls_id2neg_feats[cls_id], dim=1)
        print('\t', cls_id, cls_id2neg_feats[cls_id].shape)
    # get cell feats
    print('Get cell feats')
    valid_mask = np.ones(len(new_cls_ids), dtype=bool)
    torch_new_masks = torch.as_tensor(new_masks).to(device)  # (#masks, H, W)
    feat_maps = F.normalize(feat_maps, dim=0, p=2)
    for ind, (mask, cls_id) in enumerate(zip(torch_new_masks, new_cls_ids)):
        if not cls_id in cls_id2neg_feats:
            continue
        # mask = mask[None]
        # mask_sum = mask.sum()
        # #
        # feat = torch.sum(
        #     mask * feat_maps,
        #     dim=(1, 2)
        # ) / mask_sum  # (256, )
        # feat = F.normalize(feat, dim=0, p=2)
        ys, xs = torch.where(mask)
        feat = feat_maps[
            :,
            ys.float().mean().round().long().item(),
            xs.float().mean().round().long().item()
        ]
        #
        posenc = torch.sum(
            mask * posenc_maps,
            dim=(1, 2)
        ) / mask_sum  # (2, )
        #
        feat_sim = feat[None] @ cls_id2neg_feats[cls_id][: -2]
        posenc_sim = torch.sqrt(torch.sum((posenc[:, None] - cls_id2neg_feats[cls_id][-2 :]) ** 2, dim=0))
        # if (feat_sim.max() > 0.8) and (posenc_sim.min() < min(H, W) / 8):
        if (feat_sim.max() > 0.6):
            valid_mask[ind] = False
    print('Remove', np.sum(~valid_mask), 'instances')
    return new_masks[valid_mask], new_cls_ids[valid_mask]
