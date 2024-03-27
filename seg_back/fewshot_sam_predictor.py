import os
import random
from glob import glob
from PIL import Image
import cv2
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from tqdm import trange
import warnings
warnings.filterwarnings('ignore')
#
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from segment_anything import sam_model_registry, SamPredictor


class Mask_Weights(nn.Module):
    def __init__(self):
        super().__init__()
        # self.weights = nn.Parameter(torch.zeros(2, requires_grad=True) / 3)
        self.weights = nn.Parameter(torch.randn(3, 1, 1))


def point_selection(mask_sim, cell_radius_list, k=1):
    '''
    Params:
        mask_sim: torch.tensor(B, H, W) dtype=torch.float
        topk: int
    Return:
        topk_b: np.array(B)         dtype=int
        topk_xy: np.array(K, 2)     dtype=np.float32
            [[x_0, y_0], ...]
        topk_label: np.array(K, )   dtype=int
    '''
    # Top-1 point selection
    b, h, w = mask_sim.shape
    topk_byx = mask_sim.view(-1).topk(k)[1].cpu().numpy()  # get indices
    topk_b = (topk_byx // (h * w))
    topk_y = ((topk_byx % (h * w)) // w)[None]
    topk_x = (topk_byx % w)[None]
    topk_xy = np.concatenate((topk_x, topk_y), axis=0).T  # (K, 2)
    topk_label = np.array([1] * k)
    return topk_b, topk_xy, topk_label
    # bg_xy = []
    # neg_dist = round(cell_radius_list[topk_b[0]] * 2)
    # for dx in [-neg_dist, 0, neg_dist]:
    #     for dy in [-neg_dist, 0, neg_dist]:
    #         if (dx == 0) and (dy == 0):
    #             continue
    #         delta = np.array([[dx, dy]])  # (1, 2)
    #         bg_xy.append(topk_xy + delta)
    # bg_xy = np.concatenate(bg_xy, axis=0)  # (K * 8, 2)
    # bg_label = np.array([0] * bg_xy.shape[0])
    # return topk_b, np.concatenate([topk_xy, bg_xy], axis=0), np.concatenate([topk_label, bg_label])


def calculate_dice_loss(inputs, targets, num_masks = 1):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum()
    denominator = inputs.sum() + targets.sum()
    loss = 1 - numerator / (denominator + 1e-6)
    return loss.sum() / num_masks


def calculate_sigmoid_focal_loss(inputs, targets, num_masks = 1, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    prob = inputs.sigmoid()
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return torch.mean(loss) / num_masks


def get_cell_feats(predictor, ref_masks):
    '''
    Params:
        predictor: SAMPredictor
        ref_image: np.ndarray(H, W, 3)     dtype=np.uint8
        ref_masks: [np.ndarray(H, W), ...] dtype=np.uint8
    Retrun:
        prompts_list: [(prompt_points, prompt_labels), ...]
        gt_masks: torch.tensor(N, 1, 256, 256)  dtype=torch.float   device=gpu
        cell_feats: [torch.tensor(C, ), ...]    dtype=torch.float   device=gpu
    '''
    # Select points
    prompts_list = []
    for ref_mask in ref_masks:
        ys, xs = np.where(ref_mask > 0)
        prompts_list.append(
            (
                np.array([[np.mean(xs), np.mean(ys)], ]),  # point
                np.array([1, ])  # label
            )
        )
    # Image features encoding
    assert predictor.is_image_set
    ref_feat = predictor.features.squeeze().permute(1, 2, 0)  # (64, 64, C)
    # Prepare ref_masks and gt_masks
    ref_masks = [
        torch.as_tensor(predictor.transform.apply_image(ref_mask)).float()[None]
        for ref_mask in ref_masks
    ]
    ref_masks = torch.cat(ref_masks, dim=0)  # (N, H', W'), where max(H', W') = 1024
    padh = predictor.model.image_encoder.img_size - ref_masks.shape[1]
    padw = predictor.model.image_encoder.img_size - ref_masks.shape[2]
    ref_masks = F.pad(ref_masks, (0, padw, 0, padh))[:, None]  # (N, 1, 1024, 1024)
    #
    gt_masks = F.interpolate(ref_masks, size=(256, 256), mode="bilinear", align_corners=False).cuda()
    gt_masks = (gt_masks > 0).float()  # (N, 1, 256, 256)
    ref_masks = F.interpolate(ref_masks, size=ref_feat.shape[0: 2], mode="bilinear", align_corners=False)
    ref_masks = ref_masks[:, 0]  # (N, 64, 64)
    # Target feature extraction
    cell_feats = []
    for ref_mask in ref_masks:
        if torch.max(ref_mask) <= 0:
            cell_feats.append(None)
            continue
        cell_feat = ref_feat[ref_mask > 0]  # (M, C)
        cell_feat_mean = torch.mean(cell_feat, dim=0)  # avgpool (C, )
        cell_feat_max = torch.max(cell_feat, dim=0)[0]  # maxpool (C, )
        cell_feat = (cell_feat_max / 2 + cell_feat_mean / 2)[None]  # (1, C)
        cell_feat = F.normalize(cell_feat, p=2, dim=-1)
        cell_feats.append(cell_feat)
    return prompts_list, gt_masks, cell_feats


def get_weights(logits_list, gt_masks, base_lr, num_epochs, log_freq):
    '''
    Params:
        logits_list: torch.tensor(N, 3, 256, 256)   dtype=torch.float   device=gpu
        gt_masks: torch.tensor(N, 1, 256, 256)      dtype=torch.float   device=gpu
        base_lr:    float
        num_epochs: int
        log_freq:   int
    Return:
        weights_np: np.ndarray(3, 1, 1) dtype=np.float32
    '''
    print('======> Start Training')
    # Learnable mask weights
    mask_weights = Mask_Weights().cuda()
    mask_weights.train()
    #
    optimizer = torch.optim.Adam(mask_weights.parameters(), lr=base_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    for epoch_ind in range(num_epochs):
        for iter_ind in range(len(logits_list)):
            # Weighted sum three-scale masks
            weights = mask_weights.weights.softmax(dim=0)  # (3, 1, 1)
            logits = torch.sum(logits_list[iter_ind] * weights, dim=0)  # (256, 256)
            logits = logits[None, None]  # (1, 1, 256, 256)
            #
            dice_loss = calculate_dice_loss(logits, gt_masks[iter_ind][None, None])
            focal_loss = calculate_sigmoid_focal_loss(logits, gt_masks[iter_ind][None, None])
            loss = dice_loss + focal_loss
            #
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        #
        if epoch_ind % log_freq == 0:
            print('Train Epoch: {:} / {:}'.format(epoch_ind, num_epochs))
            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Dice_Loss: {:.4f}, Focal_Loss: {:.4f}'.format(current_lr, dice_loss.item(), focal_loss.item()))
    mask_weights.eval()
    weights = mask_weights.weights.softmax(dim=0)
    # weights = torch.cat([1 - torch.sum(mask_weights.weights, dim=0, keepdim=True), mask_weights.weights])
    weights_np = weights.detach().cpu().numpy()
    print('======> Mask weights:\n', weights_np.reshape(-1))
    return weights_np


def get_mask(predictor, topk_xy, topk_label, weights_np, cell_radius):
    '''
    Params:
        predictor: SAMPredictor
        topk_xy: np.ndarray(K, 2)   dtype=np.float32
        topk_label: np.ndarray(K, ) dtype=np.int32
        weights_np: np.ndarray(3, 1, 1) dtype=np.float32
    Return:
        mask: np.ndarray(H, W)  dtype=bool
    '''
    half_box_side = round(cell_radius * 2)
    x_inf = topk_xy[topk_label == 1, 0].min() - half_box_side
    x_sup = topk_xy[topk_label == 1, 0].max() + half_box_side
    y_inf = topk_xy[topk_label == 1, 1].min() - half_box_side
    y_sup = topk_xy[topk_label == 1, 1].max() + half_box_side
    # First-step prediction
    _, _, low_res_masks = predictor.predict(
        point_coords=topk_xy,
        point_labels=topk_label,
        # box=input_box[None]
        multimask_output=True
    )
    # Weighted sum three-scale masks
    mask_input = np.sum(low_res_masks * weights_np, axis=0)  # (256, 256)
    mask = predictor.model.postprocess_masks(
        torch.as_tensor(mask_input)[None, None],
        input_size=predictor.input_size,
        original_size=predictor.original_size
    )[0, 0]  # (ori_h, ori_w)
    mask = (mask > predictor.model.mask_threshold).numpy()
    # Cascaded Post-refinement-1
    if np.max(mask) <= 0:
        return mask
    y, x = np.nonzero(mask)
    x_min = max(x.min(), x_inf)
    x_max = min(x.max(), x_sup)
    y_min = max(y.min(), y_inf)
    y_max = min(y.max(), y_sup)
    input_box = np.array([x_min, y_min, x_max, y_max])
    masks, iou_predictions, low_res_masks = predictor.predict(
        point_coords=topk_xy,
        point_labels=topk_label,
        box=input_box[None],
        mask_input=mask_input[None],
        multimask_output=True
    )
    best_idx = np.argmax(iou_predictions)
    mask = masks[best_idx]
    mask_input = low_res_masks[best_idx]
    # Cascaded Post-refinement-2
    if np.max(mask) <= 0:
        return mask
    y, x = np.nonzero(mask)
    x_min = max(x.min(), x_inf)
    x_max = min(x.max(), x_sup)
    y_min = max(y.min(), y_inf)
    y_max = min(y.max(), y_sup)
    input_box = np.array([x_min, y_min, x_max, y_max])
    masks, iou_predictions, _ = predictor.predict(
        point_coords=topk_xy,
        point_labels=topk_label,
        box=input_box[None],
        mask_input=mask_input[None],
        multimask_output=True
    )
    best_idx = np.argmax(iou_predictions)
    return masks[best_idx]  # (ori_h, ori_w)


def find_cells(predictor, weights_list, cell_feat_matrix, cell_radius_list, test_image, max_objs):
    '''
    Params:
        predictor: SAMPredictor
        weights_list: np.ndarray(M, 3, 1, 1)    dtype=np.float32
        cell_feat_matrix: torch.tensor(M, N, C) dtype=torch.float   device=gpu
            M: number of cell classes
            N: number of shots
        cell_radius_list: np.ndarray(M, )    dtype=np.float32
        test_image: np.ndarray(H, W, 3)     dtype=np.uint8
        max_objs: int   maximal number of objects
    Return:
        res_mask_list: np.ndarray(M, H, W)  dtype=np.int32  instance mask
        fg_mask_list: np.ndarray(M, H, W)   dtype=np.uint8  foreground mask 
        coefs_list: [coefs_0, ...]
            coefs: [(inst_ind_0, coef_score_0), ...]    dtype=float32  coefficients of every objects
        prompts_list: [prompsts_0, ...]
            prompts_*: ([points_0, ...], [labels_0, ...]) prompts of every objects
    '''
    #
    M, N, C = cell_feat_matrix.shape
    # Prepare intermediate variables
    res_mask_list = np.zeros([M, *test_image.shape[: 2]], dtype=np.int32)
    fg_mask_list = np.zeros_like(res_mask_list, dtype=np.uint8)
    coefs_list = [[] for _ in range(M)]
    prompts_list = [([], []) for _ in range(M)]
    # Image feature encoding
    predictor.set_image(test_image)
    test_feat = predictor.features.squeeze()  # (C, 64, 64)
    # Cosine similarity
    _, feat_h, feat_w = test_feat.shape
    test_feat = F.normalize(test_feat, p=2, dim=0)
    test_feat = test_feat.view(C, -1)
    sim = cell_feat_matrix.view(-1, C) @ test_feat  # (M * N, 64 * 64)
    sim = sim.view(-1, 1, feat_h, feat_w)  # (M * N, 1, 64, 64)
    #
    sim = predictor.model.postprocess_masks(
        sim,
        input_size=predictor.input_size,
        original_size=predictor.original_size
    )[:, 0]  # (M * N, ori_h, ori_w)
    sim = torch.mean(sim.view(M, N, *sim.shape[1 :]), dim=1)  # (M, ori_h, ori_w)
    for inst_ind in range(max_objs):
        # Positive location prior
        topk_mn, topk_xy, topk_label = point_selection(sim, cell_radius_list, k=1)
        # class_ind = (topk_mn // N)[0]
        class_ind = topk_mn[0]
        new_mask = get_mask(predictor, topk_xy, topk_label, weights_list[class_ind], cell_radius_list[class_ind])
        # if (np.sum(new_mask & fg_mask_list[class_ind]) / np.sum(new_mask)) > 0.8:
        #     break
        # Store the intermediate results
        res_mask_list[class_ind, new_mask] = inst_ind
        fg_mask_list[class_ind] = fg_mask_list[class_ind] | new_mask
        coefs_list[class_ind].append((inst_ind, sim.max()))
        prompts_list[class_ind][0].append(topk_xy)
        prompts_list[class_ind][1].append(topk_label)
        # Post process the similarity mask
        remove_mask = F.max_pool2d(
            torch.as_tensor(new_mask).float().to(sim.device)[None, None],
            kernel_size=5, stride=1, padding=2
        )[0, 0]
        sim[class_ind, remove_mask > 0] = torch.min(sim)
        if sim.max() < 0.5:
            break
    for class_ind in range(len(prompts_list)):
        prompts = prompts_list[class_ind]
        if len(prompts[0]) > 0:
            prompts_list[class_ind] = (
                np.concatenate(prompts[0], axis=0), np.concatenate(prompts[1])
            )
    return res_mask_list, fg_mask_list, coefs_list, prompts_list


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='MoNuSAC')
    parser.add_argument('--outdir', type=str, default='fewshot_sam_f')
    parser.add_argument('--sam_type', type=str, default='vit_b')
    parser.add_argument('--nshots', type=int, default=1)
    parser.add_argument('--max_objs', type=int, default=600)
    parser.add_argument('--lr', type=float, default=1e-1) 
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--log_freq', type=int, default=200)
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    print('Args:', args)
    # Load SAM
    print('======> Load SAM')
    if args.sam_type == 'vit_h':
        sam_type, sam_ckpt = 'vit_h', 'ckpts/sam_vit_h_4b8939.pth'
    elif args.sam_type == 'vit_l':
        sam_type, sam_ckpt = 'vit_l', 'ckpts/sam_vit_l_0b3195.pth'
    elif args.sam_type == 'vit_b':
        sam_type, sam_ckpt = 'vit_b', 'ckpts/sam_vit_b_01ec64.pth'
    sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).cuda()
    sam.eval()
    print('From', sam_ckpt)
    predictor = SamPredictor(sam)
    #

    # Collect support set
    print('======> Collect support set')
    gt_mask_matrix = [[] for _ in range(num_fg_classes)]
    cell_feat_matrix = [[] for _ in range(num_fg_classes)]
    logits_matrix = [[] for _ in range(num_fg_classes)]
    cell_radius_list = np.zeros(num_fg_classes)
    for image_path in image_path_dict['train']:
        image = np.array(Image.open(image_path))
        label = np.load(image_path.replace('images', 'masks').replace('.png', '.npy'))
        inst_label = label[..., 0]
        class_label = label[..., 1]
        # Prepare ref_masks and class_inds
        ref_masks = []
        class_inds = []
        fg_class_counts = np.zeros(num_fg_classes, dtype=np.int32)
        for inst_ind in np.unique(inst_label)[1 :]:
            ref_mask = (inst_label == inst_ind)
            class_ind = round(np.median(class_label[ref_mask])) - 1
            if fg_class_counts[class_ind] >= args.nshots:
                continue
            ref_masks.append(ref_mask.astype(np.uint8))
            class_inds.append(class_ind)
            fg_class_counts[class_ind] += 1
        # Get cell_feats
        predictor.set_image(image)
        prompts_list, gt_masks, cell_feats = get_cell_feats(predictor, ref_masks)
        for cell_ind in range(len(ref_masks)):
            class_ind = class_inds[cell_ind]
            cur_support_size = len(cell_feat_matrix[class_ind])
            if cell_feats[cell_ind] is None or (cur_support_size >= args.nshots):
                continue
            gt_mask_matrix[class_ind].append(gt_masks[cell_ind][0])
            cell_feat_matrix[class_ind].append(cell_feats[cell_ind])
            _, _, low_res_masks = predictor.predict(
                point_coords=prompts_list[cell_ind][0],
                point_labels=prompts_list[cell_ind][1],
                multimask_output=True
            )
            logits = torch.as_tensor(low_res_masks).cuda()
            logits_matrix[class_ind].append(logits)
            cell_radius_list[class_ind] += np.sqrt(np.sum(ref_masks[cell_ind]) / np.pi)
        if min(
            [
                len(cell_feat_matrix[class_ind - 1])
                for class_ind in range(num_fg_classes)
            ]
        ) >= args.nshots:
            break
    for class_ind in range(num_fg_classes):
        # shape = (nshots, C)
        cell_feat_matrix[class_ind] = torch.cat(cell_feat_matrix[class_ind], dim=0)[None]
    cell_feat_matrix = torch.cat(cell_feat_matrix, dim=0)
    print('Cell feats', cell_feat_matrix.shape)
    cell_radius_list = cell_radius_list / args.nshots
    print('Cell radius', cell_radius_list)
    # Train weights
    weights_list = []
    for class_ind in range(num_fg_classes):
        weights = get_weights(
            logits_matrix[class_ind],
            gt_mask_matrix[class_ind],
            args.lr, args.num_epochs, args.log_freq
        )
        weights_list.append(weights)
    weights_list = np.array(weights_list)
    print('======> Weights')
    for weights in weights_list:
        print('\t', np.round(weights.reshape(-1), 2))
    # Inference
    print('======> Inference')
    for image_ind in trange(0, min(len(image_path_dict['test']), 100)):
        image_path = image_path_dict['test'][image_ind]
        image = np.array(Image.open(image_path))
        label = np.load(image_path.replace('images', 'masks').replace('.png', '.npy'))
        #
        inst_label = label[..., 0]
        class_label = label[..., 1]
        inst_pred_list, fg_pred_list, coefs_list, prompts_list = find_cells(
            predictor, weights_list, cell_feat_matrix, cell_radius_list, image, args.max_objs
        )


if __name__ == '__main__':
    main()
