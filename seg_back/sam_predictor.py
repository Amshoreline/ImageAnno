import numpy as np
import cv2
import torch
import torchvision
from segment_anything import sam_model_registry, SamPredictor


def get_predictor(model_type='vit_b'):
    if model_type == 'vit_b':
        sam_checkpoint = 'ckpts/sam_vit_b_01ec64.pth'
    elif model_type == 'vit_h':
        sam_checkpoint = 'ckpts/sam_vit_h_4b8939.pth'
    elif model_type == 'med-vit_b':
        model_type = 'vit_b'
        sam_checkpoint = 'ckpts/medsam_vit_b.pth'
    # 
    device = 'cuda'
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.image_path = ''
    predictor.low_res_mask = None
    return predictor
