import numpy as np
import cv2
import torch
import torchvision
#
from segment_anything import sam_model_registry, SamPredictor


def compress(p_list, compress_degree=20):
    if len(p_list) <= 20:
        return p_list
    written_list = []
    c = 1e-3
    for i in range(0, p_list.shape[0]):
        ele = p_list[i, :]
        if i > 0 and i < p_list.shape[0] - 1:
            first_dir = ele - p_list[i - 1, :]  # 向量1数值
            second_dir = p_list[i + 1, :] - ele  # 向量2数值
            vector = first_dir * second_dir
            v = np.sqrt((vector * vector).sum())
            ab = v / (
                    np.sqrt((first_dir * first_dir).sum())
                    * np.sqrt((second_dir * second_dir).sum())
                )
            if ab >= 1 - c and ab <= 1 + c:
                continue
            elif ab >= -1 - c and ab <= -1 + c:
                continue
            elif ab >= 0 - c and ab <= 0 + c:
                continue
            last_p = written_list[len(written_list) - 1]
            dis = ele - last_p
            if np.sqrt((dis * dis).sum()) < compress_degree * np.sqrt(2):
                continue
        written_list.append(ele)
    ret = np.array(written_list)
    return ret


def get_contour(mask, compress_degree):
    H, W = mask.shape
    # Get contour
    h = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = list(h[0])
    contours = [compress(contour.astype(np.float32), compress_degree) for contour in contours]
    contours.sort(key=lambda x : cv2.contourArea(x), reverse=True)
    contour = contours[0]
    contour = {
        'path': [
            {
                'x': min(W - 1, max(1, int(point[0, 0]))),
                'y': min(H - 1, max(1, int(point[0, 1]))),
            }
            for point in contour
        ]
    }
    return contour
    # contours = contours[: 1]
    # contours = [cv2.convexHull(contour) for contour in contours]
    # contour = cv2.convexHull(contours[0]).astype(np.float32)
    # contour = contours[0].astype(np.float32)
    # print('Contour area is', cv2.contourArea(contour))
    # contour = compress(contour, compress_degree)
    # contour = [
    #     {'x': int(point[0, 0]), 'y': int(point[0, 1])}
    #     for point in contour
    # ]
    # return contours


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
    sam.eval()
    predictor = SamPredictor(sam)
    predictor.image_path = ''
    predictor.low_res_mask = None
    return predictor
