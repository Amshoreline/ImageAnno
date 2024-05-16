import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import timm
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


class UNIPredictor:

    def __init__(self, ):
        self.model = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5,
            num_classes=0, dynamic_img_size=True
        )
        self.model.load_state_dict(torch.load('ckpts/UNI.bin', map_location='cpu'), strict=True)
        self.pixel_info = torch.as_tensor([[123.675, 116.28, 103.53], [58.395, 57.12, 57.375]])  # (3, 2)
        self.pixel_info = self.pixel_info.view(1, 3, 1, 1, 2)
        # image_info
        self.is_image_set = False
        self.original_size = None
        self.input_size = None
    
    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        self.pixel_info = self.pixel_info.to(device)
        return self

    def set_image(self, input_image):
        image_tensor = torch.as_tensor(input_image, device=self.device).float()
        image_tensor = image_tensor.permute(2, 0, 1).contiguous()[None, :, :, :]  # (1, 3, H, W)
        image_tensor = (image_tensor - self.pixel_info[..., 0]) / self.pixel_info[..., 1]
        _, _, h, w = image_tensor.shape
        self.original_size = (h, w)
        if h < 224:
            pad_h = 224 - h
        else:
            pad_h = (16 - h % 16) % 16
        if w < 224:
            pad_w = 224 - w
        else:
            pad_w = (16 - w % 16) % 16
        image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
        *_, h, w = image_tensor.shape
        self.input_size = (h, w)
        #
        sub_images = []
        offsets = []
        # 滑动窗口提取图像
        for i in range(0, h - 112, 112):
            for j in range(0, w - 112, 112):
                # 计算子图像的终止位置
                end_h = min(i + 224, h)
                end_w = min(j + 224, w)
                # 提取子图像
                sub_image = image_tensor[:, :, end_h - 224 : end_h, end_w - 224 : end_w]
                # 添加到列表中
                sub_images.append(sub_image)
                #
                offsets.append(((end_h - 224) // 16, (end_w - 224) // 16))
        # 将提取的图像堆叠成一个张量
        sub_images = torch.cat(sub_images, dim=0)  # (B, 3, 224, 224)
        #
        with torch.no_grad():
            sub_feats = self.model.forward_features(sub_images) # Extracted features (torch.Tensor) with shape [1,1024]
            sub_feats = sub_feats[:, self.model.num_prefix_tokens :].contiguous().view(-1, 14, 14, 1024)
        sub_feats = sub_feats.permute(0, 3, 1, 2)
        feat = torch.zeros(1, sub_feats.shape[1], h // 16, w // 16, device=sub_feats.device)
        cnt_map = torch.zeros(1, 1, h // 16, w // 16, device=sub_feats.device)
        for (off_y, off_x), sub_feat in zip(offsets, sub_feats):
            feat[..., off_y : off_y + sub_feat.shape[1], off_x : off_x + sub_feat.shape[2]] += sub_feat
            cnt_map[..., off_y : off_y + sub_feat.shape[1], off_x : off_x + sub_feat.shape[2]] += 1
        assert cnt_map.min() > 0
        self.features = feat / cnt_map
        self.is_image_set = True
    
    def postprocess_masks(self, features, input_size, original_size):
        features = F.interpolate(
            features,
            input_size,
            mode="bilinear",
            align_corners=False,
        )
        features = features[..., : original_size[0], : original_size[1]]
        return features


def get_predictor(model_type='vit_b'):
    print('Get predictor', model_type)
    if model_type == 'UNI':
        predictor = UNIPredictor()
        predictor = predictor.to(torch.device('cuda'))
    else:
        if model_type == 'vit_b':
            sam_checkpoint = 'ckpts/sam_vit_b_01ec64.pth'
        elif model_type == 'vit_h':
            sam_checkpoint = 'ckpts/sam_vit_h_4b8939.pth'
        elif model_type == 'med-vit_b':
            model_type = 'vit_b'
            sam_checkpoint = 'ckpts/medsam_vit_b.pth'
        # 
        device = torch.device('cuda')
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        sam.eval()
        predictor = SamPredictor(sam)
    predictor.image_path = ''
    predictor.low_res_mask = None
    return predictor


if __name__ == '__main__':
    predictor = get_predictor('UNI')
    image = np.ones((225, 225, 3))
    print(image.shape)
    predictor.set_image(image)
    print(predictor.features.shape)
    feat_maps = predictor.postprocess_masks(predictor.features, predictor.input_size, predictor.original_size)
    print(feat_maps.shape)