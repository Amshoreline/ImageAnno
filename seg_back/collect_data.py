import os
import sys
import json
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def get_mask_from_contours(contours, h, w, rgb=False):
    if rgb:
        colors = [
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255]
        ]
    else:
        colors = range(100)
    # Convert json to mask
    if rgb:
        anno_mask = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        anno_mask = np.zeros((h, w), dtype=np.uint8)
    for contour in contours:
        color = colors[int(contour['label'])]
        contour = np.round([[item['x'], item['y']] for item in contour['path']]).astype(np.int32)
        cv2.fillPoly(anno_mask, [contour], color)
    return anno_mask


def collect_3d(user):
    save_dir = f'data_for_train/{user}'
    os.makedirs(save_dir, exist_ok=True)
    #
    collection_names = [item for item in os.listdir(f'data/{user}') if os.path.exists(f'data/{user}/{item}/info.json')]
    for collection_name in collection_names:
        print(collection_name)
        with open(f'data/{user}/{collection_name}/info.json', 'r') as reader:
            infos = eval(reader.read())
        #
        height, width = None, None
        ys, xs = [], []
        files_list = []
        filenames_list = []
        #
        for info_dict in tqdm(infos):
            image_name = info_dict['image_name']
            json_name = image_name.replace('.jpg', '.json')
            json_path = f'data/{user}/{collection_name}/jsons/{json_name}'
            # Check if annotations exist
            if not os.path.exists(json_path):
                continue
            print(json_path)
            with open(json_path, 'r') as reader:
                contours = eval(reader.read())
            if len(contours) == 0:
                continue
            # Get height and width
            if not height:
                print(info_dict)
                height = info_dict['height']
                width = info_dict['width']
            else:
                assert height == info_dict['height'], f'{height} versus {info_dict["height"]}'
                assert width == info_dict['width']
            # Get array
            image = np.array(Image.open(f'data/{user}/{collection_name}/images/{image_name}'))
            mask = get_mask_from_contours(contours, height, width)
            viz_mask = get_mask_from_contours(contours, height, width, rgb=True)
            #
            local_ys, local_xs = np.where(mask)
            ys.extend(local_ys.tolist())
            xs.extend(local_xs.tolist())
            #
            files_list.append((image, mask, viz_mask))
            mask_name = image_name.replace('.jpg', '_mask.png')
            viz_mask_name = image_name.replace('.jpg', '_viz_mask.jpg')
            filenames_list.append((image_name, mask_name, viz_mask_name))
        # Crop RoI
        if len(ys) == 0:
            continue
        ymin, ymax = np.min(ys), np.max(ys)
        xmin, xmax = np.min(xs), np.max(xs)
        ymin, xmin = max(0, ymin - 20), max(0, xmin - 20)
        ymax, xmax = min(height, ymax + 20), min(width, xmax + 20)
        for (image, mask, viz_mask), (image_name, mask_name, viz_mask_name) in zip(files_list, filenames_list):
            image = image[ymin : ymax, xmin : xmax]
            mask = mask[ymin : ymax, xmin : xmax]
            viz_mask = viz_mask[ymin : ymax, xmin : xmax]
            # Save
            Image.fromarray(image).save(f'data_for_train/{user}/{collection_name}_{image_name}')
            Image.fromarray(mask).save(f'data_for_train/{user}/{collection_name}_{mask_name}')
            Image.fromarray(viz_mask).save(f'data_for_train/{user}/{collection_name}_{viz_mask_name}')


def collect_2d(user):
    save_dir = f'data_for_train/{user}'
    os.makedirs(save_dir, exist_ok=True)
    #
    collection_names = [item for item in os.listdir(f'data/{user}') if os.path.exists(f'data/{user}/{item}/info.json')]
    for collection_name in collection_names:
        print(collection_name)
        with open(f'data/{user}/{collection_name}/info.json', 'r') as reader:
            infos = eval(reader.read())
        #
        for info_dict in tqdm(infos):
            image_name = info_dict['image_name']
            json_name = image_name.replace('.jpg', '.json')
            json_path = f'data/{user}/{collection_name}/jsons/{json_name}'
            # Check if annotations exist
            if not os.path.exists(json_path):
                continue
            print(json_path)
            with open(json_path, 'r') as reader:
                contours = eval(reader.read())
            if len(contours) == 0:
                continue
            height = info_dict['height']
            width = info_dict['width']
            # Get array
            image = np.array(Image.open(f'data/{user}/{collection_name}/images/{image_name}'))
            mask = get_mask_from_contours(contours, height, width)
            viz_mask = get_mask_from_contours(contours, height, width, rgb=True)
            #
            ys, xs = np.where(mask)
            if len(ys) == 0:
                continue
            ymin, ymax = np.min(ys), np.max(ys)
            xmin, xmax = np.min(xs), np.max(xs)
            ymin, xmin = max(0, ymin - 20), max(0, xmin - 20)
            ymax, xmax = min(height, ymax + 20), min(width, xmax + 20)
            #
            image = image[ymin : ymax, xmin : xmax]
            mask = mask[ymin : ymax, xmin : xmax]
            viz_mask = viz_mask[ymin : ymax, xmin : xmax]
            #
            mask_name = image_name.replace('.jpg', '_mask.png')
            viz_mask_name = image_name.replace('.jpg', '_viz_mask.jpg')
            Image.fromarray(image).save(f'data_for_train/{user}/{collection_name}_{image_name}')
            Image.fromarray(mask).save(f'data_for_train/{user}/{collection_name}_{mask_name}')
            Image.fromarray(viz_mask).save(f'data_for_train/{user}/{collection_name}_{viz_mask_name}')


if __name__ == '__main__':
    if sys.argv[1] == 'h':
        print('2d/3d username')
    elif sys.argv[1] == '2d':
        collect_2d(sys.argv[2])
    else:
        collect_3d(sys.argv[2])
