import os
import sys
from glob import glob
import json
import string
import random
from PIL import Image
import SimpleITK as sitk
import numpy as np
import pandas as pd


def register_collection(src_dir, dst_dir):
    print(f'from {src_dir} to {dst_dir}')
    #
    image_dir = f'{dst_dir}/images'
    json_dir = f'{dst_dir}/jsons'
    info_path = f'{dst_dir}/info.json'
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    #
    info_dict = []
    for image_path in (
        glob(f'{src_dir}/*.JPG') + glob(f'{src_dir}/*.jpg')
        + glob(f'{src_dir}/*.png') + glob(f'{src_dir}/*.PNG')
    ):
        image_name = os.path.basename(image_path).split('.')[0] + '.jpg'
        image = Image.open(image_path).convert('RGB')
        image_arr = np.array(image)
        image.save(f'{image_dir}/{image_name}')
        info_dict.append(
            {'image_name': image_name, 'height': image_arr.shape[0], 'width': image_arr.shape[1]}
        )
    with open(info_path, 'w') as writer:
        writer.write(json.dumps(info_dict, indent=4))


def resample_image(itk_image, out_spacing=None, out_size=None, is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    if (
        ((out_spacing is None) and (out_size is None))
        or ((out_spacing is not None) and (tuple(out_spacing) == original_spacing))
        or ((out_size is not None) and (tuple(out_size) == original_size))
    ):
        return itk_image
    if out_size is None:
        for axis in range(3):
            if out_spacing[axis] == -1:
                out_spacing[axis] = original_spacing[axis]
        out_size = [
            round(original_size[0] * (original_spacing[0] / out_spacing[0])),
            round(original_size[1] * (original_spacing[1] / out_spacing[1])),
            round(original_size[2] * (original_spacing[2] / out_spacing[2]))
        ]
    else:
        assert out_spacing is None
        out_spacing = [
            original_size[0] * original_spacing[0] / out_size[0],
            original_size[1] * original_spacing[1] / out_size[1],
            original_size[2] * original_spacing[2] / out_size[2],
        ]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    # resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    resample.SetOutputOrigin(itk_image.GetOrigin())
    # resample.SetOutputOrigin((0.0, 0.0, 0.0))
    resample.SetTransform(sitk.Transform())
    if is_label:
        itk_image = sitk.Cast(itk_image, sitk.sitkUInt8)
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
        resample.SetDefaultPixelValue(0)
    else:
        itk_image = sitk.Cast(itk_image, sitk.sitkFloat32)
        resample.SetInterpolator(sitk.sitkLinear)
        resample.SetDefaultPixelValue(float(np.min(sitk.GetArrayFromImage(itk_image))))
    return resample.Execute(itk_image)  # Not an inplace operation


def register_itk(src_path, dst_dir, cut_axis=0):
    print(f'from {src_path} to {dst_dir} with cut_axis={cut_axis}')
    assert cut_axis in [0, 1, 2]
    #
    image_dir = f'{dst_dir}/images'
    json_dir = f'{dst_dir}/jsons'
    info_path = f'{dst_dir}/info.json'
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    #
    image_itk = sitk.ReadImage(src_path)
    print('origin spacing', image_itk.GetSpacing(), 'size', image_itk.GetSize())
    if cut_axis != 0:
        max_spac = min(image_itk.GetSpacing())
        image_itk = resample_image(image_itk, out_spacing=(max_spac, max_spac, max_spac))
        print('new spacing', image_itk.GetSpacing(), 'size', image_itk.GetSize())
    image_arr = sitk.GetArrayFromImage(image_itk)
    print('arr_shape', image_arr.shape, 'min', image_arr.min(), 'max', image_arr.max())
    min_val = np.percentile(image_arr, 0)
    max_val = np.percentile(image_arr, 100)
    image_arr = (np.clip(image_arr, a_min=min_val, a_max=max_val) - min_val) / (max_val - min_val)
    assert image_arr.min() >= -1e-6, image_arr.min()
    assert image_arr.max() <= (1 + 1e-6), image_arr.max()
    image_arr = (image_arr * 254).astype(np.uint8)
    image_arr = np.concatenate([image_arr[..., None], ] * 3, axis=-1)  # (D, H, W, 3)
    #
    info_dict = []
    for image_ind in range(image_arr.shape[cut_axis]):
        image_name = f'{image_ind:04}.jpg'
        if cut_axis == 0:
            slice_arr = image_arr[image_ind]
        elif cut_axis == 1:
            slice_arr = image_arr[:, image_ind]
        elif cut_axis == 2:
            slice_arr = image_arr[:, :, image_ind]
        Image.fromarray(slice_arr).save(f'{image_dir}/{image_name}')
        info_dict.append(
            {'image_name': image_name, 'height': slice_arr.shape[0], 'width': slice_arr.shape[1]}
        )
    with open(info_path, 'w') as writer:
        writer.write(json.dumps(info_dict, indent=4))


def add_user():
    token_len = 16
    csv_path = 'data/token2user.csv'
    token2user = pd.read_csv(csv_path)
    #
    print('Input username:')
    user = input()
    assert not user in token2user['user']
    characters = string.ascii_letters + string.digits
    token = ''.join(random.choice(characters) for _ in range(token_len))
    assert not token in token2user['token']
    print(f'Add user {user} with token {token} successfully')
    token2user.loc[len(token2user)] = {'token': token, 'user': user}
    token2user.to_csv(csv_path, index=False)
    os.makedirs(f'data/{user}', exist_ok=True)


if __name__ == '__main__':
    if sys.argv[1] == 'image':
        register_collection(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'itk':
        register_itk(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif sys.argv[1] == 'user':
        add_user()

