import os
import io
import re
import json
from glob import glob
#
import cv2
import numpy as np
import pandas as pd
import PIL.Image as Image
from io import BytesIO
import base64
#
from flask import Flask, request, send_from_directory, send_file
from flask_cors import CORS
app = Flask(__name__, static_folder='/root/ImageAnno/seg_back/dist', static_url_path='')
CORS(app)
from gevent import pywsgi
#
from sam_predictor import get_predictor, get_contour
from sam_dilator import find_remaining_cells
from sam_eroder import remove_redundant_cells


# Global variables
sam_model_dict = {}
token2user = pd.read_csv('data/token2user.csv')
token2user = {
    token2user.loc[ind, 'token'] : token2user.loc[ind, 'user']
    for ind in range(len(token2user))
}


@app.route('/')
def hello_world():
    return send_from_directory('dist', 'index.html')


@app.route('/get_collection_list', methods=['POST'])
def get_collection_list():
    params = request.get_json(silent=True)
    user = token2user[params['token']]
    #
    collection_list = [
        file_path.split('/')[-2]
        for file_path in glob(f'data/{user}/*/info.json')
    ]
    collection_list.sort(reverse=True)
    return json.dumps(collection_list).encode()


@app.route('/get_image_list', methods=['POST'])
def get_image_list():
    params = request.get_json(silent=True)
    user = token2user[params['token']]
    #
    collection_name = params['collection_name']
    with open(f'data/{user}/{collection_name}/info.json', 'r') as reader:
        image_list = eval(reader.read())
    image_list.sort(key=lambda x: x['image_name'])
    return json.dumps(image_list).encode()


@app.route('/get_image', methods=['POST'])
def get_image():
    params = request.get_json(silent=True)
    user = token2user[params['token']]
    #
    collection_name = params['collection_name']
    image_name = params['image_name']
    image_path = f'data/{user}/{collection_name}/images/{image_name}'
    with open(image_path, 'rb') as reader:
        image_data = reader.read()
    return base64.b64encode(image_data)


@app.route('/get_anno', methods=['POST'])
def get_anno():
    params = request.get_json(silent=True)
    user = token2user[params['token']]
    #
    collection_name = params['collection_name']
    image_name = params['image_name']
    json_name = image_name.replace('.jpg', '.json')
    json_path = f'data/{user}/{collection_name}/jsons/{json_name}'
    if not os.path.exists(json_path):
        with open(json_path, 'w') as writer:
            writer.write(str([]))
    with open(json_path, 'r') as reader:
        content = reader.read()
    return content.encode()


@app.route('/get_anno_png', methods=['POST'])
def get_anno_png():
    params = request.get_json(silent=True)
    user = token2user[params['token']]
    #
    collection_name = params['collection_name']
    image_name = params['image_name']
    # Get height and width
    with open(f'data/{user}/{collection_name}/info.json', 'r') as reader:
        image_list = eval(reader.read())
    for item in image_list:
        if item['image_name'] == image_name:
            height, width = item['height'], item['width']
    # Get json
    json_name = image_name.replace('.jpg', '.json')
    json_path = f'data/{user}/{collection_name}/jsons/{json_name}'
    if not os.path.exists(json_path):
        with open(json_path, 'w') as writer:
            writer.write(str([]))
    with open(json_path, 'r') as reader:
        contours = eval(reader.read())
    # Convert json to mask
    label2color = {
        '1': (255, 0, 0),
        '2': (0, 255, 0),
        '3': (0, 0, 255)
    }
    anno_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for contour in contours:
        color = label2color.get(contour['label'], (255, 0, 0))
        contour = np.round([[item['x'], item['y']] for item in contour['path']]).astype(np.int32)
        cv2.fillPoly(anno_mask, [contour], color)
    #
    img_byte_array = BytesIO()
    Image.fromarray(anno_mask).save(img_byte_array, format='PNG')
    img_byte_array.seek(0)
    return base64.b64encode(img_byte_array.getvalue()).decode('utf-8')


@app.route('/get_fg_png', methods=['POST'])
def get_fg_png():
    params = request.get_json(silent=True)
    user = token2user[params['token']]
    #
    collection_name = params['collection_name']
    image_name = params['image_name']
    # Get image, height and width
    with open(f'data/{user}/{collection_name}/info.json', 'r') as reader:
        image_list = eval(reader.read())
    for item in image_list:
        if item['image_name'] == image_name:
            height, width = item['height'], item['width']
    image_path = f'data/{user}/{collection_name}/images/{image_name}'
    image = np.array(Image.open(image_path))
    if image.shape[2] == 3:
        image = np.concatenate([image, np.zeros((height, width, 1), dtype=np.uint8)], axis=2)
    # Get json
    json_name = image_name.replace('.jpg', '.json')
    json_path = f'data/{user}/{collection_name}/jsons/{json_name}'
    if not os.path.exists(json_path):
        with open(json_path, 'w') as writer:
            writer.write(str([]))
    with open(json_path, 'r') as reader:
        contours = eval(reader.read())
    # Convert json to mask
    anno_mask = np.zeros((height, width), dtype=np.uint8)
    for contour in contours:
        contour = np.round([[item['x'], item['y']] for item in contour['path']]).astype(np.int32)
        cv2.fillPoly(anno_mask, [contour], 255)
    image[..., 3] = anno_mask
    #
    img_byte_array = BytesIO()
    Image.fromarray(image).save(img_byte_array, format='PNG')
    img_byte_array.seek(0)
    return base64.b64encode(img_byte_array.getvalue()).decode('utf-8')


@app.route('/save_anno', methods=['POST'])
def save_anno():
    params = request.get_json(silent=True)
    user = token2user[params['token']]
    #
    collection_name = params['collection_name']
    image_name = params['image_name']
    json_name = image_name.replace('.jpg', '.json')
    json_path = f'data/{user}/{collection_name}/jsons/{json_name}'
    anno = params['anno']
    with open(json_path, 'w') as writer:
        writer.write(json.dumps(anno, indent=4))
    return 'Done'.encode()


@app.route('/sam_set_image', methods=['POST'])
def sam_set_image():
    return 'Done'.encode()
    global sam_model_dict
    #
    params = request.get_json(silent=True)
    user = token2user[params['token']]
    #
    sam_type = params['sam_type']
    collection_name = params['collection_name']
    image_name = params['image_name']
    #
    if not sam_type in sam_model_dict:
        sam_model_dict[sam_type] = get_predictor(sam_type)
    sam_model = sam_model_dict[sam_type]
    # set image
    image_path = f'data/{user}/{collection_name}/images/{image_name}'
    image = np.array(Image.open(image_path))
    if len(image.shape) == 2:
        image = np.concatenate([image[..., None], ] * 3, axis=-1)
    if sam_model.image_path != image_path:
        sam_model.image_path = image_path
        sam_model.set_image(image)
    return 'Done'.encode()


@app.route('/get_sam_pred', methods=['POST'])
def get_sam_pred():
    global sam_model_dict
    #
    params = request.get_json(silent=True)
    user = token2user[params['token']]
    #
    sam_type = params['sam_type']
    collection_name = params['collection_name']
    image_name = params['image_name']
    compress_degree = params['compress_degree']
    prompt_points = params['prompt_points']
    prompt_labels = params['prompt_labels']
    prompt_bbox = params['prompt_bbox']
    #
    if not sam_type in sam_model_dict:
        sam_model_dict[sam_type] = get_predictor(sam_type)
    sam_model = sam_model_dict[sam_type]
    # set image
    image_path = f'data/{user}/{collection_name}/images/{image_name}'
    image = np.array(Image.open(image_path))
    if len(image.shape) == 2:
        image = np.concatenate([image[..., None], ] * 3, axis=-1)
    if sam_model.image_path != image_path:
        sam_model.image_path = image_path
        sam_model.set_image(image)
    # set prompt point(s)
    if len(prompt_points) > 0:
        input_point = np.array([[item['x'], item['y']] for item in prompt_points])
        input_label = np.array(prompt_labels)
    else:
        input_point = None
        input_label = None
    # set prompt bbox
    if 'xmin' in prompt_bbox:
        input_box = np.array(
            [
                prompt_bbox['xmin'],
                prompt_bbox['ymin'],
                prompt_bbox['xmax'],
                prompt_bbox['ymax'],
            ]
        )
    else:
        input_box = None
    if (input_box is not None) and (input_point is None):
        # Case when the user only provides input_box
        input_box = input_box[None]
    # set prompt mask (from previous iteration)
    if (
        ((input_box is not None) and (input_point is not None))
        or ((input_point is not None) and (input_point.shape[0] > 1))
    ):
        # Case when the user provides multiple prompts
        mask_input = sam_model.low_res_mask
    else:
        mask_input = None
    # get mask
    masks, _, low_res_masks = sam_model.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_box,
        multimask_output=False,
        mask_input=mask_input,
    )
    sam_model.low_res_mask = low_res_masks[0][None]
    sam_pred = get_contour(masks[0], compress_degree)
    return json.dumps([sam_pred]).encode()


@app.route('/get_more_sam_pred', methods=['POST'])
def get_more_sam_pred():
    global sam_model_dict
    #
    params = request.get_json(silent=True)
    user = token2user[params['token']]
    # parse parameters
    sam_type = params['sam_type']
    collection_name = params['collection_name']
    image_name = params['image_name']
    compress_degree = params['compress_degree']
    max_objs = params['max_objs']
    sim_thres = params['sim_thres']
    # get height and width
    with open(f'data/{user}/{collection_name}/info.json', 'r') as reader:
        image_list = eval(reader.read())
    for item in image_list:
        if item['image_name'] == image_name:
            height, width = item['height'], item['width']
    # prepare ori_masks
    json_name = image_name.replace('.jpg', '.json')
    json_path = f'data/{user}/{collection_name}/jsons/{json_name}'
    with open(json_path, 'r') as reader:
        ori_contours = eval(reader.read())
    ori_masks = []
    ori_cls_list = []
    for contour in ori_contours:
        mask = np.zeros((height, width), dtype=np.uint8)
        contour_arr = np.round([[item['x'], item['y']] for item in contour['path']]).astype(np.int32)
        cv2.fillPoly(mask, [contour_arr], 1)
        ori_masks.append(mask)
        ori_cls_list.append(int(contour['label']))
    # set model and image
    if not sam_type in sam_model_dict:
        sam_model_dict[sam_type] = get_predictor(sam_type)
    sam_model = sam_model_dict[sam_type]
    image_path = f'data/{user}/{collection_name}/images/{image_name}'
    image = np.array(Image.open(image_path))
    if len(image.shape) == 2:
        image = np.concatenate([image[..., None], ] * 3, axis=-1)
    if sam_model.image_path != image_path:
        sam_model.image_path = image_path
        sam_model.set_image(image)
    # genereate masks
    masks, cls_list = find_remaining_cells(sam_model, ori_masks, ori_cls_list, max_objs, sim_thres)
    # convert masks to json
    contours = []
    for mask, cls_id in zip(masks, cls_list):
        contour = get_contour(mask, compress_degree)
        contour['label'] = str(cls_id)
        contours.append(contour)
    return json.dumps(ori_contours + contours).encode()


@app.route('/get_less_sam_pred', methods=['POST'])
def get_less_sam_pred():
    global sam_model_dict
    #
    params = request.get_json(silent=True)
    user = token2user[params['token']]
    # parse parameters
    sam_type = params['sam_type']
    collection_name = params['collection_name']
    image_name = params['image_name']
    compress_degree = params['compress_degree']
    new_contours = params['anno']
    # get height and width
    with open(f'data/{user}/{collection_name}/info.json', 'r') as reader:
        image_list = eval(reader.read())
    for item in image_list:
        if item['image_name'] == image_name:
            height, width = item['height'], item['width']
    # prepare masks
    print('prepare masks')
    json_name = image_name.replace('.jpg', '.json')
    json_path = f'data/{user}/{collection_name}/jsons/{json_name}'
    with open(json_path, 'r') as reader:
        ori_contours = eval(reader.read())
    print('#New Contours', len(new_contours), '#Ori Contours', len(ori_contours))
    ori_masks, new_masks = [], []
    ori_cls_ids, new_cls_ids = [], []
    for masks, cls_ids, contours in zip([ori_masks, new_masks], [ori_cls_ids, new_cls_ids], [ori_contours, new_contours]):
        for contour in contours:
            mask = np.zeros((height, width), dtype=np.uint8)
            contour_arr = np.round([[item['x'], item['y']] for item in contour['path']]).astype(np.int32)
            cv2.fillPoly(mask, [contour_arr], 1)
            masks.append(mask)
            cls_ids.append(int(contour['label']))
    ori_masks = np.array(ori_masks).astype(bool)
    ori_cls_ids = np.array(ori_cls_ids)
    new_masks = np.array(new_masks).astype(bool)
    new_cls_ids = np.array(new_cls_ids)
    # set model and image
    print('set model and image')
    if not sam_type in sam_model_dict:
        sam_model_dict[sam_type] = get_predictor(sam_type)
    sam_model = sam_model_dict[sam_type]
    image_path = f'data/{user}/{collection_name}/images/{image_name}'
    image = np.array(Image.open(image_path))
    if len(image.shape) == 2:
        image = np.concatenate([image[..., None], ] * 3, axis=-1)
    if sam_model.image_path != image_path:
        sam_model.image_path = image_path
        sam_model.set_image(image)
    # genereate masks
    print('generate masks')
    masks, cls_list = remove_redundant_cells(sam_model, ori_masks, ori_cls_ids, new_masks, new_cls_ids)
    # convert masks to json
    print('convert masks to json')
    contours = []
    for mask, cls_id in zip(masks, cls_list):
        contour = get_contour(mask, compress_degree)
        contour['label'] = str(cls_id)
        contours.append(contour)
    return json.dumps(contours).encode()



@app.route('/upload_image', methods=['POST'])
def upload_image():
    max_len = 1024
    params = request.get_json(silent=True)
    user = token2user[params['token']]
    #
    image_bytes = base64.b64decode(params['image'].split(',')[1])
    image = np.array(Image.open(io.BytesIO(image_bytes)))[..., : 3]
    collection_name = params['collection_name']
    if collection_name != 'test':
        return 'Not support currently'.encode()
    image_name = params['image_name']
    image_name = image_name.replace(' ', '')
    image_name = '.'.join(image_name.split('.')[: -1]) + '.jpg'
    image_path = f'data/{user}/{collection_name}/images/{image_name}'
    # Get new_height, new_width and pil_image
    height, width = image.shape[: 2]
    if max(height, width) > max_len:
        ratio = max_len / max(height, width)
        new_height = int(height * ratio)
        new_width = int(width * ratio)
    else:
        new_height = height
        new_width = width
    if len(image.shape) == 2:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
    else:
        pil_image = Image.fromarray(image)
    # Update image_list
    with open(f'data/{user}/{collection_name}/info.json', 'r') as reader:
        image_list = eval(reader.read())
    assert not image_name in [item['image_name'] for item in image_list]
    image_list.append(
        {
            'image_name' : image_name,
            'height': new_height,
            'width': new_width
        }
    )
    with open(f'data/{user}/{collection_name}/info.json', 'w') as writer:
        writer.write(json.dumps(image_list, indent=4))
    # Save image
    pil_image.resize((new_width, new_height)).save(image_path)
    return image_name.encode()


@app.route('/upload_json', methods=['POST'])
def upload_json():
    params = request.get_json(silent=True)
    user = token2user[params['token']]
    #
    collection_name = params['collection_name']
    json_name = params['json_name']
    if collection_name != 'test':
        return 'Not support currently'.encode()
    json_path = f'data/{user}/{collection_name}/jsons/{json_name}'
    with open(json_path, 'w') as writer:
        writer.write(params['content'])
    return 'Done'.encode()


@app.route('/rename_image', methods=['POST'])
def rename_image():
    params = request.get_json(silent=True)
    user = token2user[params['token']]
    #
    collection_name = params['collection_name']
    if collection_name != 'test':
        return 'Not support currently'.encode()
    origin_name = params['origin_name']
    new_name = params['new_name'] + '.jpg'
    origin_base_name = '.'.join(origin_name.split('.')[: -1])
    # Update image_list.json
    with open(f'data/{user}/{collection_name}/info.json', 'r') as reader:
        image_list = eval(reader.read())
    image_names = [item['image_name'] for item in image_list]
    if new_name in image_names:
        return 'Error'.encode()
    #
    image_index = image_names.index(origin_name)
    image_list[image_index]['image_name'] = new_name
    with open(f'data/{user}/{collection_name}/info.json', 'w') as writer:
        writer.write(json.dumps(image_list, indent=4))
    # Update image file
    ori_image_path = f'data/{user}/{collection_name}/images/{origin_name}'
    new_image_path = f'data/{user}/{collection_name}/images/{new_name}'
    os.system(f'mv {ori_image_path} {new_image_path}')
    # Update json file
    ori_json_path = ori_image_path.replace('/images/', '/jsons/').replace('.jpg', '.json')
    new_json_path = new_image_path.replace('/images/', '/jsons/').replace('.jpg', '.json')
    os.system(f'mv {ori_json_path} {new_json_path}')
    return new_name.encode()


@app.route('/remove_image', methods=['POST'])
def remove_image():
    params = request.get_json(silent=True)
    user = token2user[params['token']]
    #
    collection_name = params['collection_name']
    if collection_name != 'test':
        return 'Not support currently'.encode()
    image_name = params['image_name']
    json_name = image_name.replace('.jpg', '.json')
    # Update image_list.json
    with open(f'data/{user}/{collection_name}/info.json', 'r') as reader:
        image_list = eval(reader.read())
    for item in image_list:
        if item['image_name'] == image_name:
            image_list.remove(item)
            print('Remove image {} successfully!'.format(image_name))
            break
    with open(f'data/{user}/{collection_name}/info.json', 'w') as writer:
        writer.write(json.dumps(image_list, indent=4))
    # remove image file
    os.system(f'rm data/{user}/{collection_name}/images/{image_name}')
    # remove json file
    os.system(f'rm data/{user}/{collection_name}/jsons/{json_name}')
    return image_name.encode()


@app.route('/calc_volume', methods=['POST'])
def calc_volume():
    params = request.get_json(silent=True)
    user = token2user[params['token']]
    #
    collection_name = params['collection_name']
    image_name = params['image_name']
    json_name = image_name.replace('.jpg', '.json')
    #
    json_path = f'data/{user}/{collection_name}/jsons/{json_name}'
    if not os.path.exists(json_path):
        with open(json_path, 'w') as writer:
            writer.write(str([]))
    with open(json_path, 'r') as reader:
        content = eval(reader.read())
    if len(content) == 0:
        return '当前没有标注'.encode()
    else:
        content.sort(key=lambda x: x['label'])
        labels = [dict_['label'] for dict_ in content]
        paths = [dict_['path'] for dict_ in content]
        contours = [
            np.array([
                [[item['x'], item['y']]]
                for item in path
            ]).astype(np.float32)
            for path in paths
        ]
        areas = [cv2.contourArea(contour) for contour in contours]
        res = '          标签             面积'
        for ind, (label, area) in enumerate(zip(labels, areas)):
            res += f'\n{ind}/{len(labels)}      {label}              {area:.2f}'
        return res.encode()


# @app.route('/get_poster_1')
# def get_poster_1():
#     return send_file('ICME_BACON_poster.pdf', as_attachment=True)


# @app.route('/get_poster_2')
# def get_poster_2():
#     return send_file('ICME_poster_pry.pdf', as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8002', threaded=False)
    # pywsgi.WSGIServer(('0.0.0.0', 8002), app).serve_forever()
