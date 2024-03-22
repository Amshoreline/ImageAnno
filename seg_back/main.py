import os
import io
import re
import json
import numpy as np
import PIL.Image as Image
from io import BytesIO
import base64
import cv2
from flask import Flask, request, send_file
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
# cors = CORS(app, resources={r"/test": {"origins": "*"}})
# cors = CORS(app, resources={r"/api/*": {"origins": "*"}})


# from annotator import Annotator, read_image
# print('Define annotator')
# annotator = Annotator()

from utils import get_contour
from sam_predictor import get_predictor

sam_model = None
sam_type = 'vit-b'


@app.route('/')
def hello_world():
    return 'Hello World from root!'

@app.route('/test', methods = ['POST'])
def test():
    return 'Hello World!'

@app.route('/get_collection_list', methods=['POST'])
def get_set_list():
    collection_list = [
        file_name.replace('.json', '') for file_name in os.listdir('info')
        if file_name.endswith('.json')
    ]
    collection_list.sort(reverse=True)
    return json.dumps(collection_list).encode()

@app.route('/get_image_list', methods=['POST'])
def get_image_list():
    params = request.get_json(silent=True)
    collection_name = params['collection_name']
    with open(f'info/{collection_name}.json', 'r') as reader:
        image_list = eval(reader.read())
    image_list.sort(key=lambda x: x['image_name'])
    return json.dumps(image_list).encode()


@app.route('/get_image', methods=['POST'])
def get_image():
    params = request.get_json(silent=True)
    collection_name = params['collection_name']
    image_name = params['image_name']
    image_path = f'images/{collection_name}/{image_name}'
    with open(image_path, 'rb') as reader:
        image_data = reader.read()
    return base64.b64encode(image_data)


@app.route('/get_anno', methods=['POST'])
def get_anno():
    params = request.get_json(silent=True)
    collection_name = params['collection_name']
    image_name = params['image_name']
    json_path = (
        'json/' + collection_name + '/' + image_name.split('.')[0] + '.json'
    )
    os.makedirs('json/' + collection_name, exist_ok=True)
    if not os.path.exists(json_path):
        with open(json_path, 'w') as writer:
            writer.write(str([]))
    with open(json_path, 'r') as reader:
        content = reader.read()
    return content.encode()


@app.route('/get_anno_png', methods=['POST'])
def get_anno_png():
    params = request.get_json(silent=True)
    collection_name = params['collection_name']
    image_name = params['image_name']
    # Get height and width
    with open(f'info/{collection_name}.json', 'r') as reader:
        image_list = eval(reader.read())
    for item in image_list:
        if item['image_name'] == image_name:
            height, width = item['height'], item['width']
    # Get json
    json_path = (
        'json/' + collection_name + '/' + image_name.split('.')[0] + '.json'
    )
    if not os.path.exists(json_path):
        with open(json_path, 'w') as writer:
            writer.write(str([]))
    with open(json_path, 'r') as reader:
        contours = eval(reader.read())
    # Convert json to mask
    anno_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for contour in contours:
        contour = np.round([[item['x'], item['y']] for item in contour['path']]).astype(np.int32)
        cv2.fillPoly(anno_mask, [contour], (128, 0, 0))
    #
    img_byte_array = BytesIO()
    Image.fromarray(anno_mask).save(img_byte_array, format='PNG')
    img_byte_array.seek(0)
    return base64.b64encode(img_byte_array.getvalue()).decode('utf-8')


@app.route('/save_anno', methods=['POST'])
def save_anno():
    params = request.get_json(silent=True)
    collection_name = params['collection_name']
    image_name = params['image_name']
    json_path = (
        'json/' + collection_name + '/' + image_name.split('.')[0] + '.json'
    )
    anno = params['anno']
    with open(json_path, 'w') as writer:
        writer.write(json.dumps(anno, indent=4))
    return 'Done'.encode()


@app.route('/get_sam_pred', methods=['POST'])
def get_sam_pred():
    global sam_model, sam_type
    #
    params = request.get_json(silent=True)
    print(params)
    desired_sam_type = params['sam_type']
    collection_name = params['collection_name']
    image_name = params['image_name']
    compress_degree = params['compress_degree']
    prompt_points = params['prompt_points']
    prompt_labels = params['prompt_labels']
    prompt_bbox = params['prompt_bbox']
    #
    if (sam_model is None) or (sam_type != desired_sam_type):
        sam_type = desired_sam_type
        sam_model = get_predictor(sam_type)
    #
    image_path = 'images/' + collection_name + '/' + image_name
    image = np.array(Image.open(image_path))
    if len(image.shape) == 2:
        image = np.concatenate([image[..., None], ] * 3, axis=-1)
    if sam_model.image_path != image_path:
        sam_model.image_path = image_path
        sam_model.set_image(image)
    if len(prompt_points) > 0:
        input_point = np.array([[item['x'], item['y']] for item in prompt_points])
        input_label = np.array(prompt_labels)
    else:
        input_point = None
        input_label = None
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
    if (
        ((input_box is not None) and (input_point is not None))
        or ((input_point is not None) and (input_point.shape[0] > 1))
    ):
        # Case when the user provides multiple prompts
        mask_input = sam_model.low_res_mask
    else:
        mask_input = None
    #
    masks, _, low_res_masks = sam_model.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_box,
        multimask_output=False,
        mask_input=mask_input,
    )
    sam_model.low_res_mask = low_res_masks[0][None]
    sam_pred = get_contour(masks[0], compress_degree, image.shape)
    return json.dumps(sam_pred).encode()


@app.route('/upload_image', methods=['POST'])
def upload_image():
    max_len = 800
    params = request.get_json(silent=True)
    image_bytes = base64.b64decode(params['image'].split(',')[1])
    image = np.array(Image.open(io.BytesIO(image_bytes)))[..., : 3]
    # image = base64.b64decode(params['image'].split(';base64,')[1])
    collection_name = params['collection_name']
    if collection_name != 'test':
        return 'Not support currently'.encode()
    image_name = params['image_name']
    image_name = image_name.replace(' ', '')
    image_name = '.'.join(image_name.split('.')[: -1]) + '.jpg'
    # with open('images/' + collection_name + '/' + image_name, 'wb') as writer:
    #     writer.write(image)
    height, width = image.shape[: 2]
    if max(height, width) > max_len:
        new_height = int(height / (max(height, width) / max_len))
        new_width = int(width / (max(height, width) / max_len))
    else:
        new_height = height
        new_width = width
    Image.fromarray(image).resize((new_width, new_height)).save('images/' + collection_name + '/' + image_name)
    if len(image.shape) == 2:
        Image.fromarray(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)).save(
            'images/' + collection_name + '/' + image_name
        )
    # Update image_list
    with open(f'info/{collection_name}.json', 'r') as reader:
        image_list = eval(reader.read())
    if not image_name in [item['image_name'] for item in image_list]:
        image_list.append(
            {
                'image_name' : image_name,
                'height': image.shape[0],
                'width': image.shape[1]
            }
        )
    with open(f'info/{collection_name}.json', 'w') as writer:
        writer.write(json.dumps(image_list, indent=4))
    return image_name.encode()


@app.route('/upload_json', methods=['POST'])
def upload_json():
    params = request.get_json(silent=True)
    collection_name = params['collection_name']
    if collection_name != 'test':
        return 'Not support currently'.encode()
    with open(
        'json/' + collection_name + '/' + params['json_name'], 'w'
    ) as writer:
        writer.write(params['content'])
    return 'Done'.encode()


@app.route('/rename_image', methods=['POST'])
def rename_image():
    params = request.get_json(silent=True)
    collection_name = params['collection_name']
    if collection_name != 'test':
        return 'Not support currently'.encode()
    origin_name = params['origin_name']
    new_name = params['new_name']
    suffix = '.' + origin_name.split('.')[-1]
    origin_base_name = '.'.join(origin_name.split('.')[: -1])
    # Update image_list.json
    with open(f'info/{collection_name}.json', 'r') as reader:
        image_list = eval(reader.read())
    image_names = [item['image_name'] for item in image_list]
    if new_name in image_names:
        return 'Error'.encode()
    #
    image_index = image_names.index(origin_name)
    image_list[image_index]['image_name'] = new_name + suffix
    with open(f'info/{collection_name}.json', 'w') as writer:
        writer.write(json.dumps(image_list, indent=4))
    # Update image files
    os.system(f'mv images/{collection_name}/{origin_name} images/{collection_name}/{new_name + suffix}')
    # Updage json file
    os.system(
        f'mv json/{collection_name}/{origin_base_name}.json '
        f'   json/{collection_name}/{new_name}.json'
    )
    return new_name.encode()


@app.route('/remove_image', methods=['POST'])
def remove_image():
    params = request.get_json(silent=True)
    collection_name = params['collection_name']
    if collection_name != 'test':
        return 'Not support currently'.encode()
    image_name = params['image_name']
    suffix = '.' + image_name.split('.')[-1]
    base_name = '.'.join(image_name.split('.')[: -1])
    # Update image_list.json
    with open(f'info/{collection_name}.json', 'r') as reader:
        image_list = eval(reader.read())
    for item in image_list:
        if item['image_name'] == image_name:
            image_list.remove(item)
            print('Remove image {} successfully!'.format(image_name))
            break
    with open(f'info/{collection_name}.json', 'w') as writer:
        writer.write(json.dumps(image_list, indent=4))
    # remove image file
    os.system(f'rm images/{collection_name}/{image_name}')
    # remove json file
    os.system(f'rm json/{collection_name}/{base_name}.json')
    return image_name.encode()


@app.route('/prev_json', methods=['POST'])
def prev_json():
    params = request.get_json(silent=True)
    collection_name = params['collection_name']
    image_name = params['image_name']
    print(f'prev_json: {collection_name}/{image_name}')
    with open(f'info/{collection_name}.json', 'r') as reader:
        image_list = [item['image_name'] for item in eval(reader.read())]
    image_list.sort()
    cur_idx = image_list.index(image_name)
    if cur_idx > 0:
        dst_idx = -1
        for idx in range(cur_idx - 1, -1, -1):
            json_path = (
                'json/' + collection_name + '/'
                + image_list[idx].split('.')[0] + '.json'
            )
            if os.path.exists(json_path):
                with open(json_path, 'r') as reader:
                    content = eval(reader.read())
                if len(content) > 0:
                    dst_idx = idx
                    break
        if dst_idx != -1:
            os.system(
                'cp ' + json_path + ' json/' + collection_name + '/'
                + image_name.split('.')[0] + '.json'
            )
    return 'Done'.encode()


@app.route('/next_json', methods=['POST'])
def next_json():
    params = request.get_json(silent=True)
    collection_name = params['collection_name']
    image_name = params['image_name']
    print(f'next_json: {collection_name}/{image_name}')
    with open(f'info/{collection_name}.json', 'r') as reader:
        image_list = [item['image_name'] for item in eval(reader.read())]
    image_list.sort()
    cur_idx = image_list.index(image_name)
    if cur_idx < (len(image_list) - 1):
        dst_idx = -1
        for idx in range(cur_idx + 1, len(image_list)):
            json_path = (
                'json/' + collection_name + '/'
                + image_list[idx].split('.')[0] + '.json'
            )
            if os.path.exists(json_path):
                with open(json_path, 'r') as reader:
                    content = eval(reader.read())
                if len(content) > 0:
                    dst_idx = idx
                    break
        if dst_idx != -1:
            os.system(
                'cp ' + json_path + ' json/' + collection_name + '/'
                + image_name.split('.')[0] + '.json'
            )
    return 'Done'.encode()


@app.route('/calc_volume', methods=['POST'])
def calc_volume():
    params = request.get_json(silent=True)
    collection_name = params['collection_name']
    image_name = params['image_name']
    #
    json_path = (
        'json/' + collection_name + '/' + image_name.split('.')[0] + '.json'
    )
    if not os.path.exists(json_path):
        with open(json_path, 'w') as writer:
            writer.write(str([]))
    with open(json_path, 'r') as reader:
        content = eval(reader.read())
    if len(content) == 0:
        return '当前没有标注'.encode()
    else:
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
        res = '标签             面积'
        for label, area in zip(labels, areas):
            res += f'\n  {label}              {area:.2f}'
        return res.encode()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8002')
