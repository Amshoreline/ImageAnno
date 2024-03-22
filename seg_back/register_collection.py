import os
import argparse
import json
from PIL import Image
import numpy as np


def main():
    dir_names = os.listdir('images')
    for dir_name in dir_names:
        if os.path.exists(f'info/{dir_name}.json'):
            continue
        print('Registering', dir_name)
        image_names = os.listdir(f'images/' + dir_name)
        image_names.sort()
        res = []
        for image_name in image_names:
            if image_name.endswith('.jpg'):
                image = np.array(Image.open(f'images/{dir_name}/{image_name}'))
                res.append(
                    {  
                        'image_name': image_name,
                        'height': image.shape[0],
                        'width': image.shape[1],
                    }
                )
        with open(f'info/{dir_name}.json', 'w') as writer:
            writer.write(json.dumps(res, indent=4))
        if not os.path.exists('json/' + dir_name):
            os.makedirs('json/' + dir_name)


if __name__ == '__main__':
    main()
