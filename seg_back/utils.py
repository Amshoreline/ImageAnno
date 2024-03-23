import os
import sys
import json
import string
import random
from PIL import Image
import numpy as np
import pandas as pd


def register_collection():
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


if __name__ == '__main__':
    if sys.argv[1] == 'add_user':
        add_user()

