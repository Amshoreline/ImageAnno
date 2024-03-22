# Segmentation backend
## Description
- This is the backend code for semi-automic annotation system
- Frontend code can be found here: https://162.105.89.56:7777/Amshoreline/anno_front

## Environment configuration
``` bash
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

pip install Pillow SimpleITK opencv-python flask flask_cors

cd segment-anything-main; pip install -e .
```

## How to use
```
python main.py
```