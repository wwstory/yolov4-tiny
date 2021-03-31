# %%
import json
import os, shutil
from tqdm import tqdm

DATASET_TYPE = 'train'  # 'train'/'val'
json_path = f'/home/data/datasets/bdd100k/labels/detection20/det_v2_{DATASET_TYPE}_release.json'
out_dir = f'/home/data/datasets/bdd100k/labels/{DATASET_TYPE}'
IMAGE_W, IMAGE_H = 1280, 720

if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.mkdir(out_dir)

need_classes = {
    'other person': 0,
    'car': 1,
    'bus': 2,
    'truck': 3,
    'trailer': 4,
    'bicycle': 5,
    'rider': 6,
    'motorcycle': 6,
    'train': 7,
    'other vehicle': 4,
    'pedestrian': 0,
    'traffic light': 11,
    'traffic sign': 14,
}

# need_classes = dict(zip(need_classes, range(len(need_classes))))  # list -> dict

# all_classes = [
#     'other person',
#     'bicycle',
#     'car',
#     'rider',
#     'motorcycle',
#     'bus',
#     'train',
#     'truck',
#     'trailer',
#     'other vehicle',
#     'pedestrian',
#     'traffic light',
#     'traffic sign',
# ]


with open(json_path, 'r') as f:
    js = json.load(f)


def x1y1x2y2_to_cxcywh(*box):
    assert len(box) == 4, 'max len of box is 4!'
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    return cx, cy, w, h

# %%
for m in tqdm(js):
    txt_name = m['name'].split('.')[0] + '.txt'
    txt = ''
    labels = m['labels']
    if labels is None:  # 跳过无标签的
        continue
    for label in labels:
        classes = need_classes.get(label['category'], None)
        if classes is None:
            continue
        box2d = label['box2d']
        cx, cy, w, h = x1y1x2y2_to_cxcywh(box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2'])
        cx, cy, w, h = cx / IMAGE_W, cy / IMAGE_H, w / IMAGE_W, h / IMAGE_H
        cx, cy, w, h = list(map(lambda x:round(x, 6), [cx, cy, w, h]))  # 保留6位有效数字
        txt += f'{classes} {cx} {cy} {w} {h}\n'
    # print(txt_name, txt)
    # break
    if txt != '':
        with open(os.path.join(out_dir, txt_name), 'w') as f:
            f.write(txt)