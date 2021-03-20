import os
from tqdm import tqdm
import shutil
from collections import Counter

dataset_type = 'train'  # val   # train和val都需要切换运行一遍

# images_path = f'/home/data/datasets/coco2017/coco/images/{dataset_type}2017/'
labels_path = f'/home/data/datasets/coco2017/coco/labels/{dataset_type}2017/'   # 手动备份为新名称
out_labels_path = f'/home/data/datasets/coco2017/coco/labels/{dataset_type}2017-new/'


out_dir = out_labels_path
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.mkdir(out_dir)


new_classes = ['person', 
        'bicycle', 
        'car', 
        'motorcycle', 
        'airplane', 
        'bus', 
        'train', 
        'truck', 
        'boat', 
        'light green', 
        'light red', 
        'light yellow', 
        'pedestrian crossing', 
        'parking area', 
        'traffic sign', 
        'cat', 
        'dog']

old_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']


# 0.统计old类别
counter = Counter(old_classes)
labels_name = os.listdir(labels_path)
for label_name in tqdm(labels_name):
    label_path = os.path.join(labels_path, label_name)
    with open(label_path, 'r') as f_label:
        for line in f_label:
            l = int(line.split(' ')[0])
            counter[old_classes[l]] += 1
print('---------------------')
print('old统计：')
print(counter)



# 1.根据列表创建映射字典
change_map = {'traffic light': 11, 'stop sign': 14}
# change_map = {'stop sign': 14}
change = {}
for i, new in enumerate(new_classes):
    if new in old_classes and i != old_classes.index(new):
        change[old_classes.index(new)] = i
for i, old in enumerate(old_classes):
    if old not in new_classes:
        change[i] = -1
for k, v in change_map.items(): # 手动将原有的类别设置为指定序号
    change[old_classes.index(k)] = v 
print(change)


# 2.old_classes映射到new_classes
counter_swap = 0
counter_x = 0
counter_total = 0
counter_remain = 0
labels_name = os.listdir(labels_path)
for label_name in tqdm(labels_name):
    label_path = os.path.join(labels_path, label_name)
    with open(label_path, 'r') as f_label:
        txt = ''
        for line in f_label:
            counter_total += 1
            l = int(line.split(' ')[0])
            if change.get(l) == -1:
                counter_x += 1
                continue
            elif l in change:
                counter_swap += 1
                txt += line.replace(str(l), str(change[l]), 1)
            else:
                txt += line
        if txt != '':
            counter_remain += 1
            with open(os.path.join(out_labels_path, label_name), 'w') as f_out:
                f_out.write(txt)

print('---------------------')
print(f'图片总数：{len(labels_name)}')
print(f'剩余图片：{counter_remain}')
print(f'标签：{counter_total}')
print(f'交换：{counter_swap}')
print(f'去除：{counter_x}')

# 3.统计new类别
counter = Counter(new_classes)
labels_name = os.listdir(out_labels_path)
for label_name in tqdm(labels_name):
    label_path = os.path.join(out_labels_path, label_name)
    with open(label_path, 'r') as f_label:
        for line in f_label:
            l = int(line.split(' ')[0])
            counter[new_classes[l]] += 1
print('---------------------')
print('new统计：')
print(counter)