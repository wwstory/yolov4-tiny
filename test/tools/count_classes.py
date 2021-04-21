# %%
import os
from tqdm import tqdm
from collections import Counter


dataset_type = 'train'  # 'train'/'val'
labels_path = f'/home/data/datasets/coco2017/coco/labels/{dataset_type}2017/'
# labels_path = f'/home/data/datasets/swucar/labels/{dataset_type}/'

fpaths = [os.path.join(labels_path, name) for name in os.listdir(labels_path)]

counter = []    # 统计类别
for path in tqdm(fpaths):
    with open(path, 'r') as f:
        for line in f:
            counter.append(line.split()[0])
counter = Counter(counter)
counter = sorted(counter.items(), key=lambda x:x[1])
print(counter)
