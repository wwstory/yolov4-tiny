# %%
import os
import numpy as np
from tqdm import tqdm
from scipy.cluster.vq import kmeans2
import pylab


dataset_type = 'train'  # 'train'/'val'
labels_path = f'/home/data/datasets/coco2017/coco/labels/{dataset_type}2017/'
# labels_path = f'/home/data/datasets/swucar/labels/{dataset_type}/'

fpaths = [os.path.join(labels_path, name) for name in os.listdir(labels_path)]

bbox = []    # 统计类别
for path in tqdm(fpaths):
    with open(path, 'r') as f:
        for line in f:
            bbox.append(list(map(float, line.split()[-2:])))
bbox = np.array(bbox)*416
anchors, idx = kmeans2(bbox, k=6)
pylab.scatter(bbox[:, 0], bbox[:, 1], marker='.', s=0.01)

pylab.scatter(anchors[:, 0], anchors[:, 1], marker='o', s=10, linewidths=2)
pylab.scatter(anchors[:, 0], anchors[:, 1], marker='x', s=500, linewidths=2)
# pylab.show()
pylab.savefig('../../out/anchors.png', dpi=1000)
print(anchors)
