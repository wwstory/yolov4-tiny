# %%
import os

from torchvision.ops import boxes
os.chdir('..')
print(os.path.abspath('.'))
# %%
import pickle

log_path = 'log/log-2021-03-20_20-42-04-1.pkl'
log_path = 'log/log-2021-03-21_00-04-55-4.pkl'

with open(log_path, 'rb') as f:
    pl = pickle.load(f)

# %%
import numpy as np
import matplotlib.pyplot as plt
print(pl.keys())

y = pl['loss']
x = np.arange(0, len(y), 1)

plt.plot(x, y)
plt.show()
# %%
img_label_pred = pl['img_label_pred']    # (num_batch, batch_size, [bbox])
imgs, labels, preds = img_label_pred['img'], img_label_pred['label'], img_label_pred['pred']
# print(pred[0], label[0])
# print(len(pred[0]), len(label[0]))
# print(preds)
# print(pred[45])
# print(label[45])


import colorsys
from utils import draw_multi_box, get_class_names, convert_cxcywh_to_x1y1x2y2
from config import opt
import numpy as np
import matplotlib.pyplot as plt
# 设置画框颜色
class_names = get_class_names('./cfg/swucar.txt')
hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

idx = 2

img = imgs[idx].copy()
label = labels[idx].copy()
boxes = label

plt.subplot(1, 2, 1)
boxes = convert_cxcywh_to_x1y1x2y2(boxes, img.shape, is_predict=True)
img = draw_multi_box(img, boxes, class_names=class_names, colors=colors)
plt.imshow(img)

img = imgs[idx].copy()
pred = preds[idx].copy()
boxes = pred

plt.subplot(1, 2, 2)
boxes = convert_cxcywh_to_x1y1x2y2(boxes, img.shape, is_predict=True)
img = draw_multi_box(img, boxes, class_names=class_names, colors=colors)
plt.imshow(img)
plt.show()

