# %%
import os

from torchvision.ops import boxes
os.chdir('..')
print(os.path.abspath('.'))
# %%
import pickle

log_path = 'log/log-2021-03-19_11-40-09-0.pkl'
log_path = 'log/log-2021-03-19_18-15-50-2.pkl'
log_path = 'log/log-2021-03-19_19-04-58-3.pkl'
log_path = 'log/log-2021-03-19_19-54-59-4.pkl'
log_path = 'log/log-2021-03-19_22-23-06-1.pkl'
log_path = 'log/log-2021-03-19_23-12-03-2.pkl'
log_path = 'log/log-2021-03-20_14-35-45-1.pkl'
log_path = 'log/log-2021-03-20_15-38-09-1.pkl'
log_path = 'log/log-2021-03-20_15-50-33-2.pkl'

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

idx = 6

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


# %%
# other
# 测试是否标记
import colorsys
from utils import draw_multi_box, get_class_names, convert_cxcywh_to_x1y1x2y2
from config import opt
import os
import cv2
import matplotlib.pyplot as plt
# 设置画框颜色
class_names = get_class_names('./cfg/swucar.txt')
hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

# files_name.index('000000062808')

idx = 51
label_path = f'{opt.datasets_path}/labels/val2017'
imgs_path = f'{opt.datasets_path}/images/val2017'
imgs_name = os.listdir(imgs_path)
labels_name = os.listdir(label_path)
files_name = [x.split('.')[0] for x in labels_name]

# img = cv2.imread(os.path.join(imgs_path, imgs_name[0]))
img = cv2.imread(os.path.join(imgs_path, files_name[idx]+'.jpg'))
# with open(os.path.join(label_path, labels_name[0])) as f:
with open(os.path.join(label_path, files_name[idx]+'.txt')) as f:
    txt = np.array([list(map(float, line.strip().split())) for line in f.readlines()])
    boxes = txt
# boxes = label[0][0]

boxes = convert_cxcywh_to_x1y1x2y2(boxes, img.shape, is_predict=False)
img = draw_multi_box(img, boxes, class_names=class_names, colors=colors)

plt.imshow(img[:,:,::-1])
plt.show()

# %%
idx = files_name.index(imgs_name[3].split('.')[0])   # 51, None, 3347
img = cv2.imread(os.path.join(imgs_path, files_name[idx]+'.jpg'))
boxes = label[0][3]
boxes = convert_cxcywh_to_x1y1x2y2(boxes, img.shape, is_predict=False)
img = draw_multi_box(img, boxes, class_names=class_names, colors=colors)
plt.imshow(img[:,:,::-1])
plt.show()
