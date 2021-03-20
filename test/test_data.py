# %%
import os

from PIL.Image import Image
os.chdir('..')
print(os.path.abspath('.'))

# %%
import colorsys
from utils import draw_multi_box, get_class_names, convert_cxcywh_to_x1y1x2y2
from config import opt
import os
import cv2
import matplotlib.pyplot as plt
from config import opt
import numpy as np
# 设置画框颜色
class_names = get_class_names('./cfg/swucar.txt')
hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

# %%
from utils import YoloDataset
dataset = YoloDataset(opt.datasets_path, (opt.input_shape[0], opt.input_shape[1]), is_train=True)
# %%
img, label = dataset[0]
img = np.transpose(img, (1, 2, 0))
boxes = label
boxes = convert_cxcywh_to_x1y1x2y2(boxes, img.shape)
img = draw_multi_box(img, boxes, class_names=class_names, colors=colors)

plt.imshow(img)
plt.show()

# %%
img = Image.open('/tmp/test.jpg')
print(img.size)

# img = cv2.circle(img, center=(305, 278), radius=20, color=(44, 44, 255), thickness=-1)
plt.imshow(img)
plt.show(img)