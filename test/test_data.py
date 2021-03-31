# %%
import os
if os.path.abspath('.').split('/')[-1] != 'yolov4-tiny':
    os.chdir('..')
print(os.path.abspath('.'))

# %%
import colorsys
from utils import draw_multi_box, get_class_names, convert_cxcywh_to_x1y1x2y2
import os
import matplotlib.pyplot as plt
from config import opt
import numpy as np
# 设置画框颜色
# class_names = get_class_names(opt.class_names_path)
class_names = get_class_names('./cfg/swucar.txt')
hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

# %%
from utils import YoloDataset
dataset = YoloDataset(opt.train_datasets_images_path, opt.train_datasets_labels_path, (opt.input_shape[0], opt.input_shape[1]), is_train=True)
# %%
index = 6
img, label = dataset[index]
img = np.transpose(img, (1, 2, 0))
boxes = label
boxes = convert_cxcywh_to_x1y1x2y2(boxes, img.shape)
img = draw_multi_box(img, boxes, class_names=class_names, colors=colors)

plt.imshow(img)
plt.savefig('/tmp/out.jpg')
plt.show()
print(label)