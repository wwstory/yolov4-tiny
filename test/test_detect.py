# coding=utf-8
# %%
import os
if os.path.abspath('.').split('/')[-1] != 'yolov4-tiny':
    os.chdir('..')
print(os.path.abspath('.'))
# %%
from detect import Detect

import cv2
from config import opt
import  matplotlib.pyplot as plt
from tqdm import tqdm
import random
from utils import boxes_classes_filter
# detect = Detect(is_letterbox_image=True)
# detect = Detect(class_names_path=opt.class_names_path, anchors_path=opt.anchors_path)
detect = Detect(weights_path='./weights/yolov4-tiny.pth', 
                # class_names_path=opt.class_names_path, 
                # class_names_path='./cfg/coco.txt', 
                # class_names_path='./cfg/bdd100k.txt', 
                class_names_path='./cfg/swucar.txt', 
                anchors_path=opt.anchors_path,
                is_letterbox_image=True)

# img = cv2.imread('/tmp/test.jpg')
images_dir = '/home/dejiang/Pictures/test-car-images'
image_list = os.listdir(images_dir)
for i in tqdm(random.choices(range(len(image_list)), k=10)):
    img = cv2.imread(f'{images_dir}/{image_list[i]}')
    assert img is not None, 'image file not exist!'
    boxes = detect(img)

    # boxes = boxes_classes_filter(boxes, need_list=['car'], class_names_path='./cfg/coco.txt')
    # boxes = boxes_classes_filter(boxes, filter_list=['car'], class_names_path='./cfg/coco.txt')
    boxes = boxes_classes_filter(boxes, need_list=['car', 'bus', 'truck', 'trailer'], class_names_path='./cfg/swucar.txt')

    img = detect.draw_boxes(img, boxes)
    plt.imshow(img[:,:,::-1])
    plt.show()
    print(boxes)
    cv2.imwrite(f'/tmp/test_out{i}.jpg', img)
