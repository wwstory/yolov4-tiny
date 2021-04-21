#!encoding:utf-8
# %%
import os
if os.path.abspath('.').split('/')[-1] != 'yolov4-tiny':
    os.chdir('..')
print(os.path.abspath('.'))
# %%
from utils.utils import get_class_names, get_anchors

class_names_path = './cfg/coco.txt'
class_names = get_class_names(class_names_path)
anchors = get_anchors().reshape(-1,2)
print(class_names)
print(anchors)

# %%
import numpy as np
import torch
from utils.utils import DecodeBox

with open('cfg/anchors.txt', 'r') as f:
    anchors = np.array(list(map(int, f.readline().split(',')))).reshape(-1, 3, 2)

db = DecodeBox(np.reshape(anchors, [-1, 2])[[[3,4,5]]], 80, (416, 416))
# db = DecodeBox(np.reshape(anchors, [-1, 2])[[[1,2,3]]], 80, (416, 416))
x1 = (torch.rand(1, 255, 13, 13))
x2 = (torch.rand(1, 255, 26, 26))
db(x1).shape

# %% detect.py -> out1, out2
# -----------------------------------------------------------
import torch
import os
import numpy as np
import cv2

from model import Yolo
from utils.utils import get_class_names, get_anchors, DecodeBox, letterbox_image, non_max_suppression, letter_correct_boxes


class Detect:

    def __init__(self, weights_path='./weights/yolov4-tiny.pt', confidence=0.5, iou=0.3, class_names_path='./cfg/coco.txt', anchors_path='./cfg/anchors.txt', is_letterbox_image=False):
        self.model_image_size = (416, 416, 3)
        self.confidence = confidence
        self.iou = iou
        self.is_letterbox_image = is_letterbox_image

        self.image_shape = np.array([416, 416])

        # 加载类别
        self.class_names = get_class_names(class_names_path)

        # 加载模型
        self.net = Yolo(num_classes=len(self.class_names))

        assert os.path.exists(weights_path), '训练模型不存在!'
        pth = torch.load(weights_path, map_location=lambda storage, loc: storage)
        if 'model' in pth:
            record_epoch = pth['epoch']
            pretrained_dict = pth['model']
        else:
            pretrained_dict = pth
        self.net.load_state_dict(pretrained_dict)
        if torch.cuda.is_available():
            self.net.cuda()
        print('load net completed!')

        # 特征层解码(加上anchor)
        self.yolo_decodes = []
        anchors = get_anchors(anchors_path)
        anchors_mask = [[3,4,5], [1,2,3]]   # 13x13检测的目标更大，对应的anchor更大。26x26反之。
        # for i in range(anchors.shape[0]):  # yolov4-tiny有2个输出(13x13, 26x26)
        for i in range(2):
            self.yolo_decodes.append(
                DecodeBox(
                    anchors.reshape(-1,2)[anchors_mask[i]], 
                    len(self.class_names), 
                    (self.model_image_size[1], self.model_image_size[0])
                )
            )

    def _detect(self, img):
        with torch.no_grad():
            if torch.cuda.is_available():
                img = img.cuda()
            return self.net(img)
    
    def __call__(self, img):
        img = self._preprocess(img)
        out = self._detect(img)
        # out = self._postprocess(out)  # 不后处理
        return out
    
    def _preprocess(self, img):
        self.image_shape = np.array(img.shape[:2])
        if self.is_letterbox_image:
            # cv2.imwrite('/tmp/test1.jpg', img)
            img = letterbox_image(img, (self.model_image_size[1], self.model_image_size[0]))
            # cv2.imwrite('/tmp/test2.jpg', img)
        else:
            img = cv2.resize(img, self.model_image_size[:2])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32) / 255.
        img = torch.from_numpy(img)
        img.unsqueeze_(dim=0)
        return img

    def _postprocess(self, out):
        # 解码anchor
        out_list = []
        for i in range(2):
            out_list.append(self.yolo_decodes[i](out[i]))

        # 堆叠预测框，nms
        out = torch.cat(out_list, 1)
        batch_detections = non_max_suppression(out, 
                                                len(self.class_names),
                                                conf_thres=self.confidence,
                                                nms_thres=self.iou
        )   # cx cy w h -> x1 y1 x2 y2
        detections = batch_detections[0]

        # 去除letterbox_image添加灰边造成的box偏移
        try:
            boxes = detections.cpu().numpy()
        except:
            return np.array([])
        x1, y1, x2, y2 = np.expand_dims(boxes[:,0],-1), np.expand_dims(boxes[:,1],-1), np.expand_dims(boxes[:,2],-1), np.expand_dims(boxes[:,3],-1) # xmin, ymin, xmax, ymax
        class_conf, classes = np.expand_dims(boxes[:,5], -1), np.expand_dims(boxes[:,6], -1)
        if self.is_letterbox_image:
            boxes = letter_correct_boxes(x1, y1, x2, y2, np.array([self.model_image_size[0], self.model_image_size[1]]), self.image_shape)
        else:
            x1 = x1 / self.model_image_size[1] * self.image_shape[1]
            y1 = y1 / self.model_image_size[0] * self.image_shape[0]
            x2 = x2 / self.model_image_size[1] * self.image_shape[1]
            y2 = y2 / self.model_image_size[0] * self.image_shape[0]
            boxes = np.concatenate([x1, y1, x2, y2], axis=-1)
        boxes = np.concatenate([boxes, class_conf, classes], axis=-1)

        return boxes

    def draw_boxes(self, img, boxes):
        import colorsys
        from utils.utils import draw_multi_box
        # 设置画框颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        img = draw_multi_box(img, boxes, class_names=self.class_names, colors=colors)
        return img

# -----------------------------------------------------------
# %%
import cv2
import numpy as np

detect = Detect(class_names_path=class_names_path)
class_names = get_class_names(class_names_path)

img = cv2.imread('/tmp/test.jpg')
assert img, '图片不存在'

out1, out2 = detect(img)
#%%
print(out1.shape)
print(out2.shape)

input = out1
# %%
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

anchors = get_anchors()
print(anchors.reshape(-1, 2))
anchors = anchors.reshape(-1, 2)[[3, 4, 5]]
num_anchors = len(anchors)
num_classes = len(class_names) # 80
bbox_attrs = 5 + num_classes
img_size = (416, 416)

#-----------------------------------------------#
#   输入的input一共有两个，他们的shape分别是
#   batch_size, 255, 13, 13
#   batch_size, 255, 26, 26
#-----------------------------------------------#
batch_size, _, input_height, input_width = input.shape

#-----------------------------------------------#
#   一个单元格占多少像素
#   输入为416x416时
#   stride_h = stride_w = 32、16、8
#-----------------------------------------------#
stride_h = img_size[1] / input_height
stride_w = img_size[0] / input_width
#-------------------------------------------------#
#   此时获得的scaled_anchors大小是相对于特征层的    
#-------------------------------------------------#
scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in anchors]

#-----------------------------------------------#
#   输入的input一共有两个，他们的shape分别是
#   batch_size, 3, 13, 13, 85
#   batch_size, 3, 26, 26, 85
#-----------------------------------------------#
prediction = input.view(batch_size, 
                        num_anchors, 
                        bbox_attrs, 
                        input_height, 
                        input_width).permute(0, 1, 3, 4, 2).contiguous()    # batch,num_anchors,(x,y,w,h,conf,cls),h_anchors,w_anchors

# 先验框的中心位置的调整参数
x = torch.sigmoid(prediction[..., 0])  
y = torch.sigmoid(prediction[..., 1])
# 先验框的宽高调整参数
w = prediction[..., 2]
h = prediction[..., 3]
# 获得置信度，是否有物体
conf = torch.sigmoid(prediction[..., 4])
# 种类置信度
pred_cls = torch.sigmoid(prediction[..., 5:])

FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

#----------------------------------------------------------#
#   生成网格，先验框中心，网格左上角 
#   batch_size,3,13,13
#----------------------------------------------------------#
grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
    batch_size * num_anchors, 1, 1).view(x.shape).type(FloatTensor)
grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
    batch_size * num_anchors, 1, 1).view(y.shape).type(FloatTensor)

#----------------------------------------------------------#
#   按照网格格式生成先验框的宽高
#   batch_size,3,13,13
#----------------------------------------------------------#
anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

#----------------------------------------------------------#
#   利用预测结果对先验框进行调整
#   首先调整先验框的中心，从先验框中心向右下角偏移
#   再调整先验框的宽高。
#----------------------------------------------------------#
pred_boxes = FloatTensor(prediction[..., :4].shape)
pred_boxes[..., 0] = x.data + grid_x
pred_boxes[..., 1] = y.data + grid_y
pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

#----------------------------------------------------------#
#   show
#----------------------------------------------------------#
fig = plt.figure()
ax = fig.add_subplot(121)
if input_height==13:
    plt.ylim(0,13)
    plt.xlim(0,13)
elif input_height==26:
    plt.ylim(0,26)
    plt.xlim(0,26)
elif input_height==52:
    plt.ylim(0,52)
    plt.xlim(0,52)
plt.scatter(grid_x.cpu(),grid_y.cpu())

anchor_left = grid_x - anchor_w/2 
anchor_top = grid_y - anchor_h/2 

rect1 = plt.Rectangle([anchor_left[0,0,5,5],anchor_top[0,0,5,5]],anchor_w[0,0,5,5],anchor_h[0,0,5,5],color="r",fill=False)
rect2 = plt.Rectangle([anchor_left[0,1,5,5],anchor_top[0,1,5,5]],anchor_w[0,1,5,5],anchor_h[0,1,5,5],color="r",fill=False)
rect3 = plt.Rectangle([anchor_left[0,2,5,5],anchor_top[0,2,5,5]],anchor_w[0,2,5,5],anchor_h[0,2,5,5],color="r",fill=False)

ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)

ax = fig.add_subplot(122)
if input_height==13:
    plt.ylim(0,13)
    plt.xlim(0,13)
elif input_height==26:
    plt.ylim(0,26)
    plt.xlim(0,26)
elif input_height==52:
    plt.ylim(0,52)
    plt.xlim(0,52)
plt.scatter(grid_x.cpu(),grid_y.cpu())
plt.scatter(pred_boxes[0,:,5,5,0].cpu(),pred_boxes[0,:,5,5,1].cpu(),c='r')

pre_left = pred_boxes[...,0] - pred_boxes[...,2]/2 
pre_top = pred_boxes[...,1] - pred_boxes[...,3]/2 

rect1 = plt.Rectangle([pre_left[0,0,5,5],pre_top[0,0,5,5]],pred_boxes[0,0,5,5,2],pred_boxes[0,0,5,5,3],color="r",fill=False)
rect2 = plt.Rectangle([pre_left[0,1,5,5],pre_top[0,1,5,5]],pred_boxes[0,1,5,5,2],pred_boxes[0,1,5,5,3],color="r",fill=False)
rect3 = plt.Rectangle([pre_left[0,2,5,5],pre_top[0,2,5,5]],pred_boxes[0,2,5,5,2],pred_boxes[0,2,5,5,3],color="r",fill=False)

ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)

plt.show()

#----------------------------------------------------------#
#   将输出结果调整成相对于输入图像大小
#----------------------------------------------------------#
_scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
output = torch.cat((pred_boxes.view(batch_size, -1, 4) * _scale,
                    conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, num_classes)), dim=-1)
output.data
# %%
