#!encoding:utf-8
# %%
import os
os.chdir('..')
print(os.path.abspath('.'))
# %%
from utils.utils import get_class_names, get_anchors

class_names = get_class_names('./cfg/coco.txt')
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

# %%
import cv2
import numpy as np
from detect import Detect

detect = Detect()

img = cv2.imread('/tmp/test.jpg')

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
num_classes = 80
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