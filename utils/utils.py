from matplotlib.pyplot import box
import numpy as np
import cv2
import torch
from torch import nn
from torchvision.ops import nms


def get_class_names(class_names_path='./cfg/coco.txt'):
    with open(class_names_path, 'r') as f:
        return [x.strip() for x in f.readlines()]

def get_anchors(anchors_path='./cfg/anchors.txt'):
    with open(anchors_path, 'r') as f:
        # (-1->2个输出层的anchors, 每个输出层的每个单元格有3个anchor, 每个anchor的尺寸)
        return np.array(list(map(int, f.readline().split(',')))).reshape(-1, 3, 2)
    

class DecodeBox(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super().__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size

    def forward(self, input):
        #-----------------------------------------------#
        #   输入的input一共有两个，他们的shape分别是
        #   batch_size, 255, 13, 13
        #   batch_size, 255, 26, 26
        #-----------------------------------------------#
        batch_size, _, input_height, input_width = input.shape

        #-----------------------------------------------#
        #   (一个单元格[13x13或26x26]占多少图像像素[416, 416])
        #   输入为416x416时
        #   stride_h = stride_w = 32、16、(8)
        #-----------------------------------------------#
        stride_h = self.img_size[1] / input_height      # TODO: (w, h)
        stride_w = self.img_size[0] / input_width
        #-------------------------------------------------#
        #   此时获得的scaled_anchors大小是相对于特征层的    (anchor框占一个单元格的比例)
        #-------------------------------------------------#
        scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in self.anchors]

        #-----------------------------------------------#
        #   输入的input一共有两个，他们的shape分别是
        #   batch_size, 3, 13, 13, 85
        #   batch_size, 3, 26, 26, 85
        #-----------------------------------------------#
        prediction = input.view(batch_size, 
                                self.num_anchors, 
                                self.bbox_attrs, 
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
            batch_size * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
            batch_size * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)

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
        #   左图选择其中一个单元格画出anchor框
        #   右图选择其中一个单元格画出预测的框
        #----------------------------------------------------------#
        # import matplotlib.pyplot as plt
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(121)
        # if input_height==13:
        #     plt.ylim(0,13)
        #     plt.xlim(0,13)
        # elif input_height==26:
        #     plt.ylim(0,26)
        #     plt.xlim(0,26)
        # elif input_height==52:
        #     plt.ylim(0,52)
        #     plt.xlim(0,52)
        # plt.scatter(grid_x.cpu(),grid_y.cpu())

        # anchor_left = grid_x - anchor_w/2 
        # anchor_top = grid_y - anchor_h/2 

        # rect1 = plt.Rectangle([anchor_left[0,0,5,5],anchor_top[0,0,5,5]],anchor_w[0,0,5,5],anchor_h[0,0,5,5],color="r",fill=False)
        # rect2 = plt.Rectangle([anchor_left[0,1,5,5],anchor_top[0,1,5,5]],anchor_w[0,1,5,5],anchor_h[0,1,5,5],color="r",fill=False)
        # rect3 = plt.Rectangle([anchor_left[0,2,5,5],anchor_top[0,2,5,5]],anchor_w[0,2,5,5],anchor_h[0,2,5,5],color="r",fill=False)

        # ax.add_patch(rect1)
        # ax.add_patch(rect2)
        # ax.add_patch(rect3)

        # ax = fig.add_subplot(122)
        # if input_height==13:
        #     plt.ylim(0,13)
        #     plt.xlim(0,13)
        # elif input_height==26:
        #     plt.ylim(0,26)
        #     plt.xlim(0,26)
        # elif input_height==52:
        #     plt.ylim(0,52)
        #     plt.xlim(0,52)
        # plt.scatter(grid_x.cpu(),grid_y.cpu())
        # plt.scatter(pred_boxes[0,:,5,5,0].cpu(),pred_boxes[0,:,5,5,1].cpu(),c='r')

        # pre_left = pred_boxes[...,0] - pred_boxes[...,2]/2 
        # pre_top = pred_boxes[...,1] - pred_boxes[...,3]/2 

        # rect1 = plt.Rectangle([pre_left[0,0,5,5],pre_top[0,0,5,5]],pred_boxes[0,0,5,5,2],pred_boxes[0,0,5,5,3],color="r",fill=False)
        # rect2 = plt.Rectangle([pre_left[0,1,5,5],pre_top[0,1,5,5]],pred_boxes[0,1,5,5,2],pred_boxes[0,1,5,5,3],color="r",fill=False)
        # rect3 = plt.Rectangle([pre_left[0,2,5,5],pre_top[0,2,5,5]],pred_boxes[0,2,5,5,2],pred_boxes[0,2,5,5,3],color="r",fill=False)

        # ax.add_patch(rect1)
        # ax.add_patch(rect2)
        # ax.add_patch(rect3)

        # plt.show()

        #----------------------------------------------------------#
        #   将输出结果调整成相对于输入图像大小
        #----------------------------------------------------------#
        _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
        output = torch.cat((pred_boxes.view(batch_size, -1, 4) * _scale,
                            conf.view(batch_size, -1, 1), 
                            pred_cls.view(batch_size, -1, self.num_classes)), 
                            dim=-1)
        return output.data

def letterbox_image(img, size):
    ih, iw, _ = img.shape
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    img = cv2.resize(img, (nw, nh))
    new_img = (np.ones((h, w, 3)) * 127).astype(np.uint8)
    new_img[(h-nh)//2 : nh+(h-nh)//2, (w-nw)//2 : nw+(w-nw)//2, :] = img
    return new_img

def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    #----------------------------------------------------------#
    #   将预测结果的格式转换成左上角右下角的格式。
    #   prediction  [batch_size, num_anchors, 85]
    #----------------------------------------------------------#
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        #----------------------------------------------------------#
        #   对种类预测部分取max。
        #   class_conf  [num_anchors, 1]    种类置信度
        #   class_pred  [num_anchors, 1]    种类
        #----------------------------------------------------------#
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

        #----------------------------------------------------------#
        #   利用置信度进行第一轮筛选
        #----------------------------------------------------------#
        obj_conf = image_pred[:, 4]
        conf_mask = (obj_conf * class_conf[:, 0] >= conf_thres).squeeze()

        #----------------------------------------------------------#
        #   根据置信度进行预测结果的筛选
        #----------------------------------------------------------#
        image_pred = image_pred[conf_mask]
        class_conf = class_conf[conf_mask]
        class_pred = class_pred[conf_mask]
        if not image_pred.size(0):
            continue
        #-------------------------------------------------------------------------#
        #   detections  [num_anchors, 7]
        #   7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
        #-------------------------------------------------------------------------#
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

        #------------------------------------------#
        #   获得预测结果中包含的所有种类
        #------------------------------------------#
        unique_labels = detections[:, -1].cpu().unique()

        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
            detections = detections.cuda()

        for c in unique_labels:
            #------------------------------------------#
            #   获得某一类得分筛选后全部的预测结果
            #------------------------------------------#
            detections_class = detections[detections[:, -1] == c]

            #------------------------------------------#
            #   使用官方自带的非极大抑制会速度更快一些！
            #------------------------------------------#
            keep = nms(
                detections_class[:, :4],
                detections_class[:, 4] * detections_class[:, 5],
                nms_thres
            )
            max_detections = detections_class[keep]
            
            # # 按照存在物体的置信度排序
            # _, conf_sort_index = torch.sort(detections_class[:, 4]*detections_class[:, 5], descending=True)
            # detections_class = detections_class[conf_sort_index]
            # # 进行非极大抑制
            # max_detections = []
            # while detections_class.size(0):
            #     # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
            #     max_detections.append(detections_class[0].unsqueeze(0))
            #     if len(detections_class) == 1:
            #         break
            #     ious = bbox_iou(max_detections[-1], detections_class[1:])
            #     detections_class = detections_class[1:][ious < nms_thres]
            # # 堆叠
            # max_detections = torch.cat(max_detections).data
            
            # Add max detections to outputs
            output[image_i] = max_detections if output[image_i] is None else torch.cat(
                (output[image_i], max_detections))

    return output

def letter_correct_boxes(left, top, right, bottom, input_shape, image_shape):
    new_shape = image_shape*np.min(input_shape/image_shape)

    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape

    box_yx = np.concatenate(((top+bottom)/2,(left+right)/2),axis=-1)/input_shape
    box_hw = np.concatenate((bottom-top,right-left),axis=-1)/input_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  np.concatenate([
        box_mins[:, 1:2],
        box_mins[:, 0:1],
        box_maxes[:, 1:2],
        box_maxes[:, 0:1],
    ],axis=-1)
    boxes *= np.concatenate([image_shape[::-1], image_shape[::-1]],axis=-1)
    return boxes

def get_box_from_out(out, image_shape=np.array([416, 416]), model_image_size = (416, 416, 3), confidence=0.5, iou=0.3, class_names_path='./cfg/coco.txt', anchors_path='./cfg/anchors.txt', is_letterbox_image=False):
    class_names = get_class_names(class_names_path)
    anchors = get_anchors(anchors_path)
    anchors_mask = [[3,4,5], [1,2,3]]   # 13x13检测的目标更大，对应的anchor更大。26x26反之。
    yolo_decodes = []
    for i in range(2):
        yolo_decodes.append(
            DecodeBox(
                anchors.reshape(-1,2)[anchors_mask[i]], 
                len(class_names), 
                (model_image_size[1], model_image_size[0])
            )
        )
    # 解码anchor
    out_list = []
    for i in range(2):
        out_list.append(yolo_decodes[i](out[i]))

    # 堆叠预测框，nms
    out = torch.cat(out_list, 1)
    batch_detections = non_max_suppression(out, 
                                            len(class_names),
                                            conf_thres=confidence,
                                            nms_thres=iou
    )
    batch_boxes = []
    for detections in batch_detections:
        # 去除letterbox_image添加灰边造成的box偏移
        try:
            boxes = detections.cpu().numpy()
        except:
            continue
        x1, y1, x2, y2 = np.expand_dims(boxes[:,0],-1), np.expand_dims(boxes[:,1],-1), np.expand_dims(boxes[:,2],-1), np.expand_dims(boxes[:,3],-1) # xmin, ymin, xmax, ymax
        class_conf, classes = np.expand_dims(boxes[:,5], -1), np.expand_dims(boxes[:,6], -1)
        if is_letterbox_image:
            boxes = letter_correct_boxes(x1, y1, x2, y2, np.array([model_image_size[0], model_image_size[1]]), image_shape)
        else:
            x1 = x1 / model_image_size[1] # * image_shape[1]
            y1 = y1 / model_image_size[0] # * image_shape[0]
            x2 = x2 / model_image_size[1] # * image_shape[1]
            y2 = y2 / model_image_size[0] # * image_shape[0]
            # boxes = np.concatenate([x1, y1, x2, y2], axis=-1)
            # (x1, y1, x2, y2) -> (cx, cy, w, h)
            cx = x1 + (x2-x1)/2
            cy = y1 + (y2-y1)/2
            w = x2 - x1
            h = y2 - y1
            boxes = np.concatenate([cx, cy, w, h], axis=-1)
        boxes = np.concatenate([boxes, class_conf, classes], axis=-1)
        batch_boxes.append(boxes)
    return batch_boxes

def draw_one_box(img, b, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [44, 44, 255]
    c1, c2 = (int(b[0]), int(b[1])), (int(b[2]), int(b[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img

def draw_multi_box(img, boxes, class_names=None, colors=None, line_thickness=None):
    for box in boxes:
        idx = int(box[-1])
        b = box[:4]
        draw_one_box(img, b, color=colors[idx] if colors else [44, 44, 255], label=class_names[idx] if class_names else '', line_thickness=line_thickness)
    return img

def convert_cxcywh_to_x1y1x2y2(boxes, img_shape=None, is_predict=True):
    if is_predict:  # (cx, cy, w, h, conf, classes)
        pass
    else:   # (classes, cx, cy, w, h)
        boxes = boxes[:, [1,2,3,4,0]]
    boxes[:, 0] -= boxes[:, 2] / 2
    boxes[:, 1] -= boxes[:, 3] / 2
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]
    if img_shape:
        h, w, _ = img_shape
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
        boxes = boxes.astype(int)
    return boxes

def convert_x1y1x2y2_to_cxcywh(boxes, img_shape=None):
    boxes[:, 2] -= boxes[:, 0]
    boxes[:, 3] -= boxes[:, 1]
    boxes[:, 0] += boxes[:, 2] / 2
    boxes[:, 1] += boxes[:, 3] / 2
    if img_shape:
        h, w, _ = img_shape
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
        boxes = boxes.astype(int)
    return boxes