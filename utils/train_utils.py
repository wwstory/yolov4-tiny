import cv2
import torch
import numpy as np

from utils.utils import draw_multi_box, get_class_names, get_anchors, DecodeBox, non_max_suppression, letter_correct_boxes


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
        except:
            boxes = np.array([])
        batch_boxes.append(boxes)
    return batch_boxes

def img_tensor_to_cv(img : torch.Tensor):
    img = img.numpy()
    img = img.transpose(1, 2, 0)
    img = img * 255
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def img_cv_to_tensor(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = img / 255
    img = torch.from_numpy(img)
    return img

def draw_one_box_in_tensor(img, boxes, class_names=None, colors=None, format='x1y1x2y2', resize_box=False):
    cv_img = img_tensor_to_cv(img)
    img = draw_multi_box(cv_img, boxes, class_names=class_names, colors=colors, format=format, resize_box=resize_box)
    t_img = img_cv_to_tensor(img)
    return t_img

def draw_multi_box_in_tensor(imgs, batch_boxes, class_names=None, colors=None, format='x1y1x2y2', resize_box=False):
    img_list = []
    for img, boxes in zip(imgs, batch_boxes):
        img_list.append(draw_one_box_in_tensor(img, boxes, class_names=class_names, colors=colors, format=format, resize_box=resize_box))
    return torch.stack(img_list)
