import torch
import os
import numpy as np
import cv2

from model import Yolo
from utils import get_class_names, get_anchors, DecodeBox, letterbox_image, non_max_suppression, letter_correct_boxes


class Detect:

    def __init__(self, weights_path='./weights/yolov4-tiny.pth', confidence=0.5, iou=0.3, class_names_path='./cfg/coco.txt', anchors_path='./cfg/anchors.txt', is_letterbox_image=False):
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
        pth = torch.load(opt.pretrain_model, map_location=lambda storage, loc: storage)
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
        out = self._postprocess(out)
        return out
    
    def _preprocess(self, img):
        self.image_shape = np.array(img.shape[:2])
        if self.is_letterbox_image:
            # cv2.imwrite('/tmp/test1.jpg', img)
            img = letterbox_image(img, (self.model_image_size[1],self.model_image_size[0]))
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
        from utils import draw_multi_box
        # 设置画框颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        img = draw_multi_box(img, boxes, class_names=self.class_names, colors=colors)
        return img


if __name__ == '__main__':
    import cv2
    import numpy as np
    from config import opt
    # detect = Detect(is_letterbox_image=True)
    # detect = Detect(class_names_path=opt.class_names_path, anchors_path=opt.anchors_path)
    detect = Detect(class_names_path='./cfg/swucar.txt', anchors_path=opt.anchors_path)

    img = cv2.imread('/tmp/test.jpg')

    boxes = detect(img)
    print(boxes)
    print(boxes.shape)

    img = detect.draw_boxes(img, boxes)
    cv2.imwrite('/tmp/test_out.jpg', img)