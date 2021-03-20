import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
import cv2
import os

from utils.utils import convert_cxcywh_to_x1y1x2y2, convert_x1y1x2y2_to_cxcywh


class YoloDataset(Dataset):
    def __init__(self, datasets_path, image_size, is_train=True):
        super().__init__()

        self.files_name = [x.split('.')[0] for x in os.listdir(f'{datasets_path}/labels/{"train" if is_train else "val"}2017') if x.endswith('.txt')]
        self.labels_path = [f'{datasets_path}/labels/{"train" if is_train else "val"}2017/{x}.txt' for x in self.files_name]
        self.images_path = [f'{datasets_path}/images/{"train" if is_train else "val"}2017/{x}.jpg' for x in self.files_name]
        self.image_size = image_size
        self.flag = True
        self.is_train = is_train

    def __len__(self):
        return len(self.files_name)

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, box, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        """实时数据增强的随机预处理"""
        iw, ih = image.size
        h, w = input_shape
        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            # 调整目标框坐标
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] / iw * nw + dx
                box[:, [1, 3]] = box[:, [1, 3]] / ih * nh + dy
                box[:, 0: 2][box[:, 0: 2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]  # 保留有效框

            return image_data, box

        # 调整图片大小
        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # 放置图片
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h),
                              (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
        new_image.paste(image, (dx, dy))
        image = new_image

        # 是否翻转图片
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 色域变换
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255

        # 调整目标框坐标
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] / iw * nw + dx
            box[:, [1, 3]] = box[:, [1, 3]] / ih * nh + dy
            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # 保留有效框
        return image_data, box

    def __getitem__(self, index):
        img = Image.open(self.images_path[index])
        w, h = img.size
        with open(self.labels_path[index], 'r') as f:
            boxes = np.array([list(map(float, line.strip().split())) for line in f.readlines()])
        boxes[:, [1,3]] *= w  # 0~1 -> 0~w
        boxes[:, [2,4]] *= h
        # boxes = boxes.astype(int)
        boxes = boxes[:, [1,2,3,4,0]]   # (classes, cx, cy, w, h) -> (cx, cy, w, h, classes)
        boxes = convert_cxcywh_to_x1y1x2y2(boxes)  # (cx, cy, w, h, classes) -> (x1, y1, x2, y2, classes)

        img, boxes = self.get_random_data(img, boxes, self.image_size[0:2], random=self.is_train)

        if len(boxes) != 0:
            x = np.array(boxes[:, :4], dtype=np.float32)    # 0~255 -> 0.~1.
            x[:, [0, 2]] /= self.image_size[1]
            x[:, [1, 3]] /= self.image_size[0]

            x[:, :4] = np.clip(x[:, :4], 0., 1.)
            x = convert_x1y1x2y2_to_cxcywh(x) # (x1, y1, x2, y2, classes) -> (cx, cy, w, h, classes)
            boxes = np.concatenate([x, boxes[:, -1:]], axis=-1)

        img = np.array(img, dtype=np.float32)

        imgs = np.transpose(img / 255.0, (2, 0, 1))
        labels = np.array(boxes, dtype=np.float32)
        return imgs, labels


# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    return images, bboxes
