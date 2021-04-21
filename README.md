# yolov4-tiny
pytorch implementation of yolov4-tiny

# requirements

```
torch
torchvision
opencv-python
Pillow
matplotlib
tqdm
```

# train

## add datasets

```
数据集按coco2017数据集形式存放
coco
├── labels/
│   ├── train2017/
│   │   └── 0001.txt (classes cx cy w h)
│   └── val2017/
└── images/
       ├── train2017/
       │   └── 0001.jpg
       └── val2017/
```

## set datasets

修改`config.py`中的`train_datasets_images_path`、`train_datasets_labels_path`、`valid_datasets_images_path`、`valid_datasets_labels_path`。设置类别`class_names_path`

添加类别文件`./cfg/coco.txt`（一行一个类别名）。

## add weights

下载预训练模型并修改名称为`yolov4-tiny.pt`，放在`./weights/`目录下。（否则，将会重新训练）

## start train

```sh
python3 train.py
```

## test

准备一张图片放在`/tmp/test.jpg`。

修改`detect.py`中的配置为想要识别的数据集类别：

```python
detect = Detect(weights_path='./weights/yolov4-tiny.pt', 
                class_names_path='./cfg/coco.txt', 
                # is_letterbox_image=True
)
```

执行：

```sh
python3 detect.py
```

---

**ref:** [yolov4-tiny-pytorch][101]

[101]: https://github.com/bubbliiiing/yolov4-tiny-pytorch