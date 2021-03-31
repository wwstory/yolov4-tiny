# %%
import os
if os.path.abspath('.').split('/')[-1] != 'yolov4-tiny':
    os.chdir('..')
print(os.path.abspath('.'))
# 1.模型
# %%
from model.CSPdarknet53_tiny import CSPDarkNet
csp = CSPDarkNet()

# 2.模型参数迁移
# %%
import torch
from model.yolov4_tiny import Yolo
yolo = Yolo(80, 3)
img = torch.rand(32, 3, 416, 416)
out = yolo(img)
print(out[0].shape, out[1].shape)

# %%
pretrained_weight = '/home/dejiang/data/pretrain/yolov4_tiny_weights_coco.pth'

map_keys = {
    # old layer -> new layer
    'resblock_body1' : 'resblock1',
    'resblock_body2' : 'resblock2',
    'resblock_body3' : 'resblock3',
    'conv_for_P5' : 'conv1',
    'yolo_headP5' : 'yolo_head1.head',
    'upsample.upsample' : 'upsample',
    'yolo_headP4' : 'yolo_head2.head',
}

# %%
import torch
pretrained_dict = torch.load(pretrained_weight, map_location=lambda storage, loc: storage)
model_dict = yolo.state_dict()

new_pretrained_dict = pretrained_dict.copy()
for k, v in pretrained_dict.items():
    for k2, v2 in map_keys.items():
        if k2 in k:
            new_pretrained_dict[k.replace(k2, v2)] = new_pretrained_dict.pop(k)
            print('======')
            print(k, '--->', k.replace(k2, v2))
            print('old:', v.shape,' new:', model_dict[k.replace(k2, v2)].shape)
            print(new_pretrained_dict[k.replace(k2, v2)])
            print('------')
            print(pretrained_dict[k])

# %%
new_pretrained_dict.keys()

yolo.load_state_dict(new_pretrained_dict)
torch.save(yolo.state_dict(), './weights/yolov4_tiny-coco.pth')
print('sucessfully converted old_pth to new_pth!')

