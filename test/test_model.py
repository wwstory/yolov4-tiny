# %%
import os
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
key0 = [
    'backbone.conv1.conv.weight',
    'backbone.conv1.bn.weight',
    'backbone.conv1.bn.bias',
    'backbone.conv1.bn.running_mean',
    'backbone.conv1.bn.running_var',
    'backbone.conv1.bn.num_batches_tracked',
    'backbone.conv2.conv.weight',
    'backbone.conv2.bn.weight',
    'backbone.conv2.bn.bias',
    'backbone.conv2.bn.running_mean',
    'backbone.conv2.bn.running_var',
    'backbone.conv2.bn.num_batches_tracked',
    'backbone.resblock_body1.conv1.conv.weight',
    'backbone.resblock_body1.conv1.bn.weight',
    'backbone.resblock_body1.conv1.bn.bias',
    'backbone.resblock_body1.conv1.bn.running_mean',
    'backbone.resblock_body1.conv1.bn.running_var',
    'backbone.resblock_body1.conv1.bn.num_batches_tracked',
    'backbone.resblock_body1.conv2.conv.weight',
    'backbone.resblock_body1.conv2.bn.weight',
    'backbone.resblock_body1.conv2.bn.bias',
    'backbone.resblock_body1.conv2.bn.running_mean',
    'backbone.resblock_body1.conv2.bn.running_var',
    'backbone.resblock_body1.conv2.bn.num_batches_tracked',
    'backbone.resblock_body1.conv3.conv.weight',
    'backbone.resblock_body1.conv3.bn.weight',
    'backbone.resblock_body1.conv3.bn.bias',
    'backbone.resblock_body1.conv3.bn.running_mean',
    'backbone.resblock_body1.conv3.bn.running_var',
    'backbone.resblock_body1.conv3.bn.num_batches_tracked',
    'backbone.resblock_body1.conv4.conv.weight',
    'backbone.resblock_body1.conv4.bn.weight',
    'backbone.resblock_body1.conv4.bn.bias',
    'backbone.resblock_body1.conv4.bn.running_mean',
    'backbone.resblock_body1.conv4.bn.running_var',
    'backbone.resblock_body1.conv4.bn.num_batches_tracked',
    'backbone.resblock_body2.conv1.conv.weight',
    'backbone.resblock_body2.conv1.bn.weight',
    'backbone.resblock_body2.conv1.bn.bias',
    'backbone.resblock_body2.conv1.bn.running_mean',
    'backbone.resblock_body2.conv1.bn.running_var',
    'backbone.resblock_body2.conv1.bn.num_batches_tracked',
    'backbone.resblock_body2.conv2.conv.weight',
    'backbone.resblock_body2.conv2.bn.weight',
    'backbone.resblock_body2.conv2.bn.bias',
    'backbone.resblock_body2.conv2.bn.running_mean',
    'backbone.resblock_body2.conv2.bn.running_var',
    'backbone.resblock_body2.conv2.bn.num_batches_tracked',
    'backbone.resblock_body2.conv3.conv.weight',
    'backbone.resblock_body2.conv3.bn.weight',
    'backbone.resblock_body2.conv3.bn.bias',
    'backbone.resblock_body2.conv3.bn.running_mean',
    'backbone.resblock_body2.conv3.bn.running_var',
    'backbone.resblock_body2.conv3.bn.num_batches_tracked',
    'backbone.resblock_body2.conv4.conv.weight',
    'backbone.resblock_body2.conv4.bn.weight',
    'backbone.resblock_body2.conv4.bn.bias',
    'backbone.resblock_body2.conv4.bn.running_mean',
    'backbone.resblock_body2.conv4.bn.running_var',
    'backbone.resblock_body2.conv4.bn.num_batches_tracked',
    'backbone.resblock_body3.conv1.conv.weight',
    'backbone.resblock_body3.conv1.bn.weight',
    'backbone.resblock_body3.conv1.bn.bias',
    'backbone.resblock_body3.conv1.bn.running_mean',
    'backbone.resblock_body3.conv1.bn.running_var',
    'backbone.resblock_body3.conv1.bn.num_batches_tracked',
    'backbone.resblock_body3.conv2.conv.weight',
    'backbone.resblock_body3.conv2.bn.weight',
    'backbone.resblock_body3.conv2.bn.bias',
    'backbone.resblock_body3.conv2.bn.running_mean',
    'backbone.resblock_body3.conv2.bn.running_var',
    'backbone.resblock_body3.conv2.bn.num_batches_tracked',
    'backbone.resblock_body3.conv3.conv.weight',
    'backbone.resblock_body3.conv3.bn.weight',
    'backbone.resblock_body3.conv3.bn.bias',
    'backbone.resblock_body3.conv3.bn.running_mean',
    'backbone.resblock_body3.conv3.bn.running_var',
    'backbone.resblock_body3.conv3.bn.num_batches_tracked',
    'backbone.resblock_body3.conv4.conv.weight',
    'backbone.resblock_body3.conv4.bn.weight',
    'backbone.resblock_body3.conv4.bn.bias',
    'backbone.resblock_body3.conv4.bn.running_mean',
    'backbone.resblock_body3.conv4.bn.running_var',
    'backbone.resblock_body3.conv4.bn.num_batches_tracked',
    'backbone.conv3.conv.weight',
    'backbone.conv3.bn.weight',
    'backbone.conv3.bn.bias',
    'backbone.conv3.bn.running_mean',
    'backbone.conv3.bn.running_var',
    'backbone.conv3.bn.num_batches_tracked',
    'conv_for_P5.conv.weight',
    'conv_for_P5.bn.weight',
    'conv_for_P5.bn.bias',
    'conv_for_P5.bn.running_mean',
    'conv_for_P5.bn.running_var',
    'conv_for_P5.bn.num_batches_tracked',
    'yolo_headP5.0.conv.weight',
    'yolo_headP5.0.bn.weight',
    'yolo_headP5.0.bn.bias',
    'yolo_headP5.0.bn.running_mean',
    'yolo_headP5.0.bn.running_var',
    'yolo_headP5.0.bn.num_batches_tracked',
    'yolo_headP5.1.weight',
    'yolo_headP5.1.bias',
    'upsample.upsample.0.conv.weight',
    'upsample.upsample.0.bn.weight',
    'upsample.upsample.0.bn.bias',
    'upsample.upsample.0.bn.running_mean',
    'upsample.upsample.0.bn.running_var',
    'upsample.upsample.0.bn.num_batches_tracked',
    'yolo_headP4.0.conv.weight',
    'yolo_headP4.0.bn.weight',
    'yolo_headP4.0.bn.bias',
    'yolo_headP4.0.bn.running_mean',
    'yolo_headP4.0.bn.running_var',
    'yolo_headP4.0.bn.num_batches_tracked',
    'yolo_headP4.1.weight',
    'yolo_headP4.1.bias'
]

key1 = [
    'backbone.conv1.conv.weight',
    'backbone.conv1.bn.weight',
    'backbone.conv1.bn.bias',
    'backbone.conv1.bn.running_mean',
    'backbone.conv1.bn.running_var',
    'backbone.conv1.bn.num_batches_tracked',
    'backbone.conv2.conv.weight',
    'backbone.conv2.bn.weight',
    'backbone.conv2.bn.bias',
    'backbone.conv2.bn.running_mean',
    'backbone.conv2.bn.running_var',
    'backbone.conv2.bn.num_batches_tracked',
    'backbone.resblock1.conv1.conv.weight',
    'backbone.resblock1.conv1.bn.weight',
    'backbone.resblock1.conv1.bn.bias',
    'backbone.resblock1.conv1.bn.running_mean',
    'backbone.resblock1.conv1.bn.running_var',
    'backbone.resblock1.conv1.bn.num_batches_tracked',
    'backbone.resblock1.conv2.conv.weight',
    'backbone.resblock1.conv2.bn.weight',
    'backbone.resblock1.conv2.bn.bias',
    'backbone.resblock1.conv2.bn.running_mean',
    'backbone.resblock1.conv2.bn.running_var',
    'backbone.resblock1.conv2.bn.num_batches_tracked',
    'backbone.resblock1.conv3.conv.weight',
    'backbone.resblock1.conv3.bn.weight',
    'backbone.resblock1.conv3.bn.bias',
    'backbone.resblock1.conv3.bn.running_mean',
    'backbone.resblock1.conv3.bn.running_var',
    'backbone.resblock1.conv3.bn.num_batches_tracked',
    'backbone.resblock1.conv4.conv.weight',
    'backbone.resblock1.conv4.bn.weight',
    'backbone.resblock1.conv4.bn.bias',
    'backbone.resblock1.conv4.bn.running_mean',
    'backbone.resblock1.conv4.bn.running_var',
    'backbone.resblock1.conv4.bn.num_batches_tracked',
    'backbone.resblock2.conv1.conv.weight',
    'backbone.resblock2.conv1.bn.weight',
    'backbone.resblock2.conv1.bn.bias',
    'backbone.resblock2.conv1.bn.running_mean',
    'backbone.resblock2.conv1.bn.running_var',
    'backbone.resblock2.conv1.bn.num_batches_tracked',
    'backbone.resblock2.conv2.conv.weight',
    'backbone.resblock2.conv2.bn.weight',
    'backbone.resblock2.conv2.bn.bias',
    'backbone.resblock2.conv2.bn.running_mean',
    'backbone.resblock2.conv2.bn.running_var',
    'backbone.resblock2.conv2.bn.num_batches_tracked',
    'backbone.resblock2.conv3.conv.weight',
    'backbone.resblock2.conv3.bn.weight',
    'backbone.resblock2.conv3.bn.bias',
    'backbone.resblock2.conv3.bn.running_mean',
    'backbone.resblock2.conv3.bn.running_var',
    'backbone.resblock2.conv3.bn.num_batches_tracked',
    'backbone.resblock2.conv4.conv.weight',
    'backbone.resblock2.conv4.bn.weight',
    'backbone.resblock2.conv4.bn.bias',
    'backbone.resblock2.conv4.bn.running_mean',
    'backbone.resblock2.conv4.bn.running_var',
    'backbone.resblock2.conv4.bn.num_batches_tracked',
    'backbone.resblock3.conv1.conv.weight',
    'backbone.resblock3.conv1.bn.weight',
    'backbone.resblock3.conv1.bn.bias',
    'backbone.resblock3.conv1.bn.running_mean',
    'backbone.resblock3.conv1.bn.running_var',
    'backbone.resblock3.conv1.bn.num_batches_tracked',
    'backbone.resblock3.conv2.conv.weight',
    'backbone.resblock3.conv2.bn.weight',
    'backbone.resblock3.conv2.bn.bias',
    'backbone.resblock3.conv2.bn.running_mean',
    'backbone.resblock3.conv2.bn.running_var',
    'backbone.resblock3.conv2.bn.num_batches_tracked',
    'backbone.resblock3.conv3.conv.weight',
    'backbone.resblock3.conv3.bn.weight',
    'backbone.resblock3.conv3.bn.bias',
    'backbone.resblock3.conv3.bn.running_mean',
    'backbone.resblock3.conv3.bn.running_var',
    'backbone.resblock3.conv3.bn.num_batches_tracked',
    'backbone.resblock3.conv4.conv.weight',
    'backbone.resblock3.conv4.bn.weight',
    'backbone.resblock3.conv4.bn.bias',
    'backbone.resblock3.conv4.bn.running_mean',
    'backbone.resblock3.conv4.bn.running_var',
    'backbone.resblock3.conv4.bn.num_batches_tracked',
    'backbone.conv3.conv.weight',
    'backbone.conv3.bn.weight',
    'backbone.conv3.bn.bias',
    'backbone.conv3.bn.running_mean',
    'backbone.conv3.bn.running_var',
    'backbone.conv3.bn.num_batches_tracked',
    'conv1.conv.weight',
    'conv1.bn.weight',
    'conv1.bn.bias',
    'conv1.bn.running_mean',
    'conv1.bn.running_var',
    'conv1.bn.num_batches_tracked',
    'yolo_head1.head.0.conv.weight',
    'yolo_head1.head.0.bn.weight',
    'yolo_head1.head.0.bn.bias',
    'yolo_head1.head.0.bn.running_mean',
    'yolo_head1.head.0.bn.running_var',
    'yolo_head1.head.0.bn.num_batches_tracked',
    'yolo_head1.head.1.weight',
    'yolo_head1.head.1.bias',
    'upsample.0.conv.weight',
    'upsample.0.bn.weight',
    'upsample.0.bn.bias',
    'upsample.0.bn.running_mean',
    'upsample.0.bn.running_var',
    'upsample.0.bn.num_batches_tracked',
    'yolo_head2.head.0.conv.weight',
    'yolo_head2.head.0.bn.weight',
    'yolo_head2.head.0.bn.bias',
    'yolo_head2.head.0.bn.running_mean',
    'yolo_head2.head.0.bn.running_var',
    'yolo_head2.head.0.bn.num_batches_tracked',
    'yolo_head2.head.1.weight',
    'yolo_head2.head.1.bias'
]

# %%
map_keys = {
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
pretrained_dict = torch.load('/home/dejiang/data/pretrain/yolov4_tiny_weights_coco.pth', map_location=lambda storage, loc: storage)
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

