import torch
import torch.nn as nn

from model.CSPdarknet53_tiny import BasicConv, CSPDarkNet



#---------------------------------------------------#
#   最后获得yolov4的输出
#---------------------------------------------------#
class YoloHead(nn.Module):
    def __init__(self, in_filters, filters_list):
        super().__init__()
        self.head = nn.Sequential(
            BasicConv(in_filters, filters_list[0], 3),
            nn.Conv2d(filters_list[0], filters_list[1], 1),
        )

    def forward(self, x):
        return self.head(x)

#---------------------------------------------------#
#   yolo
#---------------------------------------------------#
class Yolo(nn.Module):
    def __init__(self, num_classes=80, num_anchors=3):
        super().__init__()
        # backbone
        self.backbone = CSPDarkNet()

        # neck1
        self.conv1 = BasicConv(512, 256, 1)
        # head1
        self.yolo_head1 = YoloHead(256, [512, num_anchors * (5 + num_classes)])

        # neck2
        self.upsample = nn.Sequential(
            BasicConv(256, 128, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        # head2
        self.yolo_head2 = YoloHead(384, [256, num_anchors * (5 + num_classes)])

    def forward(self, x):
        #---------------------------------------------------#
        #   生成CSPdarknet53_tiny的主干模型
        #   feat1的shape为26,26,256
        #   feat2的shape为13,13,512
        #---------------------------------------------------#
        feat1, feat2 = self.backbone(x)
        # 13,13,512 -> 13,13,256
        x1 = self.conv1(feat2)
        # 13,13,256 -> 13,13,512 -> 13,13,255
        out1 = self.yolo_head1(x1)

        # 13,13,256 -> 13,13,128 -> 26,26,128
        x2 = self.upsample(x1)
        # 26,26,256 + 26,26,128 -> 26,26,384
        x2 = torch.cat([x2, feat1], axis=1)

        # 26,26,384 -> 26,26,256 -> 26,26,255
        out2 = self.yolo_head2(x2)
        
        return out1, out2
