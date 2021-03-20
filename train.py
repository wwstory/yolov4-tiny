# coding=utf-8
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.backends import cudnn

import os
from tqdm import tqdm

from model import Yolo, YoloLoss
from utils import YoloDataset, yolo_dataset_collate, get_class_names, get_anchors
from config import opt


def train(**args):
    # 1.参数设置
    opt.parse(**args)
    class_names = get_class_names(opt.class_names_path)
    anchors = get_anchors(opt.anchors_path)
    num_classes = len(class_names)
    num_anchors = len(anchors[0])
    record_epoch = 0
    print(f'num classes: {num_classes}   num anchors: {num_anchors}')
    
    # 2.模型
    net = Yolo(num_classes, num_anchors)
    if os.path.exists(opt.pretrain_model):
        pth = torch.load(opt.pretrain_model, map_location=lambda storage, loc: storage)
        if 'model' in pth:
            record_epoch = pth['epoch']
            pretrained_dict = pth['model']
        else:
            pretrained_dict = pth
        # load part weight
        model_dict = net.state_dict()
        changed_keys = [k for k,v in pretrained_dict.items() if k in model_dict and v.shape != model_dict[k].shape]
        if changed_keys:
            print('model changed!')
            print('--->', changed_keys)
            record_epoch = 0
        pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        # pretrained_dict.pop('yolo_head1.head.1.weight')
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
        print(f'load weight: {opt.pretrain_model}   epoch:{record_epoch}')
    else:
        print("haven't weight!")
    if torch.cuda.is_available():
        net.cuda()
        cudnn.benchmark = True
    net.train()

    # 3.损失函数、优化器、lr
    yolo_losses = YoloLoss( # YoloLoss中的get_target()会通过传入的模型预测的out尺寸选择对应的anchor(2, 3, 2)
            anchors.reshape(-1, 2), 
            num_classes,
            (opt.input_shape[1], opt.input_shape[0]), 
            opt.smooth_label, 
            torch.cuda.is_available(), 
            opt.loss_normalize
    )
    optimizer = optim.Adam(net.parameters(), opt.lr)
    if opt.use_cosine_lr:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.T_max, eta_min=opt.eta_min)
    else:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.gamma)
    
    # 4.数据集
    train_dataset = YoloDataset(opt.datasets_path, (opt.input_shape[0], opt.input_shape[1]), is_train=True)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=opt.batch_size, num_workers=opt.cpu_count, pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)
    
    # 5.训练
    print('start train...')
    loss_list = []
    for epoch in tqdm(range(opt.max_epoch), position=1):
        l_tqdm = tqdm(train_dataloader, position=0)
        for _, (imgs, labels) in enumerate(l_tqdm):
            imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
            labels = [torch.from_numpy(label).type(torch.FloatTensor) for label in labels]   # (torch.Tensor只允许生成维度一致的，torch.Tensor([[1], [2, 3]])不被允许)
            if torch.cuda.is_available():
                imgs = imgs.cuda()
            
            def closure():
                optimizer.zero_grad()
                out = net(imgs)
                losses = []
                num_pos_all = 0

                for i in range(2):
                    loss_item, num_pos = yolo_losses(out[i], labels)
                    losses.append(loss_item)
                    num_pos_all += num_pos
                loss = sum(losses) / num_pos_all
                loss.backward()

                loss_list.append(loss.detach().item())
                return loss
            optimizer.step(closure)

            # debug
            l_tqdm.set_description(f'-> epoch:{epoch} loss:{round(loss_list[-1], 7)}')
            if os.path.exists(opt.debug):
                import ipdb; ipdb.set_trace()
        record_epoch += 1
        lr_scheduler.step()
        with torch.no_grad():
            if epoch % opt.every_save == 0:
                if not os.path.exists(opt.save_folder):
                    os.mkdir(opt.save_folder)
                torch.save({
                    'model' : net.state_dict(),
                    'epoch' : record_epoch,
                }, opt.pretrain_model)
            if epoch % opt.every_valid == 0:
                import time, pickle
                prediction_list, labels_list, img_label_pred = valid(net)
                log = {'epoch': record_epoch, 'loss': loss_list, 'valid_pred_label': (prediction_list, labels_list), 'img_label_pred': img_label_pred}
                if not os.path.exists(opt.log_path):
                    os.mkdir(opt.log_path)
                with open(f'{opt.log_path}/log-{time.strftime("%Y-%m-%d_%H-%M-%S")}-{record_epoch}.pkl', 'wb') as f:
                    pickle.dump(log, f)


def valid(net):
    import numpy as np
    from utils import get_box_from_out
    val_dataset = YoloDataset(opt.datasets_path, (opt.input_shape[0], opt.input_shape[1]), is_train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=opt.cpu_count, pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)
    
    prediction_list = []
    labels_list = []
    rand_choice_batch = 0
    img_label_pred = None
    for i, (imgs, labels) in enumerate(tqdm(val_dataloader)):
        imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
        labels = [torch.from_numpy(label).type(torch.FloatTensor) for label in labels]   # (torch.Tensor只允许生成维度一致的，torch.Tensor([[1], [2, 3]])不被允许)
        if torch.cuda.is_available():
            imgs = imgs.cuda()
        
        out = net(imgs)
        prediction = get_box_from_out(out, class_names_path=opt.class_names_path)
        labels = [label.detach().cpu().numpy() for label in labels]

        if i == rand_choice_batch:
            img_label_pred = {
                'img' : np.transpose(imgs.detach().cpu().numpy()*255, (0, 2, 3, 1)).astype('uint8'),
                'label' : labels,
                'pred' : prediction,
            }

        prediction_list.append(prediction)
        labels_list.append(labels)
    return prediction_list, labels_list, img_label_pred

if __name__ == '__main__':
    '''
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
    '''
    train(
        datasets_path = '/home/data/datasets/coco2017/coco',
        class_names_path = './cfg/swucar.txt'
    )