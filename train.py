import torch
from torch import optim
from torch.utils.data import DataLoader

import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from model import Yolo, YoloLoss
from utils.utils import get_class_names, get_anchors
from utils.data import YoloDataset, yolo_dataset_collate
from config import cfg
from utils.train_utils import get_box_from_out, draw_multi_box_in_tensor


def train(**args):
    # 1.参数设置
    cfg.parse(**args)
    class_names = get_class_names(cfg.class_names_path)
    anchors = get_anchors(cfg.anchors_path)
    num_classes = len(class_names)
    num_anchors = len(anchors[0])
    record_epoch = 0
    print(f'num classes: {num_classes}   num anchors: {num_anchors}')
    
    # 2.模型
    net = Yolo(num_classes, num_anchors)
    if os.path.exists(cfg.pretrain_model):
        pth = torch.load(cfg.pretrain_model, map_location=lambda storage, loc: storage)
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
        print(f'load weight: {cfg.pretrain_model}   epoch:{record_epoch}')
    else:
        print("haven't weight!")
    if torch.cuda.is_available():
        # net = torch.nn.DataParallel(net)
        torch.backends.cudnn.benchmark = True
        net.cuda()
    net.train()

    # 3.损失函数、优化器、lr
    yolo_losses = YoloLoss( # YoloLoss中的get_target()会通过传入的模型预测的out尺寸选择对应的anchor(2, 3, 2)
            anchors.reshape(-1, 2), 
            num_classes,
            (cfg.input_shape[1], cfg.input_shape[0]), 
            cfg.smooth_label, 
            torch.cuda.is_available(), 
            cfg.loss_normalize
    )
    optimizer = optim.Adam(net.parameters(), cfg.lr)
    if cfg.use_cosine_lr:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.T_max, eta_min=cfg.eta_min)
    else:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    
    # 4.数据集
    train_dataset = YoloDataset(cfg.train_datasets_images_path, cfg.train_datasets_labels_path, (cfg.input_shape[0], cfg.input_shape[1]), is_train=True)
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        batch_size=cfg.batch_size, 
        num_workers=cfg.cpu_count, 
        pin_memory=True, 
        drop_last=True, 
        collate_fn=yolo_dataset_collate
    )
    
    # 5.其它（训练无关）
    writer = SummaryWriter(comment=f'-{record_epoch}')

    # 6.训练
    print('start train...')
    for epoch in tqdm(range(cfg.max_epoch), position=1):
        for k, (imgs, labels) in enumerate(tqdm(train_dataloader, position=0)):
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

                writer.add_scalar('Loss/train/batch', loss.detach().item(), epoch * len(train_dataloader) + k)
                return loss
            optimizer.step(closure)

            # debug
            if os.path.exists(cfg.debug):
                import ipdb; ipdb.set_trace()
        lr_scheduler.step()
        with torch.no_grad():
            record_epoch += 1
            if epoch % cfg.every_save == 0:
                if not os.path.exists(cfg.save_folder):
                    os.mkdir(cfg.save_folder)
                torch.save({
                    'model' : net.state_dict(),
                    'epoch' : record_epoch,
                }, cfg.pretrain_model)
            if epoch % cfg.every_valid == 0:
                valid(net, writer=writer, record_epoch=record_epoch)


def valid(net, **kwargs):
    writer = kwargs.get('writer', None)
    record_epoch = kwargs.get('record_epoch', 0)

    val_dataset = YoloDataset(cfg.valid_datasets_images_path, cfg.valid_datasets_labels_path, (cfg.input_shape[0], cfg.input_shape[1]), is_train=False)
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=cfg.batch_size, 
        num_workers=cfg.cpu_count, 
        pin_memory=True, 
        drop_last=False, 
        collate_fn=yolo_dataset_collate, 
        shuffle=True
    )
    
    for imgs, labels in tqdm(val_dataloader):
        imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
        labels = [torch.from_numpy(label).type(torch.FloatTensor) for label in labels]   # (torch.Tensor只允许生成维度一致的，torch.Tensor([[1], [2, 3]])不被允许)
        if torch.cuda.is_available():
            imgs = imgs.cuda()
        
        out = net(imgs)
        batch_boxes = get_box_from_out(out, class_names_path=cfg.class_names_path)
        labels = [label.detach().cpu().numpy() for label in labels]

        if writer:
            mark_imgs = draw_multi_box_in_tensor(imgs.cpu(), labels, class_names=get_class_names(cfg.class_names_path), colors=[[44, 255, 44]], format='cxcywh', resize_box=True)
            mark_imgs = draw_multi_box_in_tensor(mark_imgs, batch_boxes, class_names=get_class_names(cfg.class_names_path), colors=[[44, 44, 255]], format='cxcywh', resize_box=True)
            writer.add_images('Images/valid', mark_imgs, record_epoch)
            writer = None


if __name__ == '__main__':
    train(
        # train_datasets_labels_path = '/home/data/datasets/coco2017/coco/labels/train2017',
        # train_datasets_images_path = '/home/data/datasets/coco2017/coco/images/train2017',
        # class_names_path = './cfg/coco.txt',
    )