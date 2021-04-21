import warnings

class Config:

    # dataset path
    train_datasets_images_path = '/home/data/datasets/coco2017/coco/images/train2017' # coco
    train_datasets_labels_path = '/home/data/datasets/coco2017/coco/labels/train2017'
    valid_datasets_images_path = '/home/data/datasets/coco2017/coco/images/val2017'
    valid_datasets_labels_path = '/home/data/datasets/coco2017/coco/labels/val2017'
    class_names_path = './cfg/coco.txt'
    # train_datasets_images_path = '/home/data/datasets/bdd100k/images/train' # bdd100k
    # train_datasets_labels_path = '/home/data/datasets/bdd100k/labels/train'
    # valid_datasets_images_path = '/home/data/datasets/bdd100k/images/val'
    # valid_datasets_labels_path = '/home/data/datasets/bdd100k/labels/val'
    # class_names_path = './cfg/bdd100k.txt'
    # train_datasets_images_path = '/home/data/datasets/swucar/images/train' # swucar
    # train_datasets_labels_path = '/home/data/datasets/swucar/labels/train'
    # valid_datasets_images_path = '/home/data/datasets/swucar/images/val'
    # valid_datasets_labels_path = '/home/data/datasets/swucar/labels/val'
    # class_names_path = './cfg/swucar.txt'

    # pretrain
    save_folder = 'weights/'
    pretrain_model = save_folder + 'yolov4-tiny.pt'
    
    # net
    input_shape = (416, 416)
    max_epoch = 100
    batch_size = 64
    cpu_count = 8
    lr = 1e-3
    T_max = 5
    eta_min = 1e-5
    step_size = 1
    gamma = 0.92
    anchors_path = './cfg/anchors.txt'

    # train
    use_gpu = True
    load_part_weight = False
    loss_normalize = False
    use_cosine_lr = False
    smooth_label = 0
    # valid
    threshold = 0.5
    
    # other
    debug = '/tmp/debug'
    log_path = './log'
    every_save = 1
    every_valid = 10


    def parse(self, **kwargs):
        """
        update configuration by kwargs.
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Waning: cfg has no attribute %s" % k)
            setattr(self, k, v)

        print('User config:')
        for k, v in self.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))


cfg = Config()