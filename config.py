import warnings

class Config(object):

    # dataset path
    datasets_path = '/home/data/datasets/coco2017/coco'
    class_names_path = './cfg/coco.txt'
    anchors_path = './cfg/anchors.txt'

    # pretrain
    save_folder = 'weights/'
    pretrain_model = save_folder + 'yolov4-tiny.pth'
    
    # net
    input_shape = (416, 416)
    max_epoch = 100
    batch_size = 64
    cpu_count = 16
    lr = 1e-3
    T_max = 5
    eta_min = 1e-5
    step_size = 1
    gamma = 0.92

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
    every_valid = 1


    def parse(self, **kwargs):
        """
        update configuration by kwargs.
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Waning: opt has no attribute %s" % k)
            setattr(self, k, v)

        print('User config:')
        for k, v in self.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))


opt = Config()