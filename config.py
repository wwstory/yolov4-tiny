import warnings

class Config(object):

    # dataset path
    root_path = '/home/dejiang/datasets/'

    # dataset, pretrain
    save_folder = 'weights/'
    pretrain_model = save_folder + 'yolov4-tiny.pth'
    
    # net
    max_epoch = 100
    batch_size = 32
    cpu_count = 8
    lr = 1e-3
    betas = (0.9, 0.999)
    eps = 1e-8
    weight_decay = 0    # 5e-4
    momentum = 0.9

    # train
    use_gpu = True
    load_part_weight = False
    # valid
    threshold = 0.5
    
    # other
    debug = '/tmp/debug'
    every_save = 1
    every_valid = 1
    num_vaild = 100
    every_log = 1


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