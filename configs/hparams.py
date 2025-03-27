## The cuurent hyper-parameters values are not necessarily the best ones for a specific risk.
def get_hparams_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

class CMAPSS():
    def __init__(self):
        super(CMAPSS, self).__init__()
        self.train_params = {
            'num_epochs': 100, 
            'batch_size': 256,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0, #0.5
            'pretrain': False,
            'save': True

        }
        self.alg_hparams = {
            'NO_ADAPT': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1, 'pretrain_epochs': 20},
            'TARGET_ONLY': {'learning_rate': 1e-3, 'trg_cls_loss_wt': 1},
            "CURA": {
                "learning_rate": 5e-5,
                'init_unc_threshold': 0.1,
                'max_unc_threshold':0.5,
                'unc_threshold_step':0.1,
                'stop_ratio':0.05,
                'threshold_update_interval':20
            },
        }


class NCMAPSS():
    def __init__(self):
        super(NCMAPSS, self).__init__()
        self.train_params = {
            'num_epochs': 150, 
            'batch_size': 256,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5,
            'pretrain': False,
            'save': True
        }
        self.alg_hparams = {
            'NO_ADAPT': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1, 'pretrain_epochs': 40},
            'TARGET_ONLY': {'learning_rate': 1e-3, 'trg_cls_loss_wt': 1},
            "CURA": {
                "learning_rate": 5e-5,
                'init_unc_threshold': 0.2,
                'max_unc_threshold':0.5,
                'unc_threshold_step':0.1,
                'stop_ratio':0.05,
                'threshold_update_interval':20
            },
        }

