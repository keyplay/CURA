def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

class CMAPSS():
    def __init__(self):
        super(CMAPSS, self)
        self.source_ratio = 1
        self.target_ratio = 1
        scenarios = [("FD001", "FD002"), ("FD001", "FD003"), ("FD001", "FD004"), ("FD002", "FD001"), ("FD002", "FD003"), ("FD002", "FD004"),("FD003", "FD001"), ("FD003", "FD002"), ("FD003", "FD004"), ("FD004", "FD001"), ("FD004", "FD002"), ("FD004", "FD003")]

        self.scenarios = [(f"{pair[0]}_{self.source_ratio}", f"{pair[1]}_{self.target_ratio}") for pair in scenarios]

        self.sequence_len = 30
        self.max_rul = 130
        self.shuffle = True
        self.drop_last = True
        self.normalize = False
        self.permute = True # True for lstm

        # model configs
        self.input_channels = 14
        self.evidential = 'non' #org, quantile, unreason
        self.quantiles=[0.5] #0.25, 0.75

        self.kernel_size = 4
        self.stride = 2
        self.dropout = 0.5
        self.num_classes = 6

        # CNN and RESNET features
        self.mid_channels = 32
        self.final_out_channels = 32
        self.features_len = 64 # 150 #128 #64

        # TCN features
        self.tcn_layers = [32, 64]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 5
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 32
        self.lstm_n_layers = 5 #3 for unreason
        self.lstm_bid = True

        # discriminator
        self.disc_hid_dim = 64
        self.hidden_dim = 500
        self.DSKN_disc_hid = 128

class NCMAPSS():
    def __init__(self):
        super(NCMAPSS, self)
        self.source_ratio = 1
        self.target_ratio = 0.6
        scenarios = [("DS01", "DS02"), ("DS01", "DS03"), ("DS02", "DS01"), ("DS02", "DS03"), ("DS03", "DS01"), ("DS03", "DS02")]
        self.scenarios = [(f"{pair[0]}_{self.source_ratio}", f"{pair[1]}_{self.target_ratio}") for pair in scenarios]
        self.sequence_len = 50
        self.max_rul = 88
        self.shuffle = True
        self.drop_last = True
        self.normalize = False
        self.permute = True

        # model configs
        self.input_channels = 20
        self.evidential = 'unreason' 
        self.quantiles= [0.5]

        self.kernel_size = 3
        self.stride = 1
        self.dropout = 0.1
        self.num_classes = 6

        # CNN and RESNET features
        self.mid_channels = 20
        self.final_out_channels = 20
        self.features_len = 128 #160 for cnn, 128 for lstm, 64 for tcn, 

        # TCN features
        self.tcn_layers = [32, 64] #[75, 150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 5
        self.tcn_dropout = 0.1

        # lstm features
        self.lstm_hid = 64
        self.lstm_n_layers = 1
        self.lstm_bid = True

        # discriminator
        self.disc_hid_dim = 64
        self.hidden_dim = 500
        self.DSKN_disc_hid = 128

