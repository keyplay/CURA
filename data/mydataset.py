import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import os

class MyDataset(Dataset):
    def __init__(self, data, labels):
        """Reads source and target sequences from processing file ."""
        self.input_tensor = (torch.from_numpy(data)).float()

        self.label = (torch.torch.FloatTensor(labels))
        self.num_total_seqs = len(self.input_tensor)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        input_seq = self.input_tensor[index]
        input_labels = self.label[index]
        return input_seq, input_labels

    def __len__(self):
        return self.num_total_seqs


def create_dataset(data, batch_size, shuffle, drop_last):
    trainX, validX, testX, trainY, validY, testY = data
    train_dl = DataLoader(MyDataset(trainX, trainY), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    valid_dl = DataLoader(MyDataset(validX, validY), batch_size=10, shuffle=False, drop_last=False)
    test_dl = DataLoader(MyDataset(testX, testY), batch_size=10, shuffle=False, drop_last=False)
    return train_dl, valid_dl, test_dl


def create_dataset_full(data, batch_size=10, shuffle=True, drop_last=True):
    trainX, testX, trainY, testY = data
    print('training data size: ', trainX.shape, 'test data size: ', testX.shape)
    train_dl = DataLoader(MyDataset(trainX, trainY), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    test_dl = DataLoader(MyDataset(testX, testY), batch_size=10, shuffle=False, drop_last=False)
    return train_dl, test_dl

class Load_Dataset(Dataset):
    def __init__(self, dataset, dataset_configs):
        super().__init__()
        self.num_channels = dataset_configs.input_channels
        
        # Load samples
        x_data = dataset["samples"]

        # Load labels
        y_data = dataset.get("labels")
        if y_data is not None and isinstance(y_data, np.ndarray):
            y_data = torch.from_numpy(y_data)
        
        # Convert to torch tensor
        if isinstance(x_data, np.ndarray):
            x_data = torch.from_numpy(x_data)
        
        # Check samples dimensions.
        # The dimension of the data is expected to be (N, C, L)
        # where N is the #samples, C: #channels, and L is the sequence length
        if len(x_data.shape) == 2:
            x_data = x_data.unsqueeze(1)
        elif len(x_data.shape) == 3 and x_data.shape[1] != self.num_channels:
            x_data = x_data.transpose(1, 2)
        
        
        if dataset_configs.permute:
            x_data = x_data.transpose(1, 2)
        
        self.x_data = x_data.float()
        self.y_data = y_data.float() if y_data is not None else None
        self.len = x_data.shape[0]
         

    def __getitem__(self, index):
        x = self.x_data[index]
        y = self.y_data[index] if self.y_data is not None else None
        return x, y

    def __len__(self):
        return self.len


def data_generator(data_path, domain_id, dataset_configs, hparams, dtype):
    # loading dataset file from path
    dataset_file = torch.load(os.path.join(data_path, f"{dtype}_{domain_id}.pt"))
    print(dataset_file['samples'].shape, f"{dtype}_{domain_id}.pt")
    # Loading datasets
    dataset = Load_Dataset(dataset_file, dataset_configs)
    print(len(dataset))
    if dtype == "test":  # you don't need to shuffle or drop last batch while testing
        shuffle  = False
        drop_last = False
    else:
        shuffle = dataset_configs.shuffle
        drop_last = dataset_configs.drop_last

    # Dataloaders
    data_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                              batch_size=hparams["batch_size"],
                                              shuffle=shuffle, 
                                              drop_last=drop_last, 
                                              num_workers=0)

    return data_loader
   
class DA_TripletDataset_close(Dataset):
    def __init__(self, src_data, src_labels, src_cluster, tgt_data, tgt_labels, tgt_cluster):
        #src_labels, tgt_labels = np.round(100*src_labels), np.round(100*tgt_labels)

        self.src_data = src_data
        self.src_labels = src_labels
        self.src_cluster = torch.LongTensor(src_cluster)
        self.tgt_data = tgt_data
        self.tgt_labels = torch.from_numpy(tgt_labels).float()
        self.tgt_cluster = torch.LongTensor(tgt_cluster)

        # Group data by labels
        self.label_to_indices = {}
        for i, label in enumerate(src_cluster):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(i)

        #print('triplet dataset:', self.label_to_indices.keys())

    def __len__(self):
        return len(self.tgt_data)

    def __getitem__(self, idx):
        anchor, anchor_label, anchor_cluster = self.tgt_data[idx], self.tgt_labels[idx].item(), self.tgt_cluster[idx].item()

        positive_idx = np.random.choice(self.label_to_indices[anchor_cluster])
        
        # Randomly select a negative sample from a different label
        negative_cluster = anchor_cluster
        while negative_cluster == anchor_cluster:
            negative_cluster = np.random.choice(list(self.label_to_indices.keys()))
        threshold = np.abs(anchor_label-self.src_labels[positive_idx])
        negative_idx = self.find_closest_label_index(anchor_label, self.label_to_indices[negative_cluster], threshold)
        #negative_idx = np.random.choice(self.label_to_indices[negative_cluster])

        #print(anchor_label, positive_idx, negative_label, negative_idx)
        
        anchor, positive, negative = self.tgt_data[idx], self.src_data[positive_idx], self.src_data[negative_idx]
        anchor_label, positive_label, negative_label = self.tgt_labels[idx], self.src_labels[positive_idx], self.src_labels[negative_idx]
        return anchor, positive, negative, anchor_label, positive_label, negative_label
        

    def find_closest_label_index(self, anchor_label, indices, threshold):
        # ??????
        label_differences = np.abs(self.src_labels[indices] - anchor_label)

        # ????????????????
        valid_indices = np.where(label_differences > threshold)[0]

        # ?????????????,??????;????None
        if valid_indices.size > 0:
            # ???????????
            choice_index = np.random.choice(valid_indices)
        else:
            choice_index = np.argmax(label_differences)
        return indices[choice_index]
           
class DA_TripletDataset_label(Dataset):
    def __init__(self, src_data, src_labels, src_cluster, tgt_data, tgt_labels, tgt_cluster):

        self.src_data = src_data
        self.src_labels = src_labels
        self.src_cluster = torch.LongTensor(src_cluster)
        self.tgt_data = tgt_data
        self.tgt_labels = torch.from_numpy(tgt_labels).float()
        self.tgt_cluster = torch.LongTensor(tgt_cluster)

        # Group data by labels
        self.label_to_indices = {}
        for i, label in enumerate(src_cluster):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(i)

        print('triplet dataset:', self.label_to_indices.keys())

    def __len__(self):
        return len(self.tgt_data)

    def __getitem__(self, idx):
        anchor, anchor_label, anchor_cluster = self.tgt_data[idx], self.tgt_labels[idx].item(), self.tgt_cluster[idx].item()

        positive_idx = np.random.choice(self.label_to_indices[anchor_cluster])

        # Randomly select a negative sample from a different label
        negative_cluster = anchor_cluster
        while negative_cluster == anchor_cluster:
            negative_cluster = np.random.choice(list(self.label_to_indices.keys()))
        negative_idx = np.random.choice(self.label_to_indices[negative_cluster])

        anchor, positive, negative = self.tgt_data[idx], self.src_data[positive_idx], self.src_data[negative_idx]
        anchor_label, positive_label, negative_label = self.tgt_labels[idx], self.src_labels[positive_idx], self.src_labels[negative_idx]
      
        return anchor, positive, negative, anchor_label, positive_label, negative_label

class DA_StageDataset_label(Dataset):
    def __init__(self, src_data, src_labels, src_cluster, tgt_data, tgt_labels, tgt_cluster):

        self.src_data = src_data
        self.src_labels = src_labels
        self.src_cluster = torch.LongTensor(src_cluster)
        self.tgt_data = tgt_data
        self.tgt_labels = torch.from_numpy(tgt_labels).float()
        self.tgt_cluster = torch.LongTensor(tgt_cluster)

        # Group data by labels
        self.label_to_indices = {}
        for i, label in enumerate(src_cluster):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(i)

        print('triplet dataset:', self.label_to_indices.keys())

    def __len__(self):
        return len(self.tgt_data)

    def __getitem__(self, idx):
        anchor, anchor_label, anchor_cluster = self.tgt_data[idx], self.tgt_labels[idx].item(), self.tgt_cluster[idx].item()

        positive_idx = np.random.choice(self.label_to_indices[anchor_cluster])

        anchor, positive = self.tgt_data[idx], self.src_data[positive_idx]
        anchor_label, positive_label = self.tgt_labels[idx], self.src_labels[positive_idx]
      
        return anchor, positive, anchor_label, positive_label

def stage_dataset_generator(src_data, src_labels, src_cluster, tgt_data, tgt_labels, tgt_cluster, random_flag=True, batch_size=100, shuffle=True, drop_last=False):
    train_dl = DataLoader(DA_StageDataset_label(src_data, src_labels, src_cluster, tgt_data, tgt_labels, tgt_cluster), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    return train_dl
        
def triplet_dataset_generator(src_data, src_labels, src_cluster, tgt_data, tgt_labels, tgt_cluster, random_flag=True, batch_size=100, shuffle=True, drop_last=False):
    if random_flag:
        train_dl = DataLoader(DA_TripletDataset_label(src_data, src_labels, src_cluster, tgt_data, tgt_labels, tgt_cluster), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    else:
        train_dl = DataLoader(DA_TripletDataset_close(src_data, src_labels, src_cluster, tgt_data, tgt_labels, tgt_cluster), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    return train_dl
