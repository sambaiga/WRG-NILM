import torch
import numpy as np
from data.load_whited_data import get_whited_feature, get_train_test_leave_out_whited  
from data.load_cooll_data import get_cool_feature, generate_image_label_pair, get_train_test_leave_out_cooll, get_train_test_data
from data.load_plaid_data import get_plaid_data



datasets = {"plaid":"../../data/PLAID/",
               "whited":"../../data/WHITED/",
               "cooll":"../../data/COOLL/"}

def get_data(dataset):
    data_path = datasets[dataset]
    print("Load data")
    if dataset=="plaid":
        current, voltage, label, house_label = get_plaid_data(data_path)
        eps=1e1
        delta=20
    elif dataset=="whited":
        current, voltage, label = get_whited_feature(data_path)
        house_label=None
        eps=1e3
        steps=50
        delta=50
        fs=44.1e3
    elif dataset=="cooll":
       
        current, voltage, label = get_cool_feature(data_path)
        house_label=None
        eps=1e3
        delta=50
        fs=100e3
    return   current, voltage, label, house_label, eps, delta

def get_correct_labels_lilac(labels):
    correct_1_phase_motor = [920,923,956, 959, 961, 962, 1188]
    correct_hair = [922, 921, 957, 958,  960, 963, 1181, 1314]
    correct_bulb = [1316]
    
    correct_labels = []
    for idx, l in enumerate(labels):
        if idx in correct_1_phase_motor:
            correct_labels.append('1-phase-async-motor')
        elif idx in correct_hair:
            correct_labels.append('Hair-dryer')
        elif idx in correct_bulb:
            correct_labels.append('Bulb')
        else:
            correct_labels.append(l)
    correct_labels = np.hstack(correct_labels)
    return correct_labels





class Dataset(torch.utils.data.Dataset):
    

    def __init__(self, feature,  label, width=50):
       
        self.feature   = feature
        self.label    = label
        

        
    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
       
        feature = self.feature[index]
        label =  self.label[index]
        
        return feature, label
        
        
def get_loaders(input_tra, input_val, label_tra, label_val,
                batch_size=64):
   
    tra_data = Dataset(input_tra, label_tra)
    val_data = Dataset(input_val, label_val)
    
    tra_loader=torch.utils.data.DataLoader(tra_data, batch_size, shuffle=True, num_workers=4,drop_last=False)
    val_loader=torch.utils.data.DataLoader(val_data, batch_size, shuffle=False, num_workers=4, drop_last=False)
    
    loaders = {'train':tra_loader, 'val':val_loader}
    
    return loaders  
