
import math
import numpy as np
import torch
from torchvision import transforms
import PIL

transform = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    } 

class Dataset(torch.utils.data.Dataset):
    

    def __init__(self, img, label, transform=None):
       
        self.img = img
        self.label = label
        self.transform = transform
        
       
        

    def __len__(self):
        return len(self.label)
    
    def pil_image(self, img):
        if img.shape[-1]==1:
            img=img[:,:,0]
        img = PIL.Image.fromarray((img))
        return img.convert('RGB')
    
    
    def __getitem__(self, index):
       
        img = self.img[index]
        img = self.pil_image(img)
        img = self.transform(img)
        label = self.label[index]
        return img, label
        
            
def get_loaders(img_tra, img_val, label_tra, label_val,batch_size=32):
   
    tra_data = Dataset(img_tra, label=label_tra,  transform=transform['train'])
    val_data = Dataset(img_val, label=label_val, transform=transform['val'])
    
    tra_loader=torch.utils.data.DataLoader(tra_data, batch_size, shuffle=True, num_workers=4)
    val_loader=torch.utils.data.DataLoader(val_data, batch_size, shuffle=False, num_workers=4)
    
    loaders = {'train':tra_loader, 'val':val_loader}
    
    return loaders  


