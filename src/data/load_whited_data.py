import numpy as np
import soundfile as sf
import os
from tqdm import tqdm
from transform import *


wanted_appl = ['CFL','Charger','DrillingMachine','Fan','FlatIron','GameConsole','HairDryer','Iron','Kettle','LEDLight',
               'LightBulb', 'Massage', 'Microwave', 'Mixer', 'Monitor', 'PowerSupply','ShoeWarmer','Shredder','SolderingIron',
               'Toaster','VacuumCleaner','WaterHeater' ]

def get_whited_data(path_dir):
    
    appliance_type = []
    appliance_label = []
    region_label=[]
    data = []

    with os.scandir(path_dir) as entries:
        for entry in entries:
            if entry.is_file():
                file_name = entry.name
                ext = file_name.strip().split(".")[-1]
                if ext=="flac":
                    names = file_name.strip().split(".")[0].strip().split("_")
                    
                    appliance_type.append(names[0])
                    appliance_label.append(names[1])
                    d, fs = sf.read(entry.path)
                    data.append(d)
                    region_label.append(int(list(names[2])[-1]))
                        
    label=np.array(appliance_type)
    
                        
    return label, data

def get_VI_trajectory(data, label, fs=44.1e3, f0=50):
    
    NS = int(fs/f0) 
    NP = 20
    npts=int(NS*20)
    n = len(data)
    I = []
    V = []
    files_id=0
    with tqdm(total=n) as pbar:
        for ind in range(n):
            
            app=label[ind]
            if ind in [648, 38]:
                v=data[ind][15000:15000+NS*NP,0]
                i=data[ind][15000:15000+NS*NP,1]
                
            elif ind in [8]:
                v=data[ind][90000:90000+NS*NP,0]
                i=data[ind][90000:90000+NS*NP,1]   
                
            elif ind in [78, 28]:
                v=data[ind][10000:10000+NS*NP,0]
                i=data[ind][10000:10000+NS*NP,1]
                
            elif ind in [17]:
                i=data[17][20000:20000+NS*NP,1]
                v=data[17][20000:20000+NS*NP,0]
                
            elif ind in [17, 140, 780, 1137]:
                v=data[ind][:10000,0]
                i=data[ind][:10000,1]
                
            elif ind in [93, 274, 662]:
                v=data[ind][5000:5000+NS*NP,0]
                i=data[ind][5000:5000+NS*NP,1]
        
            else:
                v=data[ind][:NS*NP,0]
                i=data[ind][:NS*NP,1]
                
            c, u = align_IV_zero_crossing(i, v, NS, app)

            I.append(c)
            V.append(u)
            pbar.set_description('processed: %d' % (1 + files_id))
            pbar.update(1)
            files_id+=1
        pbar.close()
        
        return np.array(I), np.array(V), label

def get_trajectory(data, fs=44.1e3, f0=50):
    
    NS = int(fs/f0) 
    NP = 20
    npts=int(NS*20)
    n = len(data)
    I = np.empty([n,NS])
    V = np.empty([n,NS])
    files_id=0
    with tqdm(total=n) as pbar:
        for ind in range(n):

            tempI = np.sum(np.reshape(data[ind][:npts, 1], [NP, NS]), 0)/NP
            tempV = np.sum(np.reshape(data[ind][:npts, 0], [NP, NS]), 0)/NP

            ix = np.argsort(np.abs(tempI))
            j = 0
            while True:
                if ix[j] < NS-1 and tempI[ix[j]+1] > tempI[ix[j]]:
                    real_ix = ix[j]
                    break
                else:
                    j += 1
            c = np.hstack([tempI[real_ix:], tempI[:real_ix]])
            v = np.hstack([tempV[real_ix:], tempV[:real_ix]])
            I[ind,] = c 
            V[ind,] = v 
            pbar.set_description('processed: %d' % (1 + files_id))
            pbar.update(1)
            files_id+=1
        pbar.close()
        
    return I, V
    
def get_whited_feature(path_dir):
    
    label, data = get_whited_data(path_dir)
    #current, voltage = get_trajectory(data)
    current, voltage, label = get_VI_trajectory(data, label, fs=44.1e3, f0=50)
    
    return current, voltage, label
    
    
    

def generate_dataset_whited(label, images):
    
    dataset = {}
    for name in wanted_appl:
        index = np.where(label==name)[0]
        dataset[name]=images[index]
        
    return dataset
               
               
def get_train_test_leave_out_whited(dataset, n=9):
    
    houses = dict([(key, []) for key in range(n)])
    houses_ids = dict([(key, []) for key in range(n)])

    for name in wanted_appl:
        ids = np.array(range(len(dataset[name])))//10
        for i in np.unique(ids):
            arr = list(range(n))
            np.random.shuffle(arr)
            j = 0
            while True:
                if name in houses[arr[j]]:
                    j += 1
                else:
                    houses[arr[j]].append(name)
                    houses_ids[arr[j]].append(i)
                    break


    train_set = []
    test_set = []
    index = 0            
    for h, hi in zip(list(houses.values()), list(houses_ids.values())):
        test = {}
        train = {}
        #print(index)
        
        
        test_names = [ i + str(j) for (i,j) in zip(h,hi)]
        for name in  list(dataset.keys()):
            ids = np.array(range(len(dataset[name])))//10
            for i in range(len(dataset[name])):
                if name+str(ids[i]) in test_names:
                    if name not in test:
                        test[name] = []
                    test[name].append(dataset[name][i])
                elif name in wanted_appl:
                    if name not in train:
                        train[name] = []
                    train[name].append(dataset[name][i])
        test_set.append(test)
        train_set.append(train)
        index += 1
        
        
    return train_set, test_set


def get_train_test_data(train_set, test_set, idx=0):
    train=train_set[idx]
    test=test_set[idx]
    train_X = []
    train_y = []
    
    for key, items in train.items():
         for i in range(len(items)):
                train_X.append(items[i])
                train_y.append(key)

    test_X = []
    test_y = []
    
    for key, items in test.items():
         for i in range(len(items)):
                test_X.append(items[i])
                test_y.append(key)
                
    return np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)
