import os
import numpy as np
import pickle 
import random
import json
import subprocess
from datetime import datetime
import matplotlib.pyplot as plt



class PLAID(object):
    
    def __init__(self, path, progress=True, width=50, npts=10000, fs=30000, f0=60):
        self.path = path
        self.progress = progress
        self.npts = npts
        self.width = width
        self.sampling_frequency = fs
        self.mains_frequency = f0
        self.get_meta_parameters()
        

    def clean_meta(self, ist):
        '''remove None elements in Meta Data '''
        clean_ist = ist.copy()
        for k, v in ist.items():
            if len(v) == 0:
                del clean_ist[k]
        return clean_ist

   

    def get_meta_data(self):

        with open(self.path+'meta1.json') as data_file:
            meta1 = json.load(data_file)

        with open(self.path+'meta2.json') as data_file:
            meta2 = json.load(data_file)

        M = {}
        
        
        # consider PLAID1 and 2 [meta1, meta2]
        
        for m in[meta1]:
            for app in m:
                M[int(app['id'])] = self.clean_meta(app['meta'])

        return M

    def get_meta_parameters(self):
        # applinace types of all instances
        Meta = self.get_meta_data()
        self.appliance_types = [x['type'] for x in Meta.values()]

        # unique appliance types
        self.appliances = list(set(self.appliance_types))
        self.appliances.sort()

        self.data_IDs = list(Meta.keys())

        # households of appliances
        self.households = [x['header']['collection_time'] +
                           '_'+x['location'] for x in Meta.values()]
        #   unique households
        self.house = list(set(self.households))
        self.house.sort()

        print(f'Number of appliances:{len(self.appliances)} \nNumber of households:{len(self.house)}\
             \nNumber of total measurements:{len(self.households)}')

    def get_data(self):
        path = self.path + 'CSV/'
        last_offset = self.npts
        start = datetime.now()
        n = len(self.data_IDs)
        if n == 0:
            return {}
        else:
            data = {}
            for (i, ist_id) in enumerate(self.data_IDs, start=1):
                if self.progress and np.mod(i, np.ceil(n/10)) == 0:
                    print('%d/%d (%2.0f%s) have been read...\t time consumed: %ds'
                          % (i, n, i/n*100, '%', (datetime.now()-start).seconds))
                if last_offset == 0:
                    data[ist_id] = np.genfromtxt(path+str(ist_id)+'.csv', delimiter=',',
                                                 names='current,voltage', dtype=(float, float))
                else:
                    p = subprocess.Popen(['tail', '-'+str(int(last_offset)), path+str(ist_id)+'.csv'],
                                         stdout=subprocess.PIPE)
                    data[ist_id] = np.genfromtxt(p.stdout, delimiter=',', names='current,voltage',
                                                 dtype=(float, float))
            print('%d/%d (%2.0f%s) have been read(Done!) \t time consumed: %ds'
                  % (n, n, 100, '%', (datetime.now()-start).seconds))

        appliance_Ids = {}
        house_Ids = {}
        Mapping = {}
        n = len(data)
        appliance_label = np.zeros(n, dtype='int')
        house_label = np.zeros(n, dtype='int')
        for (ii, t) in enumerate(self.appliances):
            appliance_Ids[t] = [
                i-1 for i, j in enumerate(self.appliance_types, start=1) if j == t]
            appliance_label[appliance_Ids[t]] = ii
            Mapping[ii] = t
        for (ii, t) in enumerate(self.house):
            house_Ids[t] = [i-1 for i,
                            j in enumerate(self.households, start=1) if j == t]
            house_label[house_Ids[t]] = ii+1
        print('number of different appliances: %d' % len(self.appliances))
        print('number of different households: %d' % len(self.house))
        return data, house_label, house_Ids, appliance_label, appliance_Ids

    def get_features(self, data):
        # number of samples per period
        NS = int(self.sampling_frequency//self.mains_frequency)
        NP = int(self.npts/NS)  # number of periods for npts

        # calculate the representative one period of steady state
        # (mean of the aggregated signals over one cycle)
        n = len(data)
        rep_I = np.empty([n, NS])
        rep_V = np.empty([n, NS])
        for i in range(n):
            ind = list(data)[i]
            tempI = np.sum(np.reshape(data[ind]['current'], [NP, NS]), 0)/NP
            tempV = np.sum(np.reshape(data[ind]['voltage'], [NP, NS]), 0)/NP
            # align current to make all samples start from 0 and goes up
            ix = np.argsort(np.abs(tempI))
            j = 0
            while True:
                if ix[j] < 499 and tempI[ix[j]+1] > tempI[ix[j]]:
                    real_ix = ix[j]
                    break
                else:
                    j += 1
            rep_I[i, ] = np.hstack([tempI[real_ix:], tempI[:real_ix]])
            rep_V[i, ] = np.hstack([tempV[real_ix:], tempV[:real_ix]])
            #rep_I[i, ] = rep_I[i, ] 
            #rep_V[i, ] = rep_V[i, ] 

        return rep_I, rep_V

def get_plaid_data(path="/home/ibcn079/data/PLAID/",width=50):
    plaid = PLAID(path=path)
    
    data, house_label, house_Ids, appliance_label, appliance_Ids = plaid.get_data()
    current, voltage = plaid.get_features(data)
    
    
    return current, voltage, appliance_label, house_label
