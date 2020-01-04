import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import random
from functools import partial
from net.fit_functions import get_accuracy, CSVLogger, Checkpoint
from net.fit import perform_training, get_prediction
from net.model import Conv2D
from utils.visual_functions import plot_learning_curve, savefig
from utils.feature import  createImage, createRPImage
from utils.data_generator import get_loaders
from data.load_whited_data import generate_dataset_whited, get_train_test_leave_out_whited    
from data.load_cooll_data import get_cool_feature, generate_image_label_pair, get_train_test_leave_out_cooll, get_train_test_data
from data.load_plaid_data import get_plaid_data
import collections
seed = 4783957
print("set seed")
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
else:
    torch.manual_seed(seed)

datasets = {"plaid":"../data/PLAID/",
               "whited":"../data/WHITED/",
               "cooll":"../data/COOLL/"}



def get_images(current, voltage, width=50,image_type="vi", eps=1e-1, steps=10, distance='euclidean',rp_type="delta"):
    if image_type=="rcp_iv":
        images = createRPImage(current, voltage, width, eps, steps, distance,rp_type)
    elif image_type in ["rcp", "vi"]:
        images =createImage(current, voltage, width,image_type, eps, steps, distance,rp_type)
    else:
        raise AssertionError("Specify correct type of image among rcp_iv, rcp and vi")
    return images

        
def get_data(data_path, dataset,  width=50, eps=0.1, steps=0.15, epochs=600, loss="BCE", model_name="RF", run_num=1):
    
    print("Load data")
    if dataset=="plaid":
        current, voltage, label, house_label = get_plaid_data(data_path)
        eps=0.08
        steps=10
        fs=30e3
    elif dataset=="whited":
        current, voltage, label = get_whited_feature(data_path)
        #current, voltage = get_trajectory(data)
        house_label=None
        eps=0.001
        steps=20
        fs=44.1e3
    elif dataset=="cooll":
        print(dataset)
        current, voltage, label = get_cool_feature(data_path)
        house_label=None
        eps=0.001
        steps=20
        fs=100e3
        
    rp_type="delta"
   
    
        
    for image_type in ["rcp_iv", "vi"]:
        images = get_images(current, voltage, width,image_type, 
                        eps=eps, steps=steps, distance='euclidean', rp_type=rp_type)
        
        for run_id in range(1, run_num+1):

    
            if image_type!="vi":
                file_name = f"{dataset}_{str(run_id)}_{image_type}_{rp_type}_{str(eps)}_{str(steps)}_{model_name}_{str(run_id)}" 
            else:
                file_name = f"{dataset}_{str(run_id)}_{image_type}_{model_name}_{str(run_id)}"
        
            run_experiment_one(images, label, house_label, width=width, epochs=epochs, dataset=dataset, exp_name=file_name, loss=loss, model_name=model_name)
        


def run_experiment_one(images, y, house_label=None, width=50, epochs=50, dataset="whited", exp_name="lilac", loss="BCE", amount_houses_test = 1, model_name="RF"):
    
    if dataset=="whited":
        data  = generate_dataset_whited(y, images)
        train_set, test_set = get_train_test_leave_out_whited(data, 9)
        n =len(train_set)
        
        
    if dataset=="cooll":
        data = generate_image_label_pair(y, images)
        train_set, test_set = get_train_test_leave_out_cooll(data, 8) 
        n =len(train_set)
        
    if dataset=="plaid":
        houses = np.unique(house_label)
        n =len(houses)
        
    

    y_pred_total = []
    y_test_total = []
    f_score_total = []
    
    d = 3
    num_class = len(np.unique(y))
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    file_name=exp_name+"_"+loss+"_"+str(width)
    metric_fn = partial(get_accuracy)
    print(f"fit model for {file_name}")
    
    for i in range(n):
        
        if dataset=="plaid":
            ix_train = [j for j in range(len(house_label)) if house_label[j] not in range(i+1,i+amount_houses_test + 1)]
            ytrain = y[ix_train]
            ix_test = [j for j in range(len(house_label)) if house_label[j] in range(i+1,i+amount_houses_test + 1) and y[j] in ytrain]
            ytest  = y[ix_test]
        
            Xtrain, Xtest = images[ix_train], images[ix_test]
        if dataset in ["whited", "cooll"]:
            Xtrain,ytrain,  Xtest, ytest = get_train_test_data(train_set, test_set, idx=i)
            
        
        
        
        print(f"Xtrain:{Xtrain.shape}:Xtest:{Xtest.shape}")
        print(f"ytrain:{ytrain.shape}:ytest:{ytest.shape}")
        
        loaders=get_loaders(Xtrain, Xtest, ytrain, ytest, batch_size=16)
        model = Conv2D(in_size=d, out_size=num_class)
        saved_model_path   = '../models/{}_checkpoint.pt'.format(file_name)
        csv_logger = CSVLogger(filename=f'../logs/{file_name}.csv',
                       fieldnames=['epoch', 'train_loss', 'test_loss', 'train_acc', 'test_acc'])
        checkpoint = Checkpoint(saved_model_path, patience=20, checkpoint=True, score_mode="max",min_delta=1e-4)
        
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, nesterov=True)
        train_loss, train_acc,  test_loss, test_acc = perform_training(epochs, 
                                                        model, 
                                                        loaders['train'],
                                                        loaders['val'],
                                                        optimizer, 
                                                        criterion, 
                                                        device, 
                                                        checkpoint,
                                                        metric_fn,
                                                        csv_logger)
        plot_learning_curve(train_loss, train_acc, test_loss, test_acc)
        savefig(f"../figure/temp/{file_name}",format=".pdf")
    
        
        pred, test= get_prediction(model, loaders['val'], checkpoint, device,1)

        f1 = f1_score(test, pred, average='micro')
        f_score_total.append(f1)
        y_pred_total.append(pred)
        y_test_total.append(test)
    np.save(f"results/{file_name}_pred.npy", np.vstack(y_pred_total))
    np.save(f"results/{file_name}_true.npy", np.vstack(y_test_total))
    np.save(f"results/{file_name}_fscore.npy", np.vstack(f_score_total))

    

if __name__ == "__main__":
    
    run_num=1
    dataset="plaid"
    data_path=datasets[dataset]
    get_data(data_path, dataset, width=30, eps=None, steps=None, epochs=200, loss="CE", model_name="RF", run_num=run_num)
    


    
