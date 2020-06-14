import sys
sys.path.append("../src/")
import random
import torch
import numpy as np
import time
from functools import partial
import warnings
warnings.filterwarnings("ignore")
from functools import partial
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, zero_one_loss
from net.model import  CNN2DModel
from net.fit import perform_training, get_prediction
from net.fit_functions import get_accuracy, CSVLogger, Checkpoint

from utils.data_generator import get_loaders,  get_correct_labels_lilac, get_data


from utils.visual_functions import plot_learning_curve, savefig
from utils.feature_representation  import generate_input_feature,  get_weighted_reccurrence_graph, get_binary_reccurrence_graph

seed = 4783957
print("set seed")
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
else:
    torch.manual_seed(seed)


datasets = {"plaid":"../../data/PLAID/",
               "whited":"../../data/WHITED/",
               "cooll":"../../data/COOLL/"}



        

def train_model(Xtrain, Xtest, ytrain, ytest, width,
                 model_name, epochs=2, 
                 file_name=None,
                  in_size=1,batch_size=32):
    
    y_pred_total = []
    f_1_total = []
    mcc_total = []
    z_one_total = []
    y_test_total = []
    total_time = []
  
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    metric_fn = partial(get_accuracy)
    classes=list(np.unique(ytrain))
    num_class=len(classes)

    
    loaders=get_loaders(Xtrain, Xtest, ytrain, ytest, 
                        batch_size=batch_size)
    model = CNN2DModel(n_channels=in_size, n_kernels=64, n_layers=3, emb_size=width, dropout=0.25, output_size=num_class)

    
    saved_model_path   = '../checkpoint/{}_checkpoint.pt'.format(file_name)
    csv_logger = CSVLogger(filename=f'../logs/{file_name}.csv',
                       fieldnames=['epoch', 'train_loss', 'test_loss', 'train_acc', 'test_acc'])
    checkpoint = Checkpoint(saved_model_path, patience=25, checkpoint=True, score_mode="max",min_delta=1e-4)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
    
    start_time =  time.time()

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
    end_time    = time.time()
    time_used   = end_time - start_time
    pred, test  = get_prediction(model, loaders['val'], checkpoint, device,1)
    plot_learning_curve(train_loss, train_acc, test_loss, test_acc)
    savefig(f"../figure/temp/{file_name}",format=".pdf")
    
    f1 = f1_score(test, pred,  average='weighted')
    mcc  = matthews_corrcoef(test, pred)
    zl   = zero_one_loss(test, pred)*100
    f_1_total.append(f1)
    mcc_total.append(mcc)
    z_one_total.append(zl)
    y_pred_total.append(pred.astype(int))
    y_test_total.append(test.astype(int))
    total_time.append(time_used)

    np.save(f"../results/{file_name}_pred.npy", np.vstack(y_pred_total))
    np.save(f"../results/{file_name}_f1.npy", np.vstack(f_1_total))
    np.save(f"../results/{file_name}_mcc.npy", np.vstack(mcc_total))
    np.save(f"../results/{file_name}_z_one.npy", np.vstack(z_one_total))
    np.save(f"../results/{file_name}_true.npy", np.vstack(y_test_total))
    np.save(f"../results/{file_name}_time.npy", np.vstack(total_time))


   
   
    
def eps_delta_experiments(dataset="cooll",
                            image_type="wrg",
                            width=50,
                            run_id=1,
                            epochs=100,
                            multi_dimension=False,
                            batch_size=16,
                            model_name="CNN"):  

    current, voltage, labels, house_label, eps, delta = get_data(dataset) 

    #perform label enconding
    le = LabelEncoder()
    le.fit(labels)
    y = le.transform(labels)    

    skf = StratifiedKFold(n_splits=4,random_state=42,shuffle=True)
    train_index, test_index = next(skf.split(current, y))
    ytrain, ytest = y[train_index], y[test_index]

    input_feature = generate_input_feature(current, voltage, image_type, width, multi_dimension)
    in_size  = 1
    
    deltas = [0, 0.5, 1, 5, 10, 20, 30, width]
    #epss   = [5, 1e1, 1e2, 1e3, 1e4, 1e5]
    for delta in deltas:
        for eps in [5, 1e1, 1e2, 1e3, 1e4, 1e5]:
            file_name=f"{dataset}_{image_type}_{model_name}_parameters_delta_{str(delta)}_eps_{str(eps)}"
            
           
            wrg_feature = get_weighted_reccurrence_graph(input_feature, eps, delta)
            Xtrain, Xtest = wrg_feature[train_index], wrg_feature[test_index]

            print(f"fit model for {file_name}")
            train_model(Xtrain, Xtest, ytrain, ytest, width,
                    model_name, epochs, file_name, in_size,batch_size)
    

def embending_experiments(dataset="cooll",
                            image_type="wrg",
                            width=50,
                            run_id=1,
                            epochs=100,
                            multi_dimension=False,
                            batch_size=16,
                            model_name="CNN"):  

    current, voltage, labels, house_label, eps, delta = get_data(dataset) 

    #perform label enconding
    le = LabelEncoder()
    le.fit(labels)
    y = le.transform(labels)    

    skf = StratifiedKFold(n_splits=4,random_state=42,shuffle=True)
    train_index, test_index = next(skf.split(current, y))
    ytrain, ytest = y[train_index], y[test_index]

    in_size  = 1
    widths = [20, 30, 40, 50, 60, 80, 100]
    
    for width in widths:
        
        file_name=f"{dataset}_{image_type}_{model_name}_parameters_width_{str(width)}"
        input_feature = generate_input_feature(current, voltage, image_type, width, multi_dimension)
        wrg_feature = get_weighted_reccurrence_graph(input_feature, eps, delta)
        Xtrain, Xtest = wrg_feature[train_index], wrg_feature[test_index]

        print(f"fit model for {file_name}")
        train_model(Xtrain, Xtest, ytrain, ytest, width,
                    model_name, epochs, file_name, in_size,batch_size)    



if __name__ == "__main__":
    
    run_num=1
    dataset=["cooll", 'plaid', 'whited']
    """
    for dataset in [ 'whited']:
        eps_delta_experiments(dataset=dataset,
                            image_type="wrg",
                            width=50,
                            run_id=1,
                            epochs=100,
                            batch_size=16,
                            model_name="CNN")
    """
    for dataset in [ 'cooll', 'plaid']:
        embending_experiments(dataset=dataset,
                            image_type="wrg",
                            width=50,
                            run_id=1,
                            epochs=100,
                            batch_size=16,
                            model_name="CNN")


    
