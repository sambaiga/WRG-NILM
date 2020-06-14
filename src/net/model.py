import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPLayer(nn.Module):
    def __init__(self, in_size, hidden_arch=[128, 512, 1024], output_size=None, activation=nn.LeakyReLU(),
                 batch_norm=True):
        
        super(MLPLayer, self).__init__()
        self.in_size = in_size
        self.output_size = output_size
        layer_sizes = [in_size] + [x for x in hidden_arch]
        self.layers = []
        
        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            #self.add_module("layer"+str(i+1), layer)            
            if batch_norm and i!=0:# if used as discriminator, then there is no batch norm in the first layer
                bn = nn.BatchNorm1d(layer_sizes[i+1])
                self.layers.append(bn)
                #self.add_module("bn"+str(i+1), bn)
            self.layers.append(activation)
            #self.add_module("activation"+str(i+1), activation)
        if output_size is not None:
            layer = nn.Linear(layer_sizes[-1], output_size)
            self.layers.append(layer)
            #self.add_module("layer"+str(len(self.layers)), layer)
            self.layers.append(activation)
            #self.add_module("activation"+str(i+1), activation)
        self.init_weights()
        self.mlp_network =  nn.Sequential(*self.layers)
        
    def forward(self, z):
        return self.mlp_network(z)
        
    def init_weights(self):
        for layer in self.layers:
            try:
                if isinstance(layer, nn.Linear):
                    init.xavier_uniform_(layer.weight)
                
            except: pass

class Conv2D(nn.Module):
    """2D convolutional layer with optional batch norm and ReLU."""
    def __init__(self, n_channels, n_kernels,
                 kernel_size=3, stride=2, padding=1, last=False, activation=nn.LeakyReLU()):
        super(Conv2D, self).__init__()
        self.conv = nn.Conv2d(
            n_channels, n_kernels,
            kernel_size, stride, padding
        )
        if not last:
            self.net = nn.Sequential(
                self.conv,
                #nn.BatchNorm1d(n_kernels),
                nn.LeakyReLU()
            )
        else:
            self.net = self.conv
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        return self.net(x) 

class CNN2DModel(nn.Module):
    """Convolutional encoder for audio spectrogram slices."""
    def __init__(self, n_channels=1, n_kernels=128, n_layers=5, emb_size=99, dropout=0.5, output_size=10):
        super(CNN2DModel, self).__init__()
        # Compute final feature size assuming n_freqs = k*2^n +1 for some n, k
        self.feat_size = (emb_size-1) // 2**n_layers +1
        self.feat_dim = self.feat_size**2 * n_kernels
        
        

        self.conv_stack = nn.Sequential(
            *([Conv2D(n_channels, n_kernels // 2**(n_layers-1))] +
              [Conv2D(n_kernels//2**(n_layers-l),
                         n_kernels//2**(n_layers-l-1))
               for l in range(1, n_layers-1)] +
              [Conv2D(n_kernels // 2, n_kernels, last=True)])
        )
        
        self.dec_net = MLPLayer(in_size=self.feat_dim, hidden_arch=[1024], output_size=None)
        self.dropout = nn.Dropout(dropout)
        self.fc_out  = nn.Linear(1024, output_size)
        nn.init.xavier_normal_(self.fc_out.weight.data)
        self.fc_out.bias.data.fill_(0)

        
    def forward(self, x):
        bsz = x.size(0)
        x = self.dropout(self.conv_stack(x))
        x = self.dropout(self.dec_net(x.view(bsz, -1)))
        x = self.fc_out(x)
        return x

