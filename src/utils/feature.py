import math
import numpy as np
from scipy import signal
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler
import PIL
from tqdm import tqdm


def paa_segmentation(data, image_width, overlapping=False, n_segments=None):
    """Compute the indices for Piecewise Agrgegate Approximation.
    """
    if data.ndim==1:
        data = data[:,None]
    n_samples, n_timestamps = data.shape
    window, remainder = divmod(n_samples, image_width)
    window_size = int(np.floor(n_samples//image_width))
    
    if n_segments is None:
        quotient, remainder = divmod(n_samples, window_size)
        n_segments = quotient if remainder == 0 else quotient - 1
        
    if not overlapping:
        bounds = np.linspace(0, n_samples, n_segments + 1).astype('int64')
        start = bounds[:-1]
        end = bounds[1:]
        size = start.size
    else:
        n_overlapping = (n_segments * window_size) - n_samples
        n_overlaps = n_segments - 1
        overlaps = np.linspace(0, n_overlapping,
                               n_overlaps + 1).astype('int64')
        bounds = np.arange(0, (n_segments + 1) * window_size, window_size)
        start = bounds[:-1] - overlaps
        end = bounds[1:] - overlaps
        size = start.size
    
    return np.array([np.median(data[start[i]:end[i]]) for i in range(size)])





def generalized_recurrent_plot(s, eps=0.01, delta=20, distance='euclidean', rp_type="delta"):
    
    if s.ndim==1:
        s = s[:,None]
    
    if rp_type=="delta":
        d = pdist(s, metric=distance)
        d = np.floor(d/eps)
        d[d>delta]=delta
        Img = squareform(d)
        
    elif rp_type=="distance":
        d = pdist(s, metric=distance)
        Img = squareform(d)
            
    elif rp_type=="normalized-distance":
        d = pdist(s, metric=distance)
        d = d/np.max(d)
        Img = squareform(d)
            
    elif rp_type =="exponential":
        d = pdist(s, metric=distance)
        d = np.sqrt(np.exp(-d))
        Img = squareform(d)

    else:
        d = pdist(s, metric=distance)
        d[d>=eps]=1
        d[d<eps]=0   
        Img = squareform(d)
    return Img

def get_resized_image(img, size):
    if img.shape[-1]==1:
        img=img[:,:,0]
    img = PIL.Image.fromarray((img))
    img = img.resize((size, size),  PIL.Image.LANCZOS)
    return  np.array(img)

def generateRPImage(c, w, eps=0.1, delta=10, distance='euclidean', rp_type="delta"):
    img = generalized_recurrent_plot(c, eps, delta, distance, rp_type)
    img = rescale_image(img)
    img = get_resized_image(img, w)
    return img



def generateBinaryimage(c, v, w=16, para=0.5, threshold=0):
    
    """
    Generate I-V binary image
    Agg:import argparse
       cimport argparse
       vimport argparse
       wimport argparse
       pimport argparse
       timport argparse
       rescale:bool wether to rescale image
    Return:
       Image
    """

    #find min and max voltage
    v_min=np.min(v)
    v_max=np.max(v)
    
    #find min and max current
    #scaler = MinMaxScaler(feature_range=(-1, 1))
    #c=scaler.fit_transform(c.reshape(-1,1))
    c_min=np.min(c)
    c_max=np.max(c)

    #get max value of current and voltage
    d_c = max(abs(c_min),c_max)
    d_v = max(abs(v_min),v_max)

    

    #Resize current value to d_c value
    c[c<-d_c] = -d_c
    c[c>d_c] = d_c
    
    #Resize voltage value to d_c value
    v[v<-d_v] = -d_v
    v[v>d_v] = d_v

    d_c = (c_max-c_min)/(w-1)
    d_v = (v_max-v_min)/(w-1)


    #ind_c = np.ceil((c-np.amin(c))/d_c)
    
    ind_c = np.ceil((c-np.min(c))/d_c) if d_c>0 else np.ceil((c-np.min(c)))
    ind_v = np.ceil((v-np.min(v))/d_v) if d_v>0 else np.ceil((v-np.min(v)))
    ind_c[ind_c==w] = w-1
    ind_v[ind_v==w] = w-1
    
    #create image data
    Img = np.zeros((w,w))  
    for i in range(len(c)):
        Img[int(ind_c[i]),int(w-ind_v[i]-1)] += 1

    
    if threshold:
        Img[Img<para] = 0
        Img[Img!=0] = 1
   
    Img = rescale_image(Img)
    return Img




 
def createImage(current, voltage, width=50,image_type="vi", eps=1e-1, steps=10, distance='euclidean',rp_type="delta"):
    
    n = len(current)
    Imgs = np.empty((n,width,width), dtype=np.float64)
    for i in range(n):
        if image_type=="vi":
            Imgs[i,:,:] = generateBinaryimage(current[i],voltage[i,], width)
        else:
            Imgs[i,:,:] = generateRPImage(current[i,],  width, eps, steps, distance, rp_type)
    
    return np.reshape(Imgs,(n,width, width,1))

def createRPImage(current, voltage, width=50, eps=1e-1, steps=10, distance='euclidean', rp_type="delta"):
    
    n = len(current)
    Imgs = np.empty((n,width,width, 2), dtype=np.float64)
    for i in range(n):
       
        c_im= generateRPImage(current[i,],  width, eps, steps, distance, rp_type)
        v_im= generateRPImage(voltage[i,],  width, eps, steps, distance, rp_type)
        img = np.concatenate([c_im[:, :, np.newaxis],v_im[:, :, np.newaxis]], 2)
        Imgs[i,:,:] = img
    
    return Imgs


def rescale_image(img, range=(0, 255)):
    scaler = MinMaxScaler(feature_range=range)
    return scaler.fit_transform(img).astype(np.uint8)

def createImage(current, voltage, width=50,image_type="vi", eps=1e-1, steps=10, distance='euclidean',rp_type="delta"):
    
    n = len(current)
    Imgs = np.empty((n,width,width), dtype=np.float64)
    for i in range(n):
        if image_type=="vi":
            Imgs[i,:,:] = generateBinaryimage(current[i],voltage[i,], width,True,1)
        else:
            Imgs[i,:,:] = generateRPImage(current[i,],  width, eps, steps, distance, rp_type)
    
    return np.reshape(Imgs,(n,width, width,1))

def createRPImage(current, voltage, width=50, eps=1e-1, steps=10, distance='euclidean', rp_type="delta"):
    
    n = len(current)
    Imgs = np.empty((n,width,width, 2), dtype=np.float64)
    for i in range(n):
       
        c_im= generateRPImage(current[i,],  width, eps, steps, distance, rp_type)
        v_im= generateRPImage(voltage[i,],  width, eps, steps, distance, rp_type)
        img = np.concatenate([c_im[:, :, np.newaxis],v_im[:, :, np.newaxis]], 2)
        Imgs[i,:,:] = img
    
    return Imgs




