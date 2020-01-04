import numpy as np


def zero_crossings(voltage): 
    return np.where(np.diff(np.sign(voltage)))[0]

def get_zero_crossing(voltage, NN=1000):
      
    zero_crossing = zero_crossings(voltage)
    
    if len(zero_crossing)>0:
        if voltage[zero_crossing[0]+1] > 0:
            zero_crossing = zero_crossing[0:]
        else:
            zero_crossing = zero_crossing[1:]
        if len(zero_crossing) % 2 == 1:
            zero_crossing = zero_crossing[:-1]

        if zero_crossing[-1] + NN >= len(voltage):
            zero_crossing = zero_crossing[:-2]
    else:
        zero_crossing = None
        
    return zero_crossing

def align_IV_zero_crossing(i,v, TS, app):
    ks = []
    cs = []
    current, voltage = np.copy(i), np.copy(v)
    
    zc = get_zero_crossing(voltage, TS)[1:]
    ks = []
    crs = []
    
    for j in range(2, len(zc)-2):
        ts=zc[-j]-zc[-(j+2)]
        I=current[zc[-(j+2)]:zc[-j]]
        if app=='Iron':
            diff=round(np.max(abs(current)), 3) - round(np.max(abs(I)), 3)
            diff=diff*100/round(np.max(abs(i)), 3)
            
        ic=ic=zero_crossings(I)
        
        if ts>=TS-100:
            if len(ic)>1:
                k=j
                break
       
        elif ts>3*TS//2 and ts<TS-1:
            if len(ic)>1:
                if app=='Iron' and diff<=3:
                    k=j
                    break
                else:
                    k=j
                    break
                
        
        elif ts>TS//2:
            if len(ic)>1:
                if app=='Iron' and diff<=3:
                    k=j
                    break
                else:
                    k=j
                    break
                    
      
    
    voltage = voltage[zc[-(k+2)]:zc[-k]]
    current = current[zc[-(k+2)]:zc[-k]]
    return current, voltage




