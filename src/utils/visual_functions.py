import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score, precision_recall_curve
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import seaborn as sns
sns.color_palette('husl', n_colors=20)
from sklearn.metrics import confusion_matrix, f1_score
import itertools
nice_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "text.latex.preamble" : [r'\usepackage{amsmath}',r'\usepackage{amssymb}'],
        "font.family": "serif",
        # Always save as 'tight'
        "savefig.bbox" : "tight",
        "savefig.pad_inches" : 0.05,
        "xtick.direction" : "in",
        "xtick.major.size" : 3,
        "xtick.major.width" : 0.5,
        "xtick.minor.size" : 1.5,
        "xtick.minor.width" : 0.5,
        "xtick.minor.visible" : False,
        "xtick.top" : True,
        "ytick.direction" : "in",
        "ytick.major.size" : 3,
        "ytick.major.width" : 0.5,
        "ytick.minor.size" : 1.5,
        "ytick.minor.width" : 0.5,
        "ytick.minor.visible" : False,
        "ytick.right" : True,
        "figure.dpi" : 600,
        "font.serif" : "Times New Roman",
        "mathtext.fontset" : "dejavuserif",
        "axes.labelsize": 14,
        "font.size": 16,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        # Set line widths
        "axes.linewidth" : 0.5,
        "grid.linewidth" : 0.5,
        "lines.linewidth" : 1.,
        # Remove legend frame
        "legend.frameon" : False
}
matplotlib.rcParams.update(nice_fonts)
SPINE_COLOR="gray"
colors =[plt.cm.Blues(0.6), plt.cm.Reds(0.4), plt.cm.Greens(0.6), '#ffcc99', plt.cm.Greys(0.6)]
SPINE_COLOR="gray"


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
    plt.setp(bp['fliers'], markeredgecolor=color)

def set_figure_size(fig_width=None, fig_height=None, columns=2):
    assert(columns in [1,2])

    if fig_width is None:
        fig_width = 4.39 if columns==1 else 7.9 # width in inches

    if fig_height is None:
        golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches

    MAX_HEIGHT_INCHES = 10.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height + 
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES
    return (fig_width, fig_height)


def format_axes(ax):
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)
    
    
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)
    return ax

def figure(fig_width=None, fig_height=None, columns=2):
    """
    Returns a figure with an appropriate size and tight layout.
    """
    fig_width, fig_height =set_figure_size(fig_width, fig_height, columns)
    fig = plt.figure(figsize=(fig_width, fig_height))
    return fig

def subplots(fig_width=None, fig_height=None, *args, **kwargs):
    """
    Returns subplots with an appropriate figure size and tight layout.
    """
    fig_width, fig_height = get_width_height(fig_width, fig_height, columns=2)
    fig, axes = plt.subplots(figsize=(fig_width, fig_height), *args, **kwargs)
    return fig, axes

def legend(ax, ncol=3, loc=9, pos=(0.5, -0.1)):
    leg=ax.legend(loc=loc, bbox_to_anchor=pos, ncol=ncol)
    return leg

def savefig(filename, leg=None, format='.eps', *args, **kwargs):
    """
    Save in PDF file with the given filename.
    """
    if leg:
        art=[leg]
        plt.savefig(filename + format, additional_artists=art, bbox_inches="tight", *args, **kwargs)
    else:
        plt.savefig(filename + format,  bbox_inches="tight", *args, **kwargs)
    plt.close()


def plot_learning_curve(tra_loss_list, tra_f1_list, val_loss_list, val_f1_list):
    
    def line_plot(y_train, y_val, early_stoping, y_label="Loss", y_min=None, y_max=None, best_score=None):
        iterations = range(1,len(y_train)+1)
        if y_min is None:
            y_min = min(min(y_train), min(y_val))
            y_min = max(0, (y_min - y_min*0.01))
        if y_max is None:
            y_max = max(max(y_train), max(y_val))
            y_max = min(1, (y_max + 0.1*y_max))

       
        plt.plot(iterations, y_train, label="training " )
        plt.plot(iterations, y_val, label="validation ")

        if best_score:
            
            plt.title(r"\textbf{Learning curve}"  f": best score: {best_score}",  fontsize=8)
            #plt.axvline(early_stoping, linestyle='--', color='r',label='Early Stopping')
       
        else:
            plt.title(r'\textbf{Learning curve}')
           

        plt.ylabel(y_label)
        #plt.ylim(y_min, y_max)
        plt.xlabel(r"Iterations")
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
       
        plt.legend(loc="best")
        ax = plt.gca()
        ax.patch.set_alpha(0.0)
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))  
        format_axes(ax)
    

   

    min_val_loss_poss = val_loss_list.index(min(val_loss_list))+1 
    min_val_score_poss = val_f1_list.index(max(val_f1_list))+1 
    
    

    fig = figure(fig_width=8)
    plt.subplot(1,2,1)
    line_plot(tra_loss_list, val_loss_list, min_val_loss_poss, y_label="Loss", y_min=0)
   
    
    plt.subplot(1,2,2)
    
    line_plot(tra_f1_list, val_f1_list, min_val_score_poss, y_label="Accuracy", y_min=None, y_max=1, best_score=np.max(val_f1_list))
    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout(pad=1.0)
    
   
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',save = True,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cmNorm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    
    plt.imshow(cmNorm, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=18)
    plt.yticks(tick_marks, classes, fontsize=18)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = 0.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, int(cm[i, j]),fontsize=14,
                 horizontalalignment="center",
                 color="white" if cmNorm[i, j] > thresh else "black") #10

    plt.tight_layout()
    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predicted label', fontsize=16)
    ax = plt.gca()
    #ax.tick_params(axis="both", which="both", bottom=False, 
               #top=False, labelbottom=True, left=False, right=False, labelleft=True)
    #plt.yticks([])
    #plt.xticks([])
    if title:
        plt.title(title)


     
        
def plot_Fmeasure(cm, n, title="Fmacro"):
    av = 0
    p = []
    for i in range(len(n)):
        teller = 2 * cm[i,i]
        noemer = sum(cm[:,i]) + sum(cm[i,:])
        F = float(teller) / float(noemer)
        av += F
        #print('{0} {1:.2f}'.format(names[i],F*100))
        p.append(F*100)

    av = av/len(n)*100
    p = np.array(p)
    
    volgorde = np.argsort(p)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.set_color_codes("pastel")
    sns.barplot(x=p[volgorde], 
            y=np.array(n)[volgorde], color='b')
    plt.axvline(x=av,color='orange', linewidth=1.0, linestyle="--")
    a = '{0:0.02f}'.format(av)
    b = '$Fmacro =\ $'+a
    if av > 75:
        plt.text(av-27,0.1,b,color='darkorange', fontsize=14)
    else:
        plt.text(av+2,0.1,b,color='darkorange',  fontsize=14)
    ax.set_xlabel("$Fmacro$",fontsize=18)
    ax.set_ylabel("",fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set(xlim=(0, 100))
    if title:
        plt.title(title, fontsize=20)
    #sns.despine(left=True, bottom=True)
    
    
    
def get_Fmeasure(cm, n):
    av = 0
    p = []
    for i in range(len(n)):
        teller = 2 * cm[i,i]
        noemer = sum(cm[:,i]) + sum(cm[i,:])
        F = float(teller) / float(noemer)
        av += F
        #print('{0} {1:.2f}'.format(names[i],F*100))
        p.append(F*100)

    av = av/len(n)*100
    return p, av
    
    
def get_fscore(cm, names):
    f1 = get_Fmeasure(cm, names)
    return f1

def plot_multiple_fscore(names, cm_vi,cm_rp, labels=["V-I", "WRG"]):
    width = 0.4
    #sns.set_color_codes("pastel")
    f1_vi,av_vi = get_fscore(cm_vi, names)
    f1_rp,av_rp = get_fscore(cm_rp, names)
    av = max(av_vi, av_rp)
    width=0.4
    plt.barh(np.arange(len(f1_vi)), f1_vi, width, align='center', color=colors[0], label=labels[0])
    plt.barh(np.arange(len(f1_rp))+ width, f1_rp, width, align='center',color='darkorange', alpha=0.8, label=labels[1])
    ax = plt.gca()
    ax.set(yticks=np.arange(len(names)) + width, yticklabels=names)
    ax.set_xlabel("$F_1$ macro (\%)", fontsize=16)
    ax.axvline(x=av,color='darkorange', linewidth=1.0, linestyle="--")
    plt.setp(ax.get_yticklabels(), fontsize=16)
    a = '{0:0.2f}'.format(av)
    b = '$ $'+a
    if av > 75:
        OFFSET = -0.7
        plt.text(av-5,OFFSET,b,color='darkorange')
    else:
        OFFSET = 0
        plt.text(av,OFFSET,b,color='darkorange')
    ax.tick_params(axis='both', which='major')
    ax.set_ylabel("", fontsize=16)
    leg=legend(ax,ncol=2, pos=(0.5, -0.2))
    return leg