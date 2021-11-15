import os
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from sklearn.preprocessing import MinMaxScaler

def accuracy_diagonal_plot(save_path:str, X:np.array, Y:np.array, predictions:np.array, plotname:str) -> None:
    plt.rcParams.update({'font.size': 28})

    scaler_Y = MinMaxScaler(feature_range=(1e-3, 1))
    Y_scaled = scaler_Y.fit_transform(Y)
    predictions_scaled = scaler_Y.fit_transform(predictions)

    b           = np.true_divide(np.sum(predictions_scaled * Y_scaled), np.sum(predictions_scaled**2))    
    deltas      = np.true_divide(predictions_scaled, (b * Y_scaled))
    lnDeltas    = np.log(deltas)
    sdlnDeltas  = np.std(lnDeltas)            # (D.12, EC0)
    VarKf       = np.sqrt(np.exp(sdlnDeltas)-1)            # (D.13, EC0)
    Vrti        = 0.05;       # CoV for the sensitivity of the resistance function to slight variations of the input variables
    Vr          = np.sqrt(VarKf**2 + 1 * np.power((np.square(Vrti) + 1), X.shape[1]) - 1)

    #compute R^2 values
    # calculate R^2 values for each feature
    # calculate mse
    r_squared2 = np.corrcoef(Y.flatten(), predictions.flatten())[0, 1]**2

    plt.figure(figsize=[8, 6], dpi=100)
    plt.plot(Y, predictions, marker = 'o', ms = 10, linestyle='None')
    #plt.title(plotname)
    axa = plt.gca()
    axa.set_aspect('equal', 'box')
    axa.axis('square')
    axa.set_ylabel('Predicted '+ plotname)
    axa.set_xlabel('Reference '+ plotname)
    axa.set_xlim([np.min([np.min(Y), np.min(predictions)]), np.max([np.max(Y), np.max(predictions)])])
    axa.set_ylim([np.min([np.min(Y), np.min(predictions)]), np.max([np.max(Y), np.max(predictions)])])
    axa.grid(True, which='major', color='#666666', linestyle='-')
    at = AnchoredText('$R^2$ = ' + np.array2string(r_squared2, precision=3) +
                '\n$V_r$ = '+ np.array2string(Vr, precision=3),
                prop=dict(size=25), frameon=True,loc='upper left')
    at.patch.set_boxstyle('round,pad=0.,rounding_size=0.2')
    axa.add_artist(at)   
    plt.plot([np.min([np.min(Y), np.min(predictions)]), np.max([np.max(Y), np.max(predictions)])], [np.min([np.min(Y), np.min(predictions)]), np.max([np.max(Y), np.max(predictions)])], color='darkorange', linestyle='--',
                linewidth = 7)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'diagonal_match_'+plotname), dpi=400, bbox_inches='tight')
    plt.show()
    plt.close()


def mean_correction_factor_plot(y, pred, path, labels):
    #1: compute ratios b = FEM / AI
    b = np.divide(y, pred)

    #2: compute mean b
    b_bar = np.mean(b,0)

    plt.rcParams.update({'font.size': 28})
    plt.figure(figsize=[8, 6], dpi=100)
    plt.bar(np.linspace(1, b_bar.shape[0], b_bar.shape[0], endpoint=True), b_bar, label='b mean')
    plt.plot(np.linspace(1, b_bar.shape[0], b_bar.shape[0], endpoint=True), np.ones(b_bar.shape[0]), label='mean', linewidth=2, color='r')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel('Mean Correction Factor $\hat{b}$')
    plt.title('Mean Correction Factor $\hat{b}$')
    plt.xticks(np.linspace(1, b_bar.shape[0], b_bar.shape[0], endpoint=True))
    plt.xticklabels(labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'mean_correction_factor'), dpi=200)
    plt.show()


def qq_plot(y, pred, path, labels):
    #1: compute ratios b = FEM / AI
    b=np.divide(y, pred)
    b[~np.isfinite(b)] = 0
    plt.rcParams.update({'font.size': 28})
    for resultcomponent in range(1, b.shape[1]+1):
        plt.figure(figsize=(8, 6), dpi=100)
        sm.qqplot(b[:,resultcomponent-1], line = "45", fit = True)
        plt.title('Q-Q Plot for '+labels[resultcomponent-1])
        plt.grid(color='0.5')
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'qq_plot_'+labels[resultcomponent-1]), dpi=200)
        plt.show()


def ratio_plot(y, pred, path, labels):
    #1: compute ratios b = true / pred
    b = np.divide(y, pred)
    b[~np.isfinite(b)] = 0
    plt.rcParams.update({'font.size': 28})
    for resultcomponent in range(1, b.shape[1]+1): #
        plt.figure(figsize=(8, 6), dpi=100)
        plt.plot(np.linspace(1, b.shape[0], b.shape[0], endpoint=True).reshape(-1,1),b[:,resultcomponent-1], marker = 'o', ms = 3, linestyle='None', label='ratio')
        plt.plot(np.linspace(1, b.shape[0], b.shape[0], endpoint=True).reshape(-1,1),np.multiply(np.ones(b.shape[0]),np.mean(b[:,resultcomponent-1])), label='mean',linewidth=6)
        plt.plot(np.linspace(1, b.shape[0], b.shape[0], endpoint=True).reshape(-1,1),np.ones(b.shape[0]), label='ideal',linewidth=4, linestyle='--')
        plt.legend()
        plt.grid(color='0.5')
        plt.title('Ratio for '+labels[resultcomponent-1])
        plt.yscale('log')
        plt.ylim(10**-6,10**6)
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'ratio_true_prediction_'+labels[resultcomponent-1]), dpi=200)
        plt.show()