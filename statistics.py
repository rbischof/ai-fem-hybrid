import os
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from sklearn.preprocessing import MinMaxScaler

def accuracy_diagonal_plot(x:np.array, y:np.array, pred:np.array, labels:list, path:str) -> None:
    """
    x : input values
    y : output values
    pred : predictions of model
    labels : list of output variable names, must have same length as the second dimension of y and pred
    path : path where to store the figures
    """
    plt.rc('font', size=28) #controls default text size
    plt.rc('axes', labelsize=26)
    plt.rc('xtick', labelsize=26) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=26) #fontsize of the y tick labels

    for i in range(y.shape[1]):
        Y = y[:, i:i+1]
        P = pred[:, i:i+1]
        scaler_Y = MinMaxScaler(feature_range=(1e-3, 1))
        Y_scaled = scaler_Y.fit_transform(Y)
        P_scaled = scaler_Y.fit_transform(P)

        # compute variation coeff
        b           = np.true_divide(np.sum(P_scaled * Y_scaled), np.sum(P_scaled**2))    
        deltas      = np.true_divide(P_scaled, (b * Y_scaled))
        lnDeltas    = np.log(deltas)
        sdlnDeltas  = np.std(lnDeltas)            # (D.12, EC0)
        VarKf       = np.sqrt(np.exp(sdlnDeltas)-1)            # (D.13, EC0)
        Vrti        = 0.05;       # CoV for the sensitivity of the resistance function to slight variations of the input variables
        Vr          = np.sqrt(VarKf**2 + 1 * np.power((np.square(Vrti) + 1), x.shape[1]) - 1)

        # compute R^2 values
        r_squared2 = np.corrcoef(Y.flatten(), P.flatten())[0, 1]**2

        # plot
        plt.figure(figsize=[10, 8], dpi=100)
        plt.plot(Y, P, marker = 'o', ms = 10, linestyle='None')

        axa = plt.gca()
        axa.set_aspect('equal', adjustable='box')
        axa.set_box_aspect(1)
        axa.axis('square')
        axa.set_ylabel('Predicted '+ labels[i])
        axa.set_xlabel('Reference '+ labels[i])
        axa.set_xlim([np.min([np.min(Y), np.min(P)]), np.max([np.max(Y), np.max(P)])])
        axa.set_ylim([np.min([np.min(Y), np.min(P)]), np.max([np.max(Y), np.max(P)])])
        axa.grid(True, which='major', color='#666666', linestyle='-')
        at = AnchoredText('$R^2$ = ' + np.array2string(r_squared2, precision=3) +
                    '\n$V_r$ = '+ np.array2string(Vr, precision=3),
                    prop=dict(size=26), frameon=True,loc='upper left')
        at.patch.set_boxstyle('round,pad=0.,rounding_size=0.2')
        axa.add_artist(at)   
        plt.plot([np.min([np.min(Y), np.min(P)]), np.max([np.max(Y), np.max(P)])], [np.min([np.min(Y), np.min(P)]), np.max([np.max(Y), np.max(P)])], color='darkorange', linestyle='--',
                    linewidth = 7)
        plt.tight_layout()
        if path is not None:
            plt.savefig(os.path.join(path, 'diagonal_match_'+labels[i]), dpi=100, bbox_inches='tight')
        plt.show()
        plt.close()


def mean_correction_factor_plot(y:np.array, pred:np.array, labels:list, path:str):
    # compute ratios b = FEM / AI
    b = np.divide(y, pred)

    # compute mean b
    b_bar = np.mean(b,0)

    plt.rc('font', size=28) #controls default text size
    plt.rc('axes', labelsize=26)
    plt.rc('xtick', labelsize=26) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=26) #fontsize of the y tick labels

    plt.figure(figsize=[10, 8], dpi=100)
    plt.bar(np.linspace(1, b_bar.shape[0], b_bar.shape[0], endpoint=True), b_bar, label='b mean')
    plt.plot(np.linspace(1, b_bar.shape[0], b_bar.shape[0], endpoint=True), np.ones(b_bar.shape[0]), label='mean', linewidth=2, color='r')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel('Mean Correction Factor $\hat{b}$')
    plt.xticks(np.linspace(1, b_bar.shape[0], b_bar.shape[0], endpoint=True), labels=labels, rotation=45, ha='right')
    plt.legend(fontsize=24)
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'mean_correction_factor'), dpi=100)
    plt.show()
    plt.close()


def qq_plot(y:np.array, pred:np.array, labels:list, path:str):
    # compute ratios b = FEM / AI
    b = np.divide(y, pred)
    b[~np.isfinite(b)] = 0

    plt.rc('font', size=28) #controls default text size
    plt.rc('axes', labelsize=26)
    plt.rc('xtick', labelsize=26) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=26) #fontsize of the y tick labels
    
    for resultcomponent in range(b.shape[1]):
        plt.figure(figsize=(10, 8), dpi=100)
        sm.qqplot(b[:, resultcomponent], line = "45", fit = True)
        plt.grid(color='0.5')
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'qq_plot_'+labels[resultcomponent]), dpi=100)
        plt.show()
        plt.close()


def ratio_plot(y:np.array, pred:np.array, labels:list, path:str):
    # compute ratios b = true / pred
    b = np.divide(y, pred)
    b[~np.isfinite(b)] = 0

    plt.rc('font', size=28) #controls default text size
    plt.rc('axes', labelsize=26)
    plt.rc('xtick', labelsize=26) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=26) #fontsize of the y tick labels

    for resultcomponent in range(b.shape[1]):
        plt.figure(figsize=(10, 8), dpi=100)
        plt.plot(np.linspace(1, b.shape[0], b.shape[0], endpoint=True).reshape(-1,1),b[:,resultcomponent], marker = 'o', ms = 5, linestyle='None', label='ratio')
        plt.plot(np.linspace(1, b.shape[0], b.shape[0], endpoint=True).reshape(-1,1),np.multiply(np.ones(b.shape[0]),np.mean(b[:,resultcomponent])), label='mean', linewidth=8)
        plt.plot(np.linspace(1, b.shape[0], b.shape[0], endpoint=True).reshape(-1,1),np.ones(b.shape[0]), label='ideal', linewidth=6, linestyle='--')
        plt.legend(fontsize=18)
        plt.grid(color='0.5')
        plt.yscale('log')
        plt.ylim(10**-6,10**6)
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'ratio_true_prediction_'+labels[resultcomponent]), dpi=100)
        plt.show()
        plt.close()