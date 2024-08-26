# This program trains the HMM model used for regime analysis.
# The data for training is the implied volatility of SP500.
# The time period is.

import datetime
import pickle
import warnings
import random

from hmmlearn.hmm import GaussianHMM, GMMHMM
from hmmlearn.base import ConvergenceMonitor
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
from sklearn.utils import check_random_state

import numpy as np
import pandas as pd
import seaborn as sns


def plot_in_sample_hidden_states(hmm_model, df, rets):
    """
    Plot the adjusted closing prices masked by 
    the in-sample hidden states as a mechanism
    to understand the market regimes.
    """
    # Predict the hidden states array
    hidden_states = hmm_model.predict(rets)
    # Create the correctly formatted plot
    fig, axs = plt.subplots(
        hmm_model.n_components, 
        sharex=True, sharey=True
    )
    colours = cm.rainbow(
        np.linspace(0, 1, hmm_model.n_components)
    )
    for i, (ax, colour) in enumerate(zip(axs, colours)):
        mask = hidden_states == i
        ax.plot_date(
            df.index[mask], 
            df["Close"][mask], 
            ".", linestyle='none', 
            c=colour
        )
        ax.set_title("Hidden State #%s" % i)
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.grid(True)
    plt.show()
    
def plot_hidden_states(n_components, hidden_states, df):
    """
    Plot the adjusted closing prices masked by 
    the in-sample hidden states as a mechanism
    to understand the market regimes.
    """
    # Create the correctly formatted plot
    fig, axs = plt.subplots(
        n_components, 
        sharex=True, sharey=True
    )
    colours = cm.rainbow(
        np.linspace(0, 1, n_components)
    )
    for i, (ax, colour) in enumerate(zip(axs, colours)):
        mask = hidden_states == i
        ax.plot_date(
            df.index[mask], 
            df["Close"][mask], 
            ".", linestyle='none', 
            c=colour
        )
        ax.set_title("Hidden State #%s" % i)
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.grid(True)
    plt.show()
    

def hmm_train(x, n_components=2, n_iter=1500, M=30, rs=None):
    warnings.filterwarnings("ignore")
    randomstate = check_random_state(rs)
    # Train several times and select the model with the highest score
    best_model = None
    best_score = -np.inf
    for i in range(M):
        
        h = GaussianHMM(
            n_components=n_components, covariance_type="full", n_iter=n_iter, random_state=randomstate
        ).fit(x)
        '''
        h = GMMHMM(
            n_components=n_components, n_mix=4, covariance_type="full", n_iter=n_iter, random_state=randomstate
        ).fit(x)
        '''

        score = h.score(x)
        if score > best_score:
            best_score = score
            best_model = h
    
    # Define the state with lower volatility as state 1, and state with higher volatility as state 0
    hidden_states = h.predict(x)
        
    # state0 = x.loc[hidden_states == 0, x.columns[0]]
    # state1 = x.loc[hidden_states == 1, x.columns[0]]
    
    state0 = x[hidden_states == 0]
    state1 = x[hidden_states == 1]
    #y=(len(state0),len(state1))
    
    reverse = (np.std(state1) > np.std(state0))
    
    return h, reverse
    
def hmm_predict(hmm_model, new_x):
    """
    Given new x, return the predicted hidden state
    """
    # Predict the one_step hidden state
    next_states = hmm_model.predict(new_x)
    return next_states[-1]

def out_of_sample_test():
    pass


def decide_components(X, seeds=27):
    rs = check_random_state(seeds)
    
    aic = []
    bic = []
    lls = []
    ns = [2,3,4,5]
    
    for n in ns:
        best_ll = None
        best_model = None
        
        # Repeat training the model with different random initializations and select the best model
        for i in range(30):
            h = GaussianHMM(n, n_iter=1500, random_state=rs)
            # h = GaussianHMM(n, n_iter=1500, random_state=random.randint(0,1000))
            h.fit(X)

            score = h.score(X)
            if not best_ll or best_ll < best_ll:
                best_ll = score
                best_model = h
        
        # Collect the aic, bic and lls of the models with different n_components        
        aic.append(best_model.aic(X))
        bic.append(best_model.bic(X))
        lls.append(best_model.score(X))
        
    # Plot the results
    fig, ax = plt.subplots()
    ln1 = ax.plot(ns, aic, label="AIC", color="blue", marker="o")
    ln2 = ax.plot(ns, bic, label="BIC", color="green", marker="o")
    ax2 = ax.twinx()
    ln3 = ax2.plot(ns, lls, label="LL", color="orange", marker="o")
    # ln4 = ax2.plot(ns, converged, label="Converged", color="purple", marker="o")

    ax.legend(handles=ax.lines + ax2.lines)
    ax.set_title("Using AIC/BIC for Model Selection")
    ax.set_ylabel("Criterion Value (lower is better)")
    ax2.set_ylabel("LL (higher is better)")
    ax.set_xlabel("Number of HMM Components")
    fig.tight_layout()

    plt.show()



if __name__ == '__main__':
    # Parameters
    filepath = r"AssetData.csv"
    start_date = datetime.datetime(2017, 7, 3)
    end_date = datetime.datetime(2023, 7, 3)
    
    # Hides deprecation warnings for sklearn
    warnings.filterwarnings("ignore")
    
    # Import data
    # ------------------------------To be changed-------------------------------------------------
    vix = pd.read_csv(
        filepath, header=0, parse_dates = ["Date"]
        )
    vix.set_index("Date", inplace=True)
    
    vix = vix.loc[start_date:end_date]

    #rets = vix[["Returns", "VIX"]]
    rets = np.column_stack([vix["Returns"]])
    
    # Train the model and plot it
    rs = check_random_state(27)
    hmm_model, reverse = hmm_train(rets, n_components=2, n_iter=1500, M=50)
    print("Model Score:", hmm_model.score(rets))
    print(f'Transmission Matrix Recovered:\n{hmm_model.transmat_.round(3)}\n\n')
    print("Reverse:", reverse)
  
    # Plot the hidden states closing values on the validation set
    plot_in_sample_hidden_states(hmm_model, vix, rets)


    
    