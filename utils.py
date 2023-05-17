import torch 
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pycox import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import sksurv
import sksurv.datasets

def get_unprocessed_metabric():
    df = pd.read_csv("brca_metabric/brca_metabric_clinical_data.tsv", sep="\t")
    return df, None, "Overall Survival (Months)"

def get_metabric(missing_values_strategy="mean"):
    df = get_unprocessed_metabric()[0]
    df["Censorship"] = df["Patient's Vital Status"] == "Living"
    df_clear = handle_missing_values(df, missing_values_strategy)
    df_clear = df_clear.drop(["Study ID", "Patient ID", "Sample ID","Overall Survival Status", "Patient's Vital Status", "Cancer Type", "Number of Samples Per Patient", "Sex", "Sample Type"
    , "Cancer Type Detailed", "Tumor Other Histologic Subtype", "Oncotree Code", "Relapse Free Status", "Relapse Free Status (Months)", "TMB (nonsynonymous)"], axis = 1)
    #df_clear = df_clear.drop(["Type of Breast Surgery", "Cellularity", "HER2 Status", "Integrative Cluster", "HER2 status measured by SNP6", "Pam50 + Claudin-low subtype", "3-Gene classifier subtype", "ER Status"], axis = 1)
    # replace Pam50 + Claudin-low subtype == NC with aNC , "Pam50 + Claudin-low subtype"
    df_clear["HER2 status measured by SNP6"] = df_clear["HER2 status measured by SNP6"].replace("UNDEF", "AUNDEF")
    df_clear["Pam50 + Claudin-low subtype"] = df_clear["Pam50 + Claudin-low subtype"].replace("NC", "ANC")
    df_clear = pd.get_dummies(df_clear, drop_first=True)
    return df_clear, "Censorship", "Overall Survival (Months)"

def get_flchain(missing_values_strategy="mean"):
    df, y = sksurv.datasets.load_flchain()
    df = df.join(pd.DataFrame(y))
    df = handle_missing_values(df, missing_values_strategy)
    df = df.drop(["chapter"], axis = 1)
    return pd.get_dummies(df, drop_first=True), "death", "futime"

def get_aids(missing_values_strategy="mean"):
    df, y = sksurv.datasets.load_aids()
    df = df.join(pd.DataFrame(y))
    df = handle_missing_values(df, missing_values_strategy)
    df = pd.get_dummies(df, drop_first=True)
    df = df.drop(["txgrp_3","txgrp_4"], axis = 1)
    return df, "censor", "time"

def get_gbsg2(missing_values_strategy="mean"):
    df, y = sksurv.datasets.load_gbsg2()
    df = df.join(pd.DataFrame(y))
    df = handle_missing_values(df, missing_values_strategy)
    df = pd.get_dummies(df, drop_first=True)
    return df, "cens", "time"

def get_whas500(missing_values_strategy="mean"):
    df, y = sksurv.datasets.load_whas500()
    df = df.join(pd.DataFrame(y))
    df = handle_missing_values(df, missing_values_strategy)
    df = pd.get_dummies(df, drop_first=True)
    return df, "fstat", "lenfol"

def get_veterans_lugn_cancer(missing_values_strategy="mean"):
    df, y = sksurv.datasets.load_veterans_lung_cancer()
    df = df.join(pd.DataFrame(y))
    df = handle_missing_values(df, missing_values_strategy)
    df = pd.get_dummies(df, drop_first=True)
    return df, "Status", "Survival_in_days"

def handle_missing_values(df, strategy="mean"):
    if strategy == "mean":
        return df.fillna(df.mean())
    elif strategy == "median":
        return df.fillna(df.median())
    elif strategy == "mode":
        return df.fillna(df.mode())
    elif strategy == "drop":
        return df.dropna()
    return df

def split_dataset(df):
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    return train, test   

def discretize_time(time, num_bins=10):
    return pd.cut(time, num_bins, labels=False)

#return the mask for deep hit loss
#for uncensored mask is 1 when time is equal to the event time
# for censored mask is 1 when time is greater than censoring time
def create_mask(times, events):
    times = torch.nn.functional.one_hot(torch.tensor(times))
    mask = torch.zeros(times.shape)
    for i in range(times.shape[0]):
        if events[i] == 0:
            
            mask[i, times[i].tolist().index(1) + 1 : ] = 1
        else:
            mask[i, times[i].tolist().index(1)] = 1
    return mask

def create_surv_df(output, dt, interpolation_steps):
    output= output.detach().numpy()
    id = [i * dt for i in range(output.shape[1] )]

    interp_id =[i * dt / interpolation_steps for i in range(output.shape[1] * interpolation_steps)]
    interpolated_output = np.array([np.interp(interp_id, id, output[i]) for i in range(output.shape[0])])
    
    SurvFN = 1. - np.cumsum(interpolated_output, axis=1)

    return pd.DataFrame(SurvFN, columns=interp_id).transpose()
    
def plot_history(history, title):
    plt.figure(figsize=(18, 5))
    plt.suptitle(title)
    plt.tight_layout()

    plt.subplot(1, 3, 1)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    
    plt.subplot(1, 3, 2)
    plt.plot(history['c_index'])
    plt.plot(history['val_c_index'])
    plt.title('model c-index')
    plt.ylabel('c-index')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')

    plt.subplot(1, 3, 3)
    plt.plot(history['lr'])
    plt.title('model learning rate')
    plt.ylabel('lr')
    plt.xlabel('epoch')
    plt.show()

class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, score)
        elif score < self.best_score + self.delta:
            self.counter += 1
            #print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.save_checkpoint(model, score)
            self.counter = 0
        return False
    
    def save_checkpoint(self, model, score):
        torch.save(model.state_dict(), f'checkpoint.pt')