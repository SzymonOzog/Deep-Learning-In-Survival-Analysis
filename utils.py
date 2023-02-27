import torch 
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pycox import datasets
from sklearn.model_selection import train_test_split

def get_unprocessed_dataset():
    df = pd.read_csv("brca_metabric/brca_metabric_clinical_data.tsv", sep="\t")
    return df, None, "Overall Survival (Months)"

def get_processed_dataset():
    df = get_unprocessed_dataset()[0]
    df["Censorship"] = df["Patient's Vital Status"] == "Living"
    df_clear = df.dropna()
    df_clear = df_clear.drop(["Study ID", "Patient ID", "Sample ID","Overall Survival Status", "Patient's Vital Status", "Cancer Type", "Number of Samples Per Patient", "Sex", "Sample Type"
    , "Cancer Type Detailed", "Tumor Other Histologic Subtype", "Oncotree Code", "Relapse Free Status", "Relapse Free Status (Months)", "TMB (nonsynonymous)"], axis = 1)
    df_clear = df_clear.drop(["Type of Breast Surgery", "Cellularity", "HER2 Status", "Integrative Cluster", "HER2 status measured by SNP6", "Pam50 + Claudin-low subtype", "3-Gene classifier subtype", "ER Status"], axis = 1)
    # replace Pam50 + Claudin-low subtype == NC with aNC , "Pam50 + Claudin-low subtype"
    #df_clear["HER2 status measured by SNP6"] = df_clear["HER2 status measured by SNP6"].replace("UNDEF", "AUNDEF")
    #df_clear["Pam50 + Claudin-low subtype"] = df_clear["Pam50 + Claudin-low subtype"].replace("NC", "ANC")
    df_clear = pd.get_dummies(df_clear, drop_first=True)
    return df_clear, "Censorship", "Overall Survival (Months)"

def get_deep_surv_processed_dataset():
    return datasets.metabric.read_df(), "event", "duration"

def get_deep_hit_processed_dataset():
    return pd.read_csv("METABRIC_DeepHit/features.csv").join(pd.read_csv("METABRIC_DeepHit/labels.csv")), "event_time", "label"

def split_dataset(df):
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    return train, test