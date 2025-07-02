import sys
from data_storage import ngs_path_collection, virus_seqs, random_mutant_dataset
from bio_utils import seqlogo_from_msa
import logomaker as lm
# from src.sequencing_analysis_utils import ngs_expr_set 
from ngs_analysis import ngs_expr_set, ngs_expr
from sec_structure import natr_struc
from utils import save_fig, defaultStyle
import pandas as pd
import logomaker as lm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import product
import RNA
from scipy.stats import pearsonr, ttest_ind
import math
from scipy.spatial.distance import pdist, squareform
from scipy import stats
from sklearn.metrics import r2_score
from itertools import product
import subprocess

import torch
import os
from model_config import ModelConfig, model_collections
from data_storage import random_mutant_dataset, virus_seqs
import data
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from pathlib import Path
from scipy.stats import pearsonr, mode
from math import sqrt, floor, ceil
from tqdm import tqdm
from collections import Counter
defaultStyle(fs=18)
import warnings
warnings.filterwarnings("ignore")

virus_name, date = 'VEE', '0412'
print(f"Analysis of {virus_name} sequencing data on {date}")
ngs = ngs_expr_set(virus_name, date, downstream_seq_len=7)
original_seq = virus_seqs(virus_name).original_seq
downstream_seq  = virus_seqs(virus_name).downstream_seq 
original_seq_counts_df = pd.DataFrame(columns=['A', 'C', 'G', 'T'])
# plasmid_seq_dict = ngs.plasmid_seq_dict
# rna_seq_dict = ngs.rna_seq_dict
# calculate_replication_score = ngs.calculate_replication_score()
# filter_freq_data = ngs.filter_freq_data()
# print(rna_seq_dict)

test_path1 = f"./data/VEE_0816.csv"
data1 = pd.read_csv(test_path1)
data1 = data1[data1['rna_counts'] > 14]
data1 = data1.sort_values(by = [ 'distance' , 'rna_counts' ], ascending= ( True , False ))

test_data = pd.DataFrame()
train_data = pd.DataFrame()
for i in range(1 , 30 , 1):
    data2 = data1.loc[data1['distance'] == i]
    for j in range(int(len(data2) / 20 )):
        df1 = data2.iloc[1 + j * 19]  
        test_data = test_data.append(df1)
df2 = data1.append(test_data)
train_data = df2.drop_duplicates(subset=['seqid'],keep=False)
train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
train_size = len(train_data)
test_data.to_csv("./data/VEE_0816_test.csv", index = False)
train_data.to_csv("./data/VEE_0816_train.csv", index = False)

for num in [10,31,100,316,1000,3160,10000]:
    subprocess.run(
        f''' \
        python ./src/train.py deepcnnlstm cv -k 5 --virus-name=VEE --date=0816 --num-train={num} --save \
        ''',
        shell=True
    )