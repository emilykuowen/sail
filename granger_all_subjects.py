import pickle
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import csv

# inputs
BRAIN_DIR = "./OFC_Brain_signals_40_subj/"
EMO_DIR = "./emo_downsampled/"
ENJOY_DIR = "./enjoy_downsampled/"
CSV_DIR = "./agModels_csv/"

"""
Tasks:
1. Choose one subject, combine the subject's brain data from all 3 songs, run granger on "emotion vs. brain" and "enjoyment vs. brain"
2. Look at the plots of people who are ranked low (lowest f score and highest p value)
3. Look at only the sad long song for each subject 
4. Use features that produce p value < 0.05 in linear regression and see if that gives better results than before
"""

song_list = ["happy", "sadln", "sadsh"]
song_length = [138, 484, 225] # length of each song (excluding the first 30 seconds)

data = pd.read_csv(CSV_DIR + "train_data_sadln_all_subjects.csv")
features = list(data.columns)
features.remove("emotion")
features.remove("brain_t1")
features.remove("brain_t2")
features.remove("brain_t3")
features.remove("brain_t4")
features.remove("brain_t5")
features.remove("LPC0") # because it is constant

num_lag = 8
f_list = []
f_num_lag_list = []
p_list = []
p_num_lag_list = []

for feature in features:
    two_columns = data[["emotion", feature]]
    result = grangercausalitytests(two_columns, num_lag, verbose=False)

    max_f_score = 0
    max_f_score_num_lag = 0
    min_p_value = 10
    min_p_value_num_lag = 0

    for i in range(1, num_lag+1):
        f_score = result[i][0]['ssr_ftest'][0] #ssr based F test
        p_value = result[i][0]['ssr_ftest'][1]
        if f_score > max_f_score:
            max_f_score = f_score
            max_f_score_num_lag = i
        if p_value < min_p_value:
            min_p_value = p_value
            min_p_value_num_lag = i

    f_list.append(max_f_score)
    f_num_lag_list.append(max_f_score_num_lag)
    p_list.append(min_p_value)
    p_num_lag_list.append(min_p_value_num_lag)

granger_results = list(zip(features, f_list, f_num_lag_list, p_list, p_num_lag_list))
df = pd.DataFrame(data = granger_results, columns=['feature', 'f score', 'f num lag', 'p value', 'p num lag'])

with open("granger_sorted_results_with_all_subjects.txt", "w") as f:
    print("Sorted by f score: ", file=f)
    sorted_by_f = df.sort_values(['f score'], ascending=False)
    print(sorted_by_f[:20], file=f)

    print("\nSorted by p value: ", file=f)
    sorted_by_p = df.sort_values(['p value'], ascending=True)
    print(sorted_by_p[:20], file=f)
