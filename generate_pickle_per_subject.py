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

song_list = ["happy", "sadln", "sadsh"]
song_length = [138, 484, 225] # length of each song (excluding the first 30 seconds)

for subject_num in range(1,41):
    if subject_num != 7 and subject_num != 28 and subject_num != 30:
        brain_data_from_all_songs = []
        emo_ratings_from_all_songs = []
        enjoy_ratings_from_all_songs = []

        for i in range(0, len(song_list)):
            song = song_list[i]
            length = song_length[i]
            full_length = length + 30

            if subject_num < 10: label = "0" + str(subject_num)
            else: label = str(subject_num)
            brain_filename = "sub-" + label + "_" + song + "_ts_aroma.txt"
            brain_data = []

            try:
                with open(BRAIN_DIR + brain_filename, "r") as newFile:
                    for x in newFile.readlines():
                        brain_data.append(float(x))
                brain_data = brain_data[30:full_length]
                brain_data_from_all_songs.extend(brain_data)
            except FileNotFoundError as not_found:
                print("Can't find " + not_found.filename)
            
            if song == "happy":
                emo_filename = "sub-" + label + "_hnl_n_emo_log_downsampled.p"
            elif song == "sadln":
                emo_filename = "sub-" + label + "_snl_l_emo_log_downsampled.p"
            elif song == "sadsh":
                emo_filename = "sub-" + label + "_snl_s_emo_log_downsampled.p"

            emo_ratings = pickle.load(open(EMO_DIR + song + "/" + emo_filename, "rb"))
            emo_ratings_from_all_songs.extend(emo_ratings[30:full_length])
        
            if song == "happy":
                enjoy_filename = "sub-" + label + "_hnl_n_enjoy_log_downsampled.p"
            elif song == "sadln":
                enjoy_filename = "sub-" + label + "_snl_l_enjoy_log_downsampled.p"
            elif song == "sadsh":
                enjoy_filename = "sub-" + label + "_snl_s_enjoy_log_downsampled.p"

            enjoy_ratings = pickle.load(open(ENJOY_DIR + song + "/" + enjoy_filename, "rb"))
            enjoy_ratings_from_all_songs.extend(enjoy_ratings[30:full_length])

        features = ["emotion", "enjoy", "brain"]
        with open(CSV_DIR + "sub-" + label + ".csv", "w") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(features) # write header row
            csvwriter.writerows(zip(emo_ratings_from_all_songs, enjoy_ratings_from_all_songs, brain_data_from_all_songs))