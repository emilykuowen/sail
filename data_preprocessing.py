import pickle
import csv
import pandas as pd
from os import listdir
from os.path import isfile, join
from pathlib import Path

def normalize(data):
    min_val = min(data)
    range_val = max(data) - min_val
    # print("min = " + str(min_val))
    # print("max = " + str(max(data)))
    # print("range = " + str(range_val))
    for i in range(len(data)):
        data[i] = (data[i] - min_val) / float(range_val)

# inputs
BRAIN_DIR = "./OFC_Brain_signals_40_subj/"
EMO_ENJOY_DIR = "./emo_enjoy_ratings_40_subj/"
MUSIC_FEATURES_DIR = "./music_features/"

# outputs
BRAIN_OUTPUT_DIR = "./brain_normalized/"
EMO_OUTPUT_DIR = "./emo_downsampled/"
ENJOY_OUTPUT_DIR = "./enjoy_downsampled/"
MUSIC_FEATURES_OUTPUT_DIR = "./music_features/downsampled/"

song_list = ["happy", "sadln", "sadsh"]

# Normalize brain data
filenames = [f for f in listdir(BRAIN_DIR) if isfile(join(BRAIN_DIR, f))]

# index 0 = happy, index 1 = sad long, index 2 = sad short
for i in range(3):
    song = song_list[i]
    # Create output directory for this song if not already exists
    Path(BRAIN_OUTPUT_DIR + song).mkdir(parents=True, exist_ok=True)
    Path(EMO_OUTPUT_DIR + song).mkdir(parents=True, exist_ok=True)
    Path(ENJOY_OUTPUT_DIR + song).mkdir(parents=True, exist_ok=True)

# Generate pickle files of normalized brain data in the brain_normalized folder
for filename in filenames:
    file_path = BRAIN_DIR + filename
    newFile = open(file_path, "r")
    data = []
    for x in newFile.readlines():
        data.append(float(x))
    normalize(data)

    subfolder = ""
    if "happy" in filename:
        subfolder = "happy/"
    elif "sadln" in filename:
        subfolder = "sadln/"
    elif "sadsh" in filename:
        subfolder = "sadsh/"

    filename = filename.split(".")[0]
    with open(BRAIN_OUTPUT_DIR + subfolder + filename + "_normalized.p", "wb") as f:
        pickle.dump(data, f)

# Downsample emotion ratings from 30Hz to 1Hz
filenames = [f for f in listdir(EMO_ENJOY_DIR) if isfile(join(EMO_ENJOY_DIR, f))]

for filename in filenames:
    print(filename)
    downsampled = []
    if not filename.startswith("."): # to avoid hidden files (ex. .DS_Store)
        with open(EMO_ENJOY_DIR + filename, "r") as newFile:
            line = newFile.readline()
            second = 1
            while line:
                l = line.split()
                timestamp = float(l[0])
                if timestamp > second:
                    rating = float(l[1]) # emotion/enjoyment rating from 0 to 127
                    downsampled.append(rating)
                    second += 1
                line = newFile.readline()

    length = len(downsampled)
    print("\t" + str(length))

    subfolder = ""
    if "hnl" in filename:
        subfolder = "happy/"
    elif "snl_l" in filename:
        subfolder = "sadln/"
    elif "snl_s" in filename:
        subfolder = "sadsh/"

    filename = filename.split(".")[0]
    if "emo" in filename:
        with open(EMO_OUTPUT_DIR + subfolder + filename + "_downsampled.p", "wb") as f:
            pickle.dump(downsampled, f)
    elif "enjoy" in filename:
        with open(ENJOY_OUTPUT_DIR + subfolder + filename + "_downsampled.p", "wb") as f:
            pickle.dump(downsampled, f)

# Downsample music features from 40 Hz to 1 Hz
Path(MUSIC_FEATURES_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

for song in song_list:
    raw_file = "X_matrix_" + song + ".csv"
    df = pd.read_csv(MUSIC_FEATURES_DIR + raw_file)
    features = list(df.columns)

    Hz = 40
    data = []

    for feature in features:
        downsampled = []
        start = 0
        while start + Hz < len(df):
            total = 0
            i = 0
            while i < Hz:
                total += df[feature][start + i]
                i += 1
            mean = total/Hz
            downsampled.append(mean)
            start += Hz
        data.append(downsampled)

    output_file = "music_features_1Hz_" + song
    with open(MUSIC_FEATURES_OUTPUT_DIR + output_file + ".p", "wb") as f:
        pickle.dump(data, f)

    with open(MUSIC_FEATURES_OUTPUT_DIR + output_file + ".csv", "w") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(features)
        csvwriter.writerows(zip(*data))

# Convert VGGish features' csv files to pickle files
for song in song_list:
    filename = "vggish_" + song
    df = pd.read_csv(MUSIC_FEATURES_DIR + filename + ".csv")
    features = list(df.columns)

    data = []
    for feature in features:
        column = []
        i = 0
        while i < len(df):
            value = df[feature][i]
            column.append(value)
            i = i + 1
        data.append(column)

    with open(MUSIC_FEATURES_DIR + filename + ".p", "wb") as f:
        pickle.dump(data, f)
