import pickle
import pandas as pd
from os import listdir
from os.path import isfile, join
import csv
import autogluon as ag
from autogluon import TabularPrediction as task
import matplotlib.pyplot as plt

# Goal: use brain data and musical features to predict emotions

# inputs
BRAIN_DIR = "./OFC_Brain_signals_40_subj/"
EMO_DIR = "./emo_downsampled/"
MUSIC_FEATURES_DIR = "./music_features/"

# outputs
CSV_DIR = "./agModels_csv/"

total_subjects = 37 # total number of subjects in the dataset (3 subjects are missing in the brain data)
test_subjects = 4 # the amount of subjects to be included in the test data
num_subjects = total_subjects - test_subjects

# read the filenames of brain data
brain_filenames = [f for f in listdir(BRAIN_DIR) if isfile(join(BRAIN_DIR, f))]
brain_filenames.sort()

song_list = ["happy", "sadln", "sadsh"]
song_length = [138, 484, 225] # length of each song (excluding the first 30 seconds)

# add zeros for autoregressive model
def create_column_with_zeros(data, songLength, numZero):
    column = []
    i = 0
    while(i < len(data)):
        for j in range(numZero):
            column.append(0)
        end_index = i + songLength - numZero
        column.extend(data[i:end_index])
        i += songLength
    return column

# Goal: generate/prepare csv files for autogluon models

# read music features' names (only need to read one song since all songs have the same music features)
filename = "vggish_happy.csv"
music_df = pd.read_csv(MUSIC_FEATURES_DIR + filename)

for i in range(0, len(song_list)):
    song = song_list[i]
    length = song_length[i]
    full_length = length + 30
    
    # read music features data
    all_music_data = []
    # music_features = pickle.load(open(MUSIC_FEATURES_DIR + "downsampled/music_features_1Hz_" + song + ".p", "rb"))
    music_data = pickle.load(open(MUSIC_FEATURES_DIR + "vggish_" + song + ".p", "rb"))
    num_features = len(music_data)

    for j in range(0, num_features):
        all_music_data.append(music_data[j][30:full_length] * num_subjects)
    
    all_brain_data = []
    for filename in brain_filenames:
        if song in filename:
            brain_data = []
            with open(BRAIN_DIR + filename, "r") as newFile:
                for x in newFile.readlines():
                    brain_data.append(float(x))
                all_brain_data.extend(brain_data[30:full_length]) # disregard the first 30 seconds to account for 
                                                                      # adjustment in the beginning
    emo_path = EMO_DIR + song + "/"
    emo_filenames = [f for f in listdir(emo_path) if isfile(join(emo_path, f))]
    emo_filenames.sort()

    all_emo_ratings = []

    for filename in emo_filenames:
        # skip the subjects that don't have brain data (brain data is missing subject 7, 28, 30)
        if "07" not in filename and "28" not in filename and "30" not in filename:
            emo_ratings = pickle.load(open(EMO_DIR + song + "/" + filename, "rb"))
            all_emo_ratings.extend(emo_ratings[30:full_length]) # disregard the first 30 seconds to account for 
                                                                # adjustment in the beginning

    features = ["emotion", "brain"]
    brain_data_to_csv = []
    music_features_names = list(music_df.columns)

    for i in range(0, 6):
        features.append("brain_t" + str(i))
        brain_data = create_column_with_zeros(all_brain_data, length, i)
        brain_data_to_csv.append(brain_data)
    
    music_data_to_csv = []

    for i in range(0, len(music_features_names)):
        for j in range(0, 6):
            features.append(music_features_names[i] + "_t" + str(j))
            music_data = create_column_with_zeros(all_music_data[i], length, j)
            music_data_to_csv.append(music_data)

    # with open(CSV_DIR + "train_data_" + song + "_all_subjects.csv", "w") as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     csvwriter.writerow(features) # write header row
    #     # Data from subjects #1-#40 (excluding 7, 28, 30)
    #     csvwriter.writerows(zip(all_emo_ratings, all_brain_data, brain_t1, brain_t2, 
    #                         brain_t3, brain_t4, brain_t5, *all_music_data))

    test_threshold = num_subjects * length

    # training data
    with open(CSV_DIR + "4_subjects_removed/train_data_" + song + "_vggish_autoregressive.csv", "w") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(features) # write header row
        # Data from subjects #1-#36 (excluding 7, 28, 30)
        csvwriter.writerows(zip(all_emo_ratings[0:test_threshold], *brain_data_to_csv[0:test_threshold], *music_data_to_csv))
    
    # testing data
    with open(CSV_DIR + "4_subjects_removed/test_data_" + song + "_vggish_autoregressive.csv", "w") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(features) # write header row
        # Data from subjects #37-#40
        csvwriter.writerows(zip(all_emo_ratings[test_threshold:], *brain_data_to_csv[test_threshold:], *music_data_to_csv))

# Goal: run Autogluon
with open("ag_stats_only_mir.txt", 'w') as stats:
# with open("ag_stats_vggish.txt", 'w') as stats:
    for song in song_list:
        # Using MIR features
        train_data_file = "4_subjects_removed/train_data_" + song + "_vggish_autoregressive.csv"
        # Using VGGish features
        # train_data_file = "4_subjects_removed/train_data_" + song + "_vggish.csv"

        train_data = task.Dataset(file_path=CSV_DIR+train_data_file)
        dir = "agModels_predictEmotion" # specifies folder where to store trained models
        # Run AutoGluon
        predictor = task.fit(train_data=train_data, label="emotion", problem_type="regression", hyperparameter_tune=False, output_directory=dir)

        # Using MIR features
        test_data_file = "4_subjects_removed/test_data_" + song + "_vggish_autoregressive.csv"
        # Using VGGish features
        # test_data_file = "4_subjects_removed/test_data_" + song + "_vggish.csv"

        test_data = task.Dataset(file_path=CSV_DIR+test_data_file)
        y_test = test_data["emotion"]  # values to predict
        test_data_nolab = test_data.drop(labels=["emotion"],axis=1) # delete label column to prove we're not cheating
        y_pred = predictor.predict(test_data_nolab)

        performance = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
        print(train_data_file, file=stats)
        print(test_data_file, file=stats)
        for key, value in performance.items(): 
            print("\t", key, " = ", value, file=stats) 
        print("\n", file=stats)

        # original values in green, predicted values in blue
        plt.plot(y_test, 'g:', y_pred, 'b:')
        plt.ylabel('emotion')

        # Using MIR features
        # plt.savefig("agModels_graphs/4_subjects_removed_" + song + ".png")
        # Using VGGish features
        plt.savefig("agModels_graphs/4_subjects_removed_" + song + "_vggish_autoregressive.png")

        plt.close()