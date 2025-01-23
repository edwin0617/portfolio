import os
import re
import pandas as pd
import pathlib
from typing import Dict, Tuple, List
from collections import defaultdict

from utils import *
import torch
import torch.nn as nn
import scipy.io as spio
import warnings

import sklearn.metrics
import sklearn.model_selection



# 추가
#warnings.filterwarnings("ignore", message="your specific warning message")
warnings.filterwarnings("ignore", category=FutureWarning, message="Series.__getitem__ treating keys as positions is deprecated")


# Global variables for data preprocessing
WINDOW_STEP = 0.020 # seconds
WINDOW_LENGTH = 0.050 # seconds
FREQUENCY_SAMPLING = 4000 # Hz, fixed for all recordings
FREQUENCY_HIGH = 800 # Hz
TRAINING_SEQUENCE_LENGTH = 6 # seconds



#####################################################################################################################################
##############################                       ################################################################################
##############################    Make Data Frames   ################################################################################
##############################                       ################################################################################
#####################################################################################################################################


functions = {
    "age": get_age,
    "pregnant": get_pregnancy_status,
    "height": get_height,
    "weight": get_weight,
    "sex": get_sex,
}



def load_patient_files(data_folder, start_frac=0, stop_frac=1):
    # Find the patient data files.
    patient_files = find_patient_files(data_folder) # .txt기준으로 환자파일 수 계산
    num_patient_files = len(patient_files)
    if num_patient_files == 0:
        raise Exception("No data was provided.")

    stop_index = int(stop_frac * num_patient_files)
    start_index = int(start_frac * num_patient_files)
    patient_files = patient_files[start_index: stop_index + 1]
    num_patient_files = len(patient_files)

    rows = {}
    for i in range(num_patient_files):
        # TODO: Skip unknown?

        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        num_locations = get_num_locations(current_patient_data)
        recording_information = current_patient_data.split("\n")[1 : num_locations + 1]

        rec_files = []
        for i in range(num_locations):
            recording_wav = recording_information[i].split(" ")[2]
            if recording_wav in ["50782_MV_1.wav"]:  # no segmentation
                continue # 50782_MV_1의 경우에는 segmentation이 없어서 continue로 append하지 않고 다음 루프로 넘어감
            rec_files.append(recording_wav)

        rows[get_patient_id(current_patient_data)] = {
            "murmur_label": get_murmur(current_patient_data),
            "outcome_label": get_outcome(current_patient_data),
            "systolic_timing": get_murmur_timing(current_patient_data),
            **{k: v(current_patient_data) for k, v in functions.items()},
            "murmur_locations": get_murmur_locations(current_patient_data),
            "recordings": rec_files,
        }

    return pd.DataFrame.from_dict(rows, orient="index")



def patient_df_to_recording_df(patient_df, ):
    
    recording_rows = []
    for row in patient_df.itertuples():
        for recording_path in row.recordings:
            # Get recording location from filename
            recording_loc = re.split("[_.]", recording_path)[1]  # . 또는 _를 기준으로 split
            if recording_loc not in {"AV", "MV", "PV", "TV", "Phc"}:
                raise ValueError(f"Recording loc for {recording_path} is {recording_loc}")

            # Assign murmur label to specific recording
            assert row.murmur_label in {"Present", "Absent", "Unknown"}
            if row.murmur_label == "Present" and recording_loc not in row.murmur_locations:
                rec_murmur_label = "Absent"
            else:
                rec_murmur_label = row.murmur_label
            
            recording_rows.append(
                {
                    "recording": recording_path.replace(".wav", ""), 
                    "murmur_timing": row.systolic_timing,
                    "rec_murmur_label": rec_murmur_label,
                    "patient_murmur_label": row.murmur_label,
                    "val_fold": row.val_fold,
                    
                }
            )
        
    return pd.DataFrame.from_records(recording_rows, index="recording")


# Part 3 df
## Make df for devision tree
def prepare_tree_df(pred_df, patient_df):
    pred_df.index = pred_df.index.str.split("_", expand=True)

    # Calculate difference
    pred_df["conf_difference"] = pred_df["holo_HSMM"] - pred_df["healthy_HSMM"]  # holo_hssmm과 healthy_hsmm 과의 차이 계산
    pred_df["signal_qual"] = pred_df[["holo_HSMM", "healthy_HSMM"]].max(axis=1)  # holo_hsmm vs healty_hsmm 중에서 확률값 더 큰 거

    # Average predictions for recordings at same site
    #pred_df = pred_df.groupby(level=[0, 1]).mean()
    
    # 이걸로 바꾸니까 된다...!!
    ## 추가
    pred_df = pred_df[['healthy_HSMM', 'holo_HSMM', 'conf_difference', 'signal_qual']].groupby(level=[0, 1]).mean()
        
    pred_df = pred_df.unstack()

    # Drop PhC values as not enough of them to train
    pred_df = pred_df.drop(("conf_difference", "Phc"), axis=1, errors="ignore")
    pred_df.columns = [
        "_".join(col) if len(col[1]) > 0 else col[0] for col in pred_df.columns.values
    ]

    combined_df = pd.concat([pred_df, patient_df], axis=1)
    return combined_df





#####################################################################################################################################
##############################                       ################################################################################
##############################  Data Pre-processing  ################################################################################
##############################                       ################################################################################
#####################################################################################################################################



def calc_frequency_bins():
    return int(np.ceil(FREQUENCY_HIGH * WINDOW_LENGTH))

# wav to spectogram
def calculate_features(recording: torch.Tensor, fs: int) -> Tuple[torch.Tensor, int]:
    assert fs == FREQUENCY_SAMPLING

    # Zero-mean and normalise by peak amplitude
    recording = recording.float()
    recording -= recording.mean()
    recording /= recording.abs().max()

    # Calculate spectrogram
    window_length = int(WINDOW_LENGTH * fs)
    window_step = int(WINDOW_STEP * fs)
    spectrogram = (
        torch.stft(
            recording,
            n_fft=window_length,
            hop_length=window_step,
            window=torch.hann_window(window_length),
            center=False,
            return_complex=False,
        )
        .pow(2)
        .sum(dim=-1)
    )

    # Remove high frequencies above FREQUENCY_HIGH Hz
    spectrogram = spectrogram[:calc_frequency_bins()]

    # Log and z-normalise
    spectrogram = torch.log(spectrogram)
    spectrogram = (spectrogram - spectrogram.mean(dim=-1, keepdims=True)) / spectrogram.std(
        dim=-1, keepdims=True
    )

    features_fs = int(1 / WINDOW_STEP)
    return spectrogram, features_fs




def create_folds(patient_df: pd.DataFrame, num_k):
    # deterministic?
    skf = sklearn.model_selection.StratifiedKFold(n_splits= num_k, shuffle=True, random_state=1)
    folds = skf.split(np.zeros(len(patient_df)), patient_df.murmur_label)

    patient_df["val_fold"] = np.nan
    for fold, (train, test) in enumerate(folds):
        patient_df.iloc[test, patient_df.columns.get_loc("val_fold")] = fold
    patient_df["val_fold"] = patient_df["val_fold"].astype(int)

    return patient_df




class RecordingDataset(torch.utils.data.Dataset):                           
    def __init__(self, data_folder, recording_paths, labels, timings, sequence_length, cache=None):
        self.data_folder = pathlib.Path(data_folder)  
        self.recording_paths = recording_paths   
        self.labels = labels
        self.timings = timings
        self.sequence_length = sequence_length
        
        # print(f"\n {len(self.recording_paths), len(self.labels)} \n")
        
        assert len(self.recording_paths) == len(self.labels)

        self.cache = cache
        
    def __len__(self):
        return len(self.recording_paths)

    def __getitem__(self, idx):
        
        filename = self.recording_paths[idx]  
        label = self.labels[idx] 
        timing = self.timings[idx] 

        if self.cache is None or filename not in self.cache:
            filepath = self.data_folder / filename
            fs, recording = spio.wavfile.read(filepath.with_suffix(".wav"))

            recording = torch.as_tensor(recording.copy())
            features, fs_features = calculate_features(recording, fs)

            segmentation_path = filepath.with_suffix(".tsv")
            segmentation_df = pd.read_csv(segmentation_path, sep="\t", header=None)
            segmentation_df.columns = ["start", "end", "state"]
            segmentation_df["start"] = (segmentation_df["start"] * fs_features).astype(int)  #
            segmentation_df["end"] = (segmentation_df["end"] * fs_features).astype(int)  # 

            # features.shape[-1] : frame길이
            
            #기존
            segmentation_label = torch.zeros(features.shape[-1], dtype=torch.long)
            
            
            for row in segmentation_df.itertuples():
                # IN:  noise =  0, S1 = 1, systole = 2, S2 = 3, diastole = 4
                # OUT: noise = -1, S1 = 0, systole = 1, S2 = 2, diastole = 3, murmur = 4
                if row.state == 0:
                    # Noise
                    segmentation_state = -1
                if row.state == 1:
                    # S1
                    segmentation_state = 0
                elif row.state == 3:
                    # S2
                    segmentation_state = 2
                elif row.state == 2:
                    # Systole
                    segmentation_state = 1
                elif row.state == 4:
                    # Diastole
                    segmentation_state = 3

                segmentation_label[row.start : row.end] = segmentation_state

                # TODO: handle diastolic murmurs
                if (row.state == 2) and (label == "Present"):
                    if timing == 'Early-systolic':
                        portion = [0, 0.5]
                    elif timing == 'Holosystolic':
                        portion = [0, 1]
                    elif timing == 'Mid-systolic':
                        portion = [0.25, 0.75]
                    elif timing == 'Late-systolic':
                        portion = [0.5, 1]
                    else:
                        portion = [0, 0]
                        #print(f"Warn: Got timing {timing} for file {filename}")
                        # Diastolic
                        #raise ValueError(f"ERROR: Got timing {timing} for file {filename}")

                    state_duration = row.end - row.start # TODO: inclusive?
                    start = int(row.start + portion[0] * state_duration)
                    end = int(np.ceil(row.start + portion[1] * state_duration))
                    
                    # murmur 라벨링
                    segmentation_label[start: end] = 4

            indices_to_keep = segmentation_label != -1
            segmentation_label = segmentation_label[indices_to_keep]
            
            
            ## indices_to_keep은 noise가 아닌 부분만 남기려고 쓰임 [T,F,T,F,F,...T]
            features = features[..., indices_to_keep]

            self.cache[filename] = (features, segmentation_label)
        else:
            features, segmentation_label = self.cache[filename]


        if self.sequence_length is not None:
            
            # feature.shape[-1] == frame길이
            ## 프레임이 최대길이 300을 넘어가게 되면 그 차이를 계산해서 random_start 정수 추출후 슬라이싱
            
            
            random_start = torch.randint(
                low=0, high=max(features.shape[-1] - self.sequence_length, 1), size=(1,)
            ).item()
            
            features = features[..., random_start : random_start + self.sequence_length]
            segmentation_label = segmentation_label[
                random_start : random_start + self.sequence_length
            ]

        return features, segmentation_label
    
    

def batch_whole_dataset(dataset: RecordingDataset):
    num_examples = len(dataset)
    return collate_fn([dataset[i] for i in range(num_examples)])


def collate_fn(x):
    max_length = 0
    all_features = []
    all_labels = []
    actual_lengths = []

    # x는 전처리 완료된 데이터셋, 로더에 들어가기 직전임 
    for features, labels in x:
        all_features.append(features)
        all_labels.append(labels)
        max_length = max(max_length, len(labels))
        actual_lengths.append(len(labels))
        
    padded_features = all_features[0].new_full((len(x), all_features[0].shape[0], max_length), 0.0)
    #For CRNN
    #padded_features = all_features[0].new_full((len(x), all_features[0].shape[0], 3300), 0.0)
    
    
    for i, features in enumerate(all_features):
        padded_features[i, ..., : features.shape[-1]] = features

    padded_labels = all_labels[0].new_full((len(x), max_length), -1)
    #For CRNN
    #padded_labels = all_labels[0].new_full((len(x), 3300), -1)
    
    for i, labels in enumerate(all_labels):
        padded_labels[i, : labels.shape[-1]] = labels

    return padded_features, padded_labels, torch.as_tensor(actual_lengths)



def calculate_class_weights(dataset):
    old_sequence_length = dataset.sequence_length
    dataset.sequence_length = None

    class_counts = defaultdict(int)
    for i in range(len(dataset)):
        _, label = dataset[i]
        unique, counts = torch.unique(label, return_counts=True)
        for uniq, count in zip(unique, counts):
            class_counts[uniq.item()] += count

    # May want to further adjust these weights as detecting murmur is v. important
    total_sum = sum(class_counts.values())
    class_weights = torch.zeros(5)
    for k, v in class_counts.items():
        class_weights[k] = total_sum / v

    # Restore sequence length
    dataset.sequence_length = old_sequence_length

    return class_weights
