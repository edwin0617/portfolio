# packages
import os
import re
import random
import librosa
import pandas as pd
import pathlib
from typing import Dict, Tuple, List
from collections import defaultdict
import yaml 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler

import scipy.io as spio
import warnings

import sklearn.metrics
import sklearn.model_selection


# ------------------------------------------------------------------------------------------------------------------------------
from utils import *



# 추가
#warnings.filterwarnings("ignore", message="your specific warning message")
warnings.filterwarnings("ignore", category=FutureWarning, message="Series.__getitem__ treating keys as positions is deprecated")


default_configs_dir= "/Data1/hmd2/notebooks_th/GRU_loss_add/config_default.yaml"


with open(default_configs_dir, "r") as f:
    configs_default= yaml.safe_load(f)


# # Global variables for data preprocessing
# WINDOW_STEP = 0.020 # seconds  # 0.020
# WINDOW_LENGTH = 0.050 # seconds #  0.050
# FREQUENCY_SAMPLING = 4000 # Hz, fixed for all recordings
# FREQUENCY_HIGH = 800 # Hz
# TRAINING_SEQUENCE_LENGTH = 3 # seconds


WINDOW_STEP = configs_default["data_preprocess"]["window_step"]
WINDOW_LENGTH = configs_default["data_preprocess"]["window_length"]
FREQUENCY_SAMPLING = configs_default["data_preprocess"]["sampling_rate"]
FREQUENCY_HIGH = configs_default["data_preprocess"]["frequency_high"]
TRAINING_SEQUENCE_LENGTH = configs_default["train_params"]["train_seq_length"]


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
                    "outcome_label": row.outcome_label,
                    "val_fold": row.val_fold,   
                }
            )
        
    return pd.DataFrame.from_records(recording_rows, index="recording")



def test_patient_df_to_recording_df(patient_df):
    
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
                    "outcome_label": row.outcome_label,                    
                }
            )
        
    return pd.DataFrame.from_records(recording_rows, index="recording")


# Part 3 df
## Make df for devision tree
def prepare_tree_df(pred_df, patient_df):
    pred_df.index = pred_df.index.str.split("_", expand=True)

    # Calculate difference
    pred_df["conf_difference"] = pred_df["holo_HSMM"] - pred_df["healthy_HSMM"]  # holo_hssmm과 healthy_hsmm과의 차이 계산
    pred_df["signal_qual"] = pred_df[["holo_HSMM", "healthy_HSMM"]].max(axis=1)  

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





def merge_df(val_results_df, patient_df):
    val_results_df.index = val_results_df.index.str.split("_", expand=True)
    val_results_df["conf_difference"] = val_results_df["holo_HSMM"] - val_results_df["healthy_HSMM"]
    val_results_df = val_results_df[['healthy_HSMM', 'holo_HSMM', 'conf_difference']].groupby(level=[0]).mean()
    # val_results_df = val_results_df[[ 'holo_HSMM', 'conf_difference']].groupby(level=[0]).mean()


    patient_df.index = patient_df.index.astype(int)
    val_results_df.index = val_results_df.index.astype(int) # index 데이터타입 통일
    # patient_df.index.intersection(val_results_df.index) << 이걸로 확인해보덩가

    combined_df = val_results_df.merge(patient_df, left_index=True, right_index=True)
    
    return combined_df


def merge_df_stu(val_results_df, patient_df):
    val_results_df.index = val_results_df.index.str.split("_", expand=True)
    #val_results_df["conf_difference"] = val_results_df["holo_HSMM"] - val_results_df["healthy_HSMM"]
    # val_results_df = val_results_df[['healthy_HSMM', 'holo_HSMM', 'conf_difference']].groupby(level=[0]).mean()
    val_results_df = val_results_df[['holo_HSMM']].groupby(level=[0]).max()


    patient_df.index = patient_df.index.astype(int)
    val_results_df.index = val_results_df.index.astype(int) # index 데이터타입 통일
    # patient_df.index.intersection(val_results_df.index) << 이걸로 확인해보덩가

    combined_df = val_results_df.merge(patient_df, left_index=True, right_index=True)
    
    return combined_df


def merge_df_tch(val_results_df, patient_df):
    val_results_df.index = val_results_df.index.str.split("_", expand=True)
    #val_results_df["conf_difference"] = val_results_df["holo_HSMM"] - val_results_df["healthy_HSMM"]
    # val_results_df = val_results_df[['healthy_HSMM', 'holo_HSMM', 'conf_difference']].groupby(level=[0]).mean()
    val_results_df = val_results_df[['holo_HSMM']].groupby(level=[0]).max()


    patient_df.index = patient_df.index.astype(int)
    val_results_df.index = val_results_df.index.astype(int) # index 데이터타입 통일
    # patient_df.index.intersection(val_results_df.index) << 이걸로 확인해보덩가

    combined_df = val_results_df.merge(patient_df, left_index=True, right_index=True)
    
    return combined_df






def merge_df_2(val_results_df, patient_df):
    val_results_df.index = val_results_df.index.str.split("_", expand=True)
    val_results_df = val_results_df[['holo_HSMM_mean', 'holo_HSMM_max']].groupby(level=[0]).mean()
    
    patient_df.index = patient_df.index.astype(int)
    val_results_df.index = val_results_df.index.astype(int) # index 데이터타입 통일
    
    combined_df = val_results_df.merge(patient_df, left_index=True, right_index=True)
    
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


def calculate_features_before_cut(recording: torch.Tensor, fs: int) -> Tuple[torch.Tensor, int]:
    assert fs == FREQUENCY_SAMPLING
    
    # Zero-mean and normalise by peak amplitude
    recording = recording.float()
    recording -= recording.mean()
    recording /= recording.abs().max()

    #print(f"WINDOW_LENGTH: {WINDOW_LENGTH}")
    #print(f"WINDOW_STEP: {WINDOW_STEP}")

    # Calculate spectrogram
    window_length =  int(WINDOW_LENGTH * fs)
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

    # # Log and z-normalise
    spectrogram = torch.log(spectrogram)
    # spectrogram = (spectrogram - spectrogram.mean(dim=-1, keepdims=True)) / spectrogram.std(
    #     dim=-1, keepdims=True
    # )

    features_fs = int(1 / WINDOW_STEP)
    return spectrogram, features_fs
    
    
    
    
    
    


def cal_max_frame_len(sr, sequence_len):
    recording= torch.randn(sr * sequence_len)
    feature, fs = calculate_features(recording, sr)
    max_frame_len = feature.shape[-1]
    return max_frame_len




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
    


class Stronglabeled_Dataset(torch.utils.data.Dataset):                           
    def __init__(self, data_folder, recording_paths, timings, murmur_labels, outcome_labels, sequence_length, sampling_rate, cache=None):
        self.data_folder = pathlib.Path(data_folder)  
        self.recording_paths = recording_paths   
        self.timings = timings
        self.murmur_labels = murmur_labels
        self.outcome_labels = outcome_labels
        self.sequence_length = sequence_length
        self.cache = cache
        self.sr = sampling_rate
        assert len(self.recording_paths) == len(self.murmur_labels)

    def __len__(self):
        return len(self.recording_paths)

    def __getitem__(self, idx):
        
        filename = self.recording_paths[idx]  
        label = self.murmur_labels[idx] 
        timing = self.timings[idx] 
        outcome_label = self.outcome_labels[idx] 
        
        filepath = pathlib.Path(self.data_folder / filename)
        #fs, recording = scipy.io.wavfile.read(filepath.with_suffix(".wav"))
        recording, sr = librosa.load(filepath.with_suffix(".wav"), sr=self.sr) # 수정

        recording = torch.as_tensor(recording.copy())
        features, fs_features = calculate_features(recording, sr)

        segmentation_path = filepath.with_suffix(".tsv")
        segmentation_df = pd.read_csv(segmentation_path, sep="\t", header=None)
        segmentation_df.columns = ["start", "end", "state"]
        segmentation_df["start"] = (segmentation_df["start"] * fs_features).apply(lambda x: int(x)) # 수정
        segmentation_df["end"] = (segmentation_df["end"] * fs_features).apply(lambda x: int(np.ceil(x))) # 수정
        
        
        # 수정
        segmentation_label = torch.zeros(5, features.shape[-1]) 
        
        for row in segmentation_df.itertuples():
            # IN:  noise =  0, S1 = 1, systole = 2, S2 = 3, diastole = 4
            # OUT: noise = -1, S1 = 0, systole = 1, S2 = 2, diastole = 3, murmur = 4
            if row.state == 0:
                # Noise
                #segmentation_state = -1
                # segmentation_label[row.state -1 , row.start : row.end] = -1
                pass
            if row.state == 1:
                # S1
                segmentation_label[row.state -1 , row.start : row.end] = 1.0
            elif row.state == 2:
                # Systole
                segmentation_label[row.state -1 , row.start : row.end] = 1.0    
            elif row.state == 3:
                # S2
                segmentation_label[row.state -1 , row.start : row.end] = 1.0
            elif row.state == 4:
                # Diastole
                segmentation_label[row.state -1 , row.start : row.end] = 1.0

            
            # Murmur labeling
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

                state_duration = row.end - row.start 
                start = int(row.start + portion[0] * state_duration)
                end = int(np.ceil(row.start + portion[1] * state_duration))
                segmentation_label[4, start : end] = 1.0
        
        # To erase only noise part
        non_zero_columns = segmentation_label.abs().sum(dim=0) != 0
        clean_features = features[:, non_zero_columns]
        clean_segmentation_label = segmentation_label[:, non_zero_columns]
        
        
        murmur_label = torch.zeros(2).float()
        if label == "Present":
            murmur_label[1] = 1.0
        else:
            murmur_label[0] = 1.0
                
        if outcome_label == "Normal":
            outcome_label = torch.zeros(2).float()
        else:
            outcome_label = torch.ones(2).float()

        return clean_features, clean_segmentation_label, murmur_label, outcome_label, filename
    
    
    
class Sliced_Stronglabeled_Dataset(torch.utils.data.Dataset):                           
    def __init__(self, data_folder, recording_paths, timings, murmur_labels, outcome_labels, sequence_length, sampling_rate, cache= None):
        assert len(recording_paths) == len(murmur_labels)
        self.data_folder = pathlib.Path(data_folder)  
        self.recording_paths = []  # 파일형식 제외한 오디오 파일명
        self.features= []
        self.segmentation_labels = []
        self.murmur_labels = []
        self.outcome_labels = []
        self.sr = sampling_rate
        self.sequence_length = cal_max_frame_len(sampling_rate, sequence_length)  # 슬라이싱용
        
        for filename, murmur_timing, murmur_label, outcome_label in zip(recording_paths, timings, murmur_labels, outcome_labels):
            filepath = self.data_folder / filename # pathlib.Path
            recording, sr = librosa.load(filepath.with_suffix(".wav"), sr=self.sr)
            recording = torch.as_tensor(recording.copy())
            features, fs_features = calculate_features(recording, sr)
            segmentation_path = filepath.with_suffix(".tsv")
            segmentation_df = pd.read_csv(segmentation_path, sep="\t", header=None)
            segmentation_df.columns = ["start", "end", "state"]
            segmentation_df["start"] = (segmentation_df["start"] * fs_features).apply(lambda x: int(x)) # 수정
            segmentation_df["end"] = (segmentation_df["end"] * fs_features).apply(lambda x: int(np.ceil(x))) # 수정
            segmentation_label = torch.zeros(5, features.shape[-1]) 
            
            for row in segmentation_df.itertuples():
                # IN:  noise =  0, S1 = 1, systole = 2, S2 = 3, diastole = 4
                # OUT: noise = -1, S1 = 0, systole = 1, S2 = 2, diastole = 3, murmur = 4
                if row.state == 0:
                    # Noise
                    #segmentation_state = -1
                    # segmentation_label[row.state -1 , row.start : row.end] = -1
                    pass
                if row.state == 1:
                    # S1
                    segmentation_label[row.state -1 , row.start : row.end] = 1.0
                elif row.state == 2:
                    # Systole
                    segmentation_label[row.state -1 , row.start : row.end] = 1.0    
                elif row.state == 3:
                    # S2
                    segmentation_label[row.state -1 , row.start : row.end] = 1.0
                elif row.state == 4:
                    # Diastole
                    segmentation_label[row.state -1 , row.start : row.end] = 1.0         

                        # Murmur labeling
                if (row.state == 2) and (murmur_label == "Present"):
                    if murmur_timing == 'Early-systolic':
                        portion = [0, 0.5]
                    elif murmur_timing == 'Holosystolic':
                        portion = [0, 1]
                    elif murmur_timing == 'Mid-systolic':
                        portion = [0.25, 0.75]
                    elif murmur_timing == 'Late-systolic':
                        portion = [0.5, 1]
                    else:
                        portion = [0, 0]

                    state_duration = row.end - row.start 
                    start = int(row.start + portion[0] * state_duration)
                    end = int(np.ceil(row.start + portion[1] * state_duration))
                    segmentation_label[4, start : end] = 1.0
            
            # To erase only noise part
            non_zero_columns = segmentation_label.abs().sum(dim=0) != 0
            clean_features = features[:, non_zero_columns]
            clean_segmentation_label = segmentation_label[:, non_zero_columns]

            if not features.shape[0]:
                # print(clean_features.shape)
                continue
            
            
            if murmur_label != "Present": 
                murmur_label = torch.zeros(2).float()
                murmur_label[1] = 1.0
            else:
                murmur_label = torch.zeros(2).float()
                murmur_label[0]= 1.0
                
            if outcome_label == "Normal": 
                outcome_label = torch.zeros(2).float()
                outcome_label[0] = 1.0
            else:
                outcome_label = torch.zeros(2).float()
                outcome_label[1] = 1.0
                
                
            if clean_features.shape[-1] >= self.sequence_length:
                max_slice_num = clean_features.shape[-1] // self.sequence_length

                for idx in range(max_slice_num):
                    self.features.append(clean_features[:, idx * self.sequence_length : (idx + 1) * self.sequence_length])
                    self.segmentation_labels.append(clean_segmentation_label[:, idx * self.sequence_length : (idx + 1) * self.sequence_length])
                    self.murmur_labels.append(murmur_label)
                    self.outcome_labels.append(outcome_label)
                    self.recording_paths.append(filename)
            else:
                pass
                # self.features.append(clean_features)
                # self.segmentation_labels.append(clean_segmentation_label)
                # self.murmur_labels.append(murmur_label)
                # self.outcome_labels.append(outcome_label)
                # self.recording_paths.append(filename)

    def __len__(self):
        return len(self.recording_paths)

    def __getitem__(self, idx):
        features = self.features[idx]
        sementation_label = self.segmentation_labels[idx]
        murmur_label = self.murmur_labels[idx]
        outcome_label = self.outcome_labels[idx]
        filename = self.recording_paths[idx]
        return features, sementation_label, murmur_label, outcome_label, filename
    

class Stronglabel_nomurmur(torch.utils.data.Dataset):                           
    def __init__(self, data_folder, recording_paths, timings, murmur_labels, outcome_labels, sequence_length, sampling_rate, cache=None):
        self.data_folder = pathlib.Path(data_folder)  
        self.recording_paths = recording_paths   
        self.timings = timings
        self.murmur_labels = murmur_labels
        self.outcome_labels = outcome_labels
        self.sequence_length = sequence_length
        self.cache = cache
        self.sr = sampling_rate
        assert len(self.recording_paths) == len(self.murmur_labels)

    def __len__(self):
        return len(self.recording_paths)

    def __getitem__(self, idx):
        
        filename = self.recording_paths[idx]  
        label = self.murmur_labels[idx] 
        timing = self.timings[idx] 
        outcome_label = self.outcome_labels[idx] 
        
        filepath = pathlib.Path(self.data_folder / filename)
        #fs, recording = scipy.io.wavfile.read(filepath.with_suffix(".wav"))
        recording, sr = librosa.load(filepath.with_suffix(".wav"), sr=self.sr) # 수정

        recording = torch.as_tensor(recording.copy())
        features, fs_features = calculate_features(recording, sr)

        segmentation_path = filepath.with_suffix(".tsv")
        segmentation_df = pd.read_csv(segmentation_path, sep="\t", header=None)
        segmentation_df.columns = ["start", "end", "state"]
        segmentation_df["start"] = (segmentation_df["start"] * fs_features).apply(lambda x: int(x)) # 수정
        segmentation_df["end"] = (segmentation_df["end"] * fs_features).apply(lambda x: int(np.ceil(x))) # 수정
        
        
        # 수정
        segmentation_label = torch.zeros(4, features.shape[-1]) 
        
        for row in segmentation_df.itertuples():
            # IN:  noise =  0, S1 = 1, systole = 2, S2 = 3, diastole = 4
            # OUT: noise = -1, S1 = 0, systole = 1, S2 = 2, diastole = 3, murmur = 4
            if row.state == 0:
                # Noise
                #segmentation_state = -1
                # segmentation_label[row.state -1 , row.start : row.end] = -1
                pass
            if row.state == 1:
                # S1
                segmentation_label[row.state -1 , row.start : row.end] = 1.0
            elif row.state == 2:
                # Systole
                segmentation_label[row.state -1 , row.start : row.end] = 1.0    
            elif row.state == 3:
                # S2
                segmentation_label[row.state -1 , row.start : row.end] = 1.0
            elif row.state == 4:
                # Diastole
                segmentation_label[row.state -1 , row.start : row.end] = 1.0
        
        
        # To erase only noise part
        non_zero_columns = segmentation_label.abs().sum(dim=0) != 0
        clean_features = features[:, non_zero_columns]
        clean_segmentation_label = segmentation_label[:, non_zero_columns]
        
        
        # if label == "Absent":
        #     murmur_label = torch.zeros(1).long()
        # else:
        #     murmur_label = torch.ones(1).long()
        
        murmur_label = torch.zeros(2).float()
        if label == "Present":
            murmur_label[1] = 1.0
        
        
        # if outcome_label == "Normal":
        #     outcome_label = torch.zeros(1).long()
        # else:
        #     outcome_label = torch.ones(1).long()
            

        return clean_features, clean_segmentation_label, murmur_label, outcome_label, filename





class Weaklabeled_Dataset(torch.utils.data.Dataset):                           
    def __init__(self, data_folder, mid_paths, filename_lists, murmur_labels, sampling_rate,  cache=None):
        self.data_folder = pathlib.Path(data_folder)  
        self.filename_lists = filename_lists 
        self.mid_paths = mid_paths
        self.murmur_labels = murmur_labels 
        self.sr = sampling_rate
        assert len(self.mid_paths) == len(self.murmur_labels)

    def __len__(self):
        return len(self.filename_lists)

    def __getitem__(self, idx):
        
        filename = self.filename_lists.iloc[idx]  
        mid_path = self.mid_paths.iloc[idx]
        
        filepath = self.data_folder / mid_path / filename
        label = self.murmur_labels.iloc[idx] 
        
        recording, sr = librosa.load(filepath, sr= self.sr) # 수정

        recording = torch.as_tensor(recording.copy())
        features, fs_features = calculate_features(recording, sr)
        
        segmentation_label = torch.zeros(5, features.shape[-1]) 
        
        # if label == "Present":
        #     murmur_label = torch.zeros(1).long()
        # else:
        #     murmur_label = torch.ones(1).long()
        
        murmur_label = torch.zeros(2).float()
        if label == "Present":
            murmur_label[1] = 1.0
            
        outcome_label = torch.zeros(1).long() 

        return features, segmentation_label, murmur_label, outcome_label, filename
    
    

    
    

class Unlabeled_Dataset(torch.utils.data.Dataset):                           
    def __init__(self, data_folder, mid_paths, filename_lists, sampling_rate, cache=None):
        self.data_folder = pathlib.Path(data_folder)  
        self.filename_lists = filename_lists
        self.mid_paths = mid_paths
        self.sr = sampling_rate
        assert len(self.mid_paths) == len(self.filename_lists)


    def __len__(self):
        return len(self.filename_lists)

    def __getitem__(self, idx):
        
        mid_path = self.mid_paths.iloc[idx]
        filename = self.filename_lists.iloc[idx]  
        
        filepath = self.data_folder / mid_path / filename
        recording, sr = librosa.load(filepath, sr= self.sr) # 수정

        recording = torch.as_tensor(recording.copy())
        features, fs_features = calculate_features(recording, sr)

        segmentation_label = torch.zeros(5, features.shape[-1]) 
        
        # murmur_label = torch.zeros(1).long()
        # outcome_label = torch.zeros(1).long()
        murmur_label = torch.zeros(2).float()
        outcome_label = torch.zeros(2).float()
        
        

        return features, segmentation_label, murmur_label, outcome_label, filename


class Custom_T4_dataset(torch.utils.data.Dataset):                           
    def __init__(self, data_folder, recording_paths, timings, murmur_labels, outcome_labels, sequence_length, cache=None):
        self.data_folder = pathlib.Path(data_folder)  
        self.recording_paths = recording_paths   
        self.timings = timings
        self.murmur_labels = murmur_labels
        self.outcome_labels = outcome_labels
        self.sequence_length = sequence_length
        self.cache = cache        
        assert len(self.recording_paths) == len(self.murmur_labels)

    def __len__(self):
        return len(self.recording_paths)

    def __getitem__(self, idx):
        
        filename = self.recording_paths[idx]  
        label = self.murmur_labels[idx] 
        timing = self.timings[idx] 
        outcome_label = self.outcome_labels[idx] 
        
    # if self.cache is None or filename not in self.cache:
        filepath = self.data_folder / filename
        fs, recording = spio.wavfile.read(filepath.with_suffix(".wav"))

        recording = torch.as_tensor(recording.copy())
        features, fs_features = calculate_features(recording, fs)

        segmentation_path = filepath.with_suffix(".tsv")
        segmentation_df = pd.read_csv(segmentation_path, sep="\t", header=None)
        segmentation_df.columns = ["start", "end", "state"]
        segmentation_df["start"] = (segmentation_df["start"] * fs_features).apply(lambda x: int(x)) # 수정
        segmentation_df["end"] = (segmentation_df["end"] * fs_features).apply(lambda x: int(np.ceil(x))) # 수정
        
        # features.shape[-1] : frame길이
        
        # 기존
        # segmentation_label = torch.zeros(features.shape[-1], dtype=torch.long)
        
        # 수정
        segmentation_label = torch.zeros(5, features.shape[-1]) # BCE_Loss 쓸 꺼니까 float처리
        
        for row in segmentation_df.itertuples():
            # IN:  noise =  0, S1 = 1, systole = 2, S2 = 3, diastole = 4
            # OUT: noise = -1, S1 = 0, systole = 1, S2 = 2, diastole = 3, murmur = 4
            if row.state == 0:
                # Noise
                segmentation_state = -1
            if row.state == 1:
                # S1
                segmentation_label[row.state -1 , row.start : row.end] = 1.0
            elif row.state == 2:
                # Systole
                segmentation_label[row.state -1 , row.start : row.end] = 1.0    
            elif row.state == 3:
                # S2
                segmentation_label[row.state -1 , row.start : row.end] = 1.0
            elif row.state == 4:
                # Diastole
                segmentation_label[row.state -1 , row.start : row.end] = 1.0

            #segmentation_label[row.start : row.end] = segmentation_state
            # segmentation_label[row.state, row.start : row.end] = 1.0  # 4
            
            
            
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
                #segmentation_label[row.state, start : end] = 4
                segmentation_label[4, start : end] = 1.0
        
        # To erase only noise part
        # non_zero_columns = segmentation_label.sum(dim=1) != 0
        # segmentation_label = segmentation_label[:, non_zero_columns]
                
        
        if label == "Absent":
            murmur_label = torch.zeros(1).long()
        else:
            murmur_label = torch.ones(1).long()
        
        
        if outcome_label == "Normal":
            outcome_label = torch.zeros(1).long()
        else:
            outcome_label = torch.ones(1).long()
            
        if self.sequence_length is not None:
            
            random_start = torch.randint(
                low=0, high=max(features.shape[-1] - self.sequence_length, 1), size=(1,)
            ).item()
            
            features = features[..., random_start : random_start + self.sequence_length]
            segmentation_label = segmentation_label[...,
                random_start : random_start + self.sequence_length
            ]

        return features, segmentation_label, murmur_label, outcome_label, filename


class Custom_T4_dataset_2(torch.utils.data.Dataset):                           
    def __init__(self, data_folder, recording_paths, timings, murmur_labels, outcome_labels, sequence_length= None, cache=None):
        self.data_folder = pathlib.Path(data_folder)  
        self.recording_paths = recording_paths 
        self.timings = timings
        self.murmur_labels = murmur_labels
        self.outcome_labels = outcome_labels
        self.sequence_length = sequence_length
        self.cache = cache        
        assert len(self.recording_paths) == len(self.murmur_labels)

    def __len__(self):
        return len(self.recording_paths)

    def __getitem__(self, idx):
        
        filename = self.recording_paths[idx]  
        label = self.murmur_labels[idx] 
        timing = self.timings[idx] 
        outcome_label = self.outcome_labels[idx] 
        
    # if self.cache is None or filename not in self.cache:
        filepath = self.data_folder / filename
        fs, recording = spio.wavfile.read(filepath.with_suffix(".wav"))

        recording = torch.as_tensor(recording.copy())
        features, fs_features = calculate_features(recording, fs)

        segmentation_path = filepath.with_suffix(".tsv")
        segmentation_df = pd.read_csv(segmentation_path, sep="\t", header=None)
        segmentation_df.columns = ["start", "end", "state"]
        segmentation_df["start"] = (segmentation_df["start"] * fs_features).apply(lambda x: int(x)) # 수정
        segmentation_df["end"] = (segmentation_df["end"] * fs_features).apply(lambda x: int(np.ceil(x))) # 수정
        
        # 수정
        segmentation_label = torch.zeros(5, features.shape[-1]).float() # BCE_Loss 쓸 꺼니까 float처리
        
        for row in segmentation_df.itertuples():
            # IN:  noise =  0, S1 = 1, systole = 2, S2 = 3, diastole = 4
            # OUT: noise = -1, S1 = 0, systole = 1, S2 = 2, diastole = 3, murmur = 4
            if row.state == 0:
                # Noise
                # segmentation_state = -1
                pass
            elif row.state == 1:
                # S1
                segmentation_label[row.state -1 , row.start : row.end] = 1.0
            elif row.state == 2:
                # Systole
                segmentation_label[row.state -1 , row.start : row.end] = 1.0    
            elif row.state == 3:
                # S2
                segmentation_label[row.state -1 , row.start : row.end] = 1.0
            elif row.state == 4:
                # Diastole
                segmentation_label[row.state -1 , row.start : row.end] = 1.0

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

                state_duration = row.end - row.start # TODO: inclusive?
                start = int(row.start + portion[0] * state_duration)
                end = int(np.ceil(row.start + portion[1] * state_duration))
                
                # murmur 라벨링
                #segmentation_label[row.state, start : end] = 4
                segmentation_label[4, start : end] = 1.0
        
        # To erase only noise part
        non_zero_columns = segmentation_label.abs().sum(dim=0) != 0
        clean_features = features[:, non_zero_columns]
        clean_segmentation_label = segmentation_label[:, non_zero_columns]
                
        
        if label == "Absent":
            murmur_label = torch.zeros(1).long()
        else:
            murmur_label = torch.ones(1).long()
        
        if outcome_label == "Normal":
            outcome_label = torch.zeros(1).long()
        else:
            outcome_label = torch.ones(1).long()
            
        return clean_features, clean_segmentation_label, murmur_label, outcome_label, filename

 
def add_murmur_seg(segmentation_label, timing):
    
    added_label = []
    
    if timing == 'Early-systolic':
        for state in segmentation_label:
            if state == 1: # if state is systole
                added_label.append(4) # add murmur state before systole
                added_label.append(state)
                continue
            added_label.append(state) # add systole after murmur   
        
    elif timing == 'Mid-systolic':
        for state in segmentation_label:
            if state == 1:
                added_label.append(state) # add First systole
                added_label.append(4) # add murmur state after First systole
                added_label.append(state)
                continue
            added_label.append(state) # add Second systole after murmur   
    
    elif timing == 'Late-systolic':
        for state in segmentation_label:
            if state == 1:
                added_label.append(state)
                added_label.append(4)
                continue
            added_label.append(state)
        
    elif timing == 'Holosystolic':
        for state in segmentation_label:
            if state == 1:
                added_label.append(4)
                continue
            added_label.append(state)
    
    # elif timing == 
    
    else:
        # raise Exception(f"\nThere are another kind of systolic!\n It is {timing}")
        added_label = segmentation_label
        
    
    return added_label
    




class Custom_CTCLoss_dataset_slice(torch.utils.data.Dataset):                           
    def __init__(self, data_folder, recording_paths, timings, murmur_labels, outcome_labels, seq_len,  cache):
        self.data_folder = pathlib.Path(data_folder)  
        self.recording_paths = recording_paths   
        self.timings = timings
        self.murmur_labels = murmur_labels
        self.outcome_labels = outcome_labels
        # self.sequence_length = sequence_length
        self.cache = cache
        self.seq_len = seq_len        
        assert len(self.recording_paths) == len(self.murmur_labels)

    def __len__(self):
        return len(self.recording_paths)

    def __getitem__(self, idx):
        
        filename = self.recording_paths[idx]  
        label = self.murmur_labels[idx] 
        timing = self.timings[idx] 
        outcome_label = self.outcome_labels[idx] 
        
        filepath = self.data_folder / filename
        fs, recording = spio.wavfile.read(filepath.with_suffix(".wav"))

        # Load waveform
        recording = torch.as_tensor(recording.copy())

        segmentation_path = filepath.with_suffix(".tsv")
        segmentation_df = pd.read_csv(segmentation_path, sep="\t", header=None)
        segmentation_df.columns = ["start", "end", "state"]
                
        # IN:  noise =  0, S1 = 1, systole = 2, S2 = 3, diastole = 4
        # OUT: noise = -1, S1 = 0, systole = 1, S2 = 2, diastole = 3, (additional)murmur = 4
        segmentation_df["state"] = segmentation_df["state"] - 1
        segmentation_df["state"] = segmentation_df["state"].apply(lambda x: 5 if x == -1 else x) #  CTC loss does not contain -1 index

        
        if self.seq_len is not None:
            
            # slice
            if len(recording) >= self.seq_len * FREQUENCY_SAMPLING:
                recording = recording[ : self.seq_len * FREQUENCY_SAMPLING]
            # zero-padding
            else:
                diff = self.seq_len * FREQUENCY_SAMPLING - len(recording)
                recording = F.pad(recording, (0, diff), 'constant', 0.0)
       
        # waveform to spectogram
        features, fs_features = calculate_features(recording, fs)
           
        
        max_index = segmentation_df[segmentation_df['end'] < self.seq_len].index.max()
        segmentation_df = segmentation_df.loc[ : max_index, :]
        # max_seconds= segmentation_df['end'].max()
        
         
        segmentation_label = segmentation_df["state"].to_list()
        
        
        #segmentation_label = segmentation_df["state"][segmentation_df["state"] != -1].to_list() # erase noise part
        #segmentation_label = segmentation_df["state"].to_list()
        segmentation_label = torch.as_tensor(add_murmur_seg(segmentation_label, timing))
        
        
        if label == "Absent":
            murmur_label = torch.zeros(1).long()
        else:
            murmur_label = torch.ones(1).long()
        
        if outcome_label == "Normal":
            outcome_label = torch.zeros(1).long()
        else:
            outcome_label = torch.ones(1).long()


        self.cache[filename] = {"features": features, "segmentation": segmentation_label, 
                                "murmur_label": murmur_label, "outcome": outcome_label}
        
        
        return features, segmentation_label, murmur_label, outcome_label, filename #, max_seconds #, self.cache

    

class DatasetFor_Inference(torch.utils.data.Dataset):                           
    def __init__(self, data_folder, recording_paths, timings, murmur_labels, outcome_labels, sequence_length, sampling_rate, cache=None):
        self.data_folder = pathlib.Path(data_folder)  
        self.recording_paths = recording_paths   
        self.timings = timings
        self.murmur_labels = murmur_labels
        self.outcome_labels = outcome_labels
        self.sr = sampling_rate
        assert len(self.recording_paths) == len(self.murmur_labels)

    def __len__(self):
        return len(self.recording_paths)

    def __getitem__(self, idx):
        
        filename = self.recording_paths[idx]  
        label = self.murmur_labels[idx] 
        # timing = self.timings[idx] 
        outcome_label = self.outcome_labels[idx] 
        
        filepath = pathlib.Path(self.data_folder / filename)
        #fs, recording = scipy.io.wavfile.read(filepath.with_suffix(".wav"))
        recording, sr = librosa.load(filepath.with_suffix(".wav"), sr=self.sr) # 수정

        recording = torch.as_tensor(recording.copy())
        features, _ = calculate_features(recording, sr)
            
        # valid, test_
        segmentation_label = torch.zeros(5, features.shape[-1]) 
        
        murmur_label = torch.zeros(2).float()
        if label == "Present":
            murmur_label[1] = 1.0
        else:
            murmur_label[0] = 1.0
        
        if outcome_label == "Normal":
            outcome_label = torch.zeros(2).float()
            outcome_label[0] = 1.0
        else:
            outcome_label = torch.ones(2).float()
            outcome_label[1] = 1.0

        return features, segmentation_label, murmur_label, outcome_label, filename









def batch_whole_dataset(dataset: RecordingDataset):
    num_examples = len(dataset)
    #return collate_fn_slice_add([dataset[i] for i in range(num_examples)])
    #return collate_fn_slice_add_val([dataset[i] for i in range(num_examples)])
    #return collate_fn_multi_output([dataset[i] for i in range(num_examples)])
    return collate_fn_ctc([dataset[i] for i in range(num_examples)])  # not good...







def collate_fn_slice(x):
    all_features = []
    all_frame_labels = []
    all_murmur_labels = []
    all_outcome_labels = []
    all_filenames = []
    
    frame_len = int(300)
    
    all_features = torch.zeros(len(x), 40, 300)
    all_frame_labels = torch.zeros(len(x), 5, 300)
    
    for idx, (feature, segmentation_label, murmur_label, outcome_label, filename) in enumerate(x): # train_dataset
        
        current_len = feature.size(1)
        
        # 더 짧을 때        
        if current_len < frame_len:
            
            diff = frame_len - current_len
            
            left_pad = random.randint(0, diff)
            right_pad = diff - left_pad
            
            padded_feature = F.pad(feature, (left_pad, right_pad), 'constant', 0.0)
            segmentation_label = F.pad(segmentation_label, (left_pad, right_pad), 'constant', 0.0)  # 이거 -1로 바꿔야겠네
            
            # 추가
            #all_frame_labels[idx] = segmentation_label
        else:
            diff = current_len - frame_len
            
            start = random.randint(0, diff)
            padded_feature = feature[:, start : start + 300]
            segmentation_label = segmentation_label[:, start : start + 300]
            
            # 추가
            #all_frame_labels[idx] = segmentation_label
        
        
        all_features[idx] = padded_feature
        all_murmur_labels.append(murmur_label)
        all_outcome_labels.append(outcome_label)
        all_filenames.append(filename)

    all_features = all_features.float()
    all_murmur_labels = torch.tensor(all_murmur_labels, dtype= torch.long)
    all_outcome_labels = torch.tensor(all_outcome_labels, dtype= torch.long)    
        
    return all_features, all_frame_labels, all_murmur_labels, all_outcome_labels, all_filenames



def collate_fn_slice_add(x):
    all_features = []
    all_frame_labels = []
    all_murmur_labels = []
    all_outcome_labels = []
    all_filenames = []
    
    frame_len = int(300)
    
    
    all_features = [] # [len, 40, 300]
    all_frame_labels = [] # [len, 5, 300]
    
    for idx, (feature, segmentation_label, murmur_label, outcome_label, filename) in enumerate(x): # train_dataset
        
        current_len = feature.size(1)
        
        # 더 짧을 때        
        if current_len < frame_len:
            
            diff = frame_len - current_len
            
            left_pad = random.randint(0, diff)
            right_pad = diff - left_pad
            
            padded_feature = F.pad(feature, (left_pad, right_pad), 'constant', 0.0)
            segmentation_label = F.pad(segmentation_label, (left_pad, right_pad), 'constant', -1)  # 이거 -1로 바꿔야겠네
            
            # 추가
            all_features.append(padded_feature)
            all_frame_labels.append(segmentation_label)
            all_murmur_labels.append(murmur_label)
            all_outcome_labels.append(outcome_label)
            all_filenames.append(filename)
        else:
            #diff = current_len - frame_len
        
            # start = random.randint(0, diff)
            # padded_feature = feature[:, start : start + 300]
            # segmentation_label = segmentation_label[:, start : start + 300]
            
            for j in range(current_len // 300): 
                sliced_feature = feature[ :, j * 300 : (j+1) * 300]
                sliced_seg_label = segmentation_label[ :, j * 300 : (j+1) * 300]
            
                # 추가
                all_features.append(sliced_feature)
                all_frame_labels.append(sliced_seg_label)
                all_murmur_labels.append(murmur_label)
                all_outcome_labels.append(outcome_label)
                all_filenames.append(filename)
            

    all_features = torch.stack(all_features).float() 
    all_frame_labels = torch.stack(all_frame_labels).float() # BCE_loss 
    all_murmur_labels = torch.tensor(all_murmur_labels).long() # CE_loss
    all_outcome_labels = torch.tensor(all_outcome_labels).long() # CE_loss
        
    return all_features, all_frame_labels, all_murmur_labels, all_outcome_labels, all_filenames



def collate_fn_slice_add_val(x):
    all_features = []
    all_frame_labels = []
    all_murmur_labels = []
    all_outcome_labels = []
    all_filenames = []
    
    frame_len = int(300)
    
    
    all_features = [] # [len, 40, 300]
    all_frame_labels = [] # [len, 5, 300]
    
    for idx, (feature, segmentation_label, murmur_label, outcome_label, filename) in enumerate(x): # train_dataset
        
        current_len = feature.size(1)
        
        # 더 짧을 때        
        if current_len < frame_len:
            
            diff = frame_len - current_len
            
            left_pad = random.randint(0, diff)
            right_pad = diff - left_pad
            
            padded_feature = F.pad(feature, (left_pad, right_pad), 'constant', 0.0)
            segmentation_label = F.pad(segmentation_label, (left_pad, right_pad), 'constant', -1)  # 이거 -1로 바꿔야겠네
            
            # 추가
            all_features.append(padded_feature)
            all_frame_labels.append(segmentation_label)
            all_murmur_labels.append(murmur_label)
            all_outcome_labels.append(outcome_label)
            all_filenames.append(filename)
        else:
            #diff = current_len - frame_len
        
            # start = random.randint(0, diff)
            # padded_feature = feature[:, start : start + 300]
            # segmentation_label = segmentation_label[:, start : start + 300]
            
            
            sliced_feature = feature[..., : 300]
            sliced_seg_label = segmentation_label[..., : 300]
        
            # 추가
            all_features.append(sliced_feature)
            all_frame_labels.append(sliced_seg_label)
            all_murmur_labels.append(murmur_label)
            all_outcome_labels.append(outcome_label)
            all_filenames.append(filename)
            

    all_features = torch.stack(all_features).float() 
    all_frame_labels = torch.stack(all_frame_labels).float() # BCE_loss 
    all_murmur_labels = torch.tensor(all_murmur_labels).long() # CE_loss
    all_outcome_labels = torch.tensor(all_outcome_labels).long() # CE_loss
        
    return all_features, all_frame_labels, all_murmur_labels, all_outcome_labels, all_filenames


def collate_fn_multi_output(x):
    max_length = 0
    all_features = []
    all_frame_labels = []
    all_murmur_labels = []
    all_outcome_labels = []
    #actual_lengths = []

    for feature, frame_label, murmur_label, outcome_label in x:
        all_features.append(feature)
        all_frame_labels.append(frame_label)
        all_murmur_labels.append(murmur_label)
        all_outcome_labels.append(outcome_label)
        max_length = max(max_length, len(frame_label))
        #actual_lengths.append(len(frame_label))
    
    padded_features = all_features[0].new_full((len(x), all_features[0].shape[0], max_length), 0.0)
    
    for i, feature in enumerate(all_features): 
        padded_features[i, ..., : feature.shape[-1]] = feature

        padded_labels = all_frame_labels[0].new_full((len(x), max_length), -1)
    
    for i, frame_label in enumerate(all_frame_labels):
        padded_labels[i, : frame_label.shape[-1]] = frame_label


    all_murmur_labels = torch.tensor(all_murmur_labels, dtype= torch.long)
    all_outcome_labels = torch.tensor(all_outcome_labels, dtype= torch.long)


    return padded_features, padded_labels, all_murmur_labels, all_outcome_labels  # torch.as_tensor(actual_lengths)


# 최대 길이로 패딩할 것....!
def collate_fn_ctc(x):
    max_feature_length = 0
    max_seq_length = 0
    all_features = []
    all_seq_labels = []
    all_murmur_labels = []
    all_outcome_labels = []
    actual_seq_lengths = []
    actual_label_lengths = []
    
    for feature, seq_label, murmur_label, outcome_label, filename in x:
        all_features.append(feature) # spectogram
        all_seq_labels.append(seq_label) # state_seq_label
        actual_seq_lengths.append(feature.shape[-1]) # state_seq_label_length
        actual_label_lengths.append(len(seq_label)) # 
        
        all_murmur_labels.append(murmur_label)
        all_outcome_labels.append(outcome_label)
        
        
        max_feature_length = max(max_feature_length, feature.shape[-1])
        max_seq_length= max(max_seq_length, len(seq_label))
    
    #padded_features = all_features[0].new_full((len(x), all_features[0].shape[0], max_feature_length), 0.0) # zero tensor.shape: (datasize, freq, max_frame)
    padded_features= torch.zeros((len(x), all_features[0].shape[0], max_feature_length)).float()
    padded_labels= torch.full((len(x), max_seq_length), 5).to(torch.int64)
    
    
    for idx, (feature, seq_label) in enumerate(zip(all_features, all_seq_labels)): 
        
        #print(seq_label.shape)
        padded_features[idx, ..., : feature.shape[-1]] = feature
        padded_labels[idx, : seq_label.shape[-1]] = seq_label
        


    # all_seq_labels = torch.tensor(all_seq_labels, dtype= torch.long)
    all_murmur_labels = torch.tensor(all_murmur_labels, dtype= torch.long)
    all_outcome_labels = torch.tensor(all_outcome_labels, dtype= torch.long)
    actual_seq_lengths = torch.tensor(actual_seq_lengths, dtype= torch.int64)
    actual_label_lengths = torch.tensor(actual_label_lengths, dtype= torch.int64)

    return padded_features, padded_labels, all_murmur_labels, all_outcome_labels, actual_seq_lengths, actual_label_lengths



def collate_fn_for_train(x):
    all_features = []
    all_seq_labels = []
    all_murmur_labels = []
    all_outcome_labels = []
    all_filenames= []
            
    max_seq_len = cal_max_frame_len(FREQUENCY_SAMPLING, TRAINING_SEQUENCE_LENGTH)
    
    for idx, (feature, seq_label, murmur_label, outcome_label, filename) in enumerate(x): # train_dataset
        
        current_len = feature.size(1)
        
        # 더 짧을 때
        if current_len < max_seq_len:
            diff = max_seq_len - current_len
            feature = F.pad(feature, (diff, 0),'constant', 0.0)
            seq_label = F.pad(seq_label, (diff, 0),'constant', 0.0)
            all_features.append(feature)
            all_seq_labels.append(seq_label)
            all_murmur_labels.append(murmur_label)
            all_outcome_labels.append(outcome_label)
            all_filenames.append(filename)

    all_features = torch.stack(all_features).float() 
    all_seq_labels = torch.stack(all_seq_labels).float() # BCE_loss 
    all_murmur_labels = torch.stack(all_murmur_labels).float() # CE_loss
    #all_outcome_labels = all_outcome_labels # CE_loss
        
    return all_features, all_seq_labels, all_murmur_labels, all_outcome_labels, all_filenames




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
    
    # 0.0으로 가득한 padded_features 생성: [num_of_data, freq, frame] == [num_of_data, freq, 300]
    padded_features = all_features[0].new_full((len(x), all_features[0].shape[0], max_length), 0.0)
    
    
    # feature의 frame축으로 부족한 만큼 0.0으로 패딩 
    for i, features in enumerate(all_features): 
        padded_features[i, ..., : features.shape[-1]] = features

    
    # -1로 가득한 padded_labels 생성: [num_of_data, frame] == [num_of_data, 300]
    padded_labels = all_labels[0].new_full((len(x), max_length), -1)

    
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



def calculate_class_weights_wo(dataset):
    old_sequence_length = dataset.sequence_length
    dataset.sequence_length = None

    class_counts = defaultdict(int)
    for i in range(len(dataset)):
        _, label, _, _,_ = dataset[i]
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


# th23 added--------------------------------------------------------------------------------------------------------

def check_duration_in_dataframe(filepath):
    recording, sampling_rate = librosa.load(filepath, sr= 4000)
    return len(recording) / sampling_rate



def get_pysionet_sufhsdb_df(external_datapath, external_sufhsdb_path): 
    
    # make columns
    wave_file_list = [file for file in os.listdir(external_datapath / external_sufhsdb_path) if file.endswith(".wav")]
    label_list = ["Unlabeled"] * len(wave_file_list)
    mid_path_list = [external_sufhsdb_path] * len(wave_file_list)

    # make df with columns above
    pysi_sufhsdb_df = pd.DataFrame({"label": label_list, "mid_path": mid_path_list, "filename": wave_file_list})
    
    pysi_sufhsdb_df["absolute_path"] = external_datapath / pysi_sufhsdb_df["mid_path"] / pysi_sufhsdb_df["filename"]
    pysi_sufhsdb_df["audio_len"] = pysi_sufhsdb_df["absolute_path"].apply(lambda x: check_duration_in_dataframe(x))

    
    pysi_sufhsdb_df = pysi_sufhsdb_df[pysi_sufhsdb_df["audio_len"] > 5.0]
    
    return pysi_sufhsdb_df


def get_kaggle_df(external_datapath, external_kaggle_a, external_kaggle_b):
    # kaggle dataset subpath to pathlib.Path

    
    # Filter to get only "normal", "murmur", or "unlabelled"
    set_a_filenames_list = [filename for filename in os.listdir(external_datapath / external_kaggle_a) if filter_wavfiles(filename)]
    set_b_filenames_list = [filename for filename in os.listdir(external_datapath / external_kaggle_b) if filter_wavfiles(filename)]
    mid_path_list = [external_kaggle_a] * len(set_a_filenames_list) + [external_kaggle_b] * len(set_b_filenames_list)
    # mid_path_list = ["kag_dataset_1/set_a"] * len(set_a_filenames_list) + ["kag_dataset_1/set_b"] * len(set_b_filenames_list)

    # Combine
    combined_filename_list = set_a_filenames_list + set_b_filenames_list

    # Get catergory list: "normal", "murmur", or "unlabelled"
    category_list = [define_category(filename) for filename in combined_filename_list]
    kaggle_df = pd.DataFrame({"label": category_list, "mid_path": mid_path_list , "filename": combined_filename_list})
    
    kaggle_df["absolute_path"] = external_datapath / kaggle_df["mid_path"] / kaggle_df["filename"]
    kaggle_df["audio_len"] = kaggle_df["absolute_path"].apply(lambda x: check_duration_in_dataframe(x))

    kaggle_df = kaggle_df[kaggle_df["audio_len"] > 5.0]
    
    return kaggle_df



# External_dataset
    
def get_external_df(configs, external_data_folder):
    external_subpath_config = configs["data_path"]["external_data_subpath"]
    
    external_sufhsdb_path = external_subpath_config["pysionet_sufhsdb"]
    external_kaggle_a = external_subpath_config["kaggle_set_a"]
    external_kaggle_b = external_subpath_config["kaggle_set_b"]
    
    
    kaggle_df = get_kaggle_df(external_data_folder, external_kaggle_a, external_kaggle_b)
    sufhsdb_df = get_pysionet_sufhsdb_df(external_data_folder, external_sufhsdb_path)
    
    #print(kaggle_df.shape, sufhsdb_df.shape)
    
    
    weaklabeled_df = kaggle_df[kaggle_df.label != "Unlabeled"]
    kaggle_unlabeled = kaggle_df[kaggle_df.label == "Unlabeled"]
    unlabeled_df = pd.concat([sufhsdb_df, kaggle_unlabeled], ignore_index= True)
    
    
    # # Exclude below 5 seconds wav files
    # exclude_weak = configs["exclude_filelist"]["weak_labeled"]
    # exclude_unlabeled = configs["exclude_filelist"]["unlabeled"]
    
    # weaklabeled_df = weaklabeled_df.query("filename not in @exclude_weak")
    # unlabeled_df = unlabeled_df.query("filename not in @exclude_unlabeled")
    
    return weaklabeled_df, unlabeled_df




class ConcatDatasetBatchSampler(Sampler):
    def __init__(self, samplers, batch_sizes, epoch=0):
        self.batch_sizes = batch_sizes
        self.samplers = samplers
        self.offsets = [0] + np.cumsum([len(x) for x in self.samplers]).tolist()[:-1]

        self.epoch = epoch
        self.set_epoch(self.epoch)

    def _iter_one_dataset(self, c_batch_size, c_sampler, c_offset):
        batch = []
        for idx in c_sampler:
            batch.append(c_offset + idx)
            if len(batch) == c_batch_size:
                yield batch

    def set_epoch(self, epoch):
        if hasattr(self.samplers[0], "epoch"):
            for s in self.samplers:
                s.set_epoch(epoch)

    def __iter__(self):
        iterators = [iter(i) for i in self.samplers]
        tot_batch = []
        for b_num in range(len(self)):
            for samp_idx in range(len(self.samplers)):
                c_batch = []
                while len(c_batch) < self.batch_sizes[samp_idx]:
                    c_batch.append(self.offsets[samp_idx] + next(iterators[samp_idx]))
                tot_batch.extend(c_batch)
            yield tot_batch
            tot_batch = []

    def __len__(self):
        min_len = float("inf")
        for idx, sampler in enumerate(self.samplers):
            c_len = (len(sampler)) // self.batch_sizes[idx]
            min_len = min(c_len, min_len)
        return min_len