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
from scipy.signal import resample
import warnings

import sklearn.metrics
import sklearn.model_selection

# ------------------------------------------------------------------------------------------------------------------------------
import config
from utils import *

# 추가
#warnings.filterwarnings("ignore", message="your specific warning message")
warnings.filterwarnings("ignore", category=FutureWarning, message="Series.__getitem__ treating keys as positions is deprecated")


WINDOW_STEP = config.hop_length
WINDOW_LENGTH = config.window_length
FREQUENCY_SAMPLING = config.sampling_rate
FREQUENCY_HIGH = config.freq_high
TRAINING_SEQUENCE_LENGTH = config.train_seq_len


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


class Stronglabeled_Dataset(torch.utils.data.Dataset):                           
    def __init__(self, data_folder, recording_paths, timings, murmur_labels, outcome_labels, sampling_rate, window_length, window_step, freq_bins,
                 clean_noise, slice_feature, sequence_length, cache= None):
        assert len(recording_paths) == len(murmur_labels)
        self.data_folder = pathlib.Path(data_folder)  
        self.recording_paths = recording_paths
        self.murmur_timings = timings
        self.murmur_labels = murmur_labels
        self.outcome_labels = outcome_labels
        
        self.sr = sampling_rate
        self.window_length = window_length
        self.window_step = window_step
        self.freq_bins= freq_bins
        self.sequence_length = sequence_length
        self.clean_noise = clean_noise
        self.slice_feature = slice_feature
        
    def calc_frequency_bins(self):
        return int(np.ceil(config.freq_high * config.window_length))
                
    def calculate_features(self, recording, fs):
        # assert fs == self.sr
        
        # Zero-mean and normalise by peak amplitude
        recording = recording.float()
        recording -= recording.mean()
        recording /= recording.abs().max()
        
        # Calculate spectrogram
        
        ## 1. Seconds to frame
        window_length = int(self.window_length * self.sr) 
        window_step = int(self.window_step * self.sr)
        
        ## 2. Wav to spectrogram
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
        
        # 3. Remove high frequencies above FREQUENCY_HIGH Hz
        spectrogram = spectrogram[ : self.calc_frequency_bins()]
        
        # 4. Log and z-normalise
        spectrogram = torch.log(spectrogram)
        spectrogram = (spectrogram - spectrogram.mean(dim=-1, keepdims=True)) / spectrogram.std(dim=-1, keepdims=True)
        
        features_fs = int(1 / self.window_step)        
        return spectrogram, features_fs
    
    def find_min_max_zero_indices(self, tensor):
        
        non_zero_indices = torch.where(tensor != 0)[0]
        
            # return non-zero
        if len(non_zero_indices):
            min_idx = non_zero_indices.min().item()
            max_idx = non_zero_indices.max().item() + 1
            # return nothing
        else:
            min_idx, max_idx = 0, 0 
        return min_idx, max_idx
            
    
    def __len__(self):
        return len(self.recording_paths)      
    
    def __getitem__(self, idx):
        
        filename = self.recording_paths[idx]
        murmur_timing = self.murmur_timings[idx]
        murmur_label = self.murmur_labels[idx]
        outcome_label = self.outcome_labels[idx]
        
        filepath = self.data_folder / filename
        sr, recording = spio.wavfile.read(filepath.with_suffix(".wav"))
        
        if sr != self.sr:
            num_samples = int(len(recording) * self.sr / sr)
            recording = resample(recording, num_samples)
        recording = torch.as_tensor(recording.copy())
        
        wav_len = len(recording) / self.sr
        
        features, fs_features = self.calculate_features(recording, self.sr)
        segmentation_path = filepath.with_suffix(".tsv")
        segmentation_df = pd.read_csv(segmentation_path, sep="\t", header=None)
        segmentation_df.columns = ["start", "end", "state"]
        segmentation_df["start"] = (segmentation_df["start"] * fs_features).apply(lambda x: int(x)) 
        segmentation_df["end"] = (segmentation_df["end"] * fs_features).apply(lambda x: int(np.ceil(x))) # 올림처리 해야 슬라이싱할 때 포함됨.
        segmentation_label = torch.zeros(3, features.shape[-1]) 
    
        # IN:  noise =  0, S1 = 1, systole = 2, S2 = 3, diastole = 4
        for row in segmentation_df.itertuples():
            
            if row.state == 0: 
                # Noise
                pass
            elif row.state == 1: 
                # S1
                segmentation_label[0 , row.start : row.end] = 1.0
            elif row.state == 2: 
                # Systole
                pass
            elif row.state == 3: 
                # S2
                segmentation_label[1 , row.start : row.end] = 1.0
            else: 
                # Diastole
                pass        
        
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
                segmentation_label[-1,  start : end] = 1.0                
                
        # To erase only noise part
        if self.clean_noise:
            columns = segmentation_label.sum(dim=0)
            min_idx, max_idx = self.find_min_max_zero_indices(columns)
            features = features[:, min_idx : max_idx]
            segmentation_label = segmentation_label[:, min_idx : max_idx]            
        else:
            segmentation_label = segmentation_label[:, :]
    
        if murmur_label == "Absent": 
            murmur_label = torch.zeros(2).float()
            murmur_label[0] = 1.0
        else:
            murmur_label = torch.zeros(2).float()
            murmur_label[1]= 1.0
        
        if outcome_label == "Normal":
            outcome_label = torch.zeros(2).float()
            outcome_label[0] = 1.0
        else:
            outcome_label = torch.zeros(2).float()
            outcome_label[1] = 1.0
    
        return features, segmentation_label, murmur_label, outcome_label, wav_len, filename
    
    
class Unlabeled_Dataset(torch.utils.data.Dataset):                           
    def __init__(self, data_folder, mid_paths, filenames, sampling_rate, window_length, window_step, freq_bins,
                 slice_feature, sequence_length, cache= None):
        self.data_folder = pathlib.Path(data_folder)  
        self.mid_paths = mid_paths.to_list()
        self.filenames = filenames.to_list()
        self.sr = sampling_rate
        self.window_length = window_length
        self.window_step = window_step
        self.freq_bins= freq_bins
        self.sequence_length = sequence_length 
                
    def calc_frequency_bins(self):
        return int(np.ceil(config.freq_high * config.window_length))
                
    def calculate_features(self, recording):
        # assert fs == self.sr
        
        
        # Zero-mean and normalise by peak amplitude
        recording = recording.float()
        recording -= recording.mean()
        recording /= recording.abs().max()
        
        # Calculate spectrogram
        
        ## 1. Seconds to frame
        window_length = int(self.window_length * self.sr) 
        window_step = int(self.window_step * self.sr)
        
        ## 2. Wav to spectrogram
        spectrogram = (
            torch.stft(
                recording,
                n_fft=window_length,
                hop_length=window_step,
                window=torch.hann_window(window_length),
                center=False,
                return_complex=False,
            ).pow(2).sum(dim=-1)
        )
        
        # 3. Remove high frequencies above FREQUENCY_HIGH Hz
        spectrogram = spectrogram[ : self.calc_frequency_bins()]
        
        # 4. Log and z-normalise
        spectrogram = torch.log(spectrogram)
        
        spectrogram = (spectrogram - spectrogram.mean(dim=-1, keepdims=True)) / spectrogram.std(dim=-1, keepdims=True)
        
        features_fs = int(1 / self.window_step)        
        
        # assert not torch.isnan(spectrogram).any()
        return spectrogram, features_fs
    
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        
        mid_path = self.mid_paths[idx]
        filename = self.filenames[idx]
        
        filepath = self.data_folder / mid_path / filename # pathlib.Path
        sr, recording = spio.wavfile.read(filepath)
                    
        if len(recording.shape) > 1:
            recording = np.mean(recording, axis= 1)
        
        if sr != self.sr:
            num_samples = int(len(recording) * self.sr / sr)
            recording = resample(recording, num_samples)
        recording = torch.as_tensor(recording.copy())
        
        wav_len = len(recording) / self.sr
        features, _ = self.calculate_features(recording)        
        
        if torch.isnan(features).any():
            features = features.masked_fill(torch.isnan(features), 1e-20)
        
        # empty label
        segmentation_label = torch.zeros(3, features.shape[-1]) 
        murmur_label = torch.zeros(2).float()
        outcome_label = torch.zeros(2).float()

        return features, segmentation_label, murmur_label, outcome_label, wav_len, filename    
    
def calculate_class_weights(dataset):
    
    seq_class_counts = defaultdict(int)
    mm_class_counts = defaultdict(int)
    
    for i, (feature, seq_label, _, mm_label, _, _) in enumerate(dataset):
        seq_to_cls_label = seq_label.sum(dim= -1) # [C, T] >> [C]
        
        for idx, seq_cls_counts in enumerate(seq_to_cls_label):
            seq_class_counts[idx] += seq_cls_counts
        
        mm_class_counts[0] += mm_label[0].item()
        mm_class_counts[1] += mm_label[1].item()
        
    seq_total_sum = sum(seq_class_counts.values())
    
    
    seq_cls_weights = torch.zeros(seq_label.shape[0])
    
    for k, v in seq_class_counts.items():
        seq_cls_weights[k] = seq_total_sum / v
    
    mm_total_sum = sum(mm_class_counts.values())
    mm_weights = torch.zeros(mm_label.shape)
    
    for k, v in mm_class_counts.items():
        mm_weights[k] = mm_total_sum / v
    
    return seq_cls_weights, mm_weights

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
    val_results_df.index = val_results_df.index.astype(str) # Added
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
    val_results_df = val_results_df[['holo_HSMM']].groupby(level=[0]).max()
    
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



def calculate_class_weights(dataset):
    
    seq_class_counts = defaultdict(int)
    mm_class_counts = defaultdict(int)
    
    for i, (feature, seq_label, _, mm_label, _, _) in enumerate(dataset):
        seq_to_cls_label = seq_label.sum(dim= -1) # [C, T] >> [C]
        
        for idx, seq_cls_counts in enumerate(seq_to_cls_label):
            seq_class_counts[idx] += seq_cls_counts
        
        mm_class_counts[0] += mm_label[0].item()
        mm_class_counts[1] += mm_label[1].item()
        
    seq_total_sum = sum(seq_class_counts.values())
    seq_cls_weights = torch.zeros(seq_label.shape[0])
    
    for k, v in seq_class_counts.items():
        seq_cls_weights[k] = seq_total_sum / v
    
    mm_total_sum = sum(mm_class_counts.values())
    mm_weights = torch.zeros(mm_label.shape)
    
    for k, v in mm_class_counts.items():
        mm_weights[k] = mm_total_sum / v
    
    return seq_cls_weights, mm_weights


# th23 added--------------------------------------------------------------------------------------------------------

def filter_wavfiles(filename):
    if "normal" in filename or "murmur" in filename or "unlabelled" in filename:
        return True
    else:
        return False 
    
def define_category(filename):
    if "normal" in filename:
        return "Absent"
    elif "murmur" in filename:
        return "Present"
    else:
        return "Unlabeled"

def check_duration_in_dataframe(filepath):
    recording, sampling_rate = librosa.load(filepath, sr= config.sampling_rate)
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

# Combine External dataset   
def get_external_df(external_data_folder, external_subpath_dict):
    
    external_sufhsdb_path = external_subpath_dict["pysionet_sufhsdb"]
    external_kaggle_a = external_subpath_dict["kaggle_set_a"]
    external_kaggle_b = external_subpath_dict["kaggle_set_b"]
    
    kaggle_df = get_kaggle_df(external_data_folder, external_kaggle_a, external_kaggle_b)
    sufhsdb_df = get_pysionet_sufhsdb_df(external_data_folder, external_sufhsdb_path)
    
    #print(kaggle_df.shape, sufhsdb_df.shape)
    
    
    weaklabeled_df = kaggle_df[kaggle_df.label != "Unlabeled"]
    kaggle_unlabeled = kaggle_df[kaggle_df.label == "Unlabeled"]
    unlabeled_df = pd.concat([sufhsdb_df, kaggle_unlabeled], ignore_index= True)
    
    return weaklabeled_df, unlabeled_df


####################################################################################################

data_folder = config.train_data
external_data_folder = config.external_data
test_data_folder = config.test_data

num_k = config.num_k
val_fold_num = config.val_fold_num

split_ratio= 0.1 if config.debug else 1    

patient_df = load_patient_files(data_folder, stop_frac= split_ratio)
patient_df = create_folds(patient_df, num_k)
recording_df = patient_df_to_recording_df(patient_df)

print(f"\nTraining with {len(patient_df)} patients!\n")

recording_df_gq = recording_df[recording_df["patient_murmur_label"] != "Unknown"]
recording_df_bq = recording_df[recording_df["patient_murmur_label"] == "Unknown"]

# Dev_df
train_recording_df = recording_df_gq[recording_df_gq['val_fold'] != val_fold_num]
val_recording_df = recording_df_gq[recording_df_gq["val_fold"] == val_fold_num]    

# External_df
weaklabeled_df, unlabeled_df = get_external_df(external_data_folder, config.external_data_subpath)
concat_unlabeld_df = pd.concat([weaklabeled_df, unlabeled_df], axis= 0).reset_index(drop=True)

# Valid_df
# eval_df = pd.concat([val_recording_df, recording_df_bq], axis=0)


# Test df
test_patient_df = load_patient_files(test_data_folder, stop_frac= split_ratio)
test_recording_df = test_patient_df_to_recording_df(test_patient_df)    

Df_dict = {}

Df_dict["patient_df"] = patient_df
Df_dict["recording_df"] = recording_df
Df_dict["recording_df_gq"] = recording_df_gq
Df_dict["recording_df_bq"] = recording_df_bq
Df_dict["train_recording_df"] = train_recording_df
Df_dict["val_recording_df"] = val_recording_df
Df_dict["unknown_df"] = recording_df_bq
Df_dict["weaklabeled_df"] = weaklabeled_df
Df_dict["unlabeled_df"] = concat_unlabeld_df
Df_dict["test_patient_df"] = test_patient_df
Df_dict["test_recording_df"] = test_recording_df    



train_dataset =  Stronglabeled_Dataset(data_folder, 
                            train_recording_df.index, train_recording_df.murmur_timing, 
                            train_recording_df.patient_murmur_label, train_recording_df.outcome_label,
                            sampling_rate= config.sampling_rate,
                            window_length= config.window_length,
                            window_step= config.hop_length,
                            freq_bins= None,
                            clean_noise= config.clean_noise,
                            slice_feature= False,
                            sequence_length= config.sequence_length)


val_dataset =  Stronglabeled_Dataset(data_folder, 
                            val_recording_df.index,
                            val_recording_df.murmur_timing, 
                            val_recording_df.patient_murmur_label,
                            val_recording_df.outcome_label,
                            sampling_rate= config.sampling_rate,
                            window_length= config.window_length,
                            window_step= config.hop_length,
                            freq_bins= None,
                            clean_noise= True,
                            slice_feature= False,
                            sequence_length= None)

unlabeled_dataset = Unlabeled_Dataset(external_data_folder, concat_unlabeld_df.mid_path, concat_unlabeld_df.filename, 
                                    config.sampling_rate, config.window_length, config.hop_length, config.freq_bins, config.slice_feature, config.train_seq_len)

recording_df_gq

full_dataset = Stronglabeled_Dataset(data_folder, 
                            recording_df_gq.index, recording_df_gq.murmur_timing, 
                            recording_df_gq.patient_murmur_label, recording_df_gq.outcome_label,
                            sampling_rate= config.sampling_rate,
                            window_length= config.window_length,
                            window_step= config.hop_length,
                            freq_bins= None,
                            clean_noise= config.clean_noise,
                            slice_feature= False,
                            sequence_length= config.sequence_length)


# Df_dict, train_dataset, val_dataset, Unlabeled_dataset


MAX_LENGTH = cal_max_frame_len(config.sampling_rate, config.sequence_length)
FREQ_BINS = calc_frequency_bins()

