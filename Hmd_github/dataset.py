# packages
import os
import re
import warnings
# import librosa
import pandas as pd
import pathlib
from typing import Dict, Tuple, List
# from collections import defaultdict
# import yaml 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler

import scipy.io as spio
from scipy.signal import resample

import sklearn.metrics
import sklearn.model_selection

from omegaconf import OmegaConf

from utils import *
config_dir = "/Data1/hmd2/notebooks_th/Hmd_github/config.yaml"
config = OmegaConf.load(config_dir)


# config = check_config_path(config) # Debug path
# config = make_checkpoint_dirs(config) # Make checkpoint path
# caculate_hparmas(config)


# Ignore Warning messages.
warnings.filterwarnings("ignore",  category=UserWarning, message="stft with return_complex=False is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning, message="Series.__getitem__ treating keys as positions is deprecated")


##############################    Make Data Frames   ##############################


functions = {
    "age": get_age,
    "pregnant": get_pregnancy_status,
    "height": get_height,
    "weight": get_weight,
    "sex": get_sex,
}


# Make patient_df
## Load INFO from per patient. 
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


# Make recording_df for Train model.
## Load INFO from per wav file (Train).
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



# Make recording_df for Test fitted model.
## Load INFO from per wav file (Test).
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
    def __init__(self, data_folder, recording_paths, 
                 timings, murmur_labels, outcome_labels, sampling_rate,
                 clean_noise, cache= None):
        assert len(recording_paths) == len(murmur_labels)
        self.data_folder = pathlib.Path(data_folder)  
        self.recording_paths = recording_paths
        self.murmur_timings = timings
        self.murmur_labels = murmur_labels
        self.outcome_labels = outcome_labels
        
        self.sr = sampling_rate

        self.clean_noise = clean_noise        
        self.cache = cache
    
    def find_noise_onset_offset(self, indices):
        
        non_zero_indices = torch.where(indices != 0)[0]
        
        # return non-zero
        if len(non_zero_indices):
            min_idx = non_zero_indices.min().item()
            max_idx = non_zero_indices.max().item() + 1
        # return nothing
        else:
            warnings.warn("There are only noise part...! ", category=UserWarning)
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
        
        features, fs_features = calculate_features(recording, self.sr)
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
            indices = segmentation_label.sum(dim=0)
            min_idx, max_idx = self.find_noise_onset_offset(indices)
            features = features[:, min_idx : max_idx]
            segmentation_label = segmentation_label[:, min_idx : max_idx]            
        else:
            segmentation_label = segmentation_label[:, :]
    
        if murmur_label == "Absent": 
            murmur_label = torch.zeros(1).float()
            # murmur_label = 1.0
        else:
            murmur_label = torch.ones(1).float()
            # murmur_label[1]= 1.0
        
        if outcome_label == "Normal":
            outcome_label = torch.zeros(2).float()
            outcome_label[0] = 1.0
        else:
            outcome_label = torch.zeros(2).float()
            outcome_label[1] = 1.0
            
        if self.cache is not None:
            self.cache[filename]["features"] = features
            self.cache[filename]["seq_label"] = segmentation_label
            self.cache[filename]["murmur_label"] = murmur_label
            self.cache[filename]["outcome_label"] = outcome_label
            self.cache[filename]["wav_len"] = wav_len
    
        return features, segmentation_label, murmur_label, outcome_label, wav_len, filename



class Unlabeled_Dataset(torch.utils.data.Dataset):                           
    def __init__(self, parent_paths, filenames, sampling_rate,
                cache= None):
        
        self.parent_paths = parent_paths.tolist() 
        self.filenames = filenames.tolist() 
        self.sr = sampling_rate
                
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
                
        parent_path = pathlib.Path(self.parent_paths[idx])
        filename = self.filenames[idx]
        
        filepath = parent_path / filename # pathlib.Path
        sr, recording = spio.wavfile.read(filepath)
                    
        if len(recording.shape) > 1:
            recording = np.mean(recording, axis= 1)
        
        if sr != self.sr:
            num_samples = int(len(recording) * self.sr / sr)
            recording = resample(recording, num_samples)
        recording = torch.as_tensor(recording.copy())
        
        wav_len = len(recording) / self.sr
        features, fs_features = calculate_features(recording, self.sr)        
        
        if torch.isnan(features).any():
            features = features.masked_fill(torch.isnan(features), 1e-20)
        
        # Empty label
        segmentation_label = torch.zeros(3, features.shape[-1]) 
        murmur_label = torch.zeros(1).float()
        outcome_label = torch.zeros(2).float()

        return features, segmentation_label, murmur_label, outcome_label, wav_len, filename 



def merge_df(val_results_df, patient_df):
    val_results_df.index = val_results_df.index.str.split("_", expand=True)
    val_results_df = val_results_df[['holo_HSMM']].groupby(level=[0]).max()
    
    patient_df.index = patient_df.index.astype(int)
    val_results_df.index = val_results_df.index.astype(int) # index 데이터타입 통일
    
    combined_df = val_results_df.merge(patient_df, left_index=True, right_index=True)
    
    return combined_df



##############################    Data pre-processing   ##############################

def calc_frequency_bins():
    return int(np.ceil(config.dataset.frequency_high * config.dataset.window_length))

# wav to spectogram
def calculate_features(recording: torch.Tensor, fs: int) -> Tuple[torch.Tensor, int]:
    assert fs == config.dataset.sampling_rate

    # Zero-mean and normalise by peak amplitude
    recording = recording.float()
    recording -= recording.mean()
    recording /= recording.abs().max()

    # Calculate spectrogram
    window_length = int(config.dataset.window_length * fs)
    window_step = int(config.dataset.hop_length * fs)
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

    features_fs = int(1 / config.dataset.hop_length)
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


def get_external_df(external_dirs):
    
    parent_path_list = []
    filepaths_list = []
    wav_len_list = []
    
    for external_dir in external_dirs:
        for root, _, filenames in os.walk(external_dir):
            for filename in filenames:
                # if "normal" in filename or "murmur" in filename or "unlabelled" in filename:
                if filename.endswith('.wav'):
                        
                        abs_filepath = os.path.join(root, filename)
                        
                        sr, recording = spio.wavfile.read(abs_filepath)
                        if len(recording.shape) > 1:
                            recording = np.mean(recording, axis= 1)

                        parent_path_list.append(root) # absolute filename
                        filepaths_list.append(filename) # filename
                        wav_len_list.append(len(recording) / sr)
                else:
                    continue
                    
    external_df = pd.DataFrame({"parent_path": parent_path_list, "filename": filepaths_list, "wav_len": wav_len_list})
    
    external_df = external_df.query("wav_len >= 5.0").reset_index(drop=True) 
    
    return external_df





# ##############################    Load Data !  ##############################

# For collate_fn parameters....

default_spectrogram, fs = calculate_features(torch.randn(config.dataset.sampling_rate * config.dataset.sequence_length), 
                                         config.dataset.sampling_rate)
FREQ_BINS, MAX_LENGTH  = default_spectrogram.shape


def collate_fn(x):
    # max_length = cal_max_frame_len(config.sampling_rate, config.sequence_length)
    max_length= MAX_LENGTH
    
    all_features = torch.zeros(len(x), FREQ_BINS, max_length)  # [bs, freq, frame]
    all_seq_labels = torch.ones(len(x), 3, max_length) * (-1) # S1, S2, Murmur, 
    all_murmur_labels = []
    all_outcome_labels = []
    pad_masks = []    
    all_filenames = []
    
    for idx, (features, seq_label, murmur_label, outcome_label, wav_len, filename) in enumerate(x):
        
        pad_mask = torch.ones(max_length).float()
        # pad_mask = torch.ones_like()
        # pad_mask = torch.ones(FREQ_BINS, max_length)
        
        # 같거나 짧음
        if features.shape[-1] <= max_length:
            diff = max_length - features.shape[-1]
            start = random.randint(0, diff)
            end = start + features.shape[-1]
            all_features[idx, :, start : end] = features
            all_seq_labels[idx, : , start : end] = seq_label
            
            pad_mask[start:end] = 0.0 # 데이터 있는 부분이 0.0
            # pad_mask[:, start:end] = 0.0
            pad_masks.append(pad_mask)
            # actual_seq_lengths.append(len(seq_label))
            all_filenames.append(filename)
        # 더 길면
        else:
            diff = features.shape[-1] - max_length
            start = random.randint(0, diff)
            end =  start + max_length
            all_features[idx, :, :] = features[:, start : end]
            all_seq_labels[idx, :, :] = seq_label[:, start : end]
            
            pad_mask[:] = 0.0 # 모든 시퀀스에 데이터 있으므로 모두 0.0
            # pad_mask[:, :] = 0.0
            pad_masks.append(pad_mask)
            # actual_seq_lengths.append(max_length)

        all_murmur_labels.append(murmur_label)
        all_outcome_labels.append(outcome_label)
        all_filenames.append(filename)
        
    all_features = all_features.float()
    all_seq_labels = all_seq_labels.float()
    all_murmur_labels = torch.stack(all_murmur_labels).float()
    pad_masks = torch.stack(pad_masks).bool() #.float()

    return (all_features, all_seq_labels, pad_masks, all_murmur_labels, all_outcome_labels, all_filenames)






def load_dataframes(config):
    
    # Load datapath
    train_datapath = pathlib.Path(config.datapath.train_data)
    external_datapath = pathlib.Path(config.datapath.external_data)
    test_datapath = pathlib.Path(config.datapath.test_data)
    
    # Set validation data
    num_k = config.dataset.num_k
    val_fold_num = config.dataset.val_fold_num
    
    # For debug split (0.1) of data
    split_ratio= 0.1 if config.virtual_settings.debug else 1    
    
    patient_df = load_patient_files(train_datapath, stop_frac= split_ratio)
    patient_df = create_folds(patient_df, num_k)
    recording_df = patient_df_to_recording_df(patient_df)
    
    print(f"\nTrain with {len(patient_df)} patients, {len(recording_df)} recording files.")
    
    # Except Unknown class
    recording_df_gq = recording_df[recording_df["patient_murmur_label"] != "Unknown"]
    recording_df_bq = recording_df[recording_df["patient_murmur_label"] == "Unknown"]

    train_recording_df = recording_df_gq[recording_df_gq['val_fold'] != val_fold_num]
    val_recording_df = recording_df_gq[recording_df_gq["val_fold"] == val_fold_num]    
    
    # External_df
    external_dirs = []
    for path in os.listdir(external_datapath):
        
        if os.path.isdir(external_datapath / path):
            external_dirs.append(external_datapath / path)
    
    unlabeled_df = get_external_df(external_dirs)
    
    # Test df
    test_patient_df = load_patient_files(test_datapath, stop_frac= split_ratio)
    test_recording_df = test_patient_df_to_recording_df(test_patient_df)  
    
    # Load df_dictionary
    df_dict = {}
    
    df_dict["patient_df"] = patient_df
    df_dict["recording_df"] = recording_df
    df_dict["recording_df_gq"] = recording_df_gq
    df_dict["recording_df_bq"] = recording_df_bq
    df_dict["train_recording_df"] = train_recording_df
    df_dict["val_recording_df"] = val_recording_df

    df_dict["unlabeled_df"] = unlabeled_df
    df_dict["test_patient_df"] = test_patient_df
    df_dict["test_recording_df"] = test_recording_df 
    
    return df_dict



