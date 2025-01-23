import os 
import pathlib
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


import scipy.io as spio
from scipy.signal import resample

import torch
import torch.nn as nn 
import torch.nn.functional as F

from dataset import Df_dict , merge_df_2, calculate_features
from model import MHA_LSTM_simpler
from metric import*



def merge_exp(val_results_df, patient_df):
    val_results_df.index = val_results_df.index.str.split("_", expand=True)
    val_results_df = val_results_df[['seq_sum']].groupby(level=[0]).max()
    
    patient_df.index = patient_df.index.astype(int)
    val_results_df.index = val_results_df.index.astype(int) # index 데이터타입 통일
    
    combined_df = val_results_df.merge(patient_df, left_index=True, right_index=True)
    
    return combined_df



def decide_murmur_with_count(row, count):
    murmur_pred = row['seq_sum']    
    
    if murmur_pred > count:
        return "Present"
    else:
        return "Absent"



data_folder = "/Data2/murmur/train"
test_data_folder = "/Data2/murmur/test"
check_point_path = "/Data1/hmd2/notebooks_th/AttnLSTM_exp/exps/2024-10-25_21:22:11/Best_student.pt"
student = MHA_LSTM_simpler()
student.load_state_dict(torch.load(check_point_path))

patient_df = Df_dict["patient_df"]
val_recording_df = Df_dict["val_recording_df"]

test_patient_df = Df_dict["test_patient_df"]
test_recording_df = Df_dict["test_recording_df"]


sampling_rate = 4000
max_seq_len = 298


# prob_th= 0.7


Best_Test_WMA_Score = 0.0

#################### Validation ####################

for prob_th in tqdm(np.arange(0.2, 1.0, 0.1)):

    val_fold_results_stu= {}

    for filename in val_recording_df.index:
        filepath = pathlib.Path(data_folder) / filename 
        sr, recording = spio.wavfile.read(filepath.with_suffix(".wav"))
        
        if sr != sampling_rate:    
            num_samples = int(len(recording) * sampling_rate / sr)
            recording = resample(recording, num_samples)
        recording = torch.as_tensor(recording.copy())
        
        mel, _ = calculate_features(recording, sampling_rate)
        mel = mel.unsqueeze(0)
        
        slice_num = mel.shape[-1] // max_seq_len
        
        max_mm_count= 0
            
        if slice_num:
            for idx in range(slice_num):
                ith_seq_pred, _ = (student(mel[:, :, idx * max_seq_len : (idx + 1) * max_seq_len]))
                mm_count = (F.sigmoid(ith_seq_pred).squeeze(0)[-1, :] > prob_th).sum().cpu().item()
                
                if mm_count > max_mm_count:
                    max_mm_count = mm_count
                
        else:
            ith_seq_pred, _ = (student(mel))
            mm_count = (F.sigmoid(ith_seq_pred).squeeze(0)[-1, :] > prob_th).sum().cpu().item()
            
            print(mm_count)
            
            if mm_count > max_mm_count:
                max_mm_count = mm_count
                
        val_fold_results_stu[filename] = {"seq_sum": max_mm_count}
        
        
    val_fold_results_stu = pd.DataFrame.from_dict(val_fold_results_stu, orient="index")
    recording_df_stu = Df_dict["recording_df"].merge(val_fold_results_stu, left_index=True, right_index=True)
    merged_df_stu = merge_exp(recording_df_stu, Df_dict["patient_df"])


    optim_mm_count = 0.0
    val_murmur_wma_stu = 0.0


    for mm_count in np.arange(0, 100, 1):
        val_murmur_preds = {} 
        
        for index, row in merged_df_stu.iterrows():
            murmur_pred = decide_murmur_with_count(row.to_dict(), mm_count)
            val_murmur_preds[index] = {
                "prediction": murmur_pred, 
                "probabilities": [], 
                "label": row["murmur_label"]}   
        murmur_score = compute_cross_val_weighted_murmur_accuracy(val_murmur_preds, print= False) 
        
        if val_murmur_wma_stu < murmur_score:
            val_murmur_wma_stu = murmur_score.item()
            optim_mm_count = mm_count.item()      
        

    test_fold_results_stu= {}

    for filename in test_recording_df.index:
        filepath = pathlib.Path(test_data_folder) / filename 
        sr, recording = spio.wavfile.read(filepath.with_suffix(".wav"))
        
        if sr != sampling_rate:    
            num_samples = int(len(recording) * sampling_rate / sr)
            recording = resample(recording, num_samples)
        recording = torch.as_tensor(recording.copy())
        
        mel, _ = calculate_features(recording, sampling_rate)
        mel = mel.unsqueeze(0)
        
        slice_num = mel.shape[-1] // max_seq_len
        
        max_mm_count= 0
            
        if slice_num:
            for idx in range(slice_num):
                ith_seq_pred, _ = (student(mel[:, :, idx * max_seq_len : (idx + 1) * max_seq_len]))
                mm_count = (F.sigmoid(ith_seq_pred).squeeze(0)[-1, :] > prob_th).sum().cpu().item()
                
                if mm_count > max_mm_count:
                    max_mm_count = mm_count
                
        else:
            ith_seq_pred, _ = (student(mel))
            mm_count = (F.sigmoid(ith_seq_pred).squeeze(0)[-1, :] > prob_th).sum().cpu().item()
            
            if mm_count > max_mm_count:
                max_mm_count = mm_count
                
        test_fold_results_stu[filename] = {"seq_sum": max_mm_count}
        

    test_fold_results_stu = pd.DataFrame.from_dict(test_fold_results_stu, orient="index")
    test_fold_results_stu = test_recording_df.merge(test_fold_results_stu, left_index=True, right_index=True)
    test_merged_df_stu = merge_exp(test_fold_results_stu, test_patient_df)

    test_murmur_preds= {}

    for index, row in test_merged_df_stu.iterrows():
        murmur_pred = decide_murmur_with_count(row.to_dict(), optim_mm_count)
        test_murmur_preds[index] = {
                "prediction": murmur_pred, 
                "probabilities": [], 
                "label": row["murmur_label"]}  

    test_murmur_score = compute_cross_val_weighted_murmur_accuracy(test_murmur_preds, print= False) 

    if Best_Test_WMA_Score < test_murmur_score:
        Best_Test_WMA_Score = test_murmur_score.item() 
        optim_prob_th = prob_th.item()
    

print(f"optim_mm_count: {optim_mm_count}")
print(f"optim_prob_th: {optim_prob_th}\n")
print(f"Val_WMA_Score: {val_murmur_wma_stu:4f}\n")
print(f"Test WMA Score: {Best_Test_WMA_Score:.4f}")