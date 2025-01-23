import os
import json
import pathlib
import datetime
import pytz # 추가
from copy import deepcopy

import argparse
import yaml

from typing import Dict, Tuple, List
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import *
from data_augmentation import *
from model import *
from metric import *


def main(configs, data_folder: pathlib.Path, test_data_folder: pathlib.Path, external_data_folder: pathlib.Path,  model_folder: pathlib.Path, verbose: int, num_k, val_fold_num, device):
    
    
    patient_df = load_patient_files(data_folder, stop_frac= 1)
    print(f"Training with {len(patient_df)} patients")
    patient_df = create_folds(patient_df, num_k)
    
    
    #patient_df.to_csv(model_folder / "dataframes" / "patient_df.csv")
    recording_df = patient_df_to_recording_df(patient_df)

    #recording_df.to_csv(model_folder / "dataframes" /  "recording_df.csv") # for check df
    
    
    # location_specific_murmur_label = True
    # murmur_label_col = "rec_murmur_label" if location_specific_murmur_label else "patient_murmur_label"
    
    recording_df_gq = recording_df[recording_df["patient_murmur_label"] != "Unknown"] # 'Holosystolic', 'Early-systolic', 'Mid-systolic'
    recording_df_bq = recording_df[recording_df["patient_murmur_label"] == "Unknown"] # murmur_timing : nan
    
    train_recording_df = recording_df_gq[recording_df_gq['val_fold'] != val_fold_num]
    val_recording_df = recording_df_gq[recording_df_gq["val_fold"] == val_fold_num]
    


    
    # model.eval()됨
    model, configs = train(configs, data_folder, model_folder, 
                train_recording_df,val_recording_df, verbose, device)
                


    
    val_results_df = recording_df[recording_df['val_fold'] == val_fold_num]
    
    # clean_features, clean_segmentation_label, murmur_label, outcome_label, filename
    val_dataset = Stronglabeled_Dataset(data_folder, 
                                val_results_df.index, 
                                val_results_df.murmur_timing, 
                                val_results_df.patient_murmur_label,
                                val_results_df.outcome_label,
                                sampling_rate= configs["data_preprocess"]["sampling_rate"], 
                                sequence_length= None, # Default: int(TRAINING_SEQUENCE_LENGTH / WINDOW_STEP)
                                cache={})
    
     
     
    for idx, (val_feature, val_seq_label, val_murmur_label, _, filename) in enumerate(val_dataset): 
        
        
        max_seq_len = cal_max_frame_len(TRAINING_SEQUENCE_LENGTH)
                
        student_preds = []
        teacher_preds = []
        
        if val_feature.shape[-1] >= max_seq_len:
            eval_num = val_feature.shape[-1] // max_seq_len
            
            for num in range(eval_num):
                sliced_feature = val_feature[:, num * max_seq_len : (num+1) * max_seq_len]
                sliced_feature = sliced_feature.unsqueeze(dim=0)
                sliced_feature = sliced_feature.to(device)
                _, ith_murmur_pred_stu = student(sliced_feature)
                _, ith_murmur_pred_tch = teacher(sliced_feature)
                
                # ith_murmur_pred_stu = F.softmax(ith_murmur_pred_stu.squeeze(), dim= 0)
                # ith_murmur_pred_tch = F.softmax(ith_murmur_pred_tch.squeeze(), dim= 0)
                student_preds.append(ith_murmur_pred_stu.squeeze()[1].item())
                teacher_preds.append(ith_murmur_pred_tch.squeeze()[1].item())

            murmur_pred_stu = np.mean(student_preds).item()
            murmur_pred_tch = np.mean(teacher_preds).item()
        
        else:
            val_feature = val_feature.unsqueeze(dim=0).to(device)
            _, murmur_pred_stu = student(val_feature)
            _, murmur_pred_tch = teacher(val_feature)
            
            # murmur_pred_stu = F.softmax(murmur_pred_stu.squeeze(), dim= 0)
            # murmur_pred_tch = F.softmax(murmur_pred_tch.squeeze(), dim= 0)
            
            murmur_pred_stu = murmur_pred_stu.squeeze()[1].item()
            murmur_pred_tch = murmur_pred_tch.squeeze()[1].item()
        
        # 수정대본
        val_results_stu[filename]= {"holo_HSMM": murmur_pred_stu}
        val_results_tch[filename]= { "holo_HSMM": murmur_pred_tch}
        
       
    val_results_df_stu = pd.DataFrame.from_dict(val_results_stu, orient="index")
    val_results_df_tch = pd.DataFrame.from_dict(val_results_tch, orient="index")
    #val_results_df.to_csv(model_folder / 'dataframes' /  "val_results_df.csv")
    
    # audio file 하나하나에 대한 prediction 담긴 데이터프레임
    recording_df = recording_df.merge(val_results_df, left_index=True, right_index= True)
    #recording_df.to_csv(model_folder / 'dataframes' /  "recording_df.csv")
    
    # 1명당 오디오 파일들에 대한 예측값 평균냄 
    merged_df_stu = merge_df_stu(val_results_df_stu, patient_df)
    merged_df_tch = merge_df_tch(val_results_df_tch, patient_df)
    #merged_df.to_csv(model_folder / 'dataframes' / "merged_df2.csv")

    
    # val_murmur_predictions = {}
    # val_outcome_predictions = {}
    
        
    optim_threshold_stu= 0.0
    val_murmur_score_stu = 0.0
    
    for threshold in np.arange(0, 1.01, 0.01):
        val_murmur_preds_stu = {}

        for index, row in merged_df_stu.iterrows():
            murmur_pred = decide_murmur_with_threshold(row.to_dict(), threshold)
            val_murmur_preds_stu[index] = {
                "prediction": murmur_pred, 
                "probabilities": [], 
                "label": row["murmur_label"]}
            
        murmur_score_stu = compute_cross_val_weighted_murmur_accuracy(val_murmur_preds_stu, print= False)
        
        if val_murmur_score_stu < murmur_score_stu:
            val_murmur_score_stu = murmur_score_stu.item()
            optim_threshold_stu = threshold.item()
        
    print(f"Studnet's Valid Weighted murmur accuracy = {val_murmur_score_stu:.3f}")       
    print(f"Studnet's Optimized murmur threshold: {optim_threshold_stu:.3f}")



    optim_threshold_tch = 0.0
    val_murmur_score_tch = 0.0
    
    for threshold in np.arange(0, 1.01, 0.01):
        val_murmur_preds_tch = {}

        for index, row in merged_df_tch.iterrows():
            murmur_pred = decide_murmur_with_threshold(row.to_dict(), threshold)
            val_murmur_preds_tch[index] = {
                "prediction": murmur_pred, 
                "probabilities": [], 
                "label": row["murmur_label"]}
            
        murmur_score_tch = compute_cross_val_weighted_murmur_accuracy(val_murmur_preds_tch, print= False)        
        
        if val_murmur_score_tch < murmur_score_tch:
            val_murmur_score_tch = murmur_score_tch.item()
            optim_threshold_tch = threshold.item()
            

    print(f"Teacher's Valid Weighted murmur accuracy = {val_murmur_score_tch:.4f}")       
    print(f"Teacher's Optimized murmur threshold: {optim_threshold_tch:.4f}")
    
    
    
    # outcome_score = compute_cross_val_outcome_score(val_outcome_predictions)
    # print(f"\nValid Outcome score = {outcome_score:.0f}\n")
    
    test_murmur_score_stu, test_murmur_score_tch = test(test_data_folder, model_folder, optim_threshold_stu, optim_threshold_tch, student, teacher, device)
    
    #print(f"\nTest Murmur score: {test_murmur_score:.3f}\n")
    
    print(f"\n Student test Murmur Score: {test_murmur_score_stu:.4f}")
    print(f"\n Teacher test Murmur Score: {test_murmur_score_tch:.4f}")
    
    
    
    configs["result"] = {"val_WMM_accuracy_stu": val_murmur_score_stu, 
                        "val_WMM_accuracy_tch": val_murmur_score_tch, 
                        "optim_murmur_threshold_stu": optim_threshold_stu, 
                        "optim_murmur_threshold_stu": optim_threshold_tch, 
                        "test_WMM_accuracy_stu": test_murmur_score_stu,
                        "test_WMM_accuracy_tch": test_murmur_score_tch, 
                         }
    
    return configs







def train_rnn_network(configs, 
          strong_data_folder: pathlib.Path,
            model_folder: pathlib.Path,
            stronglabeled_df: pd.DataFrame, 
            validation_df: pd.DataFrame, 
            verbose: int,
            device):
    
    # Load Training params
    train_max_epochs = configs["train_params"]["max_epoch"] if not configs["debug_true"] else 1
    training_patience = configs["train_params"]["train_patience"]   
    
    # Define neural_network_model
    Rnn_network = GRU_not_sorted().to(device)
    
    
    optim = torch.optim.Adam(Rnn_network.parameters(), lr= configs["train_params"]["train_lr"], betas=(0.9, 0.999))
    crit_seq = nn.BCELoss().to(device)
    
    
    train_dataset = Stronglabeled_Dataset(strong_data_folder, 
                                stronglabeled_df.index, 
                                stronglabeled_df.murmur_timing, 
                                stronglabeled_df.patient_murmur_label,
                                stronglabeled_df.outcome_label,
                                sequence_length= None, # Default: int(TRAINING_SEQUENCE_LENGTH / WINDOW_STEP)
                                sampling_rate= configs["data_preprocess"]["sampling_rate"], 
                                cache={})
    val_dataset = Stronglabeled_Dataset(strong_data_folder, 
                                validation_df.index, 
                                validation_df.murmur_timing, 
                                validation_df.patient_murmur_label,
                                validation_df.outcome_label,
                                sequence_length= None, # Default: int(TRAINING_SEQUENCE_LENGTH / WINDOW_STEP)
                                sampling_rate= configs["data_preprocess"]["sampling_rate"], 
                                cache={})
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size= configs["train_params"]["strong_batch_size"], shuffle=True, collate_fn=collate_fn_thth
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size= configs["train_params"]["val_batch_size"], shuffle=False, collate_fn=collate_fn_thth
    )
    
    
    best_val_loss = np.inf
    early_stop_counter = 0  
    
     # Seq_loss_list
    train_seqloss_list = []
    val_seqloss_list = []
    
    for epoch in range(1, train_max_epochs + 1):
                
        running_seq_losses = 0.0

        Rnn_network.train()
        
                                                    # train_outcome_labels, actual_lengths
        for i, (train_mels, train_seq_labels, train_mm_labels, _, train_actual_lengths) in enumerate(train_dataloader):
            
            ### get mask to slice strongly/weakly labeled data
            train_bs = train_mels.size(0)

            # get mask not to calculate padded part
            train_seq_mask = (train_seq_labels != -1).to(device)

            
            # TODO: apply data augmentations
            do_mixup = torch.randn(1).item()
            do_frameshift = torch.randn(1).item()
            do_addnoise = torch.randn(1).item()
            
            
            # # Frame_shift
            # if do_frameshift > 0:
            #     #train_mels, train_seq_labels = frame_shift(train_mels, train_seq_labels)
            #     pass
            
            # # Mix-up, default: "soft" mix-up
            
            if (do_frameshift <= 0) & (do_mixup > 0):
                
                # strong_data mix-up
                train_mels, train_seq_labels, train_mm_labels = \
                    mixup(train_mels, train_seq_labels, train_mm_labels)
            
            # # add Guassian-noise to unlabeled_data
            # if do_addnoise > 0:
            #     train_mels = add_gaussian_noise(train_mels, configs["train_params"]["w_gauss_noise"])
                #train_seq_labels[mask_unlabel] = add_gaussian_noise(train_seq_labels[mask_unlabel], device)
                #train_mm_labels[mask_unlabel] = add_gaussian_noise(train_mm_labels[mask_unlabel], device)

            # other data augmentations that does not affect labels
            # ----------------------------------------------------
            
            # cpu to gpu
            train_mels = train_mels.to(device)
            train_seq_labels = train_seq_labels.to(device)
            train_mm_labels = train_mm_labels.to(device)
            
            optim.zero_grad()
            
            # Train: Student_preds
            seq_preds, rnn_output = Rnn_network(train_mels)
            seq_preds = F.softmax(seq_preds, dim=1)
            
            # seq_loss
            
            seq_loss = F.binary_cross_entropy(seq_preds, train_seq_labels)
            #seq_loss = seq_loss[train_seq_mask].mean()
              
            # TODO: add scheduler
            seq_loss.backward()
            optim.step()
              
            # Calculate outcome loss
            running_seq_losses += seq_loss.item() * train_bs
            
        # In i'th epoch train_loss
        train_seqloss_list.append(running_seq_losses / len(train_dataset))
     
        with torch.no_grad():
            
            Rnn_network.eval()
            
            val_seq_losses = 0.0
                                                                    # val_outcome_labels, val_filenames
            for i, (val_features, val_seq_labels, val_murmur_labels, _, val_actual_lengths) in enumerate(val_dataloader):
                
                val_batch_size = val_features.shape[0]
                
                # val_feature
                val_features = val_features.to(device)
                
                # val_labels
                
                val_seq_labels = val_seq_labels.to(device)
                
                # val_preds
                val_seq_preds, rnn_output = Rnn_network(val_features)
                val_seq_preds = F.softmax(val_seq_preds, dim=1)
                
                # val_loss            
                #val_seq_mask = (val_seq_labels != -1).to(device)
                val_seq_loss = F.binary_cross_entropy(val_seq_preds, val_seq_labels)
                #val_seq_loss = val_seq_loss[:, val_seq_mask]

                val_seq_losses += val_seq_loss.item() * val_batch_size
                
                
            # val_loss = val_losses / len(val_dataset)
            val_seqloss_list.append(val_seq_losses /  len(val_dataset))

    
    
        # if verbose >= 1:
        #     print(f"epoch: {epoch:04d}/{train_max_epochs} | Train_loss: {running_loss:.4f} |  Val_loss: {val_loss:.4f}", end= "\n" )

        
        if verbose >= 1:
            print(f"epoch: {epoch:04d}/{train_max_epochs}, \
                Train_seq_loss: {running_seq_losses / len(train_dataset):.10f}, \
                    Val_seq_loss: {val_seq_losses /  len(val_dataset):.10f}", end= "\n")
  
            
        # Save best student
        if val_seq_losses < best_val_loss:

            save_model(model_folder, Rnn_network)
            best_val_loss = val_seq_losses
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter > training_patience:
                break
            


    loss_df = pd.DataFrame({"Train_seq_loss": train_seqloss_list,
                        "val_loss_stu": val_seqloss_list})
    loss_df.to_csv(model_folder / 'dataframes' /  "seqloss_df.csv")
    
    
    print(f"\nFinished training RNN network.")
    
    
    model_name = type(Rnn_network).__name__
    
    # model.eval()
    Rnn_network = load_model(model_folder, model_name).to(device)
    
    
    return Rnn_network



    
    
    

def train_CNN(configs, strong_data_folder: pathlib.Path,
            model_folder: pathlib.Path,
            stronglabeled_df: pd.DataFrame, 
            validation_df: pd.DataFrame, 
            verbose: int,
            Rnn_network,
            device):

     # Load Training params
    train_max_epochs = configs["train_params"]["max_epoch"] if not configs["debug_true"] else 1
    training_patience = configs["train_params"]["train_patience"]
    
    
    
    # Define CNN network
    Cnn_network = BinaryClassificationCNN().to(device)
    
    optim = torch.optim.Adam(Cnn_network.parameters(), lr= configs["train_params"]["train_lr"], betas=(0.9, 0.999))
    crit_murmur = nn.BCELoss().to(device)
    
    
    train_dataset = Stronglabeled_Dataset(strong_data_folder, 
                                stronglabeled_df.index, 
                                stronglabeled_df.murmur_timing, 
                                stronglabeled_df.patient_murmur_label,
                                stronglabeled_df.outcome_label,
                                sequence_length= None, # Default: int(TRAINING_SEQUENCE_LENGTH / WINDOW_STEP)
                                sampling_rate= configs["data_preprocess"]["sampling_rate"], 
                                cache={})
    val_dataset = Stronglabeled_Dataset(strong_data_folder, 
                                validation_df.index, 
                                validation_df.murmur_timing, 
                                validation_df.patient_murmur_label,
                                validation_df.outcome_label,
                                sequence_length= None, # Default: int(TRAINING_SEQUENCE_LENGTH / WINDOW_STEP)
                                sampling_rate= configs["data_preprocess"]["sampling_rate"], 
                                cache={})
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size= configs["train_params"]["strong_batch_size"], shuffle=True, collate_fn=collate_fn_thth
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size= configs["train_params"]["val_batch_size"], shuffle=False, collate_fn=collate_fn_thth
    )
    
    
    best_val_loss = np.inf
    early_stop_counter = 0   
    
    
     # Seq_loss_list
    train_murmurloss_list = []
    val_murmurloss_list = []
    
    for epoch in range(1, train_max_epochs + 1):
        
        running_murmur_losses = 0.0
        
        Cnn_network.train()
    
        for i, (train_mels, train_seq_labels, train_mm_labels, _, _) in enumerate(train_dataloader):
            
            train_bs = train_mels.shape[0]
            
            train_mels = train_mels.to(device)
            train_mm_labels = train_mm_labels.to(device)
            
            optim.zero_grad()
            
            _, rnn_output = Rnn_network(train_mels)
            
            #print(train_mels.shape, rnn_output.shape)
            
            cnn_input = torch.concat([train_mels.unsqueeze(1), rnn_output.unsqueeze(1).detach()], dim= 1)
            cnn_output = Cnn_network(cnn_input)

        
            #print(cnn_output.shape, train_mm_labels.shape)
            
            murmur_loss = crit_murmur(cnn_output, train_mm_labels)
            
            murmur_loss.backward()
            optim.step()
            
            running_murmur_losses += murmur_loss.item() * train_bs
            
        train_murmurloss_list.append(running_murmur_losses / len(train_dataset))
    
    
        with torch.no_grad():
                
            Cnn_network.eval()
            
            val_murmur_losses = 0.0
            
            for i, (val_mels, _, val_murmur_labels, _, _) in enumerate(val_dataloader):
                
                val_bs = val_mels.shape[0]
                
                val_mels = val_mels.to(device)
                val_murmur_labels = val_murmur_labels.to(device)
                
                _, rnn_output_val = Rnn_network(val_mels)
                

                cnn_input_val = torch.concat([val_mels.unsqueeze(1), rnn_output_val.unsqueeze(1).detach()], dim= 1)
                
                cnn_output_val = Cnn_network(cnn_input_val)
                
                murmur_loss = crit_murmur(cnn_output_val, val_murmur_labels)
                
                
                val_murmur_losses += murmur_loss.item() * val_bs
                
                
            val_murmurloss_list.append(val_murmur_losses / len(val_dataset))
                
                
            if verbose >= 1:
                    print(f"epoch: {epoch:04d}/{train_max_epochs},Train_seq_loss: {running_murmur_losses / len(train_dataset):.10f}, Val_seq_loss: {val_murmur_losses /  len(val_dataset):.10f}", end= "\n")

            # Save CNN
            if val_murmur_losses / len(val_dataset) < best_val_loss:
                
                save_model(model_folder, Cnn_network)
                best_val_loss = val_murmur_losses / len(val_dataset)
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter > training_patience:
                    break
                
                
        murmur_loss_df = pd.DataFrame({"Train_seq_loss": train_murmurloss_list,
                        "val_loss_stu": val_murmurloss_list})
        murmur_loss_df.to_csv(model_folder / 'dataframes' /  "murmurloss_df.csv")      
        
        print(f"\nFinished training CNN network.")
        
        model_name = type(Rnn_network).__name__
    
        # model.eval()
        Cnn_network = load_model(model_folder, model_name)
        
        
        return Cnn_network
        


def train(configs, strong_data_folder: pathlib.Path,
            model_folder: pathlib.Path,
            stronglabeled_df: pd.DataFrame, 
            validation_df: pd.DataFrame, 
            verbose: int,
            device):
    

    Rnn_network = train_rnn_network(configs, strong_data_folder,  model_folder,
            stronglabeled_df, validation_df, verbose, device)
    
    Cnn_network = train_CNN(configs, strong_data_folder, model_folder,
            stronglabeled_df, validation_df, 
            verbose,
            Rnn_network,
            device)
    
    print("여기까지 완료..!")
    
    
    
    
    
    
    
    
    
    
    
    
    
    return 
    
    # Load Training params
    train_max_epochs = configs["train_params"]["max_epoch"] if not configs["debug_true"] else 1
    training_patience = configs["train_params"]["train_patience"]
    
    
    # crit_murmur = nn.CrossEntropyLoss().to(device)
    crit_murmur = nn.BCELoss().to(device)
    
    

    

    
    



    
    
    
    
    
    
    
    
    return Rnn_network, configs


def test(test_data_folder: pathlib.Path, model_folder: pathlib.Path, threshold_stu, threshold_tch, student, teacher, device):
    
    patient_df_test = load_patient_files(test_data_folder, stop_frac= 1)
    patient_files = find_patient_files(test_data_folder)
    
    test_results_stu = {}
    test_results_tch = {}
    
    for patient_file, patient_id in zip(patient_files, patient_df_test.index):
        patient_data = load_patient_data(patient_file) # 환자 1명의 .txt파일 읽어옴
        recordings = load_recordings(test_data_folder, patient_data) # 거기서 1-D .wav file들 읽어옴

        murmur_preds_ID_stu = []
        murmur_preds_ID_tch = []
        
        max_seq_len = cal_max_frame_len(TRAINING_SEQUENCE_LENGTH)
        
        for recording in recordings:
            
            recording = torch.as_tensor(recording)
            feature, fs_features = calculate_features(recording, FREQUENCY_SAMPLING)
            
            murmur_preds_file_stu = []
            murmur_preds_file_tch = []
            
            if feature.shape[-1] >= max_seq_len:
                eval_num = feature.shape[-1] // max_seq_len

                for num in range(eval_num):
                    sliced_feature = feature[:, num * max_seq_len : (num+1) * max_seq_len]
                    sliced_feature = sliced_feature.unsqueeze(dim= 0).to(device)
                    _, ith_murmur_pred_stu = student(sliced_feature)
                    _, ith_murmur_pred_tch = teacher(sliced_feature)
                    
                    # print(f"if: {ith_murmur_pred_stu.shape}")
                    
                    # = F.softmax(ith_murmur_pred_stu.squeeze(), dim= 0)
                    #ith_murmur_pred_tch = F.softmax(ith_murmur_pred_tch.squeeze(), dim= 0)
                    murmur_preds_file_stu.append(ith_murmur_pred_stu.squeeze()[1].item())
                    murmur_preds_file_tch.append(ith_murmur_pred_tch.squeeze()[1].item())

                # murmur_pred = np.max(murmur_preds_file_stu)
                
                murmur_pred_file_stu = np.mean(murmur_preds_file_stu).item()
                murmur_pred_file_tch = np.mean(murmur_preds_file_tch).item()
                
                
            else:
                feature = feature.unsqueeze(dim= 0).to(device)
                
                _, murmur_pred_file_stu = student(feature)
                _, murmur_pred_file_tch = teacher(feature)

                # print(f"else: {murmur_pred_file_stu.shape}")
                
                # murmur_pred_file_stu = F.softmax(murmur_pred_file_stu.squeeze(), dim= 0)
                # murmur_pred_file_tch = F.softmax(murmur_pred_file_tch.squeeze(), dim= 0)
                murmur_pred_file_stu = murmur_pred_file_stu.squeeze()[1].item()
                murmur_pred_file_tch = murmur_pred_file_tch.squeeze()[1].item()

            murmur_preds_ID_stu.append(murmur_pred_file_stu)
            murmur_preds_ID_tch.append(murmur_pred_file_tch)
        
        
        murmur_pred_ID_mean_stu = np.mean(murmur_preds_ID_stu).item()
        murmur_pred_ID_mean_tch = np.mean(murmur_preds_ID_tch).item()
        
        
        
        #patient_id
        test_results_stu[patient_id] = {"holo_HSMM": murmur_pred_ID_mean_stu}
        test_results_tch[patient_id] = {"holo_HSMM": murmur_pred_ID_mean_tch}
        
    test_results_df_stu = pd.DataFrame.from_dict(test_results_stu, orient="index")
    test_results_df_tch = pd.DataFrame.from_dict(test_results_tch, orient="index")
    
    patient_df_test_stu = patient_df_test.merge(test_results_df_stu, left_index= True, right_index= True)
    patient_df_test_tch = patient_df_test.merge(test_results_df_tch, left_index= True, right_index= True)
    patient_df_test_stu.to_csv(model_folder / 'dataframes' / "test_df_stu.csv")
    patient_df_test_tch.to_csv(model_folder / 'dataframes' / "test_df_tch.csv")
    # loss_df.to_csv(model_folder / 'dataframes' /  "loss_df_BCE.csv")
    #patient_df_test.to_csv(model_folder / 'dataframes' / "test_df.csv")


    test_murmur_preds_stu = {}
    for index, row in patient_df_test_stu.iterrows():
        murmur_prediction = decide_murmur_with_threshold(row.to_dict(), threshold_stu)
        test_murmur_preds_stu[index] = {
                "prediction": murmur_prediction, 
                "probabilities": [], 
                "label": row["murmur_label"]}

    print("\nStudent test Murmur Matrix\n")
    test_murmur_score_stu = compute_cross_val_weighted_murmur_accuracy(test_murmur_preds_stu, print=True)
    
    test_murmur_preds_tch = {}
    for index, row in patient_df_test_tch.iterrows():
        murmur_prediction = decide_murmur_with_threshold(row.to_dict(), threshold_tch)
        test_murmur_preds_tch[index] = {
                "prediction": murmur_prediction, 
                "probabilities": [], 
                "label": row["murmur_label"]}

    print("\nTeacher test Murmur Matrix\n")
    test_murmur_score_tch = compute_cross_val_weighted_murmur_accuracy(test_murmur_preds_tch, print=True)
    
    
    return test_murmur_score_stu.item(), test_murmur_score_tch.item() # numpy object to float
    
        
def collate_fn_thth(x):
    all_murmur_labels = []
    all_outcome_labels = []
    actual_lengths = []
    
    max_seq_len = cal_max_frame_len(TRAINING_SEQUENCE_LENGTH)
    freq_bins = calc_frequency_bins()
    
    all_features = torch.zeros(len(x), freq_bins, max_seq_len)
    all_seq_labels = torch.zeros(len(x), 5, max_seq_len)
    
    
    for idx, (feature, seq_label, murmur_label, outcome_label, filename) in enumerate(x): # batch_size train_dataset
        
        current_len = feature.size(1)
        
        
        if current_len < max_seq_len:
            actual_lengths.append(current_len)
            
            diff = max_seq_len - current_len
            
            random_num = random.randint(0, diff)
            
            start, end = min(diff - random_num, random_num), max(diff - random_num, random_num)
            padded_feature = F.pad(feature, (start, end), 'constant', 0.0)
            seq_label = F.pad(seq_label, (start, end), 'constant', 0.0)  # After mask padded part BCE loss
        else:
            
            actual_lengths.append(max_seq_len)
            diff = current_len - max_seq_len
            
            # 얘는 해주자...
            start = random.randint(0, diff)
            padded_feature = feature[:, start : start + max_seq_len]
            seq_label = seq_label[:, start : start + max_seq_len]

        
        all_features[idx] = padded_feature
        all_murmur_labels.append(murmur_label)
        all_outcome_labels.append(outcome_label)
        # all_filenames.append(filename)

    all_features = all_features.float()
    all_murmur_labels = torch.stack(all_murmur_labels).float()
    # all_outcome_labels = torch.tensor(all_outcome_labels)
        
    return all_features, all_seq_labels, all_murmur_labels, all_outcome_labels, torch.as_tensor(actual_lengths) 







def get_models():
    
    # model_seld.py
    student_model = GRU_frame_murmur()
    
    # ema network
    #teacher_model = deepcopy(student_model)
    teacher_model = GRU_frame_murmur()
    for param in teacher_model.parameters():
        param.detach_()
            
#    # 수정
#     if multigpu and (train_cfg["n_gpu"] > 1):
#         net = nn.DataParallel(net)
#         ema_net = nn.DataParallel(ema_net)

#     net = net.to(train_cfg["device"])
#     ema_net = ema_net.to(train_cfg["device"])

    return student_model, teacher_model




# Load your trained model.            
def save_model(model_folder: pathlib.Path, model: nn.Module):
    model_name = type(model).__name__
    
    model_path = model_folder / f"best_{model_name}.pt"
    torch.save(model.state_dict(), model_path)
    


    
    
# Load your trained model.
def load_model(model_folder: pathlib.Path, model_name: str):
    #model = GRU_not_sorted()
    model = eval(model_name)()
    model.load_state_dict(torch.load(pathlib.Path(model_folder) / f"best_{model_name}.pt"))
    return model.eval()



if __name__ == '__main__':
    # parser
    parser = argparse.ArgumentParser()  
    parser.add_argument("--train_data", type=str, default= None) 
    parser.add_argument("--test_data", type=str, default= None) 
    parser.add_argument("--model_folder", type=str, default= None) 
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--gpu_index", type=str, default= None) 
    parser.add_argument("--external_data", type=str, default= None) 
    parser.add_argument("--default_config", type= str, default="/Data1/hmd2/notebooks_th/GRU_loss_add/config_default.yaml")
    parser.add_argument("--prev_run", type=str, default= None)
    parser.add_argument("--verbose", type=int, default= 1)
    args = parser.parse_args()
    
############################__Please assign PATH, GPU_index__########################################################################################
    
    #args.data = "your Train data path" : str
    #args.model_folder = "your model params path" : str
    #args.gpu_index = 'SELECT index of gpu' : str
    #args.model_name= None
    
    args.train_data = "/Data2/murmur/train" 
    args.test_data = "/Data2/murmur/test" 
    args.external_data = "/Data2/heart_sound_dataset/"
    args.model_folder = "/Data1/hmd2/notebooks_th/GRU_loss_add/models"   # /Data1/hmd2/notebooks_th/CUEDA_easy
    args.gpu_index = '0'
    #args.model_name = None # If none, model_name : datetime string
    args.model_name = "Forcheck"
    
######################################################################################################################################################    
    
    with open(args.default_config, "r") as f:
        configs = yaml.safe_load(f)
    
      
    if args.model_name is None:
        # current_time = datetime.datetime.now()
        # filename = current_time.strftime("%Y-%m-%d__%H-%M-%S") 
        
        current_time = datetime.datetime.utcnow() # 현재 UTC 시간을 가져옴
        kst_timezone = pytz.timezone('Asia/Seoul') # 대한민국 시간대 설정
        kst_now = current_time.astimezone(kst_timezone) # UTC 시간을 대한민국 시간으로 변환
        filename = kst_now.strftime("%Y-%m-%d_%H:%M:%S")
    else:
        filename = args.model_name
    
    
    if not os.path.exists(args.train_data):
        raise Exception(f"The path {args.train_data} does not exist!")
    elif not os.path.exists(args.test_data):
        raise Exception(f"The path {args.test_data} does not exist!")
    elif not os.path.exists(args.model_folder):
        raise Exception(f"The path {args.model_folder} does not exist!")
    elif not args.gpu_index:
        raise Exception(f"You didn't assigned index of GPU!")
    else:
        pass
    
    # model_folder= path: where to save model_parameter
    model_folder = args.model_folder + '/' + filename
    
    
    # Don't Change This Line, Ever, Forever!
    ## unless you want to use Multi-gpu...
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index  # Select ONE specific index of gpu
    GPU_NUM = 0 # Since we Selected ONE specific gpu in environment, index number is 0
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu') 
    torch.cuda.set_device(device) # Set gpu, Done!
    
    args.train_data = pathlib.Path(args.train_data)
    args.test_data = pathlib.Path(args.test_data)
    args.external_data = pathlib.Path(args.external_data)
    model_folder = pathlib.Path(model_folder)
    
    os.makedirs(model_folder, exist_ok=True)  # 추가!!
    os.makedirs(model_folder / 'dataframes', exist_ok=True)    
    
    
    
    # 시작 시간 기록
    start_time = datetime.datetime.now()
    
    configs = main(configs = configs, data_folder= args.train_data, test_data_folder= args.test_data, external_data_folder= args.external_data , 
                   model_folder= model_folder, verbose= args.verbose, num_k= 4, val_fold_num= 0, device= device)
    
    
    # Save config.yaml
    
    config_result_dir = model_folder / "configs_result.yaml"
    
    # if not configs["debug_true"]:
    with open(config_result_dir, "w") as file:
        yaml.dump(configs, file, default_flow_style=False, allow_unicode=True)
    
    end_time = datetime.datetime.now()
    
    # 수행 시간 계산
    elapsed_time = end_time - start_time
    print(f"작업 수행 시간 (초): {elapsed_time.total_seconds() // 60}분")
    print("Done.")
    
    # main(data_folder, model_folder, verbose: int, num_k, val_fold_num, device):
