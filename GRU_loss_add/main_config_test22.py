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
    
    
    # External_df
    weaklabeled_df, unlabeled_df = get_external_df(configs, external_data_folder)

    
    # model.eval()됨
    student, teacher, configs = train(configs,
                data_folder, model_folder, external_data_folder,
                train_recording_df, weaklabeled_df, unlabeled_df, val_recording_df,
                verbose, device)

    
    val_results_stu= {}
    val_results_tch= {}
    
    val_results_df = recording_df[recording_df['val_fold'] == val_fold_num]
    
    
    # Validation
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
                
                ith_murmur_pred_stu = F.softmax(ith_murmur_pred_stu.squeeze(), dim= -1)
                ith_murmur_pred_tch = F.softmax(ith_murmur_pred_tch.squeeze(), dim= -1)
                student_preds.append(ith_murmur_pred_stu[1].item())
                teacher_preds.append(ith_murmur_pred_tch[1].item())

            murmur_pred_stu = np.mean(student_preds).item()
            murmur_pred_tch = np.mean(teacher_preds).item()
        
        else:
            val_feature = val_feature.unsqueeze(dim=0).to(device)
            _, murmur_pred_stu = student(val_feature)
            _, murmur_pred_tch = teacher(val_feature)
            
            murmur_pred_stu = F.softmax(murmur_pred_stu.squeeze(), dim= -1)
            murmur_pred_tch = F.softmax(murmur_pred_tch.squeeze(), dim= -1)
            
            murmur_pred_stu = murmur_pred_stu[1].item()
            murmur_pred_tch = murmur_pred_tch[1].item()
        
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
        
    print(f"Student's Valid Weighted murmur accuracy = {val_murmur_score_stu:.3f}")       
    print(f"Student's Optimized murmur threshold: {optim_threshold_stu:.3f}")


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
    
    test_murmur_score_stu, test_murmur_score_tch = test(test_data_folder, model_folder, optim_threshold_stu, optim_threshold_tch, student, teacher, device)
    
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


def train(configs, 
          strong_data_folder: pathlib.Path,
            model_folder: pathlib.Path,
            external_data_folder: pathlib.Path,
            stronglabeled_df: pd.DataFrame, 
            weaklabeled_df: pd.DataFrame,
            unlabeled_df: pd.DataFrame,
            validation_df: pd.DataFrame, 
            verbose: int,
            device):
    
    shared_feature_cache = {}
    
    # Train_dataset
    strong_dataset = Stronglabeled_Dataset(strong_data_folder, 
                                stronglabeled_df.index, 
                                stronglabeled_df.murmur_timing, 
                                stronglabeled_df.patient_murmur_label,
                                stronglabeled_df.outcome_label,
                                sequence_length= None, # Default: int(TRAINING_SEQUENCE_LENGTH / WINDOW_STEP)
                                sampling_rate= configs["data_preprocess"]["sampling_rate"], 
                                cache=shared_feature_cache)
    
    # External_df
    weaklabeled_df, unlabeled_df = get_external_df(configs, external_data_folder)
    ema_factor = configs["train_params"]["ema_factor"]
    
    weaklabeled_dataset = Weaklabeled_Dataset(data_folder= external_data_folder, 
                                          mid_paths= weaklabeled_df.mid_path,
                                          filename_lists= weaklabeled_df.filename, 
                                          murmur_labels= weaklabeled_df.label, 
                                          sampling_rate= configs["data_preprocess"]["sampling_rate"], 
                                          )
    
    unlabeled_dataset = Unlabeled_Dataset(data_folder= external_data_folder, 
                                          mid_paths= unlabeled_df.mid_path,
                                          filename_lists= unlabeled_df.filename, 
                                          sampling_rate= configs["data_preprocess"]["sampling_rate"], )
    
                                  
    val_dataset = Stronglabeled_Dataset(strong_data_folder, 
                                validation_df.index, 
                                validation_df.murmur_timing, 
                                validation_df.patient_murmur_label,
                                validation_df.outcome_label,
                                sequence_length= None, # Default: int(TRAINING_SEQUENCE_LENGTH / WINDOW_STEP)
                                sampling_rate= configs["data_preprocess"]["sampling_rate"], 
                                cache=shared_feature_cache)


    # strong_bs, weak_bs, unlabeled_bs = 82, 12, 16
    strong_bs = configs["train_params"]["strong_batch_size"]
    weak_bs = configs["train_params"]["weak_batch_size"]
    unlabeled_bs= configs["train_params"]["unlabel_batch_size"]
    
    #w_strong = (strong_bs + weak_bs + unlabeled_bs) / strong_bs
    w_weak = strong_bs / weak_bs
    w_cons = strong_bs / unlabeled_bs
    
    
    strong_nums = len(strong_dataset) - (len(strong_dataset) % strong_bs)
    weak_nums = len(weaklabeled_dataset) - (len(weaklabeled_dataset) % weak_bs)
    # unlabel_nums = len(unlabeled_dataset) - (len(unlabeled_dataset) % unlabeled_bs)

    
    batch_sizes = [strong_bs, weak_bs, unlabeled_bs]
    train_data = [strong_dataset, weaklabeled_dataset, unlabeled_dataset]
    train_dataset = torch.utils.data.ConcatDataset(train_data)
    train_samplers = [torch.utils.data.RandomSampler(x) for x in train_data]
    train_batch_sampler = ConcatDatasetBatchSampler(train_samplers, batch_sizes)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers= 0 , collate_fn= collate_fn)
    
    class_weights_train_frame = None # calculate_class_weights_wo(train_dataset)
    class_weights_train_murmur = None  # torch.tensor([1.0, 5.0]) #torch.tensor([1.0, 10.0])
    class_weights_train_outcome = None # torch.tensor([1.0, 10.0])
    
    
    # Load Training params
    train_max_epochs = configs["train_params"]["max_epoch"] if not configs["debug_true"] else 1
    # train_bs = configs["train_params"]["train_bs"]
    val_batch_size = configs["train_params"]["val_batch_size"]
    train_lr = configs["train_params"]["train_lr"]
    training_patience = configs["train_params"]["train_patience"]
    

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Define neural_network_model
    student_model, teacher_model = get_models() 

    student_model = student_model.to(device)
    teacher_model = teacher_model.to(device)
    
    if class_weights_train_frame is not None:
        class_weights_train_frame = class_weights_train_frame.to(device)
        
    if class_weights_train_murmur is not None:
        class_weights_train_murmur = class_weights_train_murmur.to(device)    
        
    if class_weights_train_outcome is not None:
        class_weights_train_outcome = class_weights_train_outcome.to(device)       
    

    # TODO: ExponentialWarmup, scheduler
    
    # warmup_steps = train_cfg["n_epochs_warmup"] * len(train_cfg["trainloader"])
    # train_cfg["scheduler"] = ExponentialWarmup(train_cfg["optimizer"], configs["opt"]["lr"], warmup_steps)
    rampup_length = configs["train_params"]["rampup_length"]
    exponent = configs["train_params"]["exponent"]
    optim = torch.optim.Adam(student_model.parameters(), lr= train_lr, betas=(0.9, 0.999))
    scheduler = ExponentialWarmup(optim, train_lr, rampup_length, exponent)
    
    
    
    #crit_seq = nn.BCELoss().to(device)
    # crit_murmur = nn.CrossEntropyLoss().to(device)
    # crit_murmur = nn.BCELoss().to(device)
    
    crit_murmur = nn.BCEWithLogitsLoss(pos_weight= torch.tensor(5.0)).to(device)
    crit_consis = nn.MSELoss().to(device)
    #crit_outcome = nn.CrossEntropyLoss(weight= class_weights_train_outcome)
    
    
    
    best_val_loss_stu = np.inf
    best_val_loss_tch = np.inf
    #early_stop_counter = 0    
    
    # Total loss = seq_loss + Murmur_loss + Outcome_loss
    train_losses = []
    val_losses_stu = []
    val_losses_tch = []
    
    # frame_loss_list
    train_seq_losses = []
    val_seq_losses_stu = []
    val_seq_losses_tch = []
    
    # murmur_loss_list
    train_murmur_losses = []
    val_murmur_losses_stu = []
    val_murmur_losses_tch = []

    # cons_seq_loss
    train_cons_seq_losses = []
    
    # cons_murmur_loss_list = []
    train_cons_murmur_losses = []
    
    
    for epoch in range(1, train_max_epochs + 1):
        running_loss = 0.0
        running_seq_loss = 0.0
        running_murmur_loss = 0.0
        running_cons_seq_loss = 0.0
        running_cons_murmur_loss = 0.0
        
        student_model.train()
                                                                    # train_outcome_labels, train_filenames
        for i, (train_mels, train_seq_labels, train_mm_labels, _, _) in enumerate(train_dataloader):
            
            ### get mask to slice strongly/weakly labeled data
            train_bs = train_mels.size(0)
            mask_strong = torch.zeros(train_bs).to(train_mels).bool()
            mask_strong[:strong_bs] = 1                     
            mask_weak = torch.zeros(train_bs).to(train_mels).bool()
            # mask_weak[strong_bs:(strong_bs + weak_bs)] = 1  
            mask_weak[ : (strong_bs + weak_bs)] = 1 
            mask_unlabel = torch.zeros(train_bs).to(train_mels).bool()
            mask_unlabel[(strong_bs + weak_bs) :] = 1
            
            # TODO: apply data augmentations
            do_mixup = torch.randn(1).item()
            do_addnoise = torch.randn(1).item()
            
            
            # Mix-up, default: "soft" mix-up
            
            if do_mixup > 0:
                
                # Strong mixup
                train_mels[mask_strong], train_seq_labels[mask_strong], train_mm_labels[mask_strong] = \
                    mixup(train_mels[mask_strong], train_seq_labels[mask_strong], train_mm_labels[mask_strong])

                # Weak mixup
                train_mels[mask_weak], train_seq_labels[mask_weak], train_mm_labels[mask_weak] = \
                    mixup(train_mels[mask_weak], train_seq_labels[mask_weak], train_mm_labels[mask_weak])

            
            # time masking
            
            
            
            # add Guassian-noise to unlabeled_data
            if do_addnoise > 0:
                train_mels[mask_unlabel] = add_gaussian_noise(train_mels[mask_unlabel], configs["train_params"]["w_gauss_noise"])
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
            seq_preds_stu, murmur_preds_stu = student_model(train_mels)
            seq_preds_stu = F.softmax(seq_preds_stu, dim= 1)
            murmur_preds_stu = F.softmax(murmur_preds_stu, dim= -1)
            
            # Train: Teacher_preds
            with torch.no_grad():
                seq_preds_tch, murmur_preds_tch = teacher_model(train_mels)
                seq_preds_tch = F.softmax(seq_preds_tch, dim= 1)
                murmur_preds_tch = F.softmax(murmur_preds_tch, dim= -1)
            
            # 1. seq_loss
            #noise_mask = train_seq_labels != -1
            seq_loss = F.binary_cross_entropy(seq_preds_stu[mask_strong], train_seq_labels[mask_strong])
            
            # 2. murmur_loss
            #murmur_loss = crit_murmur(F.softmax(murmur_preds_stu[mask_weak], dim= -1), train_mm_labels[mask_weak])
            murmur_loss = crit_murmur(murmur_preds_stu[mask_weak], train_mm_labels[mask_weak])
            
            # 3. consistency_loss
            cons_seq_loss = crit_consis(seq_preds_stu, seq_preds_tch.detach())
            cons_murmur_loss = crit_consis(murmur_preds_stu, murmur_preds_tch.detach())
            #torch.Size([110, 5, 298]) torch.Size([110, 5, 298])
            #torch.Size([110, 2]) torch.Size([110, 2])


            # # total loss
            
            # 가중치...스케줄러로 계속해서 조절하는 듯....!
            # w_cons = configs["train_params"]["w_cons_max"] * scheduler._get_scaling_factor()
            # w_weak = configs["train_params"]["w_weak"]
            # w_weak_cons = configs["train_params"]["w_weak_cons"]
            
            
            # TODO: add weight to loss function
            
            # Total loss  = Classification_loss + Consistency_loss
            total_loss =  seq_loss +  murmur_loss + \
                            (cons_seq_loss + cons_murmur_loss)

            
            # TODO: add scheduler
            total_loss.backward()
            optim.step()
            scheduler.step()
              
            
            # TODO: Update Teacher model params
            teacher_model = update_ema(student_model, teacher_model, scheduler.step_num, ema_factor)

            # Calculate outcome loss
            running_loss += total_loss.item() * train_bs
            running_seq_loss += seq_loss.item() * strong_bs
            running_murmur_loss += murmur_loss.item() * (strong_bs + weak_bs)
            running_cons_seq_loss += cons_seq_loss.item() * train_bs
            running_cons_murmur_loss += cons_murmur_loss.item() * train_bs

            
        # In i'th epoch train_loss
        train_losses.append(running_loss / len(train_dataset))
        train_seq_losses.append(running_seq_loss / strong_nums)    
        train_murmur_losses.append(running_murmur_loss / (strong_nums + weak_nums))
        train_cons_seq_losses.append(running_cons_seq_loss / len(train_dataset))
        train_cons_murmur_losses.append(running_cons_murmur_loss / len(train_dataset))
        
        
     
        with torch.no_grad():
            student_model.eval()
            val_loss_stu = 0.0
            val_loss_tch = 0.0
            val_seq_loss_stu = 0.0
            val_murmur_loss_stu = 0.0
            val_seq_loss_tch = 0.0
            val_murmur_loss_tch = 0.0
                                                                    # val_outcome_labels, val_filenames
            for i, (val_features, val_seq_labels, val_murmur_labels, _, _) in enumerate(val_dataloader):
                
                val_batch_size = val_features.shape[0]
                
                # val_feature
                val_features = val_features.to(device)
            
                # val_labels
                
                val_seq_labels = val_seq_labels.to(device)
                val_murmur_labels = val_murmur_labels.to(device)
                
                # val_preds
                seq_preds_stu, murmur_preds_stu = student_model(val_features)
                seq_preds_tch, murmur_preds_tch = teacher_model(val_features)
                
                seq_preds_stu = F.softmax(seq_preds_stu, dim= 1)
                seq_preds_tch = F.softmax(seq_preds_tch, dim= 1)
                murmur_preds_stu = F.softmax(murmur_preds_stu, dim= -1)
                murmur_preds_tch = F.softmax(murmur_preds_tch, dim= -1)
                
                
                # val_loss            
                #val_seq_mask = val_seq_labels != -1 
                #seq_loss = F.binary_cross_entropy(seq_preds, val_seq_labels, reduction='none')
                #filtered_seq_loss = seq_loss[val_seq_mask].mean()
                #noise_mask = val_seq_labels != -1
                
                seq_loss_stu = F.binary_cross_entropy(seq_preds_stu, val_seq_labels)
                seq_loss_tch = F.binary_cross_entropy(seq_preds_tch, val_seq_labels)
                
                # seq_loss_stu = crit_seq(seq_preds_stu, val_seq_labels)
                # seq_loss_tch = crit_seq(seq_preds_tch, val_seq_labels)
                
                murmur_loss_stu = crit_murmur(murmur_preds_stu, val_murmur_labels)
                murmur_loss_tch = crit_murmur(murmur_preds_tch, val_murmur_labels)
                
                total_loss_stu = seq_loss_stu + murmur_loss_stu 
                total_loss_tch = seq_loss_tch + murmur_loss_tch 
                
                #val_losses += total_loss.item() / bs_features.shape[0]
                val_loss_stu += total_loss_stu.item() * val_batch_size
                val_loss_tch += total_loss_tch.item() * val_batch_size
                
                val_seq_loss_stu += seq_loss_stu.item() * val_batch_size
                val_seq_loss_tch += seq_loss_tch.item() * val_batch_size
                
                val_murmur_loss_stu += murmur_loss_stu.item() * val_batch_size
                val_murmur_loss_tch += murmur_loss_tch.item() * val_batch_size
                
                
            # val_loss = val_losses / len(val_dataset)
            val_losses_stu.append(val_loss_stu /  len(val_dataset))
            val_losses_tch.append(val_loss_tch /  len(val_dataset))

            val_seq_losses_stu.append(val_seq_loss_stu / len(val_dataset))
            val_seq_losses_tch.append(val_seq_loss_tch / len(val_dataset))
            
            val_murmur_losses_stu.append(val_murmur_loss_stu / len(val_dataset))
            val_murmur_losses_tch.append(val_murmur_loss_tch / len(val_dataset))
    
    
        # if verbose >= 1:
        #     print(f"epoch: {epoch:04d}/{train_max_epochs} | Train_loss: {running_loss:.4f} |  Val_loss: {val_loss:.4f}", end= "\n" )

        
        if verbose >= 1:
            print(f"epoch: {epoch:04d}/{train_max_epochs}\n \
                Train_loss: {running_loss / len(train_dataset):.4f}\n \
                Train_seq_loss: {running_seq_loss / strong_nums:.4f}\n \
                Train_murmur_loss: {running_murmur_loss / (strong_nums + weak_nums):.4f}\n \
                Train_seq_cons_loss: {running_cons_seq_loss / len(train_dataset):.4f}\n \
                Train_murmur_cons_loss: {running_cons_murmur_loss / len(train_dataset):.4f}\n \
                Val_loss_stu: {val_loss_stu / len(val_dataset):.4f}\n \
                Val_loss_tch: {val_loss_tch /  len(val_dataset):.4f}\n \
                Val_seq_loss_stu: {val_seq_loss_stu / len(val_dataset):.4f}\n \
                Val_seq_loss_tch: {val_seq_loss_tch / len(val_dataset):.4f}\n \
                Val_murmur_loss_stu: {val_murmur_loss_stu / len(val_dataset):.4f}\n \
                Val_murmur_loss_tch: {val_murmur_loss_tch / len(val_dataset):.4f}\n"
                , end= "\r" )
  
            
        # Save best student
        if val_loss_stu < best_val_loss_stu:
            save_student(model_folder, student_model)
            best_val_loss_stu = val_loss_stu
            #print("Best Student model updated at epoch %d" % epoch)

        # Save best teacher
        if val_loss_tch < best_val_loss_tch:
            save_teacher(model_folder, teacher_model)
            best_val_loss_tch = val_loss_tch
            #print("Best Teacher model updated at epoch %d" % epoch)


    loss_df = pd.DataFrame({"Train_loss": train_losses,
                        "Train_seq_loss": train_seq_losses,
                        "Train_Murmur_loss": train_murmur_losses,
                        "Train_cons_seq_loss": train_cons_seq_losses,
                        "Train_cons_murmur_loss": train_cons_murmur_losses,
                        "val_loss_stu": val_losses_stu,
                        "val_loss_tch": val_losses_tch,
                        "val_seq_loss_stu": val_seq_losses_stu,
                        "val_seq_loss_stu": val_seq_loss_tch,
                        "Val_Murmur_loss_stu": val_murmur_losses_stu,
                        "Val_Murmur_loss_tch": val_murmur_losses_tch,
                        })
    
    loss_df.to_csv(model_folder / 'dataframes' /  "loss_df.csv")
    
    
    print(f"\nFinished training neural network.")
    
    student_model, teacher_model = load_model(model_folder)
    student_model, teacher_model = student_model.to(device), teacher_model.to(device)
    
    return student_model, teacher_model, configs


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
                    
                    ith_murmur_pred_stu = F.softmax(ith_murmur_pred_stu.squeeze(), dim= -1)
                    ith_murmur_pred_tch = F.softmax(ith_murmur_pred_tch.squeeze(), dim= -1)
                    
                    # = F.softmax(ith_murmur_pred_stu.squeeze(), dim= 0)
                    #ith_murmur_pred_tch = F.softmax(ith_murmur_pred_tch.squeeze(), dim= 0)
                    murmur_preds_file_stu.append(ith_murmur_pred_stu[1].item())
                    murmur_preds_file_tch.append(ith_murmur_pred_tch[1].item())

                # murmur_pred = np.max(murmur_preds_file_stu)
                
                murmur_pred_file_stu = np.mean(murmur_preds_file_stu).item()
                murmur_pred_file_tch = np.mean(murmur_preds_file_tch).item()
                
                
            else:
                feature = feature.unsqueeze(dim= 0).to(device)
                
                _, murmur_pred_file_stu = student(feature)
                _, murmur_pred_file_tch = teacher(feature)
                
                # murmur_pred_file_stu = F.softmax(murmur_pred_file_stu.squeeze(), dim= 0)
                # murmur_pred_file_tch = F.softmax(murmur_pred_file_tch.squeeze(), dim= 0)
                murmur_pred_file_stu = F.softmax(murmur_pred_file_stu.squeeze(), dim= -1)[1].item()
                murmur_pred_file_tch = F.softmax(murmur_pred_file_tch.squeeze(), dim= -1)[1].item()

            murmur_preds_ID_stu.append(murmur_pred_file_stu)
            murmur_preds_ID_tch.append(murmur_pred_file_tch)
        
        
        murmur_pred_ID_mean_stu = np.max(murmur_preds_ID_stu).item()
        murmur_pred_ID_mean_tch = np.max(murmur_preds_ID_tch).item()
        
        
        
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
    
        
def collate_fn(x):
    all_features = []
    all_seq_labels = []
    all_murmur_labels = []
    all_outcome_labels = []
    all_filenames = []
    
    max_seq_len = cal_max_frame_len(TRAINING_SEQUENCE_LENGTH)
    freq_bins = calc_frequency_bins()
    
    all_features = torch.zeros(len(x), freq_bins, max_seq_len)
    all_seq_labels = torch.zeros(len(x), 5, max_seq_len)
    
    
    for idx, (feature, seq_label, murmur_label, outcome_label, filename) in enumerate(x): # batch_size train_dataset
        
        current_len = feature.size(1)
        
        
        if current_len < max_seq_len:
            diff = max_seq_len - current_len
            
            #random_num = random.randint(0, diff)
            #start, end = min(diff - random_num, random_num), max(diff - random_num, random_num)
            
            
            padded_feature = F.pad(feature, (diff, 0), 'constant', 0.0)
            seq_label = F.pad(seq_label, (diff, 0), 'constant', 0.0)  
        else:
            diff = current_len - max_seq_len
            
            # 얘는 해주자...
            start = random.randint(0, diff)
            padded_feature = feature[:, start : start + max_seq_len]
            seq_label = seq_label[:, start : start + max_seq_len]

        
        all_features[idx] = padded_feature
        all_murmur_labels.append(murmur_label)
        all_outcome_labels.append(outcome_label)
        all_filenames.append(filename)

    all_features = all_features.float()
    all_murmur_labels = torch.stack(all_murmur_labels).float()
    # all_outcome_labels = torch.tensor(all_outcome_labels)
        
    return all_features, all_seq_labels, all_murmur_labels, all_outcome_labels, all_filenames    


# def batch_whole_dataset_main(dataset):
#     num_examples = len(dataset)
#     return collate_fn_slice_clean([dataset[i] for i in range(num_examples)])


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
def save_student(model_folder: pathlib.Path, student: nn.Module):
    #model_name = type(model).__name__
    
    student_path = model_folder / "best_student.pt"
    torch.save(student.state_dict(), student_path)
    

def save_teacher(model_folder: pathlib.Path, teacher: nn.Module):
    
    teacher_path = model_folder / "best_teacher.pt"
    torch.save(teacher.state_dict(), teacher_path)


    
    
# Load your trained model.
def load_model(model_folder: pathlib.Path):
    student, teacher = get_models()
               
               
    student.load_state_dict(torch.load(pathlib.Path(model_folder) / f"best_student.pt"))
    teacher.load_state_dict(torch.load(pathlib.Path(model_folder) / f"best_teacher.pt"))
    return student.eval(), teacher.eval()



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
    args.model_name = None # If none, model_name : datetime string
    #args.model_name = "MeanTeacher"
    
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
