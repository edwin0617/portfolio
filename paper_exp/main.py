import os
import numpy as np
import pandas as pd
import torch 
import yaml 
import datetime
import warnings

import config
from dataset import Df_dict, train_dataset, unlabeled_dataset, val_dataset, full_dataset
from train_validate_test import set_seed, merge_df_2, train_validate_model, test
from metric import*

warnings.filterwarnings("ignore", message="stft with return_complex=False is deprecated")


def get_config_results():
    test_results = {"Virtual_machine_settings":
                        {"gpu_index": config.gpu_index, 
                         "num_workers": config.num_workers, 
                         "verbose": config.verbose},
                        
                    "Trainig_Hyperparameters":
                        {"random_seed": config.random_seed, 
                         "train_bs": config.strong_bs, 
                         "val_bs": config.val_bs, 
                         "learning_rate": config.learning_rate, 
                         "max_lr": config.max_lr,
                         "final_lr": config.final_lr,
                         "lambda_sch_decay_rate": config.lambda_sch_decay_rate,
                         "max_epoch": config.max_epoch, 
                         "pct_start":config.pct_start, 
                         "training_patience": config.training_patience,
                         "ema_factor": config.ema_factor, 
                         "const_max": config.const_max, 
                         "use_mix_up": config.use_mix_up,
                         },  
                    "Data_preprocessing_parameters":
                        {"sampling_rate": config.sampling_rate,
                        "window_length": config.window_length,
                        "hop_length": config.hop_length,
                        "freq_bins": config.freq_bins,
                        "max_seq_length": config.sequence_length,
                        "clean_noise": config.clean_noise,
                        "slice_feature": config.slice_feature, 
                        "use_mix_up": config.use_mix_up,
                         }
                    }
    return test_results


def main(seed_number, df_dict, test_data_folder, device, debug_mode, OneCycleLR_tune_dict=None):
    
    set_seed(seed_number)
    
    val_fold_results_dict = train_validate_model(df_dict, train_dataset, unlabeled_dataset, val_dataset, full_dataset,
                                        device, OneCycleLR_tune_dict)
    
    stop_epoch = val_fold_results_dict["stop_epoch"]
    student, teacher = val_fold_results_dict["student"], val_fold_results_dict["teacher"]
    val_fold_results_stu = val_fold_results_dict["student_fold_preds"]
    val_fold_results_tch = val_fold_results_dict["teacher_fold_preds"]
    assert len(val_fold_results_stu) == len(val_fold_results_tch)
    
    # Student
    rec_predictions_df_stu = pd.DataFrame.from_dict(val_fold_results_stu, orient="index")
    recording_df_stu = df_dict["recording_df"].merge(rec_predictions_df_stu, left_index=True, right_index=True)        
    
    # Teacher
    rec_predictions_df_tch = pd.DataFrame.from_dict(val_fold_results_tch, orient="index")
    recording_df_tch = df_dict["recording_df"].merge(rec_predictions_df_tch, left_index=True, right_index=True)
    
    merged_df_stu = merge_df_2(recording_df_stu, df_dict["patient_df"])
    merged_df_tch = merge_df_2(recording_df_tch, df_dict["patient_df"])    
    
    
    # Student
    optim_thr_stu= None 
    val_murmur_wma_stu= 0.0       
    
    for threshold in np.arange(0, 1.01, 0.01):
        val_murmur_preds = {} 
        
        for index, row in merged_df_stu.iterrows():
            murmur_pred = decide_murmur_with_threshold(row.to_dict(), threshold)
            val_murmur_preds[index] = {
            "prediction": murmur_pred, 
            "probabilities": [], 
            "label": row["murmur_label"]}    
        murmur_score = compute_cross_val_weighted_murmur_accuracy(val_murmur_preds, print= False)            
    
        if val_murmur_wma_stu < murmur_score:
            val_murmur_wma_stu = murmur_score.item()
            optim_thr_stu = threshold.item()      
    
    # test_patient_df, test_recording_df, data_folder, model, optim_threshold, device):
    
    test_murmur_wma_stu = test(df_dict["test_patient_df"], df_dict["test_recording_df"], test_data_folder, student, optim_thr_stu, device)    
    
    
    # Teacher
    optim_thr_tch= None
    val_murmur_wma_tch = 0.0
    
    for threshold in np.arange(0, 1.01, 0.01):
        val_murmur_preds = {}  
        
        for index, row in merged_df_tch.iterrows():
            murmur_pred = decide_murmur_with_threshold(row.to_dict(), threshold)
            val_murmur_preds[index] = {
                    "prediction": murmur_pred, 
                    "probabilities": [], 
                    "label": row["murmur_label"]}
            murmur_score = compute_cross_val_weighted_murmur_accuracy(val_murmur_preds, print= False)
    
            if val_murmur_wma_tch < murmur_score:
                val_murmur_wma_tch = murmur_score.item()
                optim_thr_tch = threshold.item()       
    
    test_murmur_wma_tch = test(df_dict["test_patient_df"], df_dict["test_recording_df"], test_data_folder, teacher, optim_thr_stu, device)
    
    if not config.tuning:
        print(f"Test_WMA_stu: {test_murmur_wma_stu}")
        print(f"Test_WMA_tch: {test_murmur_wma_tch}")
    
    test_results = get_config_results()
    exp_results_dict = {"stop_epoch": stop_epoch,
                        "optim_threshold_stu": optim_thr_stu,
                        "optim_threshold_tch": optim_thr_tch,
                        "Val_WMA_stu": val_murmur_wma_stu,
                        "Val_WMA_tch": val_murmur_wma_tch,
                        "Test_WMA_stu": test_murmur_wma_stu,
                        "Test_WMA_tch": test_murmur_wma_tch,
                        }
    test_results["exp_results"] = exp_results_dict    
    
    return test_results


if __name__ == '__main__':
    
    # Select only one Gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_index
    GPU_NUM = 0 # Since we Selected ONE specific gpu in environment, index number is 0
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu') 
    torch.cuda.set_device(device) # Set gpu
    
    start_time = datetime.datetime.now()
    
    seed_number = config.random_seed
    test_data_folder = config.test_data
    debug_mode = config.debug
    model_folder = config.model_folder

    config_result = main(seed_number, Df_dict, test_data_folder, device, debug_mode)
    
    config_result_dir = model_folder / "configs_result.yaml"
    
    with open(config_result_dir, "w") as file:
        yaml.dump(config_result, file, default_flow_style=False, allow_unicode=True)
        
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    print(f"작업 수행 시간 (초): {elapsed_time.total_seconds() // 60}분")
    print("Done.")
    