import os
import random
import pathlib
import numpy as np
import pandas as pd
import datetime
import pytz
from omegaconf import OmegaConf

import torch 

from exp_settings import get_config
from dataset import merge_df, load_dataframes, Stronglabeled_Dataset, Unlabeled_Dataset
from train_validate_test import train_validate_model, train_model_full, test
from metric import *


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
def set_one_gpu(gpu_index):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
    GPU_NUM = 0 # Since we Selected ONE specific gpu in environment, index number is 0
    
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu') 
    torch.cuda.set_device(device) # Set gpu

def check_config_path(config): # config["datapath"]
    if not os.path.exists(config.datapath.train_data):
        raise Exception(f"{config.datapath.train_data} does not exist!")
    elif not os.path.exists(config.datapath.test_data):
        raise Exception(f"{config.datapath.test_data} does not exist!")
    elif not os.path.exists(config.datapath.external_data):
        raise Exception(f"{config.datapath.external_data} does not exist!")
    elif not os.path.exists(config.datapath.checkpoint_path):
        raise Exception(f"{config.datapath.checkpoint_path} does not exist!")
    else:
        config.datapath.train_data = pathlib.Path(config.datapath.train_data)
        config.datapath.test_data = pathlib.Path(config.datapath.test_data)
        config.datapath.external_data = pathlib.Path(config.datapath.external_data)
        config.datapath.checkpoint_path = pathlib.Path(config.datapath.checkpoint_path)
        return config
    
def make_checkpoint_dirs(config):
    if config.virtual_settings.debug or config.virtual_settings.tuning:
        
        if not config.datapath.checkpoint_path.name.endswith("Forcheck"):
            checkpoint_path = config.datapath.checkpoint_path / "Forcheck"
        else: # Tuning mode
            checkpoint_path = config.datapath.checkpoint_path
            pass
    else:
        current_time = datetime.datetime.utcnow()
        kst_timezone = pytz.timezone('Asia/Seoul')
        kst_now = current_time.astimezone(kst_timezone)
        filename = kst_now.strftime("%Y-%m-%d_%H:%M:%S")
        checkpoint_path = config.datapath.checkpoint_path / filename
    
    os.makedirs(checkpoint_path, exist_ok=True) # make checkpoint dir
    config.datapath.checkpoint_path = checkpoint_path # load checkpoint path
    
    return config


def main(config, tune_dict=None):
    
    config = check_config_path(config)
    config = make_checkpoint_dirs(config)
    
    set_one_gpu(config.virtual_settings.gpu_index)
    
    df_dict = load_dataframes(config)
    
    set_seed(config.exp_settings.exp_params.random_seed)
    
    # Make dataset----------
    train_dataset = Stronglabeled_Dataset(config.datapath.train_data, 
                                        df_dict["train_recording_df"].index, # recording_filenames
                                        df_dict["train_recording_df"].murmur_timing, # categorical murmur
                                        df_dict["train_recording_df"].patient_murmur_label, # recording murmur label
                                        df_dict["train_recording_df"].outcome_label, # patient outcome label
                                        sampling_rate= config.data_preprocess.sampling_rate,
                                        clean_noise= config.data_preprocess.clean_noise,
                                        )
    unlabeled_dataset = Unlabeled_Dataset(config.datapath.external_data, 
                                          df_dict["unlabeled_df"].mid_path, 
                                          df_dict["unlabeled_df"].filename, 
                                          config.data_preprocess.sampling_rate
                                          )
    val_dataset = Stronglabeled_Dataset(config.datapath.train_data, 
                                        df_dict["val_recording_df"].index, # recording_filenames
                                        df_dict["val_recording_df"].murmur_timing, # categorical murmur
                                        df_dict["val_recording_df"].patient_murmur_label, # recording murmur label
                                        df_dict["val_recording_df"].outcome_label, # patient outcome label
                                        sampling_rate= config.data_preprocess.sampling_rate,
                                        clean_noise= config.data_preprocess.clean_noise,
                                        )

    
    # Train Model-----------
    val_fold_results_dict = train_validate_model(config, df_dict, train_dataset, unlabeled_dataset, val_dataset )
    
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
    
    merged_df_stu = merge_df(recording_df_stu, df_dict["patient_df"])
    merged_df_tch = merge_df(recording_df_tch, df_dict["patient_df"])    
    
    thresholds = np.arange(0, 1.01, 0.01)[::-1]
    
    # calculate Student threshold
    optim_thr_stu= None 
    val_murmur_wma_stu= 0.0     

    for threshold in thresholds:
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
        
    test_murmur_wma_stu = test(config, df_dict["test_patient_df"], df_dict["test_recording_df"], 
                               config.datapath.test_data, student, optim_thr_stu)   
        
    # calculate Teacher threshold
    optim_thr_tch= None
    val_murmur_wma_tch = 0.0
    
    for threshold in thresholds:
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
    
    test_murmur_wma_tch = test(config, df_dict["test_patient_df"], df_dict["test_recording_df"], 
                               config.datapath.test_data, teacher, optim_thr_tch)
    
    print(f"Val_WMA_stu: {val_murmur_wma_stu}")
    print(f"Val_WMA_tch: {val_murmur_wma_tch}\n")    
    
    print(f"Test_WMA_stu: {test_murmur_wma_stu}")
    print(f"Test_WMA_tch: {test_murmur_wma_tch}")
    
    exp_results_dict = {"stop_epoch": val_fold_results_dict["stop_epoch"], 
                        "optim_threshold_stu": optim_thr_stu,
                        "optim_threshold_tch": optim_thr_tch,
                        "Val_WMA_stu": val_murmur_wma_stu,
                        "Val_WMA_tch": val_murmur_wma_tch,
                        "Test_WMA_stu": test_murmur_wma_stu,
                        "Test_WMA_tch": test_murmur_wma_tch,
                        }
    
    config["exp_results"] = exp_results_dict
    
    return config








if __name__ == '__main__':
    
    # # 1. Set config.yaml path 
    # config_dir = "/Data1/hmd2/notebooks_th/Hmd_github/config.yaml"
    # config = OmegaConf.load(config_dir) # Load config settings
    
    config = get_config()
    
    config_result = main(config, tune_dict= None)
    config = OmegaConf.create(config_result)
    
    save_filename = config.datapath.checkpoint_path / "config_result.yaml"
    OmegaConf.save(config, save_filename)
    print(f"Configuration saved to {save_filename}")