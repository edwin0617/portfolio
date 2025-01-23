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
    
    split_ratio= 0.01 if configs['debug_true'] else 1    

    patient_df = load_patient_files(data_folder, stop_frac= split_ratio)
    print(f"Training with {len(patient_df)} patients")
    patient_df = create_folds(patient_df, num_k)
    recording_df = patient_df_to_recording_df(patient_df)
    
    shared_feature_cache = {}
    val_results = defaultdict(list)
    fold_names = sorted(patient_df["val_fold"].unique())
    
    recording_df_gq = recording_df[recording_df["patient_murmur_label"] != "Unknown"] # 'Holosystolic', 'Early-systolic', 'Mid-systolic'
    recording_df_bq = recording_df[recording_df["patient_murmur_label"] == "Unknown"]
    
    assert len(fold_names) == num_k
    
    for fold in fold_names:
        
        if verbose >= 1:
            print(f"****** Fold {fold+1} of {len(fold_names)} ******")

        # make i'th train_dataset, val_dataset
        train_recording_df = recording_df_gq[recording_df_gq["val_fold"] != fold]
        val_recording_df = recording_df_gq[recording_df_gq["val_fold"] == fold]
        
        model, val_fold_results = train_and_validate_model(configs, data_folder, model_folder, 
                train_recording_df, val_recording_df, fold, verbose, device)
        
        assert len(val_fold_results.keys() & val_results.keys()) == 0
        
        
        for k, v in val_fold_results.items():
            val_results[k].append(v)
                    
        unknown_posteriors = predict_single_model(configs, data_folder, recording_df_bq, model, device)  
              
        for k, v in unknown_posteriors.items():
            val_results[k].append(v)    
    
    print(f"\nTraining neural network is Done!\n")
    
    num_examples = len(val_results)
    
    results = {}    
    
    for i, (k, posteriors) in enumerate(val_results.items()):
        print(f"\rSegmenting {i + 1:03d} of {num_examples:03d} ", end="")
                
        murmur_pred = np.mean(posteriors).item()
        
        results[k] = {"holo_HSMM": murmur_pred}
        
    
    print("\nSegmenting complete!\n")
    
    # val_results_df = pd.DataFrame.from_dict(val_results_df, orient="index")
    rec_predictions_df = pd.DataFrame.from_dict(results, orient="index")
    
    recording_df = recording_df.merge(rec_predictions_df, left_index=True, right_index=True)
    
    merged_df = merge_df_stu(recording_df, patient_df)
    
    optim_threshold= 0.0
    val_murmur_score = 0.0
    
    for threshold in np.arange(0, 1.01, 0.01):
        val_murmur_preds = {}

        for index, row in merged_df.iterrows():
            murmur_pred = decide_murmur_with_threshold(row.to_dict(), threshold)
            val_murmur_preds[index] = {
                "prediction": murmur_pred, 
                "probabilities": [], 
                "label": row["murmur_label"]}
            
        murmur_score = compute_cross_val_weighted_murmur_accuracy(val_murmur_preds, print= False)
        
        if val_murmur_score < murmur_score:
            val_murmur_score = murmur_score.item()
            optim_threshold = threshold.item()
        
    print(f"Valid Weighted murmur accuracy = {val_murmur_score:.5f}")       
    print(f"Optimized murmur threshold: {optim_threshold:.5f}")
    
    test_murmur_score = test(configs, test_data_folder, model_folder, optim_threshold, fold, split_ratio, model, device)
    
    print(f"Test Weighted_murmur_accuracy= {test_murmur_score:.5f}")
    
    configs["Val_WMA_score"] = val_murmur_score
    configs["Optim_threshold"] = optim_threshold
    configs["Test_WMA_score"] = test_murmur_score
        
    return configs



def test(configs, 
         test_data_folder: pathlib.Path, model_folder: pathlib.Path, 
         optim_threshold: float, fold:int, split_ratio: float,
         model, device):
    
    test_patient_df = load_patient_files(test_data_folder, stop_frac= split_ratio)
    test_recording_df = test_patient_df_to_recording_df(test_patient_df)
    
    infer_dataset = DatasetFor_Inference(test_data_folder, 
                                test_recording_df.index, 
                                test_recording_df.murmur_timing, 
                                test_recording_df.patient_murmur_label,
                                test_recording_df.outcome_label,
                                sequence_length= configs["train_params"]["train_seq_length"], # Default: int(TRAINING_SEQUENCE_LENGTH / WINDOW_STEP)
                                sampling_rate= configs["data_preprocess"]["sampling_rate"],
                                cache={})
    test_dataloader = torch.utils.data.DataLoader(infer_dataset, batch_size= 1, shuffle=False)
    
    test_murmur_preds_per_file= {}  
    model_name = type(model).__name__
    
    # INFO
    print(f"\nInferencing {len(test_recording_df)} test patient files!\n")
    
    
    
    sequence_length= configs["train_params"]["train_seq_length"]
    
    for idx, (mel, _, _, _, filename) in tqdm(enumerate(test_dataloader)):
        
        mel = mel.to(device)
        
        B,W,H = mel.shape
        
        murmur_pred_per_fold= []
        
        for fold_num in range(1, fold+1):
            model = load_model_inmain(model_folder, model_name, fold_num).to(device)

            slice_num = W // sequence_length
            murmur_preds= []
            
            if slice_num:
                for idx in range(slice_num):
                    _, murmur_pred = model(mel[:, :, idx*sequence_length : (idx+1)* sequence_length])
                    murmur_preds.append(murmur_pred)
            else:
                _, murmur_pred = model(mel)
                murmur_preds.append(murmur_pred)
            
            murmur_pred = torch.stack(murmur_preds, dim=0).mean(dim=0)  # [slice_num, 1, 2] >> [1, 2]
            
            murmur_pred_per_fold.append(murmur_pred) 
            
        murmur_pred_ensemble = torch.stack(murmur_pred_per_fold, dim= 0).mean(dim= 0).squeeze(0) # [fold, 1, 2] >> [1,2] >> [2]

        # Tuple, (filename)
        test_murmur_preds_per_file[filename[0].split(".wav")[0]] = {"holo_HSMM": murmur_pred_ensemble[1].item()}
        
        print(f"Inferencing {filename}.wav is Done!, Total completed test file= {idx+1}", end="\r")
        
    test_predictions_df = pd.DataFrame.from_dict(test_murmur_preds_per_file, orient="index")
    test_recording_df = test_recording_df.merge(test_predictions_df, left_index=True, right_index=True)
    
    test_recording_df.to_csv(model_folder / 'dataframes' / "test_recording_df.csv")
    test_patient_df.to_csv(model_folder / 'dataframes' / "test_patient_df.csv")
    
    merged_df = merge_df_stu(test_recording_df, test_patient_df)
    
    test_murmur_preds= {}
    
    for index, row in merged_df.iterrows():
        murmur_pred =  decide_murmur_with_threshold(row.to_dict(), optim_threshold)
        test_murmur_preds[index] = {
                "prediction": murmur_pred, 
                "probabilities": [], 
                "label": row["murmur_label"]}
        
    test_murmur_score = compute_cross_val_weighted_murmur_accuracy(test_murmur_preds, print= False)
    
    return test_murmur_score.item()
            
            

def train_and_validate_model(configs, 
            strong_data_folder: pathlib.Path, model_folder: pathlib.Path,
            stronglabeled_df: pd.DataFrame,   validation_df: pd.DataFrame, 
            fold: int, verbose: int, device):
    
    train_dataset = Sliced_Stronglabeled_Dataset(strong_data_folder, 
                                stronglabeled_df.index, 
                                stronglabeled_df.murmur_timing, 
                                stronglabeled_df.patient_murmur_label,
                                stronglabeled_df.outcome_label,
                                sequence_length= configs["train_params"]["train_seq_length"], # Default: int(TRAINING_SEQUENCE_LENGTH / WINDOW_STEP)
                                sampling_rate= configs["data_preprocess"]["sampling_rate"], 
                                cache={})
    
    val_dataset = Sliced_Stronglabeled_Dataset(strong_data_folder, 
                                validation_df.index, 
                                validation_df.murmur_timing, 
                                validation_df.patient_murmur_label,
                                validation_df.outcome_label,
                                sequence_length= configs["train_params"]["train_seq_length"], # Default: int(TRAINING_SEQUENCE_LENGTH / WINDOW_STEP)
                                sampling_rate= configs["data_preprocess"]["sampling_rate"],
                                cache={})
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size= configs["train_params"]["strong_batch_size"], shuffle=True, collate_fn= collate_fn_for_train
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size= configs["train_params"]["val_batch_size"], shuffle=False, collate_fn= collate_fn_for_train
    )

     # Load Training params
    train_max_epochs = configs["train_params"]["max_epoch"] if not configs["debug_true"] else 1
    training_patience = configs["train_params"]["train_patience"]
    
    # Define neural_network
    model = GRU_frame_murmur()
    model = model.to(device)
    
    # Define Loss function & Optimizer
    optim = torch.optim.Adam(model.parameters(), lr= configs["train_params"]["train_lr"], betas=(0.9, 0.999))
    crit_seq = nn.BCELoss().to(device)
    crit_murmur = nn.BCELoss().to(device)
    
    best_seq_loss, best_murmur_loss = np.inf, np.inf
    early_stop_counter = 0
    
    # Loss log list    
    train_total_loss_list = []
    train_seq_loss_list = []
    train_murmur_loss_list = []
    
    val_total_loss_list = []
    val_seq_loss_list = []
    val_murmur_loss_list = []
    
    # Train neural network
    for epoch in range(1, train_max_epochs + 1):
        running_total_loss = 0.0
        runniing_seq_loss = 0.0
        running_murmur_loss = 0.0

        model.train()
        
        for i, (train_mels, train_seq_labels, train_mm_labels, _, _) in enumerate(train_dataloader):
            
            train_bs = train_mels.shape[0]
            
            train_mels = train_mels.to(device)
            train_seq_labels = train_seq_labels.to(device)
            train_mm_labels = train_mm_labels.to(device)
            
            optim.zero_grad()
            
            train_seq_preds, train_murmur_preds = model(train_mels)
            
            seq_loss = crit_seq(train_seq_preds, train_seq_labels)
            murmur_loss = crit_murmur(train_murmur_preds, train_mm_labels)
            
            total_loss = seq_loss + murmur_loss
            
            total_loss.backward()
            optim.step()
            
            running_total_loss += total_loss.item() * train_bs
            runniing_seq_loss += seq_loss.item() * train_bs
            running_murmur_loss += murmur_loss.item() * train_bs
            
        train_total_loss_list.append(running_total_loss / len(train_dataset))
        train_seq_loss_list.append(runniing_seq_loss / len(train_dataset))
        train_murmur_loss_list.append(running_murmur_loss / len(train_dataset))
        
        
        with torch.no_grad():
            model.eval()
            
            val_total_loss = 0.0
            val_seq_loss = 0.0
            val_murmur_loss = 0.0
            
            for i, (val_mels, val_seq_labels, val_murmur_labels, _, _) in enumerate(val_dataloader):
                
                val_bs = val_mels.shape[0]
                
                val_mels = val_mels.to(device)
                val_seq_labels = val_seq_labels.to(device)
                val_murmur_labels = val_murmur_labels.to(device)
                
                val_seq_preds, val_murmur_preds = model(val_mels)
                
                seq_loss = crit_seq(val_seq_preds, val_seq_labels)
                murmur_loss = crit_murmur(val_murmur_preds, val_murmur_labels)
                
                total_loss = seq_loss + murmur_loss
                
                val_total_loss += total_loss.item() * val_bs
                val_seq_loss += seq_loss.item() * val_bs
                val_murmur_loss += murmur_loss.item() * val_bs

            val_total_loss_all = val_total_loss / len(val_dataset)    
            val_seq_loss_all = val_seq_loss / len(val_dataset)
            val_murmur_loss_all = val_murmur_loss / len(val_dataset)
            
            val_total_loss_list.append(val_total_loss_all)
            val_seq_loss_list.append(val_seq_loss_all)
            val_murmur_loss_list.append(val_murmur_loss_all)    
                
            if verbose >= 1:
                    print(f"epoch: {epoch:04d}/{train_max_epochs},Train_seq_loss: {running_total_loss / len(train_dataset):.10f}, Val_seq_loss: {val_total_loss /  len(val_dataset):.10f}", end= "\n")

            if (val_seq_loss_all < best_seq_loss) & (val_murmur_loss_all < best_murmur_loss):
                
                save_model_inmain(model_folder, model, fold)
                best_seq_loss = val_seq_loss_all
                best_murmur_loss = val_murmur_loss_all
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter > training_patience:
                    break
    
    ith_loss_df = pd.DataFrame({"Train_total_loss": train_total_loss_list,
                            "Train_seq_loss": train_seq_loss_list,
                            "Train_murmur_loss": train_murmur_loss_list,
                            
                            "Val_total_loss": val_total_loss_list,
                            "Val_seq_loss": val_seq_loss_list,
                            "Val_murmur_loss": val_murmur_loss_list, 
                        })
    ith_loss_df.to_csv(model_folder / 'dataframes' /  f"{fold+1}-fold_loss_df.csv")      
    
    print(f"\nFinished training {fold+1}-fold neural network.\n")
    model_name = type(model).__name__
    
    # model.eval()
    model = load_model_inmain(model_folder, model_name, fold).to(device)
    
    fold_posteriors = predict_single_model(configs, strong_data_folder, validation_df, model, device)        
        
    return model, fold_posteriors



def predict_single_model(configs, data_folder, dataframe, model, device):
    
    
    infer_dataset = DatasetFor_Inference(data_folder, 
                                dataframe.index, 
                                dataframe.murmur_timing, 
                                dataframe.patient_murmur_label,
                                dataframe.outcome_label,
                                sequence_length= configs["train_params"]["train_seq_length"], # Default: int(TRAINING_SEQUENCE_LENGTH / WINDOW_STEP)
                                sampling_rate= configs["data_preprocess"]["sampling_rate"],
                                cache={})
    
    inference_dataloader = torch.utils.data.DataLoader(infer_dataset, batch_size= 1, shuffle=False)
    model = model.to(device)
    results = []
    
    sequence_length = configs["train_params"]["train_seq_length"]
    
    with torch.no_grad():
        for idx, (mels, _, murmur_label, _, _) in enumerate(inference_dataloader):
            mels, murmur_label=  map(lambda x: x.to(device), [mels, murmur_label])
            
            B, W, H = mels.shape
            murmur_preds= []
            slice_num = W // sequence_length

            if slice_num:
                for idx in range(slice_num):
                   _, murmur_pred = model(mels[:, :, idx*sequence_length : (idx+1)*sequence_length ])
                   murmur_preds.append(murmur_pred)
            else:
                _, murmur_pred = model(mels)
                murmur_preds.append(murmur_pred)
            murmur_preds = torch.stack(murmur_preds).mean(dim= 0).cpu().squeeze(0)            
            results.append(murmur_preds[1].item())
    return {n: p for n, p in zip(dataframe.index, results)}






# Load your trained model.            
def save_model_inmain(model_folder: pathlib.Path, model: nn.Module, fold: int):
    model_name = type(model).__name__
    model_path = model_folder / f"Fold_{fold+1}_{model_name}.pt"
    torch.save(model.state_dict(), model_path)

# Load your trained model.
def load_model_inmain(model_folder: pathlib.Path, model_name: str, fold: int):
    #model = GRU_not_sorted()
    model = eval(model_name)()
    model.load_state_dict(torch.load(pathlib.Path(model_folder) / f"Fold_{fold+1}_{model_name}.pt"))
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
    
######################################################################################################################################################    
    
    with open(args.default_config, "r") as f:
        configs = yaml.safe_load(f)
    
    if configs["debug_true"]:
        filename= "Forcheck"
    else:
        current_time = datetime.datetime.utcnow() # 현재 UTC 시간을 가져옴
        kst_timezone = pytz.timezone('Asia/Seoul') # 대한민국 시간대 설정
        kst_now = current_time.astimezone(kst_timezone) # UTC 시간을 대한민국 시간으로 변환
        filename = kst_now.strftime("%Y-%m-%d_%H:%M:%S")

    
    
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