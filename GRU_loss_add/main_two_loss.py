import os
import json
import pathlib
import datetime
import pytz # 추가

import argparse

from typing import Dict, Tuple, List
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import *
from model import *
from metric import *

# Global variables for Train models
TRAINING_BATCH_SIZE = 32
TRAINING_VAL_BATCH_SIZE = 128
TRAINING_LR = 1e-5
TRAINING_PATIENCE = 10
TRAINING_MAX_EPOCHS = 1000  #  1000



def main(data_folder: pathlib.Path, model_folder: pathlib.Path, verbose: int, num_k, val_fold_num, device):
    
    patient_df = load_patient_files(data_folder, stop_frac= 1)
    print(f"Training with {len(patient_df)} patients")
    patient_df = create_folds(patient_df, num_k)
    
    #os.makedirs(model_folder / "dataframes", exist_ok=True) 얘를 if __main__: 쪽으로 빼자
    
    patient_df.to_csv(model_folder / "dataframes" / "patient_df.csv")
    recording_df = patient_df_to_recording_df(patient_df)

    recording_df.to_csv(model_folder / "dataframes" /  "recording_df.csv") # for check df
    
    
    #return 
    
    # location_specific_murmur_label = True
    # murmur_label_col = "rec_murmur_label" if location_specific_murmur_label else "patient_murmur_label"
    
    recording_df_gq = recording_df[recording_df["patient_murmur_label"] != "Unknown"] # 'Holosystolic', 'Early-systolic', 'Mid-systolic'
    recording_df_bq = recording_df[recording_df["patient_murmur_label"] == "Unknown"] # murmur_timing : nan
    

    train_recording_df = recording_df_gq[recording_df_gq['val_fold'] != val_fold_num]
    val_recording_df = recording_df_gq[recording_df_gq["val_fold"] == val_fold_num]
    
    
    # model.eval()됨
    model = train(data_folder, model_folder, 
                train_recording_df.index,
                train_recording_df.murmur_timing,
                train_recording_df.patient_murmur_label,
                train_recording_df.outcome_label,  # train
                val_recording_df.index,
                val_recording_df.murmur_timing,
                val_recording_df.patient_murmur_label,
                val_recording_df.outcome_label,
                verbose,
                device)

    # 1. murmur != unknown인 애들에 대해 validation dataset에 대해서 prediction하는 코드 작성
    # 2. murmur == present, absent인 애들에 대해 validation dataset에 대해서 prediction하는 코드 작성
    ## 그냥 murmur여부가 unknown이든 아니든 한꺼번에 val_dataset으로 나눈 후 확률값 뽑아내면 됨
    
    val_results= {}
    
    val_results_df = recording_df[recording_df['val_fold'] == val_fold_num]
    
    val_dataset = Custome_T4_dataset(data_folder, 
                                val_results_df.index, 
                                val_results_df.murmur_timing, 
                                val_results_df.patient_murmur_label,
                                val_results_df.outcome_label,
                                sequence_length= None, # Default: int(TRAINING_SEQUENCE_LENGTH / WINDOW_STEP)
                                cache={})
    
    val_features, _, _, _,_ = batch_whole_dataset(val_dataset)
    
    for filename, val_feature in zip(val_results_df.index, val_features):
        
        val_feature = val_feature.unsqueeze(dim=0)
        val_feature = val_feature.to(device)
        
        # shape_1, shape_2 = val_feature.shape[0], val_feature.shape[1]
        # val_feature = val_feature.view(1, shape_1, shape_2)
        
        _, murmur_preds, outcome_preds = model(val_feature)
        
        murmur_preds =  F.softmax(murmur_preds.squeeze(), dim= 0).cpu()
        outcome_preds =  F.softmax(outcome_preds.squeeze(), dim= 0).cpu()
        
        #print(f"murmur_preds.shape is {murmur_preds.shape}")
        #print(f"outcome_preds.shape is {outcome_preds.shape}")
        # murmur_preds.squeeze()
        
        val_results[filename]= {"healthy_HSMM": murmur_preds.squeeze()[0].item(),
                                "holo_HSMM": murmur_preds.squeeze()[1].item(), 
                                 }
        
    val_results_df = pd.DataFrame.from_dict(val_results, orient="index")
    val_results_df.to_csv(model_folder / 'dataframes' /  "val_results_df.csv")
    
    # audio file 하나하나에 대한 prediction 담긴 데이터프레임
    recording_df = recording_df.merge(val_results_df, left_index=True, right_index= True)
    recording_df.to_csv(model_folder / 'dataframes' /  "recording_df.csv")
    
    #df = prepare_tree_df(val_results_df, patient_df)
    merged_df = merge_df(val_results_df, patient_df)
    merged_df.to_csv(model_folder / 'dataframes' / "merged_df2.csv")

    
    val_murmur_predictions = {}
    val_outcome_predictions = {}
    
    
    for index, row in merged_df.iterrows():
        murmur_prediction = decide_murmur(row.to_dict())
        val_murmur_predictions[index] = {
            "prediction": murmur_prediction, 
             "probabilities": [], 
            "label": row["murmur_label"], 
            "fold": row.val_fold}
    
    
    print("WITHOUT DECISION TREE")
    score = compute_cross_val_weighted_murmur_accuracy(val_murmur_predictions)
    print(f"Weighted murmur accuracy = {score:.3f}")
    
    
    
    return 




def train(data_folder: pathlib.Path,
            model_folder: pathlib.Path,
            train_files: List[str],
            train_timings: List[str],
            train_murmur_labels: List[str],
            train_outcome_labels : List[str],
            val_files: List[str],
            val_timings: List[str],
            val_murmur_labels: List[str],
            val_outcome_labels: List[str],
            verbose: int,
            device):
    
    shared_feature_cache = {}
    
    train_dataset = Custome_T4_dataset(data_folder, 
                                train_files, 
                                train_timings, 
                                train_murmur_labels,
                                train_outcome_labels,
                                sequence_length= None, # Default: int(TRAINING_SEQUENCE_LENGTH / WINDOW_STEP)
                                cache=shared_feature_cache)
                                  
    val_dataset = Custome_T4_dataset(data_folder, 
                                val_files, 
                                val_timings, 
                                val_murmur_labels,
                                val_outcome_labels,
                                sequence_length= None, # Default: int(TRAINING_SEQUENCE_LENGTH / WINDOW_STEP)
                                cache=shared_feature_cache)
    
    class_weights_train_frame = None #calculate_class_weights_wo(train_dataset)
    class_weights_train_murmur = None  # torch.tensor([1.0, 5.0]) #torch.tensor([1.0, 10.0])
    class_weights_train_outcome = None #torch.tensor([1.0, 10.0])
    
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=TRAINING_BATCH_SIZE, shuffle=True, collate_fn=collate_fn_slice
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=TRAINING_VAL_BATCH_SIZE, shuffle=False, collate_fn=collate_fn_slice
    )
    
    model = GRU_multi_output()
    model = model.to(device)
    
    if class_weights_train_frame is not None:
        class_weights_train_frame = class_weights_train_frame.to(device)
        
    if class_weights_train_murmur is not None:
        class_weights_train_murmur = class_weights_train_murmur.to(device)    
        
    if class_weights_train_outcome is not None:
        class_weights_train_outcome = class_weights_train_outcome.to(device)       
    
        
        
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of Training model {num_parameters} params")
    
    optim = torch.optim.Adam(model.parameters(), lr=TRAINING_LR)
    #crit_frame = nn.CrossEntropyLoss(ignore_index=-1, weight=class_weights_train_frame)
    crit_frame = nn.BCELoss()
    crit_murmur = nn.CrossEntropyLoss(weight= class_weights_train_murmur)
    #crit_outcome = nn.CrossEntropyLoss(weight= class_weights_train_outcome)
    
    best_val_loss = np.inf
    early_stop_counter = 0    
    
    # Total loss = frame_loss + Murmur_loss + Outcome_loss
    train_loss_list = []
    val_loss_list = []
    
    # frame_loss_list
    train_frame_loss_list = []
    val_frame_loss_list = []
    
    # murmur_loss_list
    train_murmur_loss_list = []
    val_murmur_loss_list = []
    
    # outcome_loss_list
    #train_outcome_loss_list = []
    #val_outcome_loss_list = []
    
    
    for epoch in tqdm(range(1, TRAINING_MAX_EPOCHS + 1)):
        running_loss = 0.0
        running_frame_loss = 0.0
        running_murmur_loss = 0.0
        running_outcome_loss = 0.0
        
        model.train()
            
        for i, (bs_features, bs_frame_labels, bs_murmur_labels, bs_outcome_labels, bs_filename) in enumerate(train_dataloader):
            
            train_batch_size = bs_features.shape[0]
            
            # train_feature
            bs_features = bs_features.to(device)
            
            # train_labels
            bs_frame_labels = bs_frame_labels.to(device)
            bs_murmur_labels = bs_murmur_labels.to(device)
            #bs_outcome_labels = bs_outcome_labels.to(device)
            
            
            #print(bs_features.shape, bs_frame_labels.shape, )
            
            
            optim.zero_grad()
            
            # train_preds
            frame_preds, murmur_preds, outcome_preds = model(bs_features)
            
            
            #print(frame_preds.shape)
            
            
            # train_loss
            frame_loss = crit_frame(frame_preds, bs_frame_labels)
            murmur_loss = crit_murmur(murmur_preds, bs_murmur_labels)
            #outcome_loss = crit_outcome(outcome_preds, bs_outcome_labels)
            
            total_loss = frame_loss + murmur_loss #+ outcome_loss
            
            total_loss.backward()
            optim.step()
            
            running_loss += total_loss.cpu().item() * train_batch_size
            running_frame_loss += frame_loss.cpu().item() * train_batch_size
            running_murmur_loss += murmur_loss.cpu().item() * train_batch_size
            #running_outcome_loss += outcome_loss.cpu().item() * train_batch_size
            

        # append i'th train_loss
        train_loss_list.append(running_loss / len(train_dataset))
        train_frame_loss_list.append(running_frame_loss / len(train_dataset))    
        train_murmur_loss_list.append(running_murmur_loss / len(train_dataset))
        #train_outcome_loss_list.append(running_outcome_loss / len(train_dataset))
        
            
        with torch.no_grad():
            model.eval()
            val_losses = 0.0
            val_frame_loss = 0.0
            val_murmur_loss = 0.0
            #val_outcome_loss = 0.0
            
            for i, (bs_features, bs_frame_labels, bs_murmur_labels, bs_outcome_labels, bs_filename) in enumerate(val_dataloader):
                
                val_batch_size = bs_features.shape[0]
                
                # val_feature
                bs_features = bs_features.to(device)
            
                # val_labels
                bs_frame_labels = bs_frame_labels.to(device)
                bs_murmur_labels = bs_murmur_labels.to(device)
                #bs_outcome_labels = bs_outcome_labels.to(device)
                
                # val_preds
                frame_preds, murmur_preds, outcome_preds = model(bs_features)
                
                # val_loss
                frame_loss = crit_frame(frame_preds, bs_frame_labels)
                murmur_loss = crit_murmur(murmur_preds, bs_murmur_labels)
                #outcome_loss = crit_outcome(outcome_preds, bs_outcome_labels)
                
                total_loss = frame_loss + murmur_loss #+ outcome_loss
                
                #val_losses += total_loss.item() / bs_features.shape[0]
                val_losses += total_loss.cpu().item() * val_batch_size
                val_frame_loss += frame_loss.cpu().item() * val_batch_size
                val_murmur_loss += murmur_loss.cpu().item() * val_batch_size
                #val_outcome_loss += outcome_loss.cpu().item() * val_batch_size
                
                
            loss = val_losses / len(val_dataset)


        val_loss_list.append(loss)
        val_frame_loss_list.append(val_frame_loss / len(val_dataset))
        val_murmur_loss_list.append(val_murmur_loss / len(val_dataset))
        #val_outcome_loss_list.append(val_outcome_loss / len(val_dataset))
        
  
            
        if verbose >= 2:
            print(
                f"{epoch:04d}/{TRAINING_MAX_EPOCHS} | {running_loss / i:.3f} | {loss:.3f} | {early_stop_counter:02d}"
            )
        elif verbose >= 1:
            print(
                f"\r{epoch:04d}/{TRAINING_MAX_EPOCHS} | {running_loss / i:.3f} | {loss:.3f} | {early_stop_counter:02d} ",
                end="",
            )            
  
            
        # Early-stopping    
        if loss < best_val_loss:
            # Save best model and reset patience
            save_model(model_folder, model)
            best_val_loss = loss
            early_stop_counter = 0
            
            # print(f"At {epoch}'th model parameters changed!")
            
        else:
            early_stop_counter += 1
            if early_stop_counter > TRAINING_PATIENCE:
                break
    

    
    loss_df = pd.DataFrame({'Epoch': range(1, epoch + 1),
                                  'Train Loss': train_loss_list,
                                    'Val Loss': val_loss_list,
                                    'Train Frame Loss': train_frame_loss_list,
                                    'Train Murmur Loss': train_murmur_loss_list,

                                    'Val Frame Loss': val_frame_loss_list,
                                    'Val Murmur Loss': val_murmur_loss_list,

                                    })
    
    
    
    loss_df.to_csv(model_folder / 'dataframes' /  "loss_df_BCE.csv")
    
    
    
    # GRU_multi_output    
    model_name = type(model).__name__ 
    
    print(f"\nFinished training neural_networks.")
    
    model = load_model(model_folder, model_name)
    model = model.to(device)
    
    return model





# Load your trained model.            
def save_model(model_folder: pathlib.Path, model: nn.Module): #
    model_name = type(model).__name__
    model_path = model_folder / f"{model_name}.pt"
    torch.save(model.state_dict(), model_path)
    
    
    
# Load your trained model.
def load_model(model_folder: pathlib.Path, model_name: str):
    model = eval(model_name)()
    model.load_state_dict(torch.load(pathlib.Path(model_folder) / f"{model_name}.pt"))
    model.eval()
    return model



if __name__ == '__main__':
    # parser
    parser = argparse.ArgumentParser()  
    parser.add_argument("--data", type=str, default= None) 
    parser.add_argument("--model_folder", type=str, default= None) 
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--gpu_index", type=str, default= None) 
    parser.add_argument("--prev_run", type=str, default= None)
    parser.add_argument("--verbose", type=int, default= 1)
    args = parser.parse_args()
    
############################__Please assign PATH, GPU_index__########################################################################################
    
    #args.data = "your Train data path" : str
    #args.model_folder = "your model params path" : str
    #args.gpu_index = 'SELECT index of gpu' : str
    #args.model_name= None
    
    args.data = "/Data2/murmur/train" 
    args.model_folder = "/Data1/hmd2/notebooks_th/GRU_loss_add/models"   # /Data1/hmd2/notebooks_th/CUEDA_easy
    args.gpu_index = '0'
    #args.model_name = None # If none, model_name : datetime string
    args.model_name = 'Forcheck'
    
######################################################################################################################################################    
        
    if args.model_name is None:
        # current_time = datetime.datetime.now()
        # filename = current_time.strftime("%Y-%m-%d__%H-%M-%S") 
        
        current_time = datetime.datetime.utcnow() # 현재 UTC 시간을 가져옴
        kst_timezone = pytz.timezone('Asia/Seoul') # 대한민국 시간대 설정
        kst_now = current_time.astimezone(kst_timezone) # UTC 시간을 대한민국 시간으로 변환
        filename = kst_now.strftime("%Y-%m-%d_%H:%M:%S")
    else:
        filename = args.model_name
    
    
    if not os.path.exists(args.data):
        raise Exception(f"The path {args.data} does not exist!")
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
    
    args.data = pathlib.Path(args.data)
    model_folder = pathlib.Path(model_folder)
    
    os.makedirs(model_folder, exist_ok=True)  # 추가!!
    os.makedirs(model_folder / 'dataframes', exist_ok=True)    
    
    
    
    # 시작 시간 기록
    start_time = datetime.datetime.now()
    
    
    main(data_folder= args.data, model_folder= model_folder, verbose= args.verbose, num_k= 4, val_fold_num= 0, device= device)
    
    end_time = datetime.datetime.now()
    
    # 수행 시간 계산
    elapsed_time = end_time - start_time
    print(f"작업 수행 시간 (초): {elapsed_time.total_seconds()} 초")
    print("Done.")
    
    
    
    # main(data_folder, model_folder, verbose: int, num_k, val_fold_num, device):
