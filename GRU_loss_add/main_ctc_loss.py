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
TRAINING_BATCH_SIZE = 32 #32
TRAINING_VAL_BATCH_SIZE = 128 # 128
TRAINING_LR = 1e-5
TRAINING_PATIENCE = 10
TRAINING_MAX_EPOCHS = 1000  #  1000
SEQUENCE_LENGTH = 11

# TEST_SLICE= 300


def main(data_folder: pathlib.Path, test_data_folder: pathlib.Path,  model_folder: pathlib.Path, verbose: int, num_k, val_fold_num, device):
    
    patient_df = load_patient_files(data_folder, stop_frac= 1)
    print(f"Training with {len(patient_df)} patients")
    patient_df = create_folds(patient_df, num_k)
    
    patient_df.to_csv(model_folder / "dataframes" / "patient_df.csv")
    recording_df = patient_df_to_recording_df(patient_df)

    recording_df.to_csv(model_folder / "dataframes" /  "recording_df.csv") # for check df

    recording_df_gq = recording_df[recording_df["patient_murmur_label"] != "Unknown"] # 'Holosystolic', 'Early-systolic', 'Mid-systolic',  "nan" is Normal case
    recording_df_bq = recording_df[recording_df["patient_murmur_label"] == "Unknown"] # murmur_timing : nan 
    
    train_recording_df = recording_df_gq[recording_df_gq['val_fold'] != val_fold_num]
    val_recording_df = recording_df_gq[recording_df_gq["val_fold"] == val_fold_num]
    
    # train output is model.eval()
    model = train(data_folder, model_folder, 
                train_recording_df.index,
                train_recording_df.murmur_timing,
                train_recording_df.patient_murmur_label,
                train_recording_df.outcome_label, 
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
    
    val_dataset = Custom_CTCLoss_dataset_slice(data_folder, 
                                val_results_df.index, 
                                val_results_df.murmur_timing, 
                                val_results_df.patient_murmur_label,
                                val_results_df.outcome_label,
                                SEQUENCE_LENGTH,
                                cache={})
    
    val_features, val_seq_labels, val_murmur_labels, _, val_actual_lengths, val_actual_label_lengths = batch_whole_dataset(val_dataset)
    
    for filename, val_feature, val_actual_length in zip(val_results_df.index, val_features, val_actual_lengths):
        
        val_feature = val_feature.unsqueeze(dim=0).to(device) # To extract last_hidden_state output
        actual_length = val_actual_length.clone().reshape(1)
        
        seq_preds, murmur_preds = model(val_feature, actual_length)
        
        murmur_preds =  F.softmax(murmur_preds.squeeze(), dim= 0)
        # outcome_preds =  F.softmax(outcome_preds.squeeze(), dim= 0)
        
        
        val_results[filename]= {"healthy_HSMM": murmur_preds.squeeze()[0].item(),
                                "holo_HSMM": murmur_preds.squeeze()[1].item() }
        
        
        
    val_results_df = pd.DataFrame.from_dict(val_results, orient="index")
    val_results_df.to_csv(model_folder / 'dataframes' /  "val_results_df.csv")
    
    # audio file 하나하나에 대한 prediction 담긴 데이터프레임
    recording_df = recording_df.merge(val_results_df, left_index=True, right_index= True)
    recording_df.to_csv(model_folder / 'dataframes' /  "recording_df.csv")
    
    # 1명당 오디오 파일들에 대한 예측값 평균냄 
    merged_df = merge_df(val_results_df, patient_df)
    merged_df.to_csv(model_folder / 'dataframes' / "merged_df.csv")

    
    
    print("\nDone.\n")
    
    # return 
    
    val_murmur_predictions = {}
    val_outcome_predictions = {}
    
    
    for index, row in merged_df.iterrows():
        murmur_prediction = decide_murmur(row.to_dict())
        val_murmur_predictions[index] = {
            "prediction": murmur_prediction, 
             "probabilities": [], 
            "label": row["murmur_label"], 
            "fold": row.val_fold}

        val_outcome_predictions[index] = {
            "prediction": "Normal" if murmur_prediction == "Absent" else "Abnormal",
            "probabilities": [],
            "label": row["outcome_label"],
            "fold": row.val_fold
        }
        
    optim_murmur_threshold= 0.0
    valid_murmur_score = 0.0
    
    for threshold in np.arange(0, 1.01, 0.01):
        valid_murmur_predictions = {}

        for index, row in merged_df.iterrows():
            murmur_prediction = decide_murmur_with_threshold(row.to_dict(), threshold)
            valid_murmur_predictions[index] = {
                "prediction": murmur_prediction, 
                "probabilities": [], 
                "label": row["murmur_label"]}
            
        murmur_score = compute_cross_val_weighted_murmur_accuracy(valid_murmur_predictions, print= False)
        
        if valid_murmur_score < murmur_score:
            valid_murmur_score = murmur_score.copy()
            optim_murmur_threshold = threshold.copy()
        
    print("\nWITHOUT DECISION TREE\n") 
    print(f"Valid Weighted murmur accuracy = {valid_murmur_score:.3f}")       
    print(f"Optimized murmur threshold: {optim_murmur_threshold:.3f}")
    
        
    # print("WITHOUT DECISION TREE")
    # score = compute_cross_val_weighted_murmur_accuracy(val_murmur_predictions)
    # print(f"Valid Weighted murmur accuracy = {score:.3f}")
    
    outcome_score = compute_cross_val_outcome_score(val_outcome_predictions)
    print(f"\nValid Outcome score = {outcome_score:.0f}\n")
    
    
    test_murmur_score = test(test_data_folder, model_folder, optim_murmur_threshold, model, device)
    
    print(f"\nTest Murmur score: {test_murmur_score:.3f}\n")
    
    
    # print(f"Test Weighted murmur accuracy = {optim_murmur_score:.3f}")
    # print(f"Optimized murmur threshold = {optim_murmur_threshold:.3f}")
    
    #return 


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
    
    train_dataset = Custom_CTCLoss_dataset_slice(data_folder, 
                                train_files, 
                                train_timings, 
                                train_murmur_labels,
                                train_outcome_labels,
                                SEQUENCE_LENGTH,
                                cache=shared_feature_cache)
                                  
    val_dataset = Custom_CTCLoss_dataset_slice(data_folder, 
                                val_files, 
                                val_timings, 
                                val_murmur_labels,
                                val_outcome_labels,
                                SEQUENCE_LENGTH,
                                cache=shared_feature_cache)
    
    class_weights_train_frame = None #calculate_class_weights_wo(train_dataset)
    class_weights_train_murmur = None  # torch.tensor([1.0, 5.0]) #torch.tensor([1.0, 10.0])
    class_weights_train_outcome = None #torch.tensor([1.0, 10.0])
    
    
    #print(f"Train_dataset 크기는? : {len(train_dataset)}")
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=TRAINING_BATCH_SIZE, shuffle=True, collate_fn=collate_fn_ctc
    )
    
    #print(f"Train 로더 크기는?: {len(train_dataloader)}")
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=TRAINING_VAL_BATCH_SIZE, shuffle=False, collate_fn=collate_fn_ctc
    )
    
    #model = GRU_frame_murmur()
    model = GRU_ctc_murmur()
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
    # crit_frame = nn.CrossEntropyLoss(ignore_index=-1, weight=class_weights_train_frame)
    # crit_frame = nn.BCELoss()
    crit_ctc = nn.CTCLoss(blank= 5)
    crit_murmur = nn.CrossEntropyLoss(weight= class_weights_train_murmur)
    #crit_outcome = nn.CrossEntropyLoss(weight= class_weights_train_outcome)
    
    best_val_loss = np.inf
    early_stop_counter = 0    
    
    # Total loss = frame_loss + Murmur_loss + Outcome_loss
    train_loss_list = []
    val_loss_list = []
    
    # frame_loss_list
    train_ctc_loss_list = []
    val_ctc_loss_list = []
    
    # murmur_loss_list
    train_murmur_loss_list = []
    val_murmur_loss_list = []
    
    
    for epoch in tqdm(range(1, TRAINING_MAX_EPOCHS + 1)):
        running_loss = 0.0
        running_ctc_loss = 0.0
        running_murmur_loss = 0.0
        
        model.train()
        
        # Train model                                                   # train_outcome_labels
        for idx, (train_features, train_seq_labels, train_murmur_labels, _, train_actual_lengths, train_actual_label_lengths) in enumerate(train_dataloader):
            
            train_batch_size = train_features.shape[0]
            
            # train_feature
            train_features = train_features.to(device)
            #train_actual_lengths = train_actual_lengths.cpu()
            
            # print(train_features.shape)  # [bs, 40, 300]
            
            # train_labels
            train_seq_labels = train_seq_labels.to(device)
            train_murmur_labels = train_murmur_labels.to(device)
            #train_actual_label_lengths = train_actual_label_lengths.cpu()

            optim.zero_grad()
            
            # train_preds
            seq_preds, murmur_preds= model(train_features, train_actual_lengths)
            #print(seq_preds.shape) # [bs, class, seq]
            
            seq_preds = seq_preds.log_softmax(1).permute(2, 0, 1)
            #seq_preds = seq_preds.softmax(1).permute(2, 0, 1)
            
            
            # train_loss
            ctc_loss = crit_ctc(seq_preds, train_seq_labels, train_actual_lengths, train_actual_label_lengths)
            murmur_loss = crit_murmur(murmur_preds, train_murmur_labels)
            
            total_loss = ctc_loss + murmur_loss   # outcome_loss
            
            total_loss.backward()
            optim.step()
            
            running_loss += total_loss.cpu().item() * train_batch_size
            running_ctc_loss += ctc_loss.cpu().item() * train_batch_size
            running_murmur_loss += murmur_loss.cpu().item() * train_batch_size
            #running_outcome_loss += outcome_loss.cpu().item() * train_batch_size
            

        # append i'th train_loss
        train_loss_list.append(running_loss / len(train_dataset))
        train_ctc_loss_list.append(running_ctc_loss / len(train_dataset))    
        train_murmur_loss_list.append(running_murmur_loss / len(train_dataset))
        #train_outcome_loss_list.append(running_outcome_loss / len(train_dataset))
        
            
        with torch.no_grad():
            model.eval()
            val_losses = 0.0
            val_ctc_loss = 0.0
            val_murmur_loss = 0.0
            #val_outcome_loss = 0.0
            
            # Valid                                             # val_outcome_labels
            for i, (val_features, val_seq_labels, val_murmur_labels, _, val_actual_lengths, val_actual_label_lengths) in enumerate(val_dataloader):
                
                val_batch_size = val_features.shape[0]
                
                # val_feature
                val_features = val_features.to(device)
            
                # val_labels
                val_seq_labels = val_seq_labels.to(device)
                val_murmur_labels = val_murmur_labels.to(device)
                #val_actual_lengths = val_actual_lengths.to(device)
                #val_actual_label_lengths = val_actual_label_lengths.to(device)
                
                # val_preds
                seq_preds, murmur_preds = model(val_features, val_actual_lengths)
                
                seq_preds = seq_preds.log_softmax(1).permute(2, 0, 1)
                
                # val_loss
                ctc_loss = crit_ctc(seq_preds, val_seq_labels, val_actual_lengths, val_actual_label_lengths)
                murmur_loss = crit_murmur(murmur_preds, val_murmur_labels)
                
                total_loss = ctc_loss + murmur_loss
                
                val_losses += total_loss.cpu().item() * val_batch_size
                val_ctc_loss += ctc_loss.cpu().item() * val_batch_size
                val_murmur_loss += murmur_loss.cpu().item() * val_batch_size
                
                
            loss = val_losses / len(val_dataset)
            
            #print(f"After {i+1} epoch val_loss : {loss}", end='\r')
            


        val_loss_list.append(loss)
        val_ctc_loss_list.append(val_ctc_loss / len(val_dataset))
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
                                  'Train_total_Loss': train_loss_list,
                                    'Val_total_Loss': val_loss_list,
                                    'Train_CTC_Loss': train_ctc_loss_list,
                                    'Train_Murmur_Loss': train_murmur_loss_list,
                                    'Val_CTC_Loss': val_ctc_loss_list,
                                    'Val_Murmur_Loss': val_murmur_loss_list,
                                    })
 
    loss_df.to_csv(model_folder / 'dataframes' /  "loss_df_BCE.csv")
    
    #
    model_name = type(model).__name__ 
    
    print(f"\nFinished training {model_name}.")
    
    
    model = load_model(model_folder, model_name)
    model = model.to(device)
    
    return model


def test(test_data_folder: pathlib.Path, model_folder: pathlib.Path, murmur_threshold, model, device):
    patient_df_test = load_patient_files(test_data_folder, stop_frac= 1)

    patient_files = find_patient_files(test_data_folder)
    
    test_results = {}
    
    for patient_file, patient_id in zip(patient_files, patient_df_test.index):
        patient_data = load_patient_data(patient_file) # 환자 1명의 .txt파일 읽어옴
        recordings = load_recordings(test_data_folder, patient_data) # 거기서 1-D .wav file들 읽어옴

        features_list = []
        actual_len_list = []

        for recording in recordings:
            recording = torch.as_tensor(recording)
            
            if len(recording) < SEQUENCE_LENGTH * 4000:
                
                diff = (SEQUENCE_LENGTH * 4000) - len(recording)
                
                recording = F.pad(recording, (0, diff), 'constant', 0.0)
            
            else:
                recording = recording[: (SEQUENCE_LENGTH * 4000) + 1]
            
            
            
            feature, fs_features = calculate_features(recording, FREQUENCY_SAMPLING) # 4000
            
            
            features_list.append(feature)
            actual_len_list.append(feature.shape[-1])
            
            
            
            
            
        features = torch.stack(features_list)
        features = features.to(device)
        actual_len_list = torch.as_tensor(actual_len_list, dtype= torch.int64)
        
        
        # _: frame_preds
        _, murmur_preds = model(features, actual_len_list)    

        murmur_preds = F.softmax(murmur_preds, dim= 1).mean(dim=0)
        
        #patient_id
        test_results[patient_id] = {"holo_HSMM": murmur_preds.squeeze()[1].item()}
        
    test_results_df = pd.DataFrame.from_dict(test_results, orient="index")
    patient_df_test = patient_df_test.merge(test_results_df, left_index= True, right_index= True)
    # loss_df.to_csv(model_folder / 'dataframes' /  "loss_df_BCE.csv")
    patient_df_test.to_csv(model_folder / 'dataframes' / "test_df.csv")

    
    # Now choose murmur threshold!
    optim_murmur_score = 0
    
    test_murmur_predictions = {}
    
    for index, row in patient_df_test.iterrows():
        murmur_prediction = decide_murmur_with_threshold(row.to_dict(), murmur_threshold)
        test_murmur_predictions[index] = {
                "prediction": murmur_prediction, 
                "probabilities": [], 
                "label": row["murmur_label"]}

    test_murmur_score = compute_cross_val_weighted_murmur_accuracy(test_murmur_predictions, print=True)

    
    return test_murmur_score    
    
    #for index, row in patient_df_test.iterrows():
        
    
    

# Load your trained model.            
def save_model(model_folder: pathlib.Path, model: nn.Module): #
    model_name = type(model).__name__
    model_path = model_folder / f"{model_name}.pt"
    torch.save(model.state_dict(), model_path)
    
    #print(f"\nSaved {model_name} parameters!")
    
    
    
# Load your trained model.
def load_model(model_folder: pathlib.Path, model_name: str):
    model = eval(model_name)()
    model.load_state_dict(torch.load(pathlib.Path(model_folder) / f"{model_name}.pt"))
    model.eval()
    return model



if __name__ == '__main__':
    # parser
    parser = argparse.ArgumentParser()  
    parser.add_argument("--train_data", type=str, default= None) 
    parser.add_argument("--test_data", type=str, default= None) 
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
    
    args.train_data = "/Data2/murmur/train" 
    args.test_data = "/Data2/murmur/test" 
    args.model_folder = "/Data1/hmd2/notebooks_th/GRU_loss_add/models"   # /Data1/hmd2/notebooks_th/CUEDA_easy
    args.gpu_index = '0'
    #args.model_name = None # If none, model_name : datetime string
    args.model_name = "GRU_CTC_Loss_11" #"GRU_CTC_Loss"
    
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
    model_folder = pathlib.Path(model_folder)
    
    os.makedirs(model_folder, exist_ok=True)  # 추가!!
    os.makedirs(model_folder / 'dataframes', exist_ok=True)    
    
    
    
    # 시작 시간 기록
    start_time = datetime.datetime.now()
    
    main(data_folder= args.train_data, test_data_folder= args.test_data,  model_folder= model_folder, verbose= args.verbose, num_k= 4, val_fold_num= 0, device= device)
    
    end_time = datetime.datetime.now()
    
    # 수행 시간 계산
    elapsed_time = end_time - start_time
    print(f"작업 수행 시간 (초): {elapsed_time.total_seconds()} 초")
    print("Done.")
    
    
    
    # main(data_folder, model_folder, verbose: int, num_k, val_fold_num, device):
