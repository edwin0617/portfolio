import os
import json
import pathlib
import datetime
import pytz # 추가

import argparse

from typing import Dict, Tuple, List
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import *
from metric import *

from train import *
import decision_tree
import segmenter


def train_challenge_model_full(data_folder, model_folder, verbose, device, num_k= 5):
    model_folder = pathlib.Path(model_folder)
    
    patient_df = load_patient_files(data_folder, stop_frac= 1)
    print(f"Training with {len(patient_df)} patients")
    patient_df = create_folds(patient_df, num_k)

    patient_df.to_csv(model_folder / "patient_df.csv")
    
    recording_df = patient_df_to_recording_df(patient_df)
    recording_df.to_csv(model_folder / "recording_df.csv") # for check df

    
    shared_feature_cache = {}
    val_results = defaultdict(list)
    fold_names = sorted(patient_df["val_fold"].unique())

    location_specific_murmur_label = True
    murmur_label_col = "rec_murmur_label" if location_specific_murmur_label else "patient_murmur_label"

    # We only want to train the murmur segmentation on recordings that have a murmur label
    # (i.e. not those recordings labelled as 'Unknown')
    recording_df_gq = recording_df[recording_df["patient_murmur_label"] != "Unknown"]
    recording_df_bq = recording_df[recording_df["patient_murmur_label"] == "Unknown"] # murmur가 Unknwon이면 murmur_timing도 없음!
    

    assert len(fold_names) == num_k
    
    
    # K-fold 각각에서의 k-val_fold_results
    for fold in fold_names:
        
        if verbose >= 1:
            print(f"****** Fold {fold} of {len(fold_names)} ******")

        train_recording_df = recording_df_gq[recording_df_gq["val_fold"] != fold]
        val_recording_df = recording_df_gq[recording_df_gq["val_fold"] == fold]

    
        # 각 fold에서의 학습완료된 모델.to(device), val_fold_results
        model, val_fold_results = train_and_validate_model(
            model_folder=model_folder,
            fold=fold,
            data_folder=data_folder,
            train_files=train_recording_df.index,
            train_labels=train_recording_df[murmur_label_col],
            train_timings=train_recording_df.murmur_timing,
            val_files=val_recording_df.index,
            val_labels=val_recording_df[murmur_label_col],
            val_timings=val_recording_df.murmur_timing,
            shared_feature_cache=shared_feature_cache,
            verbose=verbose,
            device=device,
        )
        assert len(val_fold_results.keys() & val_results.keys()) == 0
        # val_fold_results, val_results key값들이 겹치면 에러!!

        
        
        # murmur_label 있는 파일들에 대한 Fold 각각의 output저장
        for k, v in val_fold_results.items():
            val_results[k].append(v)
        
        
        # murmur_label "있는" 파일들에 대한 Fold 각각의 output
        unknown_posteriors = predict_files(
                model=model,
                data_folder=data_folder,
                files=recording_df_bq.index,
                labels=recording_df_bq[murmur_label_col],
                timings=recording_df_bq.murmur_timing,
                cache=shared_feature_cache,
                device= device
            )
        
        # murmur_label "없는" 파일들에 대한 Fold 각각의 output저장
        for k, v in unknown_posteriors.items():
            
            val_results[k].append(v)
            
    
        print(f"\nTraining {num_k}-fold is Done!\n")        
            
    
    # ### Part 2 ###
    # Segment recordings using neural network predictions as observation probabilities for
    # two HSMMs    

    num_examples = len(val_results)
    
    
    # print(f"\n 0 fold val_sample_nums: {num_examples}")
    
    
    val_results_df = {} # 추가
    results = {}

    
    for i, (k, posteriors) in enumerate(val_results.items()):
        print(f"\rSegmenting {i + 1:03d} of {num_examples:03d}", end= "\r")
        print(f"\rStacking {len(posteriors)} files", end= "\r")
        
 
        # frame 단위 확률값 확인 실험용
        val_results_df[k] = {"murmur_cpf_max_before_stack": F.softmax(torch.stack(posteriors), dim= 1).mean(dim=0)[-1, :].max().item(), 
                             "murmur_cpf_mean_before_stack": F.softmax(torch.stack(posteriors), dim= 1).mean(dim=0)[-1, :].mean().item(), } 
          
        
        
        posteriors = torch.stack(posteriors).mean(dim=0) # mean predictions on 'Unknown' recs
                                                             # 앙상블, dim=0 차원은 없어짐
            
        #추가
        # val_results_df[k] = {"murmur_conf_per_frame_max" : F.softmax(posteriors, dim= 0)[-1, :].max().item(), 
        #                      "murmur_conf_per_frame_mean" : F.softmax(posteriors, dim= 0)[-1, :].mean().item()
        #                             }    
        
        # val_results_df[k] = {"murmur_conf_per_frame_max" : F.softmax(posteriors, dim= 0)[-1, :].max().item(), 
        #                      "murmur_conf_per_frame_mean" : F.softmax(posteriors, dim= 0)[-1, :].mean().item()
        #                             }
        
        val_results_df[k]['murmur_conf_per_frame_max'] = F.softmax(posteriors, dim= 0)[-1, :].max().item()
        val_results_df[k]['murmur_conf_per_frame_mean'] = F.softmax(posteriors, dim= 0)[-1, :].mean().item()

        
        
        
        posteriors = posteriors.T  # [C, T] to [T, C]
        posteriors = posteriors.numpy()  # TODO: fs not always 50
        _, _, healthy_conf, murmur_conf, murmur_timing = segmenter.double_duration_viterbi(posteriors, 50)
        
        
        results[k] = {"healthy_HSMM": healthy_conf, 
                        "holo_HSMM": murmur_conf,
                        "murmur_timing": murmur_timing}
        
    print("\nSegmenting complete!\n")
    
    val_results_df = pd.DataFrame.from_dict(val_results_df, orient="index")
    
    rec_predictions_df = pd.DataFrame.from_dict(results, orient="index")
    
    #추가
    rec_predictions_df.to_csv(model_folder / "rec_predictions_df.csv")
    val_results_df.to_csv(model_folder / "val_results_df.csv")

    
    recording_df = recording_df.merge(rec_predictions_df, left_index=True, right_index=True)
    recording_df.to_csv(model_folder / "recordings.csv")
    
    
    
    # HSMM 모델 output까지 완성!!
    ### Part 3 ###
    # Train cross-validated gradient boosted decision trees to use segmentation confidences
    # and other patient biometrics to predict final class
    
    df = prepare_tree_df(rec_predictions_df, patient_df)
    for index, row in df.iterrows():
        df.at[index, "num_rec"] = len(row.recordings)
    df["num_rec"] = df["num_rec"].astype(int)
    # TODO: use num rec?
    df.to_csv(model_folder / "tree_inputs.csv")
    
    
    val_murmur_predictions = {}
    val_outcome_predictions = {}

    for index, row in df.iterrows():
        prediction = decide_murmur_outcome(row.to_dict())
        val_murmur_predictions[index] = {
            "prediction": prediction,
            "probabilities": [],
            "label": row["murmur_label"],
            "fold": row.val_fold
        }
        val_outcome_predictions[index] = {
            "prediction": "Normal" if prediction == "Absent" else "Abnormal",
            "probabilities": [],
            "label": row["outcome_label"],
            "fold": row.val_fold
        }
    
    print("WITHOUT DECISION TREE")
    score = compute_cross_val_weighted_murmur_accuracy(val_murmur_predictions)
    print(f"Weighted murmur accuracy = {score:.3f}")


    outcome_score = compute_cross_val_outcome_score(val_outcome_predictions)
    print(f"Outcome score = {outcome_score:.0f}")
    
    val_murmur_predictions = pd.DataFrame.from_dict(val_murmur_predictions, orient="index")
    val_outcome_predictions = pd.DataFrame.from_dict(val_outcome_predictions, orient="index")
    
    val_murmur_predictions.to_csv(model_folder / "val_murmur_predictions.csv")
    val_outcome_predictions.to_csv(model_folder / "val_outcome_predictions.csv")
    
    #print(f"\nAbove result is on train_data set\n")    
    
    
    print("WITH DECISION TREE")
    
    val_murmur_predictions = {}
    for fold in fold_names:
        val_fold_predictions = decision_tree.train_and_validate_model(
            train_df=df.loc[df["val_fold"] != fold],
            val_df=df.loc[df["val_fold"] == fold],
            model_folder=model_folder,
            fold=fold,
            target_name="murmur_label",
            class_weights={"Present": 20, "Unknown": 5, "Absent": 1}
        )
        for k, v in val_fold_predictions.items():
            val_murmur_predictions[k] = {**v, "fold": fold}
    
    score = compute_cross_val_weighted_murmur_accuracy(val_murmur_predictions)
    print(f"Weighted murmur accuracy = {score:.3f}")
    
    
    val_outcome_predictions = {}
    for fold in fold_names:
        val_fold_outcome_predictions = decision_tree.train_and_validate_model(
            train_df=df.loc[df["val_fold"] != fold],
            val_df=df.loc[df["val_fold"] == fold],
            model_folder=model_folder,
            fold=fold,
            target_name="outcome_label",
            class_weights={"Abnormal": 1.8, "Normal": 1}
        )
        for k, v in val_fold_outcome_predictions.items():
            val_outcome_predictions[k] = {**v, "fold": fold}
    
    
    df = pd.DataFrame.from_dict(val_outcome_predictions, orient="index")
    df.to_csv(model_folder / "outcome_predictions.csv")

    outcome_score, threshold = choose_outcome_threshold(df)
    print(f"Outcome score = {outcome_score:.0f} (threshold = {threshold:.03f})")

    settings = {
        "threshold": threshold
    }

    with (model_folder / "settings.json").open("w") as f:
        json.dump(settings, f)

    if verbose >= 1:
        print("Done.")

    return score, outcome_score
    
    

    
if __name__ == '__main__':
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
    args.model_folder = "/Data1/hmd2/notebooks_th/CUEDA_easy/models"   # /Data1/hmd2/notebooks_th/CUEDA_easy
    args.gpu_index = '0'
    args.model_name = None # If none, model_name : datetime string
    
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
    
    os.makedirs(model_folder, exist_ok=True)  # 추가!!
    
    
    # Don't Change This Line, Ever, Forever!
    ## unless you want to use Multi-gpu...
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index  # Select ONE specific index of gpu
    GPU_NUM = 0 # Since we Selected ONE specific gpu in environment, index number is 0
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu') 
    torch.cuda.set_device(device) # Set gpu, Done!
    
    
    score, outcome_score = train_challenge_model_full(data_folder= args.data,
                                model_folder= model_folder,   
                                verbose= args.verbose,
                                device= device,
                                num_k= 4,
        )
    
    
    
    
    
    #def train_challenge_model_full(data_folder, model_folder, verbose, device, num_k= 5):

    
    # murmur_score, outcome_score = train_challenge_model_full(
    #                             data_folder= args.data,
    #                             model_folder= model_folder,   
    #                             verbose= args.verbose,
    #                             gpu= device,
    #                             load_old_file=args.prev_run is not None,
    #                             quick=args.quick)