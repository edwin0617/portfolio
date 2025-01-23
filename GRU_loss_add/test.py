import argparse
import pathlib
from utils import *
from dataset import *
from model import *
from metric import compute_cross_val_weighted_murmur_accuracy, decide_murmur_with_threshold

import torch



def load_model(model_folder: pathlib.Path, model_name: str):
    model = eval(model_name)()
    model.load_state_dict(torch.load(pathlib.Path(model_folder) / f"{model_name}.pt"))
    model.eval()
    return model


    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()  
    parser.add_argument("--test_data_path", type=str, default= None) 
    parser.add_argument("--model_folder", type=str, default= None) 
    parser.add_argument("--model_name", type= str, default=None)
    parser.add_argument("--murmur_threshold", type=float, default= None) 
    parser.add_argument("--gpu_index", type=str, default= None) 
    args = parser.parse_args()
    
    
    
    
############################__Assign args_options__########################################################################################
    
    args.test_data_path = "/Data2/murmur/test" 
    args.model_folder = "/Data1/hmd2/notebooks_th/GRU_loss_add/models/2024-07-18_22:05:34/"
    args.model_name = "GRU_frame_murmur"
    args.murmur_threshold = None
    args.gpu_index= "0"

###########################################################################################################################################

    # str to pathlib.Path
    test_data_folder = pathlib.Path(args.test_data_path)
    model_folder = pathlib.Path(args.model_folder)
    model_name = args.model_name
    murmur_threshold = args.murmur_threshold
    
    
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index
    GPU_NUM = 0 # Since we Selected ONE specific gpu in environment, index number is 0
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu') 
    torch.cuda.set_device(device) # Set gpu, Done!
    
    
    model = load_model(model_folder, model_name)
    model = model.to(device)
    
    
    # Load Test data    
    patient_df_test = load_patient_files(test_data_folder, stop_frac= 1)

    patient_files = find_patient_files(test_data_folder)
    num_patient_files = len(patient_files)
    
    
    print(f"Evaluate {model_name} for {num_patient_files} patients!")
    
    
    test_results = {}
    
    for patient_file, patient_id in zip(patient_files, patient_df_test.index):
        patient_data = load_patient_data(patient_file) # 환자 1명의 .txt파일 읽어옴
        recordings = load_recordings(test_data_folder, patient_data) # 거기서 1-D .wav file들 읽어옴
        
        features_list = []
        
        for recording in recordings:
            recording = torch.as_tensor(recording)
            feature, fs_features = calculate_features(recording, 4000)
            
            current_len = feature.shape[-1]
            
            if current_len < 300:
                diff = 300 - current_len
                feature = F.pad(feature, (0, diff), 'constant', 0.0)
                
            else:
                feature = feature[:, :300]
            
            features_list.append(feature)
            
        
        features = torch.stack(features_list)
        features = features.to(device)
        
        
        _, murmur_preds = model(features)
        
        
        #print(murmur_preds.shape)
        
        
        murmur_preds_mean = F.softmax(murmur_preds, dim= 1).mean(dim=0)
        mm_preed_max_value, indice = F.softmax(murmur_preds, dim= 1).max(dim=0)
        
        #print(murmur_preds_max.shape)
        
        # patient_id: ex) 2530
        
        #test_results[patient_id] = {"holo_HSMM": murmur_preds_mean.squeeze()[1].item()}
        test_results[patient_id] = {"holo_HSMM": mm_preed_max_value[1].item()}
        
        
    test_results_df = pd.DataFrame.from_dict(test_results, orient="index")
    patient_df_test = patient_df_test.merge(test_results_df, left_index= True, right_index= True)
    
    
    
    # murmur thershold가 주어져 있지 않다면 최적의 threshold 구하기
    if not murmur_threshold:
        
        # Choose threshold
        optim_murmur_threshold= 0
        optim_murmur_score = 0
        
        for threshold in np.arange(0, 1.01, 0.01):
            test_murmur_predictions = {}
            
            for index, row in patient_df_test.iterrows():
                murmur_prediction = decide_murmur_with_threshold(row.to_dict(), threshold)
                test_murmur_predictions[index] = {
                        "prediction": murmur_prediction, 
                        "probabilities": [], 
                        "label": row["murmur_label"]}
                    
            murmur_score = compute_cross_val_weighted_murmur_accuracy(test_murmur_predictions, print=False)
            
            if optim_murmur_score < murmur_score:
                optim_murmur_score = murmur_score.copy()
                optim_murmur_threshold = threshold.copy()
                
        print(f"\nBest murmur score: {optim_murmur_score}")
        print(f"Best murmur threshold: {optim_murmur_threshold}\n")
        
        
    # 정해줬다면 그걸로 계산해주기    
    else:
        test_murmur_predictions = {}
        
        for index, row in patient_df_test.iterrows():
            murmur_prediction = decide_murmur_with_threshold(row.to_dict(), murmur_threshold)
            test_murmur_predictions[index] = {"prediction": murmur_prediction, 
                                            "probabilities": [], 
                                            "label": row["murmur_label"]}
            
        murmur_score = compute_cross_val_weighted_murmur_accuracy(test_murmur_predictions, print=False)
        
        print(f"Murmur score: {murmur_score}")
        print(f"Murmur threshold: {murmur_threshold}")