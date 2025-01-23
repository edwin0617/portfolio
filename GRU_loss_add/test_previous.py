import os
import pathlib
import argparse
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, Tuple, List

import torch
import torch.nn.functional as F

from utils import*
from dataset import*
from model import GRU_multi_output, load_GRU_multi_output_params
from metric import*

USE_MURMUR_DECISION_TREE= False



def inference_model(model_folder: pathlib.Path, data_folder: pathlib.Path, output_folder: pathlib.Path,
                    allow_failures: bool, verbose: int, device): # cuda device
    
    # Load model 
    if verbose >= 1:
        print('Loading Challenge model...')
        
        
        
    # @@@@@@@@  파라미터 할당해야지??
         
    model = load_GRU_multi_output_params(model_folder)
    
    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)
    
    if num_patient_files==0:
        raise Exception('No data was provided.')

    # Create a folder for the Challenge outputs if it does not already exist.
    os.makedirs(output_folder, exist_ok=True)

    if verbose >= 1:
        print('Running model on Challenge data...')

    # Iterate over the patient files.
    for i in tqdm(range(num_patient_files)):
        # if verbose >= 2:
        #     print('    {}/{}...'.format(i+1, num_patient_files))

        patient_data = load_patient_data(patient_files[i])
        recordings = load_recordings(data_folder, patient_data) # wav파일로 변환. recordings: 부위별 1개~4개
        
        #print(len(recordings))
        
        
        
        #여기서부터 다시 짜야 한다......
        
        classes, labels, probabilities = run_challenge_model(model, patient_data, recordings, verbose, device)
        
        
        # try:
        #     classes, labels, probabilities = run_challenge_model(model, patient_data, recordings, verbose, device) ### Teams: Implement this function!!!
        # except:
        #     if allow_failures:
        #         if verbose >= 2:
        #             print('... failed.')
        #         classes, labels, probabilities = list(), list(), list()
        #     else:
        #         raise
        
        
        # Save Challenge outputs.
        head, tail = os.path.split(patient_files[i])
        root, extension = os.path.splitext(tail)
        output_file = os.path.join(output_folder, root + '.csv')
        patient_id = get_patient_id(patient_data)
        save_challenge_outputs(output_file, patient_id, classes, labels, probabilities)
        
        
        
    if verbose >= 1:
        print("Done.") 



# data: 환자별 .txt파일, recordings: 환자별 오디오_파일(1 ~ 4개)
def run_challenge_model(model, data, recordings, verbose, device):
        
    num_locations = get_num_locations(data)  # 환자별 오디오 파일 개수 확인(AV, MV, TV, ...)
    
     # print(num_locations)  4, 2, 3, 1, ..., ..
    
    recording_information = data.split("\n")[1 : num_locations + 1]
    recording_name = [r.split(" ")[0] for r in recording_information]
    
    # print(recording_name)  ['AV', 'PV', 'TV', 'MV']

    fs = get_frequency(data)

    # for model_fold in model:
    #     model_fold["network"].to(device)
    model = model.to(device)
    model.eval()


    location_predictions = defaultdict(list)
    location_signal_quals = defaultdict(list)
    
    
    for recording, name in zip(recordings, recording_name):
        
        #print(len(name))  #AV, MV, PV, ... 중 하나임
        
        # Load features.
        recording = torch.as_tensor(recording.copy())
        features, fs_features = calculate_features(recording, fs) #[40, 723], 50 == 1 / WINDOW_STEP
        features = features.unsqueeze(0)
        # lengths = torch.as_tensor([features.shape[-1]])

        # fold_predictions = []
        
        if features.shape[-1] < 300:
            diff = 300 - features.shape[-1]
            features = F.pad(features, (0, diff), 'constant', 0.0)
        else:
            features = features[..., :300]
        
        features = features.to(device)
               
        frame_preds, murmur_preds, outcome_preds = model(features)
        
        murmur_preds =  F.softmax(murmur_preds.squeeze(), dim= 0)[1].item()
        # outcome_preds =  F.softmax(outcome_preds.squeeze(), dim= 0)
 
        # posteriors = torch.mean(torch.stack(fold_predictions, dim=0), dim=0)
        # posteriors = posteriors.T  # [C, T] to [T, C]
        # posteriors = posteriors.numpy()
        # _, _, healthy_conf, murmur_conf, murmur_timing = segmenter.double_duration_viterbi(posteriors, model[0]["network"].output_fs)
        
        location_predictions[name].append(murmur_preds)   
        # location_signal_quals[name].append(max(murmur_conf, healthy_conf))


    print(len(location_predictions[name]))
        
        

    # #  TODO: perhaps do not mean. ㅇㅇ 나같은 경우에는 할 필요 없음...!!
    # full_features = {f"conf_difference_{k}": np.mean(v) for k, v in location_predictions.items()}
    # for k, v in location_signal_quals.items():
    #     full_features[f"signal_qual_{k}"] = np.mean(v)
    
    # print(len(location_predictions)) 1 ~ 4개
    
    full_features = {f"holo_HSMM_{k}" : np.mean(v) for k, v in location_predictions.items()}  # holo_HSMM__AV: 0.78, holo_HSMM__MV: 0.89, ..., ...

    for k, v in functions.items():
        full_features[k] = v(data)
    full_features["num_rec"] = len(recordings)
    
    
    prediction = decide_murmur_outcome(full_features) # "Present", "Unknown" or "Absent"
    murmur_probabilities = np.zeros(3)
    prediction_to_index = {"Present": 0, "Unknown": 1, "Absent": 2}
    murmur_probabilities[prediction_to_index[prediction]] = 1   # Present: index 0번째 1 할당, Unknown: index 1번째 1 할당, Absent: index 2번째 1 할당
    murmur_labels = np.zeros(3, dtype=np.int_)
    murmur_labels[prediction_to_index[prediction]] = 1
    
    
    class_order = ["Present", "Unknown", "Absent"] # + outcome_class_order
    labels = list(murmur_labels) # +  list(outcome_labels)
    probabilities = list(murmur_probabilities) # + list(outcome_probabilities)
    
    
    return class_order, labels, probabilities

    
    
    
# # functions = {
# #     "age": get_age,
# #     "pregnant": get_pregnancy_status,
# #     "height": get_height,
# #     "weight": get_weight,
# #     "sex": get_sex,
# # }    
    
#     # functions란 무엇이냐? .txt파일 하나로부터 정보 가져오는 함수가 모여있는 딕셔너리임
#     # ex) {AV : {"age": 28, ... "pregnant", "height", "weight", ...}, ....}
        
        
#     for k, v in functions.items():
#         full_features[k] = v(data)  # "age": get_age(data)
#     full_features["num_rec"] = len(recordings)


#     # print(full_features.keys())    




    
#     prediction = decide_murmur_outcome(full_features)
    
    
#     murmur_probabilities = np.zeros(3)
#     prediction_to_index = {"Present": 0, "Unknown": 1, "Absent": 2}
#     murmur_probabilities[prediction_to_index[prediction]] = 1
#     murmur_labels = np.zeros(3, dtype=np.int_)
#     murmur_labels[prediction_to_index[prediction]] = 1

#     ordered_array = [full_features.get(k, None) for k in model[0]["tree_outcome"].feature_names_]
#     probabilities = []
#     for model_fold in model:
#         probabilities.append(
#             model_fold["tree_outcome"].predict(ordered_array, prediction_type="Probability")
#         )
#     outcome_probabilities = np.mean(probabilities, axis=0)

#     # Choose label with higher probability.
#     outcome_labels = np.zeros(len(outcome_probabilities), dtype=np.int_)
#     outcome_class_order = list(model_fold["tree_outcome"].classes_)
#     abnormal_idx = outcome_class_order.index("Abnormal")
#     idx = abnormal_idx if outcome_probabilities[abnormal_idx] > model_fold["outcome_threshold"] else 1 - abnormal_idx
#     outcome_labels[idx] = 1

#     if USE_MURMUR_DECISION_TREE:
#         class_order = list(model_fold["tree_murmur"].classes_) + outcome_class_order
#     else:
#         class_order = ["Present", "Unknown", "Absent"] + outcome_class_order
#     labels = list(murmur_labels) + list(outcome_labels)
#     probabilities = list(murmur_probabilities) + list(outcome_probabilities)
#     return class_order, labels, probabilities





# /Data1/hmd2/notebooks_th/GRU_loss_add/models/Forcheck/GRU_multi_output.pt




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", type=str, default= None) 
    parser.add_argument("--data_folder", type=str, default= None) 
    parser.add_argument("--output_folder", type=str, default= None) 
    parser.add_argument("--allow_failures", type=str, default= True)
    parser.add_argument("--gpu_index", type=str, default= None) 
    parser.add_argument("--verbose", type=str, default= 1)
    args = parser.parse_args()
    
    
##################################__Please assign PATH, GPU_index__####################################################################

    # args.model_folder =  "your trained_model.pt file"
    # args.data_folder = "test dataset folder"
    # args.output_folder = "prediction of test dataset folder"
    # args.gpu_index = "number of gpu index"
    
    args.model_folder =  "/Data1/hmd2/notebooks_th/GRU_loss_add/models/Forcheck/"
    args.data_folder = "/Data2/murmur/test"
    args.output_folder = "/Data1/hmd2/notebooks_th/GRU_loss_add/models/Forcheck/output"
    args.gpu_index = "0"

#######################################################################################################################################


    if not os.path.exists(args.model_folder):
        raise Exception(f"The path {args.model_folder} does not exist!")
    elif not os.path.exists(args.data_folder):
        raise Exception(f"The path {args.data_folder} does not exist!")
    elif not args.gpu_index:
        raise Exception(f"You didn't assigned index of GPU!")
    else:
        pass
    
    
    args.model_folder = pathlib.Path(args.model_folder)
    args.data_folder = pathlib.Path(args.data_folder)
    args.output_folder = pathlib.Path(args.output_folder)
    
    
    # Don't Change This Line, Ever, Forever!
    ## unless you want to use Multi-gpu...
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index  # Select ONE specific index of gpu
    GPU_NUM = 0 # Since we Selected ONE specific gpu in environment, index number is 0
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu') 
    torch.cuda.set_device(device) # Set gpu, Done!
    
    
# def inference_model(model_folder: pathlib.Path, data_folder: pathlib.Path, output_folder: pathlib.Path,
#                     allow_failures: bool, verbose: int, device): # cuda device
    
    inference_model(model_folder= args.model_folder,
                    data_folder= args.data_folder,
                    output_folder= args.output_folder,
                    allow_failures= args.allow_failures,
                    verbose= args.verbose, 
                    device= device)