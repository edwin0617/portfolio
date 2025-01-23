import os
import json
import pathlib
import argparse
from tqdm import tqdm

from typing import Dict, Tuple, List
from collections import defaultdict

# from utils import*
from dataset import*
from metric import *


from model import*
from train import predict_single_model
import segmenter
import decision_tree


USE_MURMUR_DECISION_TREE = False



def load_challenge_model(model_folder, verbose, num_k):
    with (pathlib.Path(model_folder) / "settings.json").open("r") as f:
        settings = json.load(f)

    return [
        {
            "network": load_single_network_fold(model_folder, fold),
            "tree_murmur": load_catboost_model(model_folder, fold, "murmur_label") ,
            "tree_outcome": load_catboost_model(model_folder, fold, "outcome_label"),
            "outcome_threshold": settings["threshold"]
        }
        for fold in range(num_k)
    ]



# Load Challenge outputs.
def load_challenge_outputs(filename):
    with open(filename, 'r') as f:
        for i, l in enumerate(f):
            if i==0:
                patient_id = l.replace('#', '').strip()
            elif i==1:
                classes = tuple(entry.strip() for entry in l.split(','))
            elif i==2:
                labels = tuple(sanitize_binary_value(entry) for entry in l.split(','))
            elif i==3:
                probabilities = tuple(sanitize_scalar_value(entry) for entry in l.split(','))
            else:
                break
    return patient_id, classes, labels, probabilities



# run_challenge_model(model, patient_data, recordings, verbose)



# data: 환자별 .txt파일, recordings: 환자별 오디오_파일(1 ~ 4개)
## 환자별 .txt파일과 오디오_파일(1 ~ 4개) 읽어들인 후 
def run_challenge_model(model, data, recordings, verbose, device):
    num_locations = get_num_locations(data) # 환자별 오디오 파일 개수 확인(AV, MV, TV, ...)
    
    # print(num_locations)  4, 2, 3, 1, ..., ..
    
    recording_information = data.split("\n")[1 : num_locations + 1]
    recording_name = [r.split(" ")[0] for r in recording_information]
    
    # print(recording_name)  ['AV', 'PV', 'TV', 'MV']
    
    fs = get_frequency(data)

    for model_fold in model:
        model_fold["network"].to(device)

    location_predictions = defaultdict(list)
    location_signal_quals = defaultdict(list)
    
    for recording, name in zip(recordings, recording_name):
        # Load features.
        recording = torch.as_tensor(recording.copy())
        features, fs_features = calculate_features(recording, fs)
        features = features.unsqueeze(0)
        lengths = torch.as_tensor([features.shape[-1]])

        # Predict using each cross validated model, but now average before segmentation
        fold_predictions = []
        for model_fold in model:
            results = predict_single_model(model_fold["network"], features, lengths, device)
            fold_predictions.append(results[0])
        
        #print(len(fold_predictions))    
        
        posteriors = torch.mean(torch.stack(fold_predictions, dim=0), dim=0)
        posteriors = posteriors.T  # [C, T] to [T, C]
        posteriors = posteriors.numpy()
        _, _, healthy_conf, murmur_conf, murmur_timing = segmenter.double_duration_viterbi(posteriors, model[0]["network"].output_fs)
        location_predictions[name].append(murmur_conf - healthy_conf)
        location_signal_quals[name].append(max(murmur_conf, healthy_conf))

    #  TODO: perhaps do not mean.
    full_features = {f"conf_difference_{k}": np.mean(v) for k, v in location_predictions.items()}
    for k, v in location_signal_quals.items():
        full_features[f"signal_qual_{k}"] = np.mean(v)
        
    for k, v in functions.items():
        full_features[k] = v(data)
    full_features["num_rec"] = len(recordings)

    if USE_MURMUR_DECISION_TREE:
        ordered_array = [full_features.get(k, None) for k in model[0]["tree_murmur"].feature_names_]
        probabilities = []
        
        for model_fold in model:
            probabilities.append(
                model_fold["tree_murmur"].predict(ordered_array, prediction_type="Probability")
            )
        murmur_probabilities = np.mean(probabilities, axis=0)

        # Choose label with higher probability.
        murmur_labels = np.zeros(len(murmur_probabilities), dtype=np.int_)
        idx = np.argmax(murmur_probabilities)
        murmur_labels[idx] = 1
    else:
        prediction = decide_murmur_outcome(full_features)
        murmur_probabilities = np.zeros(3)
        prediction_to_index = {"Present": 0, "Unknown": 1, "Absent": 2}
        murmur_probabilities[prediction_to_index[prediction]] = 1
        murmur_labels = np.zeros(3, dtype=np.int_)
        murmur_labels[prediction_to_index[prediction]] = 1


    ordered_array = [full_features.get(k, None) for k in model[0]["tree_outcome"].feature_names_]
    probabilities = []
    for model_fold in model:
        probabilities.append(
            model_fold["tree_outcome"].predict(ordered_array, prediction_type="Probability")
        )
    outcome_probabilities = np.mean(probabilities, axis=0)

    # Choose label with higher probability.
    outcome_labels = np.zeros(len(outcome_probabilities), dtype=np.int_)
    outcome_class_order = list(model_fold["tree_outcome"].classes_)
    abnormal_idx = outcome_class_order.index("Abnormal")
    idx = abnormal_idx if outcome_probabilities[abnormal_idx] > model_fold["outcome_threshold"] else 1 - abnormal_idx
    outcome_labels[idx] = 1

    if USE_MURMUR_DECISION_TREE:
        class_order = list(model_fold["tree_murmur"].classes_) + outcome_class_order
    else:
        class_order = ["Present", "Unknown", "Absent"] + outcome_class_order
    labels = list(murmur_labels) + list(outcome_labels)
    probabilities = list(murmur_probabilities) + list(outcome_probabilities)
    
    
    # print(len(class_order), len(labels), len(probabilities))
    
    return class_order, labels, probabilities




# run_model(model_folder, data_folder= args.data, 
#                         output_folder= args.output_folder, 
#                         allow_failures= allow_failures, 
#                         device = device
#                         num_k = args.num_of_fold
#                         verbose= args.verbose)




# Run model.
def run_model(model_folder, data_folder, output_folder, allow_failures, device, num_k, verbose):
    # Load models.
    if verbose >= 1:
        print('Loading Challenge model...')

    model = load_challenge_model(model_folder, verbose, num_k) ### Teams: Implement this function!!!

    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)

    if num_patient_files==0:
        raise Exception('No data was provided.')

    # Create a folder for the Challenge outputs if it does not already exist.
    os.makedirs(output_folder, exist_ok=True)

    # Run the team's model on the Challenge data.
    if verbose >= 1:
        print('Running model on Challenge data...')

    # Iterate over the patient files.
    for i in tqdm(range(num_patient_files)):
        
        
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patient_files))

        patient_data = load_patient_data(patient_files[i]) # 환자별 .txt파일 읽어온 것
        recordings = load_recordings(data_folder, patient_data) # 읽어온 .txt파일로부터 오디오파일 읽어온 것(1-D)
        
        # print(patient_data, recordings)
        


        # Allow or disallow the model to fail on parts of the data; helpful for debugging.
        try:
            classes, labels, probabilities = run_challenge_model(model, patient_data, recordings, verbose, device) ### Teams: Implement this function!!!
        except:
            if allow_failures:
                if verbose >= 2:
                    print('... failed.')
                classes, labels, probabilities = list(), list(), list()
            else:
                raise
        
        
        #print(f"len of class: {len(classes)}, len of labels: {len(labels)}, len of prob:{len(probabilities)}")
        
        
        # Save Challenge outputs.
        head, tail = os.path.split(patient_files[i])
        root, extension = os.path.splitext(tail)
        output_file = os.path.join(output_folder, root + '.csv')
        patient_id = get_patient_id(patient_data)
        save_challenge_outputs(output_file, patient_id, classes, labels, probabilities)

    if verbose >= 1:
        print('Done.')
        
        
        
# if __name__ == '__main__':
#     # Parse the arguments.
#     if not (len(sys.argv) == 4 or len(sys.argv) == 5):
#         raise Exception('Include the model, data, and output folders as arguments, e.g., python run_model.py model data outputs.')

#     # Define the model, data, and output folders.
#     model_folder = sys.argv[1] 
#     data_folder = sys.argv[2]    # /Data2/murmur/test
#     output_folder = sys.argv[3]  # /Data1/hmd2/notebooks_th/연습용2/models/practice/result

#     # Allow or disallow the model to fail on parts of the data; helpful for debugging.
#     allow_failures = False

#     # Change the level of verbosity; helpful for debugging.
#     if len(sys.argv)==5 and is_integer(sys.argv[4]):
#         verbose = int(sys.argv[4])
#     else:
#         verbose = 1

#     run_model(model_folder, data_folder, output_folder, allow_failures, verbose)
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()  
    parser.add_argument("--data", type=str, default= None) 
    parser.add_argument("--model_folder", type=str, default= None) 
    parser.add_argument("--gpu_index", type=str, default= None) 
    parser.add_argument("--prev_run", type=str, default= None)
    parser.add_argument("--verbose", type=int, default= 1)
    args = parser.parse_args()
    
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()  
    parser.add_argument("--data", type=str, default= None) 
    parser.add_argument("--model_folder", type=str, default= None) 
    parser.add_argument("--result_folder", type=str, default=None)
    parser.add_argument("--num_of_fold", type=str, default=None)
    parser.add_argument("--gpu_index", type=str, default= None) 
    parser.add_argument("--verbose", type=int, default= None)
    args = parser.parse_args()
    
############################__Please assign PATH, GPU_index__########################################################################################
    
    args.data = "/Data2/murmur/test" 
    args.model_folder = "/Data1/hmd2/notebooks_th/CUEDA_easy/models/2024-10-03_18:21:09" 
    args.output_folder = "/Data1/hmd2/notebooks_th/CUEDA_easy/models/2024-10-03_18:21:09/result_wo_tree"  # /Data1/hmd2/notebooks_th/연습용2/models/practice/settings.json
    args.gpu_index = '0'
    args.num_of_fold = 4
    args.verbose = 1
    allow_failures = False
    
######################################################################################################################################################    

    # if there are no output path exist
    os.makedirs(args.output_folder, exist_ok=True)
    
    
    if not os.path.exists(args.data):
        raise Exception(f"The path {args.data} does not exist!")
    elif not os.path.exists(args.model_folder):
        raise Exception(f"The path {args.model_folder} does not exist!")
    elif not args.gpu_index:
        raise Exception(f"You didn't assigned index of GPU!")
    else:
        pass
    
    # model_folder= path: where to save model_parameter
    model_folder = args.model_folder
        
    
    
    
    # Don't Change This Line, Ever, Forever!
    ## unless you want to use Multi-gpu...
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index  # Select ONE specific index of gpu
    GPU_NUM = 0 # Since we Selected ONE specific gpu in environment, index number is 0
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu') 
    torch.cuda.set_device(device) # Set gpu, Done!
    
    run_model(model_folder, data_folder= args.data, 
                            output_folder= args.output_folder, 
                            allow_failures= allow_failures, 
                            device = device,
                            num_k = args.num_of_fold,
                            verbose= args.verbose)
    
