import os
import pathlib
import datetime
import pytz

# Virtual_machine_settings
debug = False # If debug: do not ; else: make new_folder where to save exp_results
tuning= False
gpu_index= '0'
num_workers= 8
verbose= 1 

# wandb setting
project_name = "paper_exp"

#### Set Data & model parameters Path ####
train_data = pathlib.Path("/Data2/murmur/train")
test_data = pathlib.Path("/Data2/murmur/test")
external_data = pathlib.Path("/Data2/heart_sound_dataset")
external_data_subpath = {"pysionet_sufhsdb": "pysionet_sufhsdb", "kaggle_set_a": "kag_dataset_1/set_a", "kaggle_set_b": "kag_dataset_1/set_b"}
model_folder= pathlib.Path("/Data1/hmd2/notebooks_th/paper_exp/exps")


# For debug
if not os.path.exists(train_data):
    raise Exception(f"{train_data} does not exist!")
elif not os.path.exists(test_data):
    raise Exception(f"{test_data} does not exist!")
elif not os.path.exists(model_folder):
    raise Exception(f"{model_folder} does not exist!")


# If model folder does not exist, Make folder directory
if debug:
    model_folder = model_folder / "Forcheck"
elif tuning:
    model_folder = model_folder / "Forcheck"
else:
    current_time = datetime.datetime.utcnow()
    kst_timezone = pytz.timezone('Asia/Seoul')
    kst_now = current_time.astimezone(kst_timezone)
    filename = kst_now.strftime("%Y-%m-%d_%H:%M:%S")
    model_folder = model_folder / filename
    
if not debug:    
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(model_folder / 'dataframes', exist_ok=True) # To plot murmur preds probability


# Trainig Hyperparameters 
random_seed= 29 # 0, 1996, 2024

strong_bs = 80 # training batch size: 40, 120
# weak_bs = 60
unlabel_bs = 160 # unlabeled batch size: 80, 240s
val_bs = 120  # validation batch size : 120, 200
learning_rate= 1e-4 * 5 # 1e-5
base_lr= 4e-5
max_lr = 1e-3  #    1e-3 * 2
final_lr = 1e-5
max_epoch= 250   # training max epoch
# max_sch_step = 23 * max_epoch
pct_start = 10  # 전체 학습 기간 중 10 epoch에서 최대 학습률 도달.
lambda_sch_decay_rate =  1e-5  #  1e-4*6 #  1e-4*5   #  1e-4*3

div_factor= 25
final_div_factor= 1000
training_patience= 10 # early-stopping condition.
sequence_length = 6 # seconds
ema_factor = 0.99
const_max= 1  # 2
# n_epochs_warmup= 20 # 50
mixup_alpha= 0.2
mixup_beta = 0.2
mixup_label_type= "soft" # "hard"
use_mix_up= True


pos_weight = 4.0


# Split number ror valid data and k-fold
num_k= 5 #   4
val_fold_num= 3

## Data preprocessing Parameters
sampling_rate= 4000
window_length = 0.050
hop_length= 0.020 # window_step
freq_high= 800
freq_bins= 40
train_seq_len= 6
clean_noise= True
slice_feature = False  # If True, slice spectrogram and add to dataloader. Not recommended...







