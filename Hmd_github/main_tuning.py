import numpy as np
import pathlib
import pandas as pd

from omegaconf import OmegaConf

# Load python files.
from main import main


# 1. Set config.yaml path 
config_dir = "/Data1/hmd2/notebooks_th/Hmd_github/config.yaml"
config = OmegaConf.load(config_dir) # Load config settings


# 2. Set tuning result save path
tuning_results_savepath = pathlib.Path("/Data1/hmd2/notebooks_th/Hmd_github/Tuning_results")


# 3. Set Tuning lists....
pos_weight_list = [4.0]
learning_rate_list = [1e-3, 5e-4, 1e-4]
const_max_list = [1.0, 2.0]
# mix_up_list = [False, True]
# val_fold_num_list = [0, 1, 2, 4]


# convert Tuning_mode not to use wandb package
config.virtual_settings.tuning = True

epx_num = 1

tuning_dict = {'epx_num': [], 
               "lr": [], 
               'pos_weight': [], 
            #    'mix_up_bool': [], 
               "const_max": [],
               
               'optim_threshold_stu': [], 
               "optim_threshold_tch": [],
               "Val_WMA_stu": [], 
               "Val_WMA_tch": [], 
               "Test_WMA_stu": [], 
               "Test_WMA_tch": []
               }


for lr in learning_rate_list:
    for pos_weight in pos_weight_list:
        for const_max in const_max_list:
            # for mix_up_bool in mix_up_list: 
            # for val_fold_num in val_fold_num_list:
        
            config.exp_params.learning_rate = lr
            config.exp_params.pos_weight = pos_weight
            config.exp_params.const_max = const_max
            # config.exp_params.use_mix_up = mix_up_bool
            # config.dataset.val_fold_num = val_fold_num
            
            config_result = main(config)
            
            config = OmegaConf.create(config_result)
                        
            tuning_dict["epx_num"].append(epx_num)
            tuning_dict["lr"].append(lr)
            tuning_dict["pos_weight"].append(pos_weight)
            tuning_dict["const_max"].append(const_max)
            # tuning_dict["mix_up_bool"].append(mix_up_bool)
            
            tuning_dict["optim_threshold_stu"].append(config.exp_results.optim_threshold_stu)
            tuning_dict["optim_threshold_tch"].append(config.exp_results.optim_threshold_tch)
            tuning_dict["Val_WMA_stu"].append(config.exp_results.Val_WMA_stu)
            tuning_dict["Val_WMA_tch"].append(config.exp_results.Val_WMA_tch)
            tuning_dict["Test_WMA_stu"].append(config.exp_results.Test_WMA_stu)
            tuning_dict["Test_WMA_tch"].append(config.exp_results.Test_WMA_tch)
            
            print(f"Exp Number {epx_num} is Done!\n")
            epx_num += 1  # +1 stack!
                
                
# Save Tuning result.                
tuning_results_df = pd.DataFrame(tuning_dict)                
tuning_results_df.to_csv(tuning_results_savepath / "tuning_results_3.csv")
print(f"CSV 파일 저장 완료: {tuning_results_savepath}/tuning_results_3.csv")
                