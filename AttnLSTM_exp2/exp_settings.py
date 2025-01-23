from omegaconf import OmegaConf


def get_config():
    config = OmegaConf.create(
        {"virtual_settings": {"debug": False, # If True, use small data for debug
                            "tuning": False, # If True, wandb package will not be used
                            "gpu_index": '0', # Select gpu index, if you have at least 2 gpus
                            "num_workers": 8, # cpu num workers
                            "verbose": 1, 
                            },
        # Dataset path
        "datapath": {"train_data": "/Data2/murmur/train", # Train dataset path to fit model
                    "test_data": "/Data2/murmur/test", # Test dataset path to test model
                    "external_data": "/Data2/heart_sound_dataset", # External dataset path to use Semi-supervised method.
                    "external_data_subpath": {"pysionet_sufhsdb": "pysionet_sufhsdb", # External dataset subpath
                                              "kaggle_set_a": "kag_dataset_1/set_a", 
                                              "kaggle_set_b": "kag_dataset_1/set_b"},
                    "checkpoint_path": "/Data1/hmd2/notebooks_th/AttnLSTM_exp2/exps/MHA_LSTM_Semi", # Savepath for model checkpoint.
                    },               # "/Data1/AttnLSTM_exp2/exps/MHA_LSTM_Semi"
        # Feature extract parameters
        "data_preprocess": {"frequency_high": 800,
                    "sampling_rate": 4000,
                    "window_length": 0.050,
                    "hop_length": 0.020,
                    "sequence_length": 6, 
                    "clean_noise": True, 
                    "num_k": 5, 
                    "val_fold_num": 3
                                    }, 
        # Experiment settings.
        "exp_settings": {
                    'exp_params': {"random_seed": 0,
                                "total_epoch": 250, # 250
                                "strong_bs": 80, 
                                "unlabel_bs": 160, 
                                "val_bs": 120, 
                                "base_lr": 1e-5,
                                "learning_rate": 1e-4 * 5, 
                                "weight_decay": 1e-2, 
                                "betas": (0.9, 0.999),
                                "training_patience": 10, 
                                "mixup_alpha": 0.2,
                                "mixup_beta": 0.2,
                                "mixup_label_type": "soft", 
                                "use_mix_up": False, 
                                "pos_weight": 4.0},
                    "mean_teacher": {"ema_factor": 0.99, 
                                    "const_max": 1, 
                                    "rampup_length": 50}
                }                         
        })
    
    return config















# # 1. 설정을 딕셔너리 형식으로 정의한 후, OmegaConf 객체로 변환합니다.
# config = OmegaConf.create({
#     "database": {
#         "host": "localhost",
#         "port": 3306,
#         "user": "admin",
#         "password": "secret"
#     },
#     "logging": {
#         "level": "INFO",
#         "format": "[%(asctime)s] %(levelname)s - %(message)s"
#     },
#     "features": {
#         "enable_feature_x": True,
#         "max_items": 100
#     }
# })

# # 2. 설정 값에 접근하는 방법
# print("데이터베이스 호스트:", config.database.host)
# print("로그 레벨:", config.logging.level)
# print("Feature X 활성화:", config.features.enable_feature_x)

# # 3. 기존 설정에 새로운 키/값 추가
# config.new_section = {"new_key": "new_value"}
# print("새로운 섹션:", config.new_section.new_key)

# # 4. 설정 내용을 YAML 형식의 문자열로 출력
# yaml_string = OmegaConf.to_yaml(config)
# print("\nYAML 형식의 설정 내용:\n", yaml_string)

# # 5. 파일로부터 YAML 설정 로드하기 (예시)
# # 만약 'config.yaml' 파일이 존재한다면 아래와 같이 로드할 수 있습니다.
# # config_from_file = OmegaConf.load("config.yaml")
# # print("파일로부터 로드한 설정:", config_from_file)