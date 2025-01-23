import math

import random
import numpy as np
import pandas as pd 
import pathlib
from tqdm import tqdm
from copy import deepcopy
from itertools import cycle

import scipy.io as spio
from scipy.signal import resample

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR


from dataset import MAX_LENGTH, FREQ_BINS, collate_fn, calculate_features
from data_augment import mixup_strong
from model import MHA_LSTM_simpler, MHA_LSTM_simpler22 
# from neural_networks.MHA_LSTM_hidden import MHA_LSTM_hidden
# from neural_networks.MHA_blocks import MHA_blocks


from metric import*
import wandb


def lr_lambda(epoch):
    """
    epoch에 따른 LR 스케줄링 함수.
    
    - 0 <= epoch <= 10: 기초 lr(1e-5)에서 10 epoch에 1e-3이 되도록 지수적으로 증가
      1e-3 / 1e-5 = 100배 증가해야 하므로, 증가율은 exp(ln(100) * epoch/10)
    
    - epoch > 10: 10 epoch 시점의 학습률(1e-3)부터 total_epochs epoch 시점에 다시 1e-5로
      감소하도록 지수적으로 감소.
    
    lr_factor = base_lr * scale, 즉 scheduler에서는 scale만 반환합니다.
    """
    
    total_epoch = 250
    
    if epoch <= 10:
        # 증가: 0 epoch 에는 factor=1, 10 epoch에는 factor=100가 되어야 함.
        scale = math.exp(math.log(100) * (epoch / 10))
    else:
        # 10 epoch 부터 total_epochs epoch까지 1e-3에서 1e-5로 감소
        # 1e-5 / 1e-3 = 0.01
        # decay_epochs = config.exp_params.total_epoch - 10  # 감소하는 epoch 수
        
        decay_epochs = total_epoch - 10
        # 현재 감소 단계: 0부터 decay_epochs 까지
        current_decay_epoch = epoch - 10
        # decay factor: 10 epoch시점의 factor는 100, total_epochs의 factor는 1.
        # 100 * (0.01)^(current_decay_epoch / decay_epochs)
        # scale = 100 * math.exp(math.log(0.01) * (current_decay_epoch / decay_epochs))
        scale = 100 * math.exp(math.log(0.2) * (current_decay_epoch / decay_epochs))
    return scale




def train_validate_model(config, df_dict, train_dataset, unlabeled_dataset, val_dataset):
                                            
    sampler_unlabeled = RandomSampler(unlabeled_dataset, replacement=True)
    train_dataloader = DataLoader(train_dataset, batch_size= config.exp_params.strong_bs, shuffle=True, collate_fn= collate_fn, num_workers=config.virtual_settings.num_workers)
    unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size= config.exp_params.unlabel_bs, sampler= sampler_unlabeled, collate_fn= collate_fn, num_workers=config.virtual_settings.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size= config.exp_params.val_bs, shuffle=False,  collate_fn= collate_fn, num_workers=config.virtual_settings.num_workers) 
    
    student = MHA_LSTM_simpler22().cuda()
    teacher = deepcopy(student).cuda()
    
    if (not config.virtual_settings.debug) and (not config.virtual_settings.tuning):
        wandb.init(project= "For_journal", dir=f"{config.datapath.checkpoint_path}")
        wandb.watch(student)
    
    optim = torch.optim.AdamW(student.parameters(), 
                            lr= config.exp_params.learning_rate,
                            weight_decay=1e-5, betas=(0.9, 0.999), eps=1e-8) #  weight_decay=1e-2, betas=(0.9, 0.999), eps=1e-8)
    # exp_scheduler = ReduceLROnPlateau(optim, 
    #                         mode= 'min', 
    #                         patience= 3, 
    #                         cooldown= 0,
    #                         factor= 0.9,
    #                         threshold_mode= 'rel', 
    #                         min_lr= 1e-5, )
    exp_scheduler = LambdaLR(optim, lr_lambda=lr_lambda)
    
    
    consis_weight = config.exp_params.const_max
    
    best_total_loss, best_seq_loss, best_murmur_loss= tuple([np.inf] * 3)
    early_stop_counter = 0
    optim_count= 0
    
    # class_weights = torch.tensor([1.0, config.exp_params.pos_weight]).float().cuda()
    pos_weight = torch.tensor([config.exp_params.pos_weight]).float().cuda()
    # crit_murmur = torch.nn.CrossEntropyLoss(weight = class_weights)
    crit_murmur = torch.nn.BCEWithLogitsLoss(pos_weight= pos_weight)
    
    for epoch in tqdm(range(1, config.exp_params.total_epoch + 1)):
                
        student.train()
        current_lr = optim.param_groups[0]['lr']
        consis_weight *= 0.99
        
        # per epoch losses
        train_total_losses, train_seq_losses, train_mm_losses, cons_losses = tuple([0] * 4)

        total_stack_strong, total_stack_unlabel = 0, 0
    
        # batch train
        for i, (batch_strong, batch_unlabel) in enumerate(zip(train_dataloader, cycle(unlabeled_dataloader))):
            
            batch_num_strong, batch_num_unlabel = batch_strong[0].shape[0], batch_unlabel[0].shape[0]
            total_stack_strong += batch_num_strong
            total_stack_unlabel += batch_num_unlabel
            
            la_mels, la_seq_labels, la_pad_masks, la_mm_labels, _, _ = batch_strong
            ul_mels, ul_seq_labels, ul_pad_masks, ul_mm_labels, _, _ = batch_unlabel
            
            batch_mels = torch.concat([la_mels, ul_mels], dim= 0)
            batch_seq_labels = torch.concat([la_seq_labels, ul_seq_labels], dim= 0)
            batch_pad_masks = torch.concat([la_pad_masks, ul_pad_masks], dim= 0)
            batch_mm_labels = torch.concat([la_mm_labels, ul_mm_labels], dim= 0)
            
            batch_num = batch_mels.shape[0]
            label_mask = torch.zeros(batch_num).to(batch_mels).bool()
            unlabel_mask = torch.zeros(batch_num).to(batch_mels).bool()
            
            label_mask[ : la_mels.shape[0]] = 1
            unlabel_mask[la_mels.shape[0] : ] = 1  
            
            
            if config.exp_params.use_mix_up:
                if random.random() > 0.5 :
                    batch_mels[label_mask], batch_seq_labels[label_mask], batch_mm_labels[label_mask]\
                        = mixup_strong(batch_mels[label_mask], batch_seq_labels[label_mask], batch_mm_labels[label_mask])
                    
            batch_mels, batch_seq_labels = batch_mels.cuda(), batch_seq_labels.cuda()
            batch_pad_masks,  batch_mm_labels = batch_pad_masks.cuda(),  batch_mm_labels.cuda()
            
            optim.zero_grad()
            
            # Student
            seq_pred_stu, mm_pred_stu = student(batch_mels, batch_pad_masks)
            
            # Teacher
            with torch.no_grad():
                seq_pred_tch, mm_pred_tch = teacher(batch_mels, batch_pad_masks)
            
            # Seq pred loss
            seq_loss = F.binary_cross_entropy_with_logits(seq_pred_stu[label_mask], batch_seq_labels[label_mask], reduction= 'none')
            label_pad_mask = batch_seq_labels[label_mask] != -1
            seq_loss = seq_loss[label_pad_mask].mean()
            
            # Murmur pred loss
            murmur_loss = crit_murmur(mm_pred_stu[label_mask], batch_mm_labels[label_mask])
            
            # Consistency loss
            cons_seq_loss = F.mse_loss(F.sigmoid(seq_pred_stu[unlabel_mask]), F.sigmoid(seq_pred_tch[unlabel_mask].detach()))
            cons_mm_loss = F.mse_loss(F.sigmoid(mm_pred_stu[unlabel_mask]), F.sigmoid(mm_pred_tch[unlabel_mask].detach()))   
            cons_loss = cons_seq_loss + cons_mm_loss
            
            total_loss = (seq_loss + murmur_loss)  +  cons_loss #(cons_loss * consis_weight)
            
            train_total_losses += total_loss.cpu().item() * (batch_num_strong + batch_num_unlabel)
            train_seq_losses += seq_loss.cpu().item() * batch_num_strong
            train_mm_losses += murmur_loss.cpu().item() * batch_num_strong
            cons_losses += cons_loss.cpu().item() * batch_num_unlabel
            
            # Update Student model parameters
            total_loss.backward()
            optim.step()
            optim_count += 1 # For EMA step
            
            # Update EMA
            student, teacher = update_ema(config.exp_params.ema_factor, optim_count, student, teacher)
        
            
        # per epoch is done.
        with torch.no_grad():
            
            student.eval()
            val_total_losses, val_seq_losses, val_mm_losses = tuple([0] * 3)
            
            val_total_num = 0
            
            for i, (batch_val) in enumerate(val_dataloader):
                
                val_batch_num = batch_val[0].shape[0]
                val_total_num += val_batch_num
                
                val_mels, val_seq_labels, val_pad_masks, val_mm_labels, _, _ = batch_val
                val_mels, val_seq_labels = val_mels.cuda(), val_seq_labels.cuda()
                val_pad_masks, val_mm_labels = val_pad_masks.cuda(), val_mm_labels.cuda()
                
                val_seq_preds, val_murmur_preds = student(val_mels, val_pad_masks)  
                
                val_seq_loss = F.binary_cross_entropy_with_logits(val_seq_preds, val_seq_labels, reduction= 'none')
                val_pad_mask = val_seq_labels != -1
                val_seq_loss = val_seq_loss[val_pad_mask].mean()   
                
                val_mm_loss = crit_murmur(val_murmur_preds, val_mm_labels)
                val_total_loss = val_seq_loss + val_mm_loss
                
                val_total_losses += val_total_loss.cpu().item() * val_batch_num
                val_seq_losses += val_seq_loss.cpu().item() * val_batch_num
                val_mm_losses += val_mm_loss.cpu().item() * val_batch_num
            
                
            # if val_mm_losses < best_murmur_loss:
            if val_total_losses < best_total_loss:
                # print(config.datapath.checkpoint_path)
                
                save_student_teacher_both(config, student, teacher)
                # best_murmur_loss = val_mm_losses
                best_total_loss = val_total_losses
                early_stop_counter = 0 
            else:
                early_stop_counter += 1
                if early_stop_counter > config.exp_params.training_patience:
                    break

            exp_scheduler.step()
                
        # wandb        
        
        # total_stack_strong, total_stack_unlabel
        
        if (not config.virtual_settings.debug) and (not config.virtual_settings.tuning):
            wandb.log({
                "Train_total_loss": train_total_losses / (total_stack_strong + total_stack_unlabel), 
                "Train_seq_loss": train_seq_losses / total_stack_strong, 
                "Train_murmur_loss": train_mm_losses / total_stack_strong,
                "Train_consis_loss": cons_losses / total_stack_unlabel,
                "Consis_weight": consis_weight,
                "Current_lr": current_lr,
                "Val_total_loss": val_total_losses / val_total_num, 
                "Val_seq_loss": val_seq_losses / val_total_num, 
                "Val_murmur_loss": val_mm_losses / val_total_num,
                                })
        
        
    model_name = type(student).__name__
    student, teacher = load_student_teacher_both(config, model_name)
    
    eval_df = pd.concat([df_dict["val_recording_df"], df_dict["recording_df_bq"]], axis= 0)
    
    student_fold_preds = predict_single_model(config, eval_df, config.datapath.train_data, student)
    teacher_fold_preds = predict_single_model(config, eval_df, config.datapath.train_data, teacher)   

    return {"stop_epoch": epoch, "student": student, "teacher": teacher, 
            "student_fold_preds": student_fold_preds, "teacher_fold_preds": teacher_fold_preds}



def update_ema(alpha, global_step, model, ema_model): # student, teacher
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_params, params in zip(ema_model.parameters(), model.parameters()):
        ema_params.data.mul_(alpha).add_(params.data, alpha=1 - alpha)
    return model, ema_model



def save_student_teacher_both(config, student: torch.nn.Module, teacher: torch.nn.Module):
    student_path = config.datapath.checkpoint_path / f"Best_student_{config.dataset.val_fold_num}.pt"
    teacher_path = config.datapath.checkpoint_path / f"Best_teacher_{config.dataset.val_fold_num}.pt"
    
    torch.save(student.state_dict(), student_path)
    torch.save(teacher.state_dict(), teacher_path)



def load_student_teacher_both(config, model_name: str):
    student, teacher = eval(model_name)(), eval(model_name)()
    student.load_state_dict(torch.load(config.datapath.checkpoint_path / f"Best_student_{config.dataset.val_fold_num}.pt"))
    teacher.load_state_dict(torch.load(config.datapath.checkpoint_path / f"Best_teacher_{config.dataset.val_fold_num}.pt"))
    return student, teacher

    
    
def predict_single_model(config, eval_recording_df, data_folder, model):
    
    sampling_rate = config.dataset.sampling_rate
    
    model = model.cuda()
    result = {}
    
    for filename in eval_recording_df.index:
        # per_file_probs = []
                
        filepath = pathlib.Path(data_folder) / filename
        sr, recording = spio.wavfile.read(filepath.with_suffix(".wav"))
        
        if sr != sampling_rate:    
            num_samples = int(len(recording) * sampling_rate / sr)
            recording = resample(recording, num_samples)
        recording = torch.as_tensor(recording.copy())
        
        mel, _ = calculate_features(recording, sampling_rate)
        mel = mel.unsqueeze(0).cuda()

        _, murmur_pred = model(mel)
        # murmur_pred = F.softmax(murmur_pred, dim= -1)
        murmur_pred = F.sigmoid(murmur_pred)
        
        # result[filename] = {"holo_HSMM": murmur_pred.squeeze(0)[1].detach().cpu().item()}
        result[filename] = {"holo_HSMM": murmur_pred.squeeze(0).detach().cpu().item() }
        
    return result
        
        # B, _, T = mel.shape
        
        # if T > MAX_LENGTH:
        #     for i_t in range(T // MAX_LENGTH):
        #         _, ith_mm_prob = model(mel[... , i_t * MAX_LENGTH : (i_t + 1) * MAX_LENGTH])
        #         per_file_probs.append(ith_mm_prob.squeeze(0)[1].detach().cpu().item())
        # else:
        #     _, mm_prob = model(mel)
        #     per_file_probs.append(mm_prob.squeeze(0)[1].detach().cpu().item())
        # result[filename] = {"holo_HSMM": max(per_file_probs)}        
    # return result



def test(config, test_patient_df, test_recording_df, test_data_folder, model, optim_threshold):
    
    test_results = predict_single_model(config, test_recording_df, test_data_folder, model)
    
    test_predictions_df = pd.DataFrame.from_dict(test_results, orient="index")
    test_recording_df = test_recording_df.merge(test_predictions_df, left_index=True, right_index=True)
    
    merged_df = merge_df(test_recording_df, test_patient_df)
    
    test_murmur_preds= {}
    
    for index, row in merged_df.iterrows():
        murmur_pred =  decide_murmur_with_threshold(row.to_dict(), optim_threshold)
        test_murmur_preds[index] = {
                "prediction": murmur_pred, 
                "probabilities": [], 
                "label": row["murmur_label"]}
    # return test_murmur_preds
    
    test_murmur_score = compute_cross_val_weighted_murmur_accuracy(test_murmur_preds, print= True)
    return test_murmur_score.item() 



def test_CV(config, test_patient_df, test_recording_df, test_data_folder, cv_models, optim_threshold):
    
    test_results_dict = {}
    
    for _ , model in cv_models.items():
        test_results = predict_single_model(config, test_recording_df, test_data_folder, model)
        # result[filename] = {"holo_HSMM": murmur_pred.squeeze(0).detach().cpu().item() }
    
        for filename, prob_dict in test_results.items():
            if filename in test_results_dict:
                test_results_dict[filename].append(prob_dict["holo_HSMM"])
            else:
                test_results_dict[filename] = [prob_dict["holo_HSMM"]]
    
    for filename, prob_list in test_results_dict.items():
        test_results_dict[filename] = {"holo_HSMM": np.mean(prob_list).item()}
        
    
    test_predictions_df = pd.DataFrame.from_dict(test_results_dict, orient="index")
    test_recording_df = test_recording_df.merge(test_predictions_df, left_index=True, right_index=True)
    
    merged_df = merge_df(test_recording_df, test_patient_df)
    
    test_murmur_preds= {}
    
    for index, row in merged_df.iterrows():
        murmur_pred =  decide_murmur_with_threshold(row.to_dict(), optim_threshold)
        test_murmur_preds[index] = {
                "prediction": murmur_pred, 
                "probabilities": [], 
                "label": row["murmur_label"]}
    # return test_murmur_preds
    
    test_murmur_score = compute_cross_val_weighted_murmur_accuracy(test_murmur_preds, print= True)
    return test_murmur_score.item() 


def merge_df(val_results_df, patient_df):
    val_results_df.index = val_results_df.index.str.split("_", expand=True)
    val_results_df = val_results_df[['holo_HSMM']].groupby(level=[0]).max()
    
    patient_df.index = patient_df.index.astype(int)
    val_results_df.index = val_results_df.index.astype(int) # index 데이터타입 통일
    
    combined_df = val_results_df.merge(patient_df, left_index=True, right_index=True)
    
    return combined_df