import pathlib
from typing import Dict, Tuple, List

import torch
from dataset import *
from model import instantiate_model
from tqdm import tqdm

# Global variables for Train models
TRAINING_BATCH_SIZE = 32
TRAINING_VAL_BATCH_SIZE = 128
TRAINING_LR = 1e-4   # 1e-4
TRAINING_PATIENCE = 10
TRAINING_MAX_EPOCHS = 1000  #  1000



# Save your trained model.
def save_model(model_folder, model, fold):
    model_path = pathlib.Path(model_folder) / f"model_{fold}.pt"
    torch.save(model.state_dict(), model_path)
            
            
# Load your trained model.
def load_model(model_folder, fold):
    model = instantiate_model()
    model.load_state_dict(torch.load(pathlib.Path(model_folder) / f"model_{fold}.pt"))
    model.eval()
    return model

            

# train k'th fold model
def train_model(model_folder, fold, train_dataset, valid_dataset, verbose, device):
    
    class_weights_train = calculate_class_weights(train_dataset)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=TRAINING_BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=TRAINING_VAL_BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )
    
    model = instantiate_model()
    
    model = model.to(device)
    
    if class_weights_train is not None:
        class_weights_train = class_weights_train.to(device)
    
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of Training model {num_parameters} params")
    
    optim = torch.optim.Adam(model.parameters(), lr=TRAINING_LR)
    crit = nn.CrossEntropyLoss(ignore_index=-1, weight=class_weights_train)
    
    best_val_loss = np.inf
    early_stop_counter = 0
    
    
    for epoch in tqdm(range(TRAINING_MAX_EPOCHS)):
        running_loss = 0.0

        model.train()
        
        # Training model part
        for i, (train_features, train_labels, train_lengths) in enumerate(train_dataloader):
            
            train_features = train_features.to(device)
            train_labels = train_labels.to(device)

            optim.zero_grad()
            
            out = model(train_features, train_lengths) # train_lengths? for sorted...
            loss = crit(out, train_labels)
            
            loss.backward()
            optim.step()
            running_loss += loss.item()
            
            
        # Calculate val loss
        with torch.no_grad():
            model.eval()
            val_losses = 0.0
            
            for val_features, val_labels, val_lengths in val_dataloader:
                val_features = val_features.to(device)
                val_labels = val_labels.to(device)

                out = model(val_features, val_lengths)
                
                loss = crit(out, val_labels)
                val_losses += loss * val_features.shape[0] # because CE is batch size average. Times bs 
            
            loss = val_losses / len(valid_dataset)
            
            
        if verbose >= 2:
            print(
                f"{epoch:04d}/{TRAINING_MAX_EPOCHS} | {running_loss / i:.3f} | {loss.item():.3f} | {early_stop_counter:02d}"
            )
        elif verbose >= 1:
            print(
                f"\r{epoch:04d}/{TRAINING_MAX_EPOCHS} | {running_loss / i:.3f} | {loss.item():.3f} | {early_stop_counter:02d} ",
                end="",
            )
        
        
        # Early-stopping    
        if loss < best_val_loss:
            # Save best model and reset patience
            save_model(model_folder, model, fold)
            best_val_loss = loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter > TRAINING_PATIENCE:
                break
            
    print(f"\nFinished training {fold + 1}'th neural network.")

    # Now run best model on validation set
    # Re-load best model
    model = load_model(model_folder, fold)
    model = model.to(device)
    
    return model  # return k'th trained model

            
def predict_single_model(model, features, lengths, device):
    
    features = features.to(device)

    results = []
    with torch.no_grad():
        # Compute HSMM observations(input for HSMM) using neural network
        out = model(features, lengths)
        out = out.cpu()

        for posteriors, lengths in zip(out, lengths):            
            posteriors = posteriors[:, :lengths]
            posteriors = torch.nn.functional.softmax(posteriors, dim=0)
            results.append(posteriors)

    return results



def train_and_validate_model(
    model_folder: pathlib.Path,
    fold: int,
    data_folder: pathlib.Path,
    train_files: List[str],
    train_labels: List[str],
    train_timings,
    val_files: List[str],
    val_labels: List[str],
    val_timings,
    shared_feature_cache: Dict[str, torch.Tensor],
    verbose: int,
    device  # cuda, not cpu
):
    
    train_dataset = RecordingDataset(
        data_folder, train_files, train_labels, train_timings,
        sequence_length= int(TRAINING_SEQUENCE_LENGTH / WINDOW_STEP), # Default: int(TRAINING_SEQUENCE_LENGTH / WINDOW_STEP)
        cache=shared_feature_cache
    )

    valid_dataset = RecordingDataset(
        data_folder, val_files, val_labels, val_timings, 
        sequence_length= int(TRAINING_SEQUENCE_LENGTH / WINDOW_STEP), 
        cache=shared_feature_cache
    )
    
    val_features, val_labels, val_lengths = batch_whole_dataset(valid_dataset)
    
    #print(f"val_lengths is {val_lengths}!!") 다 다름
    
    model = train_model(
        model_folder, fold, train_dataset, valid_dataset, verbose, device
    )
    
    fold_posteriors = predict_single_model(model, val_features, val_lengths, device)
    return model, {n: p for n, p in zip(val_files, fold_posteriors)}
    
    
def predict_files(model, data_folder, files, labels, timings, cache, device):  # sequence_length= int(TRAINING_SEQUENCE_LENGTH / WINDOW_STEP)
    dataset = RecordingDataset(data_folder, files, labels, timings, sequence_length= int(TRAINING_SEQUENCE_LENGTH / WINDOW_STEP), cache=cache)
    features, labels, lengths = batch_whole_dataset(dataset)
    posteriors = predict_single_model(model, features, lengths, device)
    return {n: p for n, p in zip(files, posteriors)}
    
      