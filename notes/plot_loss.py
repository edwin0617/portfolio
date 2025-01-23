import pandas as pd
import matplotlib.pyplot as plt


def plot_loss_function(df: pd.DataFrame, num_subplot: int):
    fig, axs = plt.subplots(num_subplot, 1, figsize=(8,12))

    num_epochs = df.shape[0]

    axs[0].plot(range(1, num_epochs + 1), df['Train_total_loss'], label='Train Loss')
    axs[0].plot(range(1, num_epochs + 1), df['Val_total_loss'], label='Val Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Total Loss')
    axs[0].legend()

    axs[1].plot(range(1, num_epochs + 1), df['Train_seq_loss'], label='Train Loss')
    axs[1].plot(range(1, num_epochs + 1), df['Val_seq_loss'], label='Validation Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Frame Loss')
    axs[1].legend()

    axs[2].plot(range(1, num_epochs + 1), df['Train_murmur_loss'], label='Train Loss')
    axs[2].plot(range(1, num_epochs + 1), df['Val_murmur_loss'], label='Validation Loss')
    
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Loss')
    axs[2].set_title('Murmur Loss')
    axs[2].legend()
    
    plt.tight_layout()
    plt.show()