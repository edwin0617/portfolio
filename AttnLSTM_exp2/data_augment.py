import torch
import numpy as np

def mixup_strong(data, seq_label, murmur_laebl, alpha=0.2, beta=0.2, mixup_label_type="soft"):
    
    with torch.no_grad():
        batch_size = data.size(0)
        c = np.random.beta(alpha, beta)

        perm = torch.randperm(batch_size)

        mixed_data = c * data + (1 - c) * data[perm, :]
        if seq_label is not None:
            if mixup_label_type == "soft":
                mixed_seq_label = torch.clamp(
                    c * seq_label + (1 - c) * seq_label[perm, :], min=0, max=1
                )
                mixed_murmur_label = torch.clamp(
                    c * murmur_laebl + (1 - c) * murmur_laebl[perm, :], min=0, max=1
                )
                
            elif mixup_label_type == "hard":
                mixed_seq_label = torch.clamp(seq_label + seq_label[perm, :], min=0, max=1)
                mixed_murmur_label = torch.clamp(murmur_laebl + murmur_laebl[perm, :], min=0, max=1)
            else:
                raise NotImplementedError(
                    f"mixup_label_type: {mixup_label_type} not implemented. choice in "
                    f"{'soft', 'hard'}"
                )  
            return mixed_data, mixed_seq_label, mixed_murmur_label
        else:
            return mixed_data
        


def mixup(data, target=None, alpha=0.2, beta=0.2, mixup_label_type="soft"):
    """Mixup data augmentation by permuting the data.

    Args:
        data: input tensor, must be a batch so data can be permuted and mixed.
        target: tensor of the target to be mixed, if None, do not return targets.
        alpha: float, the parameter to the np.random.beta distribution
        beta: float, the parameter to the np.random.beta distribution
        mixup_label_type: str, the type of mixup to be used choice between {'soft', 'hard'}.
    Returns:
        torch.Tensor of mixed data and labels if given
    """
    with torch.no_grad():
        batch_size = data.size(0)
        c = np.random.beta(alpha, beta)

        perm = torch.randperm(batch_size)

        mixed_data = c * data + (1 - c) * data[perm, :]
        if target is not None:
            if mixup_label_type == "soft":
                mixed_target = torch.clamp(
                    c * target + (1 - c) * target[perm, :], min=0, max=1
                )
            elif mixup_label_type == "hard":
                mixed_target = torch.clamp(target + target[perm, :], min=0, max=1)
            else:
                raise NotImplementedError(
                    f"mixup_label_type: {mixup_label_type} not implemented. choice in "
                    f"{'soft', 'hard'}"
                )

            return mixed_data, mixed_target
        else:
            return mixed_data