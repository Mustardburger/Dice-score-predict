import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import torch

def create_folder_for_run(base_dir):
    experiments_run = len(os.listdir(base_dir))
    save_dir = os.path.join(base_dir, f"run-{experiments_run+1}")
    os.mkdir(save_dir)
    return save_dir

def find_best_model_across_epochs(run_id, model_save_freq):
    """
    Find the model with the lowest val score across epochs
    """
    loss_dir = os.path.join(os.path.join("/mnt/beegfs/scratch/phuc/trained_resnets/", run_id), "loss.pkl")
    with open(loss_dir, "rb") as f:
        loss_list = pickle.load(f)["val_loss"]

    shortened_list = [dsc for i, dsc in enumerate(loss_list) if (i==1 or (i+1)%model_save_freq==0)]
    ind = np.argmin(shortened_list)
    if (ind == 0): return 1
    else: return ind*model_save_freq

def process_tensor_for_visualization(tensor, special=False):
    """
    Reshape and normalize the tensor for visualization
    Input: torch.Tensor. Output: np array
    """
    if len(tensor.size())==4: tensor = tensor[0, ...]
    tensor = tensor.detach().cpu().numpy()
    if tensor.shape[-1] not in [1, 3]: tensor = np.swapaxes(tensor, 0, -1)

    # Normalize the tensor if necessary (if tensor is a 3-channel image)
    if special:
        tensor = np.expand_dims(np.mean(tensor, axis=2), -1)
    else:
        if tensor.shape[-1] > 1:
            if np.amax(tensor) <= 1.0 and np.amin(tensor) < 0.0: tensor = np.expand_dims(np.mean(((tensor / 2) + 0.5), axis=2), -1)
            else: tensor = np.expand_dims(np.mean(tensor, axis=2), -1)
    
    # else, this tensor is a seg, so no need for further processing
    return tensor
