import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from dataset import DiceDataset, FinetuneDiceDataset, get_dataloader
from data_transforms import DataTransform
from models.cnn import ResNet, get_pretrained_resnet
from tools.otherutils import find_best_model_across_epochs
from tools.torchutils import create_mask
from tools.vizutils import *
from tools.valutils import *

if __name__ == "__main__":

    # Some constants
    base_dir = "/scratch/phuc/trained_resnets"
    run_id = "run-10"
    best_model = find_best_model_across_epochs(run_id, 5)
    model_id = f"ep-{best_model}.pt"
    DATASET_ID = "model_4"
    MODEL_DIR = os.path.join(os.path.join(base_dir, run_id), model_id)
    DATA_DIR = "/mnt/beegfs/scratch/phuc/seg-quality-control/experiments"
    VAL_MRI_DIR = os.path.join(DATA_DIR, f"{DATASET_ID}/Ground-truth-images-test/")
    VAL_SEG_DIR = os.path.join(DATA_DIR, f"{DATASET_ID}/Predicted-segs-test/")
    TRUE_SEG_DIR = os.path.join(DATA_DIR, f"{DATASET_ID}/True-segs-test/")
    DICE_DIR = os.path.join(DATA_DIR, f"{DATASET_ID}/dices_test.pkl")
    BATCH_SIZE = 16
    experiments = {
        "true_vs_pred_dsc": False, "using_real_seg": False, "shift_experiment": False, "using_mask": True,
        "visualize_seg": False, "tumor_volume": False, "swap_label": False, "feature_importance": False
    }

    # Connect to device
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else 'cpu')

    # Define model to load
    #model = ResNet(4).to(device)
    model = get_pretrained_resnet("resnet50", 4).to(device)
    model.load_state_dict(torch.load(MODEL_DIR))

    # Define dataset
    data_transform = DataTransform().get("norm_no_crop")

    # Define some objects to pass to the experiments
    dir_dict = {"VAL_MRI_DIR": VAL_MRI_DIR, "VAL_SEG_DIR": VAL_SEG_DIR, "DICE_DIR": DICE_DIR}
    torch_stuff = {"data_transform": data_transform, "device": device, "model": model}
    hyperparams = {"batch_size": BATCH_SIZE}

    # True vs pred dsc
    if experiments["true_vs_pred_dsc"]:
        true_vs_pred_dsc(dir_dict, torch_stuff, hyperparams, viz="err")

    # Tumor volume and prediction error
    if experiments["tumor_volume"]:
        tumor_volume(dir_dict, torch_stuff, hyperparams)

    # True vs pred dsc but now using real seg (true dsc now are all 1)
    if experiments["using_real_seg"]:
        dir_dict["VAL_SEG_DIR"] = TRUE_SEG_DIR
        true_vs_pred_dsc(dir_dict, torch_stuff, hyperparams, real_seg=True)

    # Masking modalities experiment
    if experiments["using_mask"]:
        masks = [
            [], [0], [1], [2], [0, 1], [0, 2], [1, 2]
        ]
        use_mask(dir_dict, torch_stuff,hyperparams, masks)

    # Shift experiment configurations
    if experiments["shift_experiment"]:
        shift_schedule = range(0, 50, 5)
        num_images = 10
        shift_experiments(dir_dict, num_images, shift_schedule, torch_stuff)

    # Swap the labels
    if experiments["swap_label"]:
        swap_label_dict = {
            1.0: [2.0, 3.0],
            2.0: [1.0, 3.0],
            3.0: [1.0, 2.0]
        }
        swap_label_experiment(swap_label_dict, dir_dict, torch_stuff)

    # Visualize the segmentations
    if experiments["visualize_seg"]:
        err_thred = 0.2
        visualize_mri_and_seg(dir_dict, torch_stuff, hyperparams, err_thred)

    # Visualize the feature importance
    if experiments["feature_importance"]:
        hyperparams["batch_size"] = 20
        thred_list = [0.0002, 0.0004, 0.0006, 0.0008, 0.001]
        feature_importance(dir_dict, torch_stuff, hyperparams, num_background=100, num_test=2, thred_list=thred_list)



