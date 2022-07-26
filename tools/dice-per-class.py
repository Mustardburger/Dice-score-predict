from re import A
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import pickle
import tqdm

from torchutils import DiceCoeffPerLabel, make_one_hot

def dice_per_label_dist(base_dir, subfolder="train"):
    """
    Calculate the Dice score per label and see how it matches with the average Dice
    """
    dice_name = "dices_m_" + subfolder + ".pkl"
    if subfolder=="train": subfolder = ""
    else: subfolder = "-" + subfolder

    pred_folder = "Predicted-segs" + subfolder
    true_folder = "True-segs" + subfolder

    pred_seg_dir = os.path.join(base_dir, pred_folder)
    true_seg_dir = os.path.join(base_dir, true_folder)

    # Average dice and per label dice
    dice_dict = {1: [], 2: [], 3: []}
    with open(os.path.join(base_dir, dice_name), "rb") as f:
        avg_dice = pickle.load(f)
        avg_dice = list(avg_dice.values())


    for file in tqdm.tqdm(os.listdir(pred_seg_dir)):
        full_pred_dir = os.path.join(pred_seg_dir, file)
        full_true_dir = os.path.join(true_seg_dir, file)

        # Read tensor to CPU
        pred_seg = torch.load(full_pred_dir)
        true_seg = torch.load(full_true_dir)
        if len(pred_seg.size()) == 4: pred_seg = torch.squeeze(pred_seg, dim=0)
        if len(true_seg.size()) == 4: true_seg = torch.squeeze(true_seg, dim=0)

        # One hot encode
        pred_seg = make_one_hot(pred_seg[:, None, :, :], pred_seg.size(1))
        true_seg = make_one_hot(true_seg[:, None, :, :], true_seg.size(1))

        # Pass it to the Dice
        dices = DiceCoeffPerLabel(4, 1).forward(true_seg, pred_seg)
        for k, v in dices.items(): dice_dict[k].append(v)

    # Visualize
    fig, axes = plt.subplots(nrows=2, ncols=2)
    sns.histplot(data=avg_dice, stat="percent", ax=axes[0][0])
    sns.histplot(data=dice_dict[1], stat="percent", ax=axes[0][1])
    sns.histplot(data=dice_dict[2], stat="percent", ax=axes[1][0])
    sns.histplot(data=dice_dict[3], stat="percent", ax=axes[1][1])
    axes[0][0].set_title("avg")
    axes[0][1].set_title("1")
    axes[1][0].set_title("2")
    axes[1][1].set_title("3")
    plt.tight_layout()
    plt.savefig("avg dice vs dice per label.png")



if __name__ == "__main__":
    dice_per_label_dist("/mnt/beegfs/scratch/phuc/seg-quality-control/model_4", "train")
        


