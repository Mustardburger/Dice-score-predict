import torch
import torch.nn as nn
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import pickle

class DiceDataset(nn.Module):

    def __init__(self, mri_img_dir, seg_dir, dice_scores, img_transform="default", ratio=None, subset_dir=None):
        self.mri_img_folder = mri_img_dir
        self.seg_folder = seg_dir
        self.ratio = ratio
        self.subset_dir = subset_dir

        # Img transforms, based on user's input
        # Currently, default means nothing
        if img_transform == "default":
            self.img_transform = None
        else:
            self.img_transform = None        

        self.mri_img_dirs = sorted(os.listdir(self.mri_img_folder))
        self.seg_dirs = sorted(os.listdir(self.seg_folder))

        # If ratio, then only take a subset of the images
        if self.ratio:
            self.mri_img_dirs, self.seg_dirs = self._take_subset()
            # The subset should be saved, because if not, then after training we will not know which images were selected
            if self.subset_dir:
                self._save_subset()

        if isinstance(dice_scores, str):
            with open(dice_scores, "rb") as f:
                self.dice_scores = pickle.load(f)
        else:
            self.dice_scores = dice_scores

        # Unravel the dict
        self.dice_scores_list = self._unravel_dict()

        # Before doing anything, make sure that all the indices are correct
        self._check_correct_idx()

    def __len__(self):
        return len(self.mri_img_dirs)

    def __getitem__(self, idx):
        img_dir = os.path.join(self.mri_img_folder, self.mri_img_dirs[idx])
        seg_dir = os.path.join(self.seg_folder, self.seg_dirs[idx])

        # Not sure if Pytorch read_image works, but just try it for now
        # Read images are only for model_6
        #img = read_image(img_dir)
        #seg = read_image(seg_dir)
        img = torch.load(img_dir)
        seg = torch.load(seg_dir)
        if self.img_transform:
            img = self.img_transform(img)
            seg = self.img_transform(seg)
        dsc = self.dice_scores_list[idx]

        return {"img": img, "seg": seg, "dsc": dsc}

    def _take_subset(self):
        """
        If user defines ratio, then take a subset of the dataset
        """
        new_img, new_seg = [], []
        np.random.seed(999)
        for i in range(len(self.mri_img_dirs)):
            ran = np.random.rand()
            if (ran <= self.ratio):
                new_img.append(self.mri_img_dirs[i])
                new_seg.append(self.seg_dirs[i])
        
        return new_img, new_seg

    def _save_subset(self):
        """
        Save the subset into the directory
        """
        name_file = "name-list.pkl"
        with open(os.path.join(self.subset_dir, name_file), "wb") as f:
            pickle.dump(self.mri_img_dirs, f)

    def _unravel_dict(self):
        """
        Create a list from the dict that aligns the value with the directories
        """
        l = []
        for mri in self.mri_img_dirs:
            e_mri, id_mri = mri.split("_")[0][5:], mri.split("_")[1][3:-3]
            #e_mri, id_mri = mri.split("_")[0][5:], mri.split("_")[1][3:-4]
            name = "Epoch" + e_mri + "-" + id_mri
            l.append(self.dice_scores[name])
        return l


    def _check_correct_idx(self):
        """
        Used to check whether the names of files in self.mri_dir, self.seg_dir, and self.dice_scores align
        """
        assert (len(self.mri_img_dirs) == len(self.seg_dirs)) and (len(self.mri_img_dirs) == len(self.dice_scores_list))

        # Check if the images, segs and dice scores align
        for i in range(len(self.mri_img_dirs)):
            mri, seg, dice = self.mri_img_dirs[i], self.seg_dirs[i], self.dice_scores_list[i]
            #e_mri, id_mri = mri.split("_")[0][5:], mri.split("_")[1][3:-4]
            #e_seg, id_seg = seg.split("_")[0][5:], seg.split("_")[1][3:-4]
            e_mri, id_mri = mri.split("_")[0][5:], mri.split("_")[1][3:-3]
            e_seg, id_seg = seg.split("_")[0][5:], seg.split("_")[1][3:-3]
            assert ((e_mri==e_seg) and (id_mri==id_seg))
            name = "Epoch" + e_mri + "-" + id_mri
            assert name in self.dice_scores
            assert self.dice_scores[name] == dice


class FinetuneDiceDataset(DiceDataset):

    def __init__(self, mri_img_dir, seg_dir, dice_scores, img_transform="default", ratio=None, subset_dir=None):
        self.img_transform = img_transform
        super().__init__(mri_img_dir, seg_dir, dice_scores, img_transform="default", ratio=ratio, subset_dir=subset_dir)

    def __getitem__(self, idx):
        img_dir = os.path.join(self.mri_img_folder, self.mri_img_dirs[idx])
        seg_dir = os.path.join(self.seg_folder, self.seg_dirs[idx])

        # This is the difference between this class and its parent's class
        img = self._normalize(torch.load(img_dir))
        seg = torch.load(seg_dir)
        assert (torch.max(img) <= 1.0) and (torch.min(img) >= 0.0)
        if (len(img.size()) > 3):
            img = torch.squeeze(img, dim=0)
        if (len(seg.size()) > 3):
            seg = torch.squeeze(seg, dim=0)

        if self.img_transform:
            img = self.img_transform(img)
        dsc = self.dice_scores_list[idx]

        assert len(img.size()) == 3
        assert len(seg.size()) == 3

        return {"img": img, "seg": seg, "dsc": dsc} 

    def _normalize(self, tensor):
        """
        Normalize the tensor from (-1, 1) to (0, 1)
        """
        return (tensor * 0.5) + 0.5


def get_dataset_dirs(data_root, dataset_id, run_id):
    subset_train_dir, subset_val_dir = make_subset_dirs(run_id)
    return {
        "IMG_TRAIN_DIR": os.path.join(data_root, f"{dataset_id}/Ground-truth-images/"),
        "SEG_TRAIN_DIR": os.path.join(data_root, f"{dataset_id}/Predicted-segs/"),
        "IMG_VAL_DIR": os.path.join(data_root, f"{dataset_id}/Ground-truth-images-val/"),
        "SEG_VAL_DIR": os.path.join(data_root, f"{dataset_id}/Predicted-segs-val/"),
        "DICE_TRAIN_DIR": os.path.join(data_root, f"{dataset_id}/dices_train.pkl"),
        "DICE_VAL_DIR": os.path.join(data_root, f"{dataset_id}/dices_val.pkl"),
        "SUBSET_TRAIN_DIR": subset_train_dir,
        "SUBSET_VAL_DIR": subset_val_dir
    }

def make_subset_dirs(run_id):
    subset_train_dir = f"/home/phuc/my-code/dsc-predict/data-subsets/{run_id}/train/"
    subset_val_dir = f"/home/phuc/my-code/dsc-predict/data-subsets/{run_id}/val/"
    os.makedirs(subset_train_dir)
    os.makedirs(subset_val_dir)
    return subset_train_dir, subset_val_dir

def get_dataloader(data, batch_size, shuffle=True):
    return DataLoader(
        data, batch_size=batch_size, shuffle=shuffle
    )



    



