import torch
import os
import pickle
import argparse
import tqdm
import numpy as np
from torchvision.transforms import RandomAffine

from dataset import SegDataset
from tools.torchutils import make_one_hot, load_without_data_parallel, load_seg_to_torch
from tools.torchutils import DiceCoeffPerLabel, WholeTumorDice
from models.unet import Modified2DUNet

class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.hyp = None

    def read_arguments(self):
        self.parser.add_argument("--aug_rate", type=float, help="The probability a seg is augmented")

    def get_hyperparameters(self):
        self.read_arguments()
        self.hyp = self.parser.parse_known_args()[0]
        return self.hyp


class DataAugmentation:

    def __init__(self, verti_shift, hori_shift, affine):
        self.verti_shift = verti_shift
        self.hori_shift = hori_shift
        self.affine = affine

    def _determine_tumor_border(self, start_ind, seg, dim, increment=1):

        if dim == 0: slice = seg[start_ind, :]
        else: slice = seg[:, start_ind]
        count = start_ind
        while not(torch.any(slice > 0).item()):
            count += increment
            if dim == 0: slice = seg[count, :]
            else: slice = seg[:, count]
        
        return count

    def add_vertical_translating_segs(self, orig_seg, margin=20, roll_mode="2exp"):
        """
        Translating the tumor to a new place vertically
        margin denotes 
        orig_seg is a torch tensor
        """
        if roll_mode == "2exp":
            roll_schedule = 2 ** np.arange(7)
        else:
            roll_schedule = np.arange(5, 75, 10)

        temp = orig_seg.squeeze()
        roll_seg_list = []

        nrows = temp.size(0)
        border_top = self._determine_tumor_border(0, temp, 0, 1)
        border_bottom = self._determine_tumor_border(nrows-1, temp, 0, -1)

        for r in roll_schedule:
            # roll down
            if (border_bottom + r) < nrows-1:
                new_seg = torch.roll(temp, r, 0)
                roll_seg_list.append(new_seg)

            # roll up
            if (border_top - r) > 0:
                new_seg = torch.roll(temp, -r, 0)
                roll_seg_list.append(new_seg)

        return roll_seg_list


    def add_horizontal_translating_segs(self, orig_seg, margin=20, roll_mode="2exp"):
        """
        Translating the tumor to a new place horizontally
        margin denotes 
        orig_seg is a torch tensor
        """
        if roll_mode == "2exp":
            roll_schedule = 2 ** np.arange(7)
        else:
            roll_schedule = np.arange(5, 75, 10)
        temp = orig_seg.squeeze()
        roll_seg_list = []

        ncols = temp.size(1)
        border_top = self._determine_tumor_border(0, temp, 1, 1)
        border_bottom = self._determine_tumor_border(ncols-1, temp, 1, -1)

        for r in roll_schedule:
            # roll down
            if (border_bottom + r) < ncols-1:
                new_seg = torch.roll(temp, r, 1)
                print(f"Orig size: {orig_seg.size()}")
                new_seg = new_seg.view(orig_seg.size())
                roll_seg_list.append(new_seg)

            # roll up
            if (border_top - r) > 0:
                new_seg = torch.roll(temp, -r, 1)
                print(f"Orig size: {orig_seg.size()}")
                new_seg = new_seg.view(orig_seg.size())
                roll_seg_list.append(new_seg)

        return roll_seg_list


    def random_affine(self, orig_seg, num_output=4, degree_range=(30, 60), scale=(0.7, 0.9)):
        affine_transformer = RandomAffine(degrees=degree_range, scale=scale)
        affine_imgs = [affine_transformer(orig_seg) for _ in range(num_output)]
        return affine_imgs

    def data_augmentation(self, orig_seg, vert_shift=False, hori_shift=False, affine=False):
        """
        Perform data augmentations to create more diverse segmentations
        """
        seg_aug_list = []
        if vert_shift:
            seg_aug_list.extend(self.add_vertical_translating_segs(orig_seg))
        if hori_shift:
            seg_aug_list.extend(self.add_horizontal_translating_segs(orig_seg))
        if affine:
            seg_aug_list.extend(self.random_affine(orig_seg, 5))

        return seg_aug_list

    


def save_seg(seg, seg_name, save_dir):
    """
    Save the segmentations
    """
    seg = seg.detach().cpu()
    torch.save(seg, os.path.join(save_dir, seg_name))


def dice_dist_for_one_model_for_one_data(data_dir, model_dir, save_dir, modalities=['T2', 'T1', 'T1c', 'Flair', 'Seg'], kernel_size=3):
    """
    Return the Dice distribution for this model
    """
    data_aug = DataAugmentation(True, True, True)
    parser = Parser()
    par_args = parser.get_hyperparameters()

    device = torch.device(f"cuda:0" if torch.cuda.is_available() else 'cpu')
    full_model_dir = os.path.join(model_dir, os.path.join("netG", "ep_final.pt"))
    model = Modified2DUNet(4, 4, 8, kernel_size=kernel_size).to(device)
    model_weights = load_without_data_parallel(full_model_dir)
    model.load_state_dict(model_weights)

    # Some model is not trained on every modality
    per_label_dice = DiceCoeffPerLabel(4, 1)
    whole_tumor_dice = WholeTumorDice()
    dice_dict_dict = {}

    dataloader = SegDataset(data_dir, modalities, batch_size=1).get_train_dataloader(shuffle=False)

    with torch.no_grad():
        for data in dataloader:
            source, target, name = data["B"].to(device), data["S"].to(device), data["id"][0]
            probs = model(source)
            pred = probs.argmax(dim=1)

            # applying data aug for the predictions
            augment_or_not = np.random.rand()
            if augment_or_not <= par_args.aug_rate:

                augmented_preds = [pred]
                augmented_preds.extend(data_aug.data_augmentation(pred, True, True, True))

                for i, aug_pred in enumerate(augmented_preds):
                    pre1hot = make_one_hot(aug_pred[:, None, :, :], probs.size(1))
                    target1hot = make_one_hot(target, probs.size(1))

                    # Save the segs
                    #save_seg(pred, name, save_dir)

                    dice_dict = per_label_dice(pre1hot, target1hot)
                    wt_dice = whole_tumor_dice(aug_pred, target)
                    dice_dict[0] = wt_dice
                    dice_dict_dict[f"{name}.{i}"] = dice_dict

            elif par_args.aug_rate <= 0.01:
                pre1hot = make_one_hot(pred[:, None, :, :], probs.size(1))
                target1hot = make_one_hot(target, probs.size(1))

                # Save the segs
                #save_seg(pred, name, save_dir)

                dice_dict = per_label_dice(pre1hot, target1hot)
                wt_dice = whole_tumor_dice(pred, target)
                dice_dict[0] = wt_dice
                dice_dict_dict[name] = dice_dict

    return dice_dict_dict    


def add_ground_truth(data_dir, name_file):
    """
    Add the ground truth seg to the model
    """
    true_seg_dir = os.path.join(os.path.join(data_dir, "Seg"), name_file)
    seg = load_seg_to_torch(true_seg_dir)
    return seg

def dice_dist_for_one_model(model_dir, model_name, save_dir, data_base_dir, modalities=['T2', 'T1', 'T1c', 'Flair', 'Seg'], kernel_size=3):
    # When generating data, a lot of things to consider:
        # Dice balance for whole tumor and each type of class (across all models)
        # Select from the dist to balance out the Dice

    # Data augmentation:
        # Add in real segmentations
        # Manually shift the segmentations
        # Add in slices with no tumors (how??) (maybe forget about this and see how performance hurts on 3D data)

    # generate histogram for each model
    full_save_dir = os.path.join(save_dir, model_name)
    os.makedirs(full_save_dir)

    data_list = os.listdir(data_base_dir)
    for di in data_list:
        print(f"    Working on dataset {di}")
        seg_save_dir = os.path.join(full_save_dir, di)
        os.makedirs(seg_save_dir)
        data_dir = os.path.join(data_base_dir, di)
        dice_dict = dice_dist_for_one_model_for_one_data(data_dir, model_dir, seg_save_dir, modalities=modalities, kernel_size=kernel_size)
        with open(os.path.join(full_save_dir, f"{di}_dice_list.pkl"), "wb") as f:
            pickle.dump(dice_dict, f)


if __name__ == "__main__":
    base_model_dir = "/mnt/beegfs/scratch/phuc/seg-quality-control/multiple-unet-exp/experiment-2/unet"
    save_dir = "/mnt/beegfs/scratch/phuc/seg-quality-control/multiple-unet-exp/experiment-2/unet-results-with-aug"
    data_dir = "/mnt/beegfs/scratch/phuc/datasets/full_2d_dataset"
    
    modal_dist = {
        "exp_6": ['null', 'T1', 'T1c', 'Flair', 'Seg'],
        "exp_7": ['T2', 'null', 'T1c', 'Flair', 'Seg'],
        "exp_8": ['T2', 'T1', 'null', 'Flair', 'Seg'],
        "exp_9": ['T2', 'T1', 'T1c', 'null', 'Seg'],
    }
    kernel_size_dict = {
        "exp_0": 3, "exp_1": 5, "exp_2": 7, "exp_3": 3, "exp_4": 7, "exp_5": 3, "exp_6": 3,
        "exp_7": 3, "exp_8": 3, "exp_9": 3, "exp_10": 5
    }

    model_dirs = sorted(os.listdir(base_model_dir))
    for m in model_dirs:
        print(f"Working on model {m}")
        model_dir = os.path.join(base_model_dir, m)
        if m in modal_dist:
            modalities = modal_dist[m]
        else:
            modalities = ['T2', 'T1', 'T1c', 'Flair', 'Seg']
        
        dice_dist_for_one_model(model_dir, m, save_dir, data_dir, modalities=modalities, kernel_size=kernel_size_dict[m])
    
    """
    # Test the data augmentation method
    true_seg_dir = "/mnt/beegfs/scratch/phuc/datasets/full_2d_dataset/train/Seg/10006.png"
    true_seg = load_seg_to_torch(true_seg_dir)
    print(true_seg.size())

    data_aug = data_augmentation(true_seg, True, True, True)
    print("Oke")

    
    dataloader = SegDataset(data_dir + "/test", ['T2', 'T1', 'T1c', 'Flair', 'Seg'], batch_size=1).get_train_dataloader(shuffle=False)
    seg = next(iter(dataloader))["S"]
    print(seg.size())
    """

        




    

