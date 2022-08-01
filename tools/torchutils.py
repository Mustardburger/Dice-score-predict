import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from collections import OrderedDict

class DiceCoeff(nn.Module):
    def __init__(self, smooth=1e-8):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_true, y_pred):
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = (y_true_f * y_pred_f).sum()
        return (2. * intersection) / (y_true_f.sum() + y_pred_f.sum() + self.smooth)

class ModifiedDiceCoeffMultiLabel(nn.Module):
    def __init__(self, n_classes, start_idx=0):
        super().__init__()
        self.n_classes = n_classes
        self.start_idx = start_idx
        self.n_classes_present = 0
        self.dice_coeff = DiceCoeff()
    
    def forward(self, y_true, y_pred):
        dice = 0
        self.n_classes_present = 0

        for idx in range(self.start_idx, self.n_classes):
            label_present = torch.count_nonzero(y_true[:, idx, :, :]).item()
            if label_present > 0: self.n_classes_present += 1
            dice += self.dice_coeff(y_true[:, idx, :, :], y_pred[:, idx, :, :])

        return dice / self.n_classes_present


class DiceCoeffPerLabel(nn.Module):
    def __init__(self, n_classes, start_idx=0):
        super().__init__()
        self.n_classes = n_classes
        self.start_idx = start_idx
        self.n_classes_present = 0
        self.dice_coeff = DiceCoeff()
    
    def forward(self, y_true, y_pred):
        dice = {}
        self.n_classes_present = 0

        for idx in range(self.start_idx, self.n_classes):
            label_present = torch.count_nonzero(y_true[:, idx, :, :]).item()
            if label_present > 0:
                dice[idx] = self.dice_coeff(y_true[:, idx, :, :], y_pred[:, idx, :, :]).item()
            else:
                label_count_in_pred = torch.count_nonzero(y_pred[:, idx, :, :]).item()
                if label_count_in_pred == 0: dice[idx] = 1
                else: dice[idx] = 0

        return dice


class WholeTumorDice(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_coeff = DiceCoeff()

    def forward(self, y_true, y_pred):
        """
        Convert all non-zero labels in y_true and y_pred to 1
        Here y_true and y_pred has not been one-hot encoded
        """
        y_true_c = torch.clone(y_true)
        y_pred_c = torch.clone(y_pred)

        y_true_c[y_true_c > 0.] = 1
        y_pred_c[y_pred_c > 0.] = 1
        return self.dice_coeff(y_true_c, y_pred_c).item()


def make_one_hot(labels, classes=4):
    one_hot = torch.FloatTensor(labels.size(0), classes, labels.size(2), labels.size(3)).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data.type(torch.int64), 1)
    return target

def create_mask(input_shape, dropped_modalities=None, background=-1.0):
    """
    Create a mask that excludes a modality from the input image
    """
    if not dropped_modalities:
        dropped_modalities = []

    mask_0 = np.ones(input_shape)
    mask_minus1 = np.zeros(input_shape)
    for ind in dropped_modalities:
        mask_0[:, ind, :, :] = np.zeros((input_shape[0], input_shape[2], input_shape[3]))
        mask_minus1[:, ind, :, :] = background*np.ones((input_shape[0], input_shape[2], input_shape[3]))

    return (mask_0, mask_minus1)


def load_without_data_parallel(file_path):
    # original saved file with DataParallel
    state_dict = torch.load(file_path)
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v

    return new_state_dict


def load_seg_to_torch(file_dir):

    seg = Image.open(file_dir)
    truth = np.array(seg) / 85.0
    S = truth.astype(np.float32)[np.newaxis, ...]

    return torch.from_numpy(S.copy())


if __name__ == "__main__":
    a = load_seg_to_torch("/mnt/beegfs/scratch/phuc/datasets/full_2d_dataset/test/Seg/1000.png")
    print(a.size())
