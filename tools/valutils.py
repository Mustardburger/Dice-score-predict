import torch
import numpy as np
import os
import pickle
import shap
from tqdm import tqdm
from captum.attr import GradientShap

from tools.otherutils import *
from tools.vizutils import *
from tools.torchutils import *
from dataset import FinetuneDiceDataset, get_dataloader


def shift_experiments(dir_dict, num_images, shift_schedule, torch_stuff):
    """
    In this experiment, the segmentation is shifted left and right, up and down, for some pixels
    The change of DSC is recorded
    """
    # Read the images and do stuff
    name_files = np.random.choice(os.listdir(dir_dict["VAL_SEG_DIR"]), num_images)
    model = torch_stuff["model"]
    err_hori_l, err_verti_l = [], []
    with open(dir_dict["DICE_DIR"], "rb") as f:
        dice_dict = pickle.load(f)

    shifted_list = []
    dices = []
    mri = None
    true_seg, ori_seg = None, None

    model.eval()
    with torch.no_grad():
        for name_file in name_files:
            ehl, evl = [], []
            epoch, id = name_file.split("_")[0][5:], name_file.split("_")[1][3:-3]
            name_img = "Epoch" + epoch + "_Img" + id + ".pt"
            name_dice = "Epoch" + epoch + "-" + id
            true_dsc = dice_dict[name_dice]
            dices.append(true_dsc)
            seg_full_dir = os.path.join(dir_dict["VAL_SEG_DIR"], name_file)
            true_full_dir = os.path.join(dir_dict["VAL_TRUE_DIR"], name_file)
            img_full_dir = os.path.join(dir_dict["VAL_MRI_DIR"], name_img)
        
            seg = torch.load(seg_full_dir).to(torch_stuff["device"])
            img = torch.load(img_full_dir).to(torch_stuff["device"])
            true = torch.load(true_full_dir).to(torch_stuff["device"])

            if torch_stuff["data_transform"]:
                img = (img * 0.5) + 0.5
                img = torch_stuff["data_transform"](img)

            if len(img.size()) == 3: img = torch.unsqueeze(img, dim=0)
            if len(seg.size()) == 3: seg = torch.unsqueeze(seg, dim=0)
            
            mri = img
            ori_seg = seg
            true_seg = true
            
            for roll in tqdm(shift_schedule):
                seg_rolled_hori = torch.roll(seg, roll, 2)
                seg_rolled_verti = torch.roll(seg, roll, 3)

                input_hori = torch.cat((img, seg_rolled_hori), dim=1).float()
                input_verti = torch.cat((img, seg_rolled_verti), dim=1).float()
                pred_hori = torch_stuff["model"](input_hori)[0].item()
                pred_verti = torch_stuff["model"](input_verti)[0].item()
                #err_hori_l.append(abs(pred_hori - true_dsc))
                #err_verti_l.append(abs(pred_verti - true_dsc))
                shifted_list.append(seg_rolled_hori)
                shifted_list.append(seg_rolled_verti)
                dices.append(pred_hori)
                dices.append(pred_verti)
                ehl.append(abs(pred_hori - true_dsc))
                evl.append(abs(pred_verti - true_dsc))
            err_hori_l.append(ehl)
            err_verti_l.append(evl)

    #plot_multiple_shift_errors(err_hori_l, err_verti_l, shift_schedule)
    #plot_shift_error(err_hori_l, err_verti_l, shift_schedule, img, seg)
    plot_shift_segs(mri, true_seg, ori_seg, shifted_list, dices)


def swap_label_experiment(label_swap_dict, data_dir, torch_stuff):
    """
    In this experiment, the tumor label of the segmentation is flipped
    The change of DSC is recorded
    """
    # Get the directory of the images
    name_seg = np.random.choice(os.listdir(data_dir["VAL_SEG_DIR"]))
    epoch, id = name_seg.split("_")[0][5:], name_seg.split("_")[1][3:-3]
    name_img = "Epoch" + epoch + "_Img" + id + ".pt"
    seg_full_dir = os.path.join(data_dir["VAL_SEG_DIR"], name_seg)
    img_full_dir = os.path.join(data_dir["VAL_MRI_DIR"], name_img)

    seg = torch.load(seg_full_dir).to(torch_stuff["device"])
    img = torch.load(img_full_dir).to(torch_stuff["device"])
    seg_list = []
    original_seg = seg.detach().clone()
    seg_list.append(original_seg)
    dice_list = []

    model = torch_stuff["model"]

    for label_swapped, label_new_list in label_swap_dict.items():
        for label_new in label_new_list:
            seg = original_seg.clone()
            #print(f"Before swapped: {torch.unique(seg)}")

            # Swap the labels
            if label_swapped in seg:
                #print(label_swapped)
                seg[seg == label_swapped] = label_new
                seg_list.append(seg)

    if torch_stuff["data_transform"]:
        img = (img * 0.5) + 0.5
        print(f"max {torch.max(img)} min {torch.min(img)}")

    model.eval()
    with torch.no_grad():

        for seg in seg_list:

            if len(img.size()) == 3: img = torch.unsqueeze(img, dim=0)
            if len(seg.size()) == 3: seg = torch.unsqueeze(seg, dim=0)
            
            print(f"After swapped: {torch.unique(seg)}")

            input_tensor = torch.cat((img, seg), dim=1).float()
            pred_dice = torch_stuff["model"](input_tensor)[0].item()
            dice_list.append(pred_dice)

    label_swap_viz(img, seg_list, dice_list)


def true_vs_pred_dsc(data_dirs, torch_stuff, hyperparams, viz="scatter", real_seg=False):
    """
    Some experiments to show correlation between true and pred dsc
    viz="scatter" means generating a scatterplot
    viz="err" means generating an error hist
    """

    true_dice_list, pred_dice_list = [], []
    model = torch_stuff["model"]
    device = torch_stuff["device"]
    data_transform = torch_stuff["data_transform"]

    VAL_MRI_DIR, VAL_SEG_DIR, DICE_DIR = data_dirs["VAL_MRI_DIR"], data_dirs["VAL_SEG_DIR"], data_dirs["DICE_DIR"]
    BATCH_SIZE = hyperparams["batch_size"]
    val_data = FinetuneDiceDataset(VAL_MRI_DIR, VAL_SEG_DIR, DICE_DIR, img_transform=data_transform)
    val_dataloader = get_dataloader(val_data, BATCH_SIZE)

    model.eval()
    with torch.no_grad():
        for count, data in enumerate(tqdm(val_dataloader)):
            imgs, segs, dices = data["img"].to(device), data["seg"].to(device), data["dsc"].to(device)
            input_tensor = torch.cat((imgs, segs), dim=1).float()
            prediced_dsc = model(input_tensor)

            for i in range(len(dices)):
                if i < len(dices) and i < len(prediced_dsc):
                    true_dice_list.append(dices[i].item())
                    pred_dice_list.append(prediced_dsc[i].item())

    if real_seg:
        box_and_hist(pred_dice_list)
    else:
        if viz == "scatter":
            scatterplot_dsc(true_dice_list, pred_dice_list, "True DSC", "Pred DSC")
        elif viz == "err":
            histogram_dsc(true_dice_list, pred_dice_list)
        else:
            print("Unknown argument")


def use_mask(data_dirs, torch_stuff, hyperparams, masks):
    """
    In this experiment, some of the modalities are masked out (set equal to background)
    The scatterplot is drawn for each masked modality
    """
    true_dice_dict, pred_dice_dict =  {i: [] for i in range(len(masks))}, {i: [] for i in range(len(masks))}
    model = torch_stuff["model"]
    device = torch_stuff["device"]
    data_transform = torch_stuff["data_transform"]

    VAL_MRI_DIR, VAL_SEG_DIR, DICE_DIR = data_dirs["VAL_MRI_DIR"], data_dirs["VAL_SEG_DIR"], data_dirs["DICE_DIR"]
    BATCH_SIZE = hyperparams["batch_size"]
    val_data = FinetuneDiceDataset(VAL_MRI_DIR, VAL_SEG_DIR, DICE_DIR, img_transform=data_transform)
    val_dataloader = get_dataloader(val_data, BATCH_SIZE)

    model.eval()
    with torch.no_grad():
        for count, data in enumerate(tqdm(val_dataloader)):
            imgs, segs, dices = data["img"].to(device), data["seg"].to(device), data["dsc"].to(device)
            dices = dices.float()

            for ind, mask_id in enumerate(masks):
                mask0, mask_minus1 = create_mask(imgs.size(), mask_id)
                mask0, mask_minus1 = torch.from_numpy(mask0).to(device), torch.from_numpy(mask_minus1).to(device)
                imgs_masked = torch.mul(imgs, mask0)
                input_tensor = torch.cat((imgs_masked, segs), dim=1).float()
                predicted_dsc = model(input_tensor)

                for i in range(len(dices)):

                    if i < len(predicted_dsc):
                        true_dice_dict[ind].append(dices[i].item())
                        pred_dice_dict[ind].append(predicted_dsc[i].item())

    scatterplot_dsc_with_mask(true_dice_dict, pred_dice_dict, masks)


def visualize_mri_and_seg(data_dirs, torch_stuff, hyperparams, err_thred):
    """
    Visualize the mri and segmentations of certain cases in which the difference between
    the true and predicted seg is greater than err_thred
    """
    
    true_dice_list, pred_dice_list, img_list, seg_list = [], [], [], []
    model = torch_stuff["model"]
    device = torch_stuff["device"]
    data_transform = torch_stuff["data_transform"]

    VAL_MRI_DIR, VAL_SEG_DIR, DICE_DIR = data_dirs["VAL_MRI_DIR"], data_dirs["VAL_SEG_DIR"], data_dirs["DICE_DIR"]
    BATCH_SIZE = hyperparams["batch_size"]
    val_data = FinetuneDiceDataset(VAL_MRI_DIR, VAL_SEG_DIR, DICE_DIR, img_transform=data_transform)
    val_dataloader = get_dataloader(val_data, BATCH_SIZE)

    model.eval()
    with torch.no_grad():
        for count, data in enumerate(tqdm(val_dataloader)):
            imgs, segs, dices = data["img"].to(device), data["seg"].to(device), data["dsc"].to(device)
            input_tensor = torch.cat((imgs, segs), dim=1).float()
            prediced_dsc = model(input_tensor)
            dices = dices.float()

            for i in range(len(dices)):
                dice, pred_dice = dices[i].item(), prediced_dsc[i].item()
                if abs(dice - pred_dice) >= err_thred and i < len(prediced_dsc):
                    img_list.append(imgs[i, :, :, :].detach().cpu().numpy())
                    seg_list.append(segs[i, :, :, :].detach().cpu().numpy())
                    true_dice_list.append(dice)
                    pred_dice_list.append(pred_dice)

    print(len(true_dice_list))
    print(len(pred_dice_list))
    visualize_val_results({
        "img": img_list[:12], "seg": seg_list[:12], "true_dice": true_dice_list[:12], "pred_dice": pred_dice_list[:12]
    }, True)


def tumor_volume(data_dirs, torch_stuff, hyperparams):
    """
    Scatterplot shows the relationship between the tumor volume and the difference between
    true and predicted DSC
    """
    
    true_dice_list, pred_dice_list, = [], []
    model = torch_stuff["model"]
    device = torch_stuff["device"]
    data_transform = torch_stuff["data_transform"]

    VAL_MRI_DIR, VAL_SEG_DIR, DICE_DIR = data_dirs["VAL_MRI_DIR"], data_dirs["VAL_SEG_DIR"], data_dirs["DICE_DIR"]
    BATCH_SIZE = hyperparams["batch_size"]
    val_data = FinetuneDiceDataset(VAL_MRI_DIR, VAL_SEG_DIR, DICE_DIR, img_transform=data_transform)
    val_dataloader = get_dataloader(val_data, BATCH_SIZE)
    
    label_intensity = {
        -1.0: [], 1.0: [], 2.0: [], 3.0: []
    }

    model.eval()
    with torch.no_grad():
        for count, data in enumerate(tqdm(val_dataloader)):
            imgs, segs, dices = data["img"].to(device), data["seg"].to(device), data["dsc"].to(device)
            input_tensor = torch.cat((imgs, segs), dim=1).float()
            prediced_dsc = model(input_tensor)
            dices = dices.float()
            for i in range(len(dices)):
                try:
                    seg_slice = segs[i, :, :, :].detach().cpu()
                    true_dice_list.append(dices[i].detach().cpu().numpy())
                    pred_dice_list.append(prediced_dsc[i].detach().cpu().numpy())
                    total_vol = torch.numel(seg_slice)
                    whole_tumor_vol = torch.count_nonzero(seg_slice) / total_vol
                    label_intensity[-1.0].append((whole_tumor_vol).item())
                    tumor_classes, tumor_classes_count = torch.unique(seg_slice, return_counts=True)
                    lost_classes = []
                    for clas in label_intensity.keys():
                        if clas not in tumor_classes and clas != -1.0:
                            label_intensity[clas].append(0.0)
                    for clas, clas_count in zip(tumor_classes, tumor_classes_count):
                        clas = clas.item()
                        if clas != 0.0: 
                            #print(label_intensity[clas])
                            label_intensity[clas].append((clas_count / total_vol).item())
                except:
                    pass

    scatterplot_dscerror_tumorvol(true_dice_list, pred_dice_list, label_intensity)


def feature_importance(data_dirs, torch_stuff, hyperparams, num_background=100, num_test=3, channel=-1, n_samples=10, stdev=0.0001, thred_list=[0.2, 0.5, 0.8]):
    """
    Attempting to explain what the model is seeing when making predictions
    """
    device = torch_stuff["device"]
    model = torch_stuff["model"]
    data_transform = torch_stuff["data_transform"]

    VAL_MRI_DIR, VAL_SEG_DIR, DICE_DIR = data_dirs["VAL_MRI_DIR"], data_dirs["VAL_SEG_DIR"], data_dirs["DICE_DIR"]
    BATCH_SIZE = hyperparams["batch_size"]
    val_data = FinetuneDiceDataset(VAL_MRI_DIR, VAL_SEG_DIR, DICE_DIR, img_transform=data_transform)
    val_dataloader = get_dataloader(val_data, BATCH_SIZE)

    gradient_shap = GradientShap(model)
    img, seg = None, None
    stop = False

    for count, data in enumerate(val_dataloader):

        if (count)*BATCH_SIZE == num_background: break
        elif (count+1)*BATCH_SIZE > num_background:
            left = num_background - (count*BATCH_SIZE)
            img_slice, seg_slice = data["img"][:left, ...], data["seg"][:left, ...]
            stop = True
        else:
            img_slice, seg_slice = data["img"], data["seg"]

        if img is None: img = img_slice
        else: img = torch.cat((img, img_slice), dim=0)

        if seg is None: seg = seg_slice
        else: seg = torch.cat((seg, seg_slice), dim=0)

        if stop: break

    baseline = torch.cat((img, seg), dim=1).to(device)

    test_sample = next(iter(val_dataloader))
    img, seg = torch.unsqueeze(test_sample["img"][0], dim=0), torch.unsqueeze(test_sample["seg"][0], dim=0)
    test = torch.cat((img, seg), dim=1).to(device)

    attribution = gradient_shap.attribute(test, baseline, n_samples=n_samples, stdevs=stdev)

    #show_feature_importance(attribution, img, seg, channel)
    tumor_reconstruction(attribution, img, seg, thred_list)


