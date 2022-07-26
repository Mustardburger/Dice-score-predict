import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import torch
from sklearn.metrics import r2_score, mean_squared_error
from captum.attr import visualization as viz

from tools.otherutils import process_tensor_for_visualization


def scatterplot_dsc(x, y, name_x, name_y):

    # Some stuff to draw the reference line
    max_val = max(np.amax(x), np.amax(y))
    line = np.linspace(0, max_val, 150)

    # Adding a trendline
    """ z = np.polyfit(x, y, 1)
    print(f"z shape: {z.shape}")
    print(f"x shape: {x}")
    print(f"y shape: {y[0].shape}")
    p = np.poly1d(z)
    yhat = p(x)
    r2 = r2_score(y, yhat) """
    
    plt.figure()
    ax = plt.axes()
    ax.scatter(x, y, c="red")
    ax.plot(line, line, "k--")
    # ax.plot(x, yhat, "b--")
    # ax.text(0.1, 0.8, f"$r^2={r2}$", fontsize=15)
    ax.set_xlabel(name_x)
    ax.set_ylabel(name_y)
    plt.savefig("true-vs-pred.png")

def scatterplot_dsc_with_mask(true_dsc_dict, pred_dsc_dict, mask_list):
    """
    Create a scatterplot of true-pred dsc with different masks
    """
    mask_ind = {0: "T2", 1: "T1c", 2: "Flair"}
    plt.figure()
    if len(mask_list) % 2 == 0: ncols = len(mask_list) // 2
    else: ncols = (len(mask_list) // 2) + 1
    fig, axe = plt.subplots(nrows=2, ncols=ncols, figsize=(15, 8))
    for i in range(len(mask_list)):
        true_dsc = true_dsc_dict[i]
        pred_dsc = pred_dsc_dict[i]
        mse = mean_squared_error(true_dsc, pred_dsc)
        if (ncols>1):
            axe[i%2][i//2].scatter(true_dsc, pred_dsc, s=5)
            if len(mask_list[i]) == 0: title = "Nothing"
            else:
                title = ""
                for m in mask_list[i]:
                    title = title + mask_ind[m] + ","
                title = title[:-1]
            axe[i%2][i//2].set_title(f"{title} masked, MSE = {mse:.3f}", fontsize=15)
        else:
            axe[i%2 + i//2].scatter(true_dsc, pred_dsc, s=5)
            axe[i%2 + i//2].set_title(f"Ind {mask_list[i]} masked, MSE = {mse:.3f}", fontsize=15)

    if len(mask_list) % 2 != 0: 
        if ncols>1: plt.delaxes(axe[1][ncols-1])
        else: plt.delaxes(axe[-1])
    fig.supxlabel("True DSC")
    fig.supylabel("Pred DSC")
    plt.tight_layout()
    plt.savefig("Mask.png")


def scatterplot_dscerror_tumorvol(true_dsc, pred_dsc, tumor_vol):
    name_label = {
        -1.0: "Whole tumor", 1.0: "Non-enhancing", 2.0: "Edema", 3.0: "Enhancing"
    }
    plt.figure()
    err = np.array([
        np.absolute(tr - pr) for tr, pr in zip(true_dsc, pred_dsc)
    ])
    fig, axe = plt.subplots(nrows=2, ncols=2)
    for i, label in enumerate(tumor_vol.keys()):
        axe[i//2][i%2].scatter(err, tumor_vol[label], c="red", s=5)
        axe[i//2][i%2].set_title(name_label[label])
    
    fig.supxlabel("Error of DSC")
    fig.supylabel("Volume ratio of tumor")
    plt.tight_layout()
    plt.savefig("tumor-volume.png")

def histogram_dsc(true, pred):
    """
    Plot the histogram of the L1 errors between true and pred dices
    """
    err = np.array([
        abs(tr - pr) for tr, pr in zip(true, pred)
    ])
    plt.figure()
    sns.histplot(data=err, stat="percent")
    plt.title("Distribution of DSC errors")
    plt.ylabel("Percentage")
    plt.xlabel("Difference of true and predicted DSC")
    plt.savefig(f"hist-error.png")

def box_and_hist(dsc):
    """
    Plot the box plot for predicted DSC
    If we want the histplot to lie on the y-axis of the boxplot,
    check out this link: https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html#sphx-glr-gallery-lines-bars-and-markers-scatter-hist-py
    """
    plt.figure()
    fig, axe = plt.subplots(nrows=1, ncols=2)
    new_dsc = []
    for d in dsc:
        try:
            if len(d) != 0: new_dsc.extend(list(d))
        except:
            new_dsc.append(d)

    axe[0].boxplot(new_dsc)
    axe[0].set_ylabel("Pred DSC")
    sns.histplot(data=new_dsc, ax=axe[1])
    axe[1].set_xlabel("Pred DSC")
    plt.suptitle("Distribution of DSC scores using true segs")
    plt.tight_layout()
    plt.savefig("DSC-for-true-segs.png")

def visualize_val_results(params, renormalize=False):
    """
    params is a dict with the following key-value pairs:
        "img": img_list
        "seg": seg_list
        "true_dice": true_dice_list
        "pred_dice": pred_dice_list
    """
    fig = plt.figure(figsize=(15, 15))

    if len(params["pred_dice"]) % 2 == 0: ncols = len(params["pred_dice"]) // 2
    else: ncols = (len(params["pred_dice"]) // 2) + 1
    subfigs = fig.subfigures(2, ncols)

    for outerind, subfig in enumerate(subfigs.flat):
        subfig.suptitle(f'Pred DSC: {params["pred_dice"][outerind]:.3f}\nTrue DSC: {params["true_dice"][outerind]:.3f}')
        axs = subfig.subplots(2, 1)
        for innerind, ax in enumerate(axs.flat):
            if innerind % 2 == 0:
                img = params["img"][outerind]
                if img.shape[-1] != 1 or img.shape[-1] != 3: img = np.swapaxes(img, 0, -1)
                if renormalize: 
                    img = np.expand_dims(np.mean(((img / 2) + 0.5), axis=2), -1)
                    ax.imshow(img, cmap="gray")
                else:
                    ax.imshow(img)
            else:
                seg = params["seg"][outerind]
                if seg.shape[-1] != 1 or seg.shape[-1] != 3: seg = np.swapaxes(seg, 0, -1)
                ax.imshow(seg)
            ax.set_axis_off()
        subfig.subplots_adjust(top=0.95)
    
    fig.tight_layout()
    plt.savefig("segs.png")


def plot_shift_error(err_hori_l, err_verti_l, shift_schedule, img, seg):

    fig, axe = plt.subplots(nrows=2, ncols=2)

    # Plot the images
    if len(img.size())==4: img = torch.squeeze(img, dim=0)
    if len(seg.size())==4: seg = torch.squeeze(seg, dim=0)
    img = img.detach().cpu().numpy()
    seg = seg.detach().cpu().numpy()
    img = np.swapaxes(img, 0, -1)
    seg = np.swapaxes(seg, 0, -1)
    # Normalize the image if necessary
    if np.amax(img) <= 1.0 and np.amin(img) < 0.0: img = np.expand_dims(np.mean(((img / 2) + 0.5), axis=2), -1)
    else: img = np.expand_dims(np.mean(img, axis=2), -1)
    axe[0][0].imshow(img, cmap="gray")
    axe[0][0].set_title("MRI image")
    axe[0][0].set_axis_off()
    axe[0][1].imshow(seg)
    axe[0][1].set_title("Segmentation")
    axe[0][1].set_axis_off()

    # Plot the errors
    axe[1][0].plot(shift_schedule, err_hori_l)
    axe[1][0].set_title("Error of DSC horizontal shift")
    axe[1][0].set_ylabel("True DSC - Pred DSC")
    axe[1][1].plot(shift_schedule, err_verti_l)
    axe[1][1].set_title("Error of DSC vertical shift")
    fig.supxlabel("Amount of shift in pixels")
    fig.tight_layout()
    plt.savefig("Shift.png")


def plot_multiple_shift_errors(err_hori_l, err_verti_l, shift_schedule):

    fig, axe = plt.subplots(nrows=1, ncols=2)
    for i in range(len(err_hori_l)):
        err_hori_residual = [abs(err - err_hori_l[i][0]) for err in err_hori_l[i]]
        err_verti_residual = [abs(err - err_verti_l[i][0]) for err in err_verti_l[i]]
        axe[0].plot(shift_schedule, err_hori_residual)
        axe[1].plot(shift_schedule, err_verti_residual)

    axe[0].set_title("Horizontal shifts")
    axe[1].set_title("Vertical shifts")
    axe[0].set_xlabel("Amount of shifts in pixels")
    axe[1].set_xlabel("Amount of shifts in pixels")
    axe[0].set_ylabel("True DSC - Pred DSC")

    fig.tight_layout()
    plt.savefig("Multiple-shift.png")


def label_swap_viz(img, seg_list, dice_list):

    num_subplots = len(dice_list) + 1
    if num_subplots % 2 == 0: ncols = num_subplots // 2
    else: ncols = (num_subplots // 2) + 1
    fig, axe = plt.subplots(nrows=2, ncols=ncols)
    for i in range(num_subplots):
        cmap = None
        if (i == 0): 
            curr_img = process_tensor_for_visualization(img)
            title = "MRI image"
            cmap = "gray"
        elif (i == 1):
            curr_img = process_tensor_for_visualization(seg_list[i-1])
            title = f"Orig, DSC = {dice_list[i-1]:.2f}"
        else:
            curr_img = process_tensor_for_visualization(seg_list[i-1])
            title = f"Swap, DSC = {dice_list[i-1]:.2f}"

        if cmap:
            axe[i%2][i//2].imshow(curr_img, cmap=cmap)
        else:
            axe[i%2][i//2].imshow(curr_img)

        axe[i%2][i//2].set_title(title)
        axe[i%2][i//2].set_axis_off()

    if num_subplots % 2 != 0: plt.delaxes(axe[1][-1])

    fig.tight_layout()
    plt.savefig("Swap-classes.png")


def show_feature_importance(attribution, img, seg, channel):
    
    attribution = process_tensor_for_visualization(attribution)
    img = process_tensor_for_visualization(img)
    seg = process_tensor_for_visualization(seg)

    fig, axes = plt.subplots(nrows=2, ncols=2)
    axes[0][0].imshow(img, cmap="gray")
    axes[0][1].imshow(seg)
    axes[0][0].set_axis_off()
    axes[0][1].set_axis_off()
    if (channel != -1): attribution = np.expand_dims(attribution[..., channel], axis=-1)
    viz.visualize_image_attr(attribution, img, method='heat_map', show_colorbar=True, sign='positive', outlier_perc=1, plt_fig_axis=(fig, axes[1][0]))
    plt.delaxes(axes[1][-1])
    plt.tight_layout()
    plt.savefig("shap.png")

def tumor_reconstruction(attribution, img, seg, thred_list):

    attribution = process_tensor_for_visualization(attribution, special=True)
    img = process_tensor_for_visualization(img)
    seg = process_tensor_for_visualization(seg)

    print(np.amax(attribution))
    print(np.amin(attribution))

    attr_list = []
    for t in thred_list:
        attr_copy = np.copy(attribution)
        attr_copy[attr_copy > t] = 1
        #attr_copy[attr_copy < 0] = 0.5
        attr_copy[attr_copy <= t] = 0
        attr_list.append(attr_copy)
    
    n_axes = 2 + len(thred_list)
    if n_axes % 2 == 0: ncols = n_axes // 2
    else: ncols = (n_axes // 2) + 1
    fig, axes = plt.subplots(nrows=2, ncols=ncols)
    axes[0][0].imshow(img, cmap="gray")
    axes[0][0].set_axis_off()
    axes[1][0].imshow(seg)
    axes[1][0].set_axis_off()

    print(np.unique(attr_list[0]))

    for i in range(n_axes - 2):
        axes[i%2][(i//2)+1].imshow(attr_list[i], cmap="gray")
        axes[i%2][(i//2)+1].set_title(f"Thred = {thred_list[i]}")
        axes[i%2][(i//2)+1].set_axis_off()

    if n_axes % 2 != 0: plt.delaxes(axes[1][-1])

    plt.tight_layout()
    plt.savefig("tumor-reconstruct.png")

        
if __name__ == "__main__":
    pass