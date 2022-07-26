import os
from matplotlib.pyplot import get
import torch
import pickle
import torch.nn as nn
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import transforms
import random
import numpy as np
from tqdm import tqdm
from data_transforms import DataTransform

from models.cnn import get_pretrained_resnet, save_network_config
from tools.lossutils import MSE
from dataset import DiceDataset, FinetuneDiceDataset, get_dataloader, get_dataset_dirs, make_subset_dirs
from tools.otherutils import create_folder_for_run
from tools.torchutils import *
from hyperparams import Hyperparameters

if __name__ == "__main__":
    
    # Let's see how it goes
    # Some constants (which will be put in parser later)
    parser = Hyperparameters()
    parser_args = parser.get_hyperparameters()
    dir_dict = get_dataset_dirs(parser_args.data_root, parser_args.data_id, parser.run_id)

    """
    BATCH_SIZE = 16
    DATA_TRAIN_RATIO = 0.35
    DATA_VAL_RATIO = 0.3
    NUM_IN_CHANNELS = 4
    LR = 0.0002
    BETA_1 = 0.5
    NUM_EPOCH = 150
    MODALITY_DROPOUT_RATE = 1.0
    MODALITY_DROPPED = [0, 1]
    """

    # Random seed for deterministic setting 
    seed = 999
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Connect to device
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else 'cpu')

    # Since the pytorch ResNet requires correct normalization of the data, here are some transforms
    data_transform = DataTransform().get(parser_args.data_transform)

    # Load training and val datasets
    training_data = FinetuneDiceDataset(dir_dict["IMG_TRAIN_DIR"], dir_dict["SEG_TRAIN_DIR"], dir_dict["DICE_TRAIN_DIR"], img_transform=data_transform, ratio=parser_args.data_train_ratio, subset_dir=dir_dict["SUBSET_TRAIN_DIR"])
    val_data = FinetuneDiceDataset(dir_dict["IMG_VAL_DIR"], dir_dict["SEG_VAL_DIR"], dir_dict["DICE_VAL_DIR"], img_transform=data_transform, ratio=parser_args.data_val_ratio, subset_dir=dir_dict["SUBSET_VAL_DIR"])
    train_dataloader = get_dataloader(training_data, batch_size=parser_args.batch_size)
    val_dataloader = get_dataloader(val_data, batch_size=parser_args.batch_size)
    print("Done loading data")

    # Load model
    model = get_pretrained_resnet(parser_args.pretrained_model, parser_args.num_in_channels).to(device)
    save_network_config(model)

    # Optimizers, losses, ...
    optimizer = torch.optim.Adam(model.parameters(), lr=parser_args.lr, betas=(parser_args.beta_1, parser_args.beta_2), weight_decay=parser_args.weight_decay)
    loss_fn = nn.MSELoss()
    print("Done loading model")

    # Create new folder for this run that contains the model save checkpoints, loss values,...
    save_dir = create_folder_for_run(parser_args.model_save_dir)

    # Storing the losses for tracing later
    train_losses = []
    val_losses = []

    # Begin training
    print("Training starting: ")
    for ep in tqdm(range(parser_args.num_epoch)):
        
        train_loss = 0.
        model.train()
        for data in train_dataloader:
            imgs, segs, dices = data["img"].to(device), data["seg"].to(device), data["dsc"].to(device)
            dices = dices.float()

            # if modality_dropout is true, then randomly cancel one or more modalities from the slice
            if parser_args.modality_dropout_rate > 0.0:
                ran = np.random.rand()  
                if ran <= parser_args.modality_dropout_rate:     # Do the dropout
                    mask0, mask_minus1 = create_mask(imgs.size(), parser_args.modality_dropped, background=0.0)
                    mask0, mask_minus1 = torch.from_numpy(mask0).to(device), torch.from_numpy(mask_minus1).to(device)
                    imgs = (imgs*mask0) + mask_minus1

            # Concat imgs and segs at channel dim, feed input in model and take loss
            input_tensor = torch.cat((imgs, segs), dim=1).float()
            prediced_dsc = model(input_tensor)
            dices = torch.unsqueeze(dices, dim=1)
            loss = loss_fn(dices, prediced_dsc)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Store loss values
            train_loss += loss.item()

        train_loss_avg = train_loss / len(train_dataloader)
        train_losses.append(train_loss_avg)
        print(f"Train loss after {ep+1} epochs: {train_loss_avg:.3f}")

        # Get val loss (later, take model at epoch that has lowest val loss)
        model.eval()
        with torch.no_grad():
            val_loss = 0.
            for data in val_dataloader:
                imgs, segs, dices = data["img"].to(device), data["seg"].to(device), data["dsc"].to(device)
                dices = dices.float()

                # Concat imgs and segs at channel dim, feed input in model and take loss
                input_tensor = torch.cat((imgs, segs), dim=1).float()
                prediced_dsc = model(input_tensor)
                dices = torch.unsqueeze(dices, dim=1)
                loss = loss_fn(dices, prediced_dsc)
                val_loss += loss.item()

            # Store val loss for each iteration
            val_loss_avg = val_loss / len(val_dataloader)
            val_losses.append(val_loss_avg)
            print(f"Val loss after {ep+1} epochs: {val_loss_avg:.3f}")

        # Save the model after a certain frequency
        if (ep+1 == 1 or (ep+1) % parser_args.save_freq == 0):
            torch.save(model.state_dict(), f'{save_dir}/ep-{ep+1}.pt')

    # After training, take the model checkpoint with the lowest val loss
    ep_lowest_val_loss = (np.argmin(np.array(val_losses)))*parser_args.save_freq
    print(f"Model id {ep_lowest_val_loss} has the lowest val loss") 

    # After training, save the train and val losses
    with open(f"{save_dir}/loss.pkl", 'wb') as f:
        pickle.dump({'train_loss': train_losses,
                     'val_loss': val_losses}, f)