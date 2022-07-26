import torch
from dataset import DiceDataset, get_dataloader
import matplotlib.pyplot as plt
import seaborn as sns

model_id = "model_4"
run_id = "run-3"
IMG_TRAIN_DIR = f"/mnt/beegfs/scratch/phuc/seg-quality-control/{model_id}/Ground-truth-images/"
SEG_TRAIN_DIR = f"/mnt/beegfs/scratch/phuc/seg-quality-control/{model_id}/Predicted-segs/"
IMG_VAL_DIR = f"/mnt/beegfs/scratch/phuc/seg-quality-control/{model_id}/Ground-truth-images-val/"
SEG_VAL_DIR = f"/mnt/beegfs/scratch/phuc/seg-quality-control/{model_id}/Predicted-segs-val/"
DICE_TRAIN_DIR = f"/mnt/beegfs/scratch/phuc/seg-quality-control/{model_id}/dices_train.pkl"
DICE_VAL_DIR = f"/mnt/beegfs/scratch/phuc/seg-quality-control/{model_id}/dices_val.pkl"
BASE_DIR = f"/mnt/beegfs/home/phuc/my-code/dsc-predict"
SUBSET_TRAIN_DIR = f"/home/phuc/my-code/dsc-predict/data-subsets/{run_id}/train/"
SUBSET_VAL_DIR = f"/home/phuc/my-code/dsc-predict/data-subsets/{run_id}/val/"
BATCH_SIZE = 16

training_data = DiceDataset(IMG_TRAIN_DIR, SEG_TRAIN_DIR, DICE_TRAIN_DIR, ratio=0.3)
val_data = DiceDataset(IMG_VAL_DIR, SEG_VAL_DIR, DICE_VAL_DIR, ratio=0.4)
train_dataloader = get_dataloader(training_data, batch_size=BATCH_SIZE)
val_dataloader = get_dataloader(val_data, batch_size=BATCH_SIZE)

print(len(training_data))
print(len(val_data))

plt.figure()
sns.histplot(data=training_data.dice_scores_list)
plt.savefig("temp1.png")
plt.figure()
sns.histplot(data=val_data.dice_scores_list)
plt.savefig("temp2.png")

print("Run successful")