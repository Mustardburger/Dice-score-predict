import argparse
import os
import pickle

class Hyperparameters():

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.hyp = None
        self.save_dir = "/mnt/beegfs/home/phuc/my-code/dsc-predict/hyperparameters"
        self.run_id = None

    def read_arguments(self):

        # About directories
        self.parser.add_argument("--data_id", type=str, required=True, help="The id of the training set used")
        self.parser.add_argument("--model_save_dir", default="/mnt/beegfs/scratch/phuc/trained_resnets", help="The dir where all trained models are saved")
        self.parser.add_argument("--base_dir", default="/mnt/beegfs/home/phuc/my-code/dsc-predict", help="The dir where all the code files are stored")

        # About the dataset
        self.parser.add_argument("--data_root", type=str, required=True, help="The directory of the data")
        self.parser.add_argument("--data_train_ratio", type=float, required=True, help="The percentage of data from the training set used to train the model")
        self.parser.add_argument("--data_val_ratio", type=float, required=True, help="The percentage of data from the val set used to pick the best model")
        self.parser.add_argument("--num_in_channels", type=int, required=True, help="The number of input channels of images, depending on how many modalities are used")
        self.parser.add_argument("--modality_dropout_rate", type=float, default=0.0, help="With probability modality_dropout_rate, drop the modalities specified in modality_dropped")
        self.parser.add_argument("--modality_dropped", nargs="*", type=int, default=[], help="A list of modalities that can be dropped")
        self.parser.add_argument("--data_transform", type=str, default="", help="Data transform technique before feeding the images to the network")

        # About network hyperparams
        self.parser.add_argument("--batch_size", type=int, default=16, help="Amount of images per batch")
        self.parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
        self.parser.add_argument("--beta_1", type=float, default=0.5, help="Beta 1 coefficient in Adam optimizer")
        self.parser.add_argument("--beta_2", type=float, default=0.999, help="Beta 2 coefficient in Adam optimizer")
        self.parser.add_argument("--weight_decay", type=float, default=2e-5, help="Weight decay in Adam optimizer")

        # About training
        self.parser.add_argument("--num_epoch", type=int, required=True, help="The number of epochs for training the model")

        # Miscellaneous
        self.parser.add_argument("--save_freq", type=int, default=5, help="The number of epochs after which a model is saved")
        self.parser.add_argument("--pretrained_model", type=str, default="", help="The name of the pretrained model. Only for finetuning")

    def get_hyperparameters(self, save=True):
        self.read_arguments()
        self.hyp = self.parser.parse_known_args()[0]
        self.run_id = f"run-{len(os.listdir(self.save_dir))+1}"
        if save:
            self.save_arguments()
        return self.hyp

    def save_arguments(self):
        """
        Save the list of hyperparameters in a text file
        """
        curr_dir = os.path.join(self.save_dir, self.run_id)
        os.makedirs(curr_dir)

        if not(self.hyp):
            print("Arguments not read. Run get_hyperparameters() first")
            return 0

        hyp_dict = vars(self.hyp)
        with open(os.path.join(curr_dir, "hyperparams.txt"), "w") as f:
            for k, v in sorted(hyp_dict.items()):
                f.write(f"{k}: {v}\n")

if __name__ == "__main__":
    a = Hyperparameters()
    hyp = a.get_hyperparameters()
    a.save_arguments()
