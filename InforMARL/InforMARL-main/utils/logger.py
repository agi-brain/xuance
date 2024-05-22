import os
import glob
import argparse
import time
import wandb
import torch
from torch.utils.tensorboard import SummaryWriter

import sys

sys.path.append(os.path.abspath(os.getcwd()))
from utils.utils import connected_to_internet


class WandbLogger:
    def __init__(
        self,
        experiment_name: str,
        save_folder: str,
        project: str,
        entity: str,
        args: argparse.Namespace,
        **kwargs,
    ):
        """
        Wandb Logger Wrapper
        Parameters:
        –––––––––––
        experiment_name: str
            Name for logging the experiment on Wandboard
            Will save logs with the name {experiment_name}_{start_time}
        save_folder: str
            Name of the folder to store wandb run files
            Will save all the logs in a folder with name `save_folder`
        project: str
            Project name for wandboard
            Example: 'My Repo Name'
            This is for the Wandboard.
            Logs will get logged to this project name on the wandb cloud.
        entity: str
            Entity/username for wandboard
            Example: 'marl'
            Your wandb userid
        args: argparse.Namespace
            Experiment arguments to save
            The arguments you want to log for the experiment
        ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        Usage:
        ––––––
        ##### args #####
        import argparse
        parser = argparse.ArgumentParser(description='Experiment args')
        parser.add_argument('--nodes', default=50, help='No. of nodes')
        parser.add_argument('--epoch', type=int, help='No. of epochs')
        args = parser.parse_args()
        ###############
        # init the logger and save the args used in the experiment
        logger = WandbLogger(experiment_name='myExpName',
                        save_folder='trafficGNN', project='myProjectName',
                        entity='marl', args=args)
        ##### run the training loop #####
        for epoch_num in range(args.epoch):
            loss = model(x,y)
            # log the epoch loss
            logger.writer.add_scalar('Epoch Loss', loss, epoch_num)
            # can use anything compatible with torch.utils.tensorboard.SummaryWriter
            # https://pytorch.org/docs/stable/tensorboard.html
            val_loss = model(x_val, y_val)
            logger.writer.add_scalar('Val Epoch Loss', val_loss, epoch_num)
            logger.save_checkpoint(network=model,train_step_num=epoch_num)
        """
        self.args = args
        # check if internet is available; if not then change wandb mode to dryrun
        if not connected_to_internet():
            import json

            # save a json file with your wandb api key in your
            # home folder as {'my_wandb_api_key': 'INSERT API HERE'}
            # NOTE this is only for running on systems without internet access
            # have to run `wandb sync wandb/run_name` to sync logs to wandboard
            with open(os.path.expanduser("~") + "/keys.json") as json_file:
                key = json.load(json_file)
                my_wandb_api_key = key["my_wandb_api_key"]  # NOTE change here as well
            os.environ["WANDB_API_KEY"] = my_wandb_api_key  # my Wandb api key
            os.environ["WANDB_MODE"] = "dryrun"
            os.environ["WANDB_SAVE_CODE"] = "true"

        start_time = time.strftime("%H_%M_%S-%d_%m_%Y", time.localtime())
        # experiment_name = f"{experiment_name}_{start_time}"

        print("_" * 50)
        print("Creating wandboard...")
        print("_" * 50)

        # wandb_save_dir = os.path.join(os.path.abspath(os.getcwd()), save_folder)
        # if not os.path.exists(wandb_save_dir):
        #     os.makedirs(wandb_save_dir)

        wandb.init(
            project=project,
            entity=entity,
            sync_tensorboard=True,
            config=vars(args),
            name=experiment_name,
            save_code=True,
            dir=save_folder,
            **kwargs,
        )

        self.writer = SummaryWriter(f"{wandb.run.dir}/{experiment_name}")
        self.wandb_dir = wandb.run.dir
        self.weight_save_path = os.path.join(wandb.run.dir, "model.ckpt")

    # TODO make this next part more elegant by inheriting from SummaryWriter
    # Was too lazy to do this more elegantly
    ############# convert all tensorboard methods to self methods #############
    def add_scalar(
        self, tag, scalar_value, global_step=None, walltime=None, new_style=False
    ):
        self.writer.add_scalar(tag, scalar_value, global_step, walltime)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        self.writer.add_scalars(main_tag, tag_scalar_dict, global_step, walltime)

    def add_histogram(
        self,
        tag,
        values,
        global_step=None,
        bins="tensorflow",
        walltime=None,
        max_bins=None,
    ):
        self.writer.add_histogram(tag, values, global_step, bins, walltime, max_bins)

    def add_histogram_raw(
        self,
        tag,
        min,
        max,
        num,
        sum,
        sum_squares,
        bucket_limits,
        bucket_counts,
        global_step=None,
        walltime=None,
    ):
        self.writer.add_histogram_raw(
            tag,
            min,
            max,
            num,
            sum,
            sum_squares,
            bucket_limits,
            bucket_counts,
            global_step,
            walltime,
        )

    def add_image(
        self, tag, img_tensor, global_step=None, walltime=None, dataformats="CHW"
    ):
        self.writer.add_image(tag, img_tensor, global_step, walltime, dataformats)

    def add_images(
        self, tag, img_tensor, global_step=None, walltime=None, dataformats="NCHW"
    ):
        self.writer.add_images(tag, img_tensor, global_step, walltime, dataformats)

    def add_image_with_boxes(
        self,
        tag,
        img_tensor,
        box_tensor,
        global_step=None,
        walltime=None,
        rescale=1,
        dataformats="CHW",
        labels=None,
    ):
        self.writer.add_image_with_boxes(
            tag,
            img_tensor,
            box_tensor,
            global_step,
            walltime,
            rescale,
            dataformats,
            labels,
        )

    def add_figure(self, tag, figure, global_step=None, close=True, walltime=None):
        self.writer.add_figure(tag, figure, global_step, close, walltime)

    def add_video(self, tag, vid_tensor, global_step=None, fps=4, walltime=None):
        self.writer.add_video(tag, vid_tensor, global_step, fps, walltime)

    def add_audio(
        self, tag, snd_tensor, global_step=None, sample_rate=44100, walltime=None
    ):
        self.writer.add_audio(tag, snd_tensor, global_step, sample_rate, walltime)

    def add_text(self, tag, text_string, global_step=None, walltime=None):
        self.writer.add_text(tag, text_string, global_step, walltime)

    def add_onnx_graph(self, prototxt):
        self.writer.add_onnx_graph(prototxt)

    def add_embedding(
        self,
        mat,
        metadata=None,
        label_img=None,
        global_step=None,
        tag="default",
        metadata_header=None,
    ):
        self.writer.add_embedding(
            mat, metadata, label_img, global_step, tag, metadata_header
        )

    def add_pr_curve(
        self,
        tag,
        labels,
        predictions,
        global_step=None,
        num_thresholds=127,
        weights=None,
        walltime=None,
    ):
        self.writer.add_pr_curve(
            tag, labels, predictions, global_step, num_thresholds, weights, walltime
        )

    def add_pr_curve_raw(
        self,
        tag,
        true_positive_counts,
        false_positive_counts,
        true_negative_counts,
        false_negative_counts,
        precision,
        recall,
        global_step=None,
        num_thresholds=127,
        weights=None,
        walltime=None,
    ):
        self.writer.add_pr_curve_raw(
            tag,
            true_positive_counts,
            false_positive_counts,
            true_negative_counts,
            false_negative_counts,
            precision,
            recall,
            global_step,
            num_thresholds,
            weights,
            walltime,
        )

    def add_custom_scalars_multilinechart(
        self, tags, category="default", title="untitled"
    ):
        self.writer.add_custom_scalars_multilinechart(tags, category, title)

    def add_custom_scalars_marginchart(
        self, tags, category="default", title="untitled"
    ):
        self.writer.add_custom_scalars_marginchart(tags, category, title)

    def add_custom_scalars(self, layout):
        self.writer.add_custom_scalars(layout)

    def add_mesh(
        self,
        tag,
        vertices,
        colors=None,
        faces=None,
        config_dict=None,
        global_step=None,
        walltime=None,
    ):
        self.writer.add_mesh(
            tag, vertices, colors, faces, config_dict, global_step, walltime
        )

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()

    def __enter__(self):
        return self.writer

    def __exit__(self):
        self.close()

    def save_checkpoint(self, network, path: str = None, epoch: int = 0):
        """
        Saves the model in the wandb experiment run directory
        This will store the
            • model state_dict
            • args:
                Will save this as a Dict as well as argparse.Namespace
        Parameters:
        –––––––––––
        network: nn.Module
            The network to be saved
        path: str
            path to the wandb run directory
            Example: wandb.run.dir
        epoch: int
            The epoch number at which model is getting saved
        ––––––––––––––––––––––––––––––––––––––––––––
        """
        if path is None:
            # this will set the path to be the same folder
            # as the one where other logs are stored
            path = self.weight_save_path
        checkpoint = {}
        checkpoint["args"] = self.args
        checkpoint["args_dict"] = vars(self.args)
        checkpoint["state_dict"] = network.state_dict()
        checkpoint["epoch"] = epoch
        torch.save(checkpoint, path)
