import os
from abc import ABC, abstractmethod
from argparse import Namespace
from xuance.common import Union, Optional

from xuance.tensorflow import tf, keras, Module


class Learner(ABC):
    def __init__(self,
                 config: Namespace,
                 model: Module,
                 callback):
        self.value_normalizer = None
        self.config = config
        self.distributed_training = config.distributed_training

        self.episode_length = config.episode_length
        self.learning_rate = config.learning_rate if hasattr(config, 'learning_rate') else None
        self.use_linear_lr_decay = config.use_linear_lr_decay if hasattr(config, 'use_linear_lr_decay') else False
        self.end_factor_lr_decay = config.end_factor_lr_decay if hasattr(config, 'end_factor_lr_decay') else 1.0
        self.gamma = config.gamma if hasattr(config, 'gamma') else 0.99
        self.use_cnn = getattr(config, "use_cnn", False)
        self.use_rnn = config.use_rnn if hasattr(config, 'use_rnn') else False
        self.use_actions_mask = config.use_actions_mask if hasattr(config, 'use_actions_mask') else False
        self.model = model
        self.optimizer: Union[dict, list, Optional[keras.optimizers.Optimizer]] = None
        self.scheduler: Union[dict, list, Optional[keras.optimizers.schedules.LinearLR]] = None
        self.callback = callback

        self.use_grad_clip = config.use_grad_clip
        self.grad_clip_norm = config.grad_clip_norm
        self.device = config.device
        self.model_dir = config.model_dir
        self.total_iters = self.estimate_total_iterations()
        self.iterations = 0

    def estimate_total_iterations(self):
        """Estimated total number of training iterations"""
        start_training = getattr(self.config, "start_training", 0)
        training_frequency = getattr(self.config, "training_frequency", 1)
        total_iters = (self.config.running_steps - start_training) // (training_frequency * self.config.parallels)
        return total_iters

    def save_model(self, model_path):
        model_path += ".weights.h5"
        self.model.save_weights(model_path)

    def load_model(self, path, model=None):
        file_names = os.listdir(path)
        if model is not None:
            path = os.path.join(path, model)
            if model not in file_names:
                raise RuntimeError(f"The folder '{path}' does not exist, please specify a correct path to load model.")
        else:
            for f in file_names:
                if "seed_" not in f:
                    file_names.remove(f)
            file_names.sort()
            path = os.path.join(path, file_names[-1])

        model_names = os.listdir(path)
        if os.path.exists(path + "/obs_rms.npy"):
            model_names.remove("obs_rms.npy")
        if len(model_names) == 0:
            raise RuntimeError(f"There is no model file in '{path}'!")
        model_names.sort()
        model_path = os.path.join(path, model_names[-1])
        latest = tf.train.latest_checkpoint(model_path)
        try:
            self.model.load_weights(latest)
        except:
            raise "Failed to load model! Please train and save the model first."

    @abstractmethod
    def update(self, *args, **kwargs):
        raise NotImplementedError

