from xuance.tensorflow.learners import *
from argparse import Namespace


class Learner(ABC):
    def __init__(self,
                 policy: tk.Model,
                 optimizer: Union[tk.optimizers.Optimizer, Sequence[tk.optimizers.Optimizer]],
                 device: str = "cpu:0",
                 model_dir: str = "./"):
        self.policy = policy
        self.optimizer = optimizer
        self.device = device
        self.model_dir = model_dir
        self.iterations = 0

    def save_model(self, model_path):
        self.policy.save_weights(model_path)

    def load_model(self, path, seed=1):
        try: file_names = os.listdir(path)
        except: raise "Failed to load model! Please train and save the model first."

        for f in file_names:
            '''Change directory to the specified seed (if exists)'''
            if f"seed_{seed}" in f:
                path = os.path.join(path, f)
                break
        latest = tf.train.latest_checkpoint(path)
        try:
            self.policy.load_weights(latest)
        except:
            raise "Failed to load model! Please train and save the model first."

    @abstractmethod
    def update(self, *args):
        raise NotImplementedError


class LearnerMAS(ABC):
    def __init__(self,
                 config: Namespace,
                 policy: tk.Model,
                 optimizer: Union[tk.optimizers.Optimizer, Sequence[tk.optimizers.Optimizer]],
                 device: str = "cpu:0",
                 model_dir: str = "./"):
        self.args = config
        self.handle = config.handle
        self.n_agents = config.n_agents
        self.agent_keys = config.agent_keys
        self.agent_index = config.agent_ids
        self.dim_obs = self.args.dim_obs
        self.dim_act = self.args.dim_act
        self.dim_id = self.n_agents
        self.device = device

        self.policy = policy
        self.optimizer = optimizer
        self.device = device
        self.model_dir = model_dir
        self.running_steps = config.running_steps
        self.iterations = 0

    def onehot_action(self, actions_int, num_actions):
        return tf.one_hot(indices=actions_int, depth=num_actions, axis=-1, dtype=tf.float32)

    def save_model(self, model_path):
        self.policy.save_weights(model_path)

    def load_model(self, path, seed=1):
        try: file_names = os.listdir(path)
        except: raise "Failed to load model! Please train and save the model first."
        model_path = ''

        for f in file_names:
            '''Change directory to the specified seed (if exists)'''
            if f"seed_{seed}" in f:
                model_path = os.path.join(path, f)
                if os.listdir(model_path).__len__() == 0:
                    continue
                else:
                    break
        if model_path == '':
            raise RuntimeError("Failed to load model! Please train and save the model first.")
        latest = tf.train.latest_checkpoint(model_path)
        try:
            self.policy.load_weights(latest)
        except:
            raise RuntimeError("Failed to load model! Please train and save the model first.")

    @abstractmethod
    def update(self, *args):
        raise NotImplementedError

    def update_recurrent(self, *args):
        pass

    def act(self, *args, **kwargs):
        pass

    def get_hidden_states(self, *args):
        pass

    def lr_decay(self, *args):
        pass

