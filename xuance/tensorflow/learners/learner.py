from xuance.tensorflow.learners import *
from argparse import Namespace


class Learner(ABC):
    def __init__(self,
                 policy: tk.Model,
                 optimizer: Union[tk.optimizers.Optimizer, Sequence[tk.optimizers.Optimizer]],
                 device: str = "cpu:0",
                 modeldir: str = "./"):
        self.policy = policy
        self.optimizer = optimizer
        self.device = device
        self.modeldir = modeldir
        self.iterations = 0

    def save_model(self, model_name):
        model_path = self.modeldir + model_name
        self.policy.save(model_path)

    def load_model(self, path):
        model_names = os.listdir(path)
        try:
            model_names.remove('obs_rms.npy')
            model_names.sort()
            model_path = path + model_names[-1]
            self.policy = tk.models.load_model(model_path, compile=False)
            # self.policy.load_weights(model_path)
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
                 modeldir: str = "./"):
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
        self.modeldir = modeldir
        self.iterations = 0

    def onehot_action(self, actions_int, num_actions):
        return tf.one_hot(indices=actions_int, depth=num_actions, axis=-1, dtype=tf.float32)

    def save_model(self):
        model_path = self.modeldir + "model-%s-%s" % (time.asctime(), str(self.iterations))
        self.policy.save(model_path)

    def load_model(self, path):
        model_names = os.listdir(path)
        try:
            model_names.sort()
            model_path = path + model_names[-1]
            print(model_path)
            # self.policy = tk.models.load_model(model_path, compile=False)
            self.policy.load_weights(model_path)
        except:
            raise "Failed to load model! Please train and save the model first."

    @abstractmethod
    def update(self, *args):
        raise NotImplementedError