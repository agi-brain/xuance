import numpy as np
import torch
from baselines.offpolicy.algorithms.mqmix.algorithm.agent_q_function import (
    AgentQFunction,
)
from torch.distributions import Categorical, OneHotCategorical
from baselines.offpolicy.utils.util import (
    get_dim_from_space,
    is_discrete,
    is_multidiscrete,
    make_onehot,
    DecayThenFlatSchedule,
    avail_choose,
    to_torch,
    to_numpy,
    get_dim_from_space,
)
from baselines.offpolicy.algorithms.base.mlp_policy import MLPPolicy


class M_QMixPolicy(MLPPolicy):
    """
    QMIX/VDN Policy Class to compute Q-values and actions (MLP). See parent class for details.
    :param config: (dict) contains information about hyperparameters and algorithm configuration
    :param policy_config: (dict) contains information specific to the policy (obs dim, act dim, etc)
    :param train: (bool) whether the policy will be trained.
    """

    def __init__(self, config, policy_config, train=True):
        self.args = config["args"]
        self.device = config["device"]
        self.obs_space = policy_config["obs_space"]
        self.obs_dim = get_dim_from_space(self.obs_space)
        self.act_space = policy_config["act_space"]
        self.act_dim = get_dim_from_space(self.act_space)
        self.output_dim = (
            sum(self.act_dim) if isinstance(self.act_dim, np.ndarray) else self.act_dim
        )
        self.central_obs_dim = policy_config["cent_obs_dim"]
        self.discrete_action = is_discrete(self.act_space)
        self.multidiscrete = is_multidiscrete(self.act_space)
        self.q_network_input_dim = self.obs_dim

        # Local recurrent q network for the agent
        self.q_network = AgentQFunction(
            self.args, self.q_network_input_dim, self.act_dim, self.device
        )

        if train:
            self.exploration = DecayThenFlatSchedule(
                self.args.epsilon_start,
                self.args.epsilon_finish,
                self.args.epsilon_anneal_time,
                decay="linear",
            )

    def get_q_values(self, obs_batch, action_batch=None):
        """
        Computes q values using the given information.
        :param obs_batch: (np.ndarray)
            agent observations from which to compute q values
        :param action_batch: (np.ndarray)
            if not None, then only return the q values
            corresponding to actions in action_batch

        :return q_values: (torch.Tensor) computed q values
        """
        q_batch = self.q_network(obs_batch)
        if action_batch is not None:
            if type(action_batch) == np.ndarray:
                action_batch = torch.FloatTensor(action_batch)
            if self.multidiscrete:
                all_q_values = []
                for i in range(len(self.act_dim)):
                    curr_q_batch = q_batch[i]
                    curr_action_batch = action_batch[i]
                    curr_q_values = torch.gather(
                        curr_q_batch, 1, curr_action_batch.unsqueeze(dim=-1)
                    )
                    all_q_values.append(curr_q_values)
                return torch.cat(all_q_values, dim=-1)
            else:
                q_values = torch.gather(q_batch, 1, action_batch.unsqueeze(dim=-1))
                # q_values is a column vector containing q values
                # for the actions specified by action_batch
                return q_values
        return q_batch

    def get_actions(self, obs_batch, available_actions=None, t_env=None, explore=False):
        """
        See parent class.
        """
        batch_size = obs_batch.shape[0]
        q_values_out = self.get_q_values(obs_batch)

        # mask the available actions by giving -inf q values to unavailable actions
        if available_actions is not None:
            q_values = q_values_out.clone()
            q_values = avail_choose(q_values, available_actions)
        else:
            q_values = q_values_out
        # greedy_Qs, greedy_actions = list(map(lambda a: a.max(dim=-1), q_values))
        if self.multidiscrete:
            onehot_actions = []
            greedy_Qs = []
            for i in range(len(self.act_dim)):
                greedy_Q, greedy_action = q_values[i].max(dim=-1)

                if explore:
                    eps = self.exploration.eval(t_env)
                    rand_number = np.random.rand(batch_size)
                    # random actions sample uniformly from action space
                    random_action = (
                        Categorical(logits=torch.ones(batch_size, self.act_dim[i]))
                        .sample()
                        .numpy()
                    )
                    take_random = (rand_number < eps).astype(int)
                    action = (1 - take_random) * to_numpy(
                        greedy_action
                    ) + take_random * random_action
                    onehot_action = make_onehot(action, self.act_dim[i])
                else:
                    greedy_Q = greedy_Q.unsqueeze(-1)
                    onehot_action = make_onehot(greedy_action, self.act_dim[i])

                onehot_actions.append(onehot_action)
                greedy_Qs.append(greedy_Q)

            onehot_actions = np.concatenate(onehot_actions, axis=-1)
            greedy_Qs = torch.cat(greedy_Qs, dim=-1)
        else:
            greedy_Qs, greedy_actions = q_values.max(dim=-1)
            if explore:
                eps = self.exploration.eval(t_env)
                rand_numbers = np.random.rand(batch_size)
                # random actions sample uniformly from action space
                logits = avail_choose(
                    torch.ones(batch_size, self.act_dim), available_actions
                )
                random_actions = Categorical(logits=logits).sample().numpy()
                take_random = (rand_numbers < eps).astype(int)
                actions = (1 - take_random) * to_numpy(
                    greedy_actions
                ) + take_random * random_actions
                onehot_actions = make_onehot(actions, self.act_dim)
            else:
                greedy_Qs = greedy_Qs.unsqueeze(-1)
                onehot_actions = make_onehot(greedy_actions, self.act_dim)

        return onehot_actions, greedy_Qs

    def get_random_actions(self, obs, available_actions=None):
        """See parent class."""
        batch_size = obs.shape[0]

        if self.multidiscrete:
            random_actions = [
                OneHotCategorical(logits=torch.ones(batch_size, self.act_dim[i]))
                .sample()
                .numpy()
                for i in range(len(self.act_dim))
            ]
            random_actions = np.concatenate(random_actions, axis=-1)
        else:
            if available_actions is not None:
                logits = avail_choose(
                    torch.ones(batch_size, self.act_dim), available_actions
                )
                random_actions = OneHotCategorical(logits=logits).sample().numpy()
            else:
                random_actions = (
                    OneHotCategorical(logits=torch.ones(batch_size, self.act_dim))
                    .sample()
                    .numpy()
                )

        return random_actions

    def parameters(self):
        return self.q_network.parameters()

    def load_state(self, source_policy):
        self.q_network.load_state_dict(source_policy.q_network.state_dict())
