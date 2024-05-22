import numpy as np
import torch
from baselines.offpolicy.algorithms.qmix.algorithm.agent_q_function import (
    AgentQFunction,
)
from baselines.offpolicy.algorithms.base.recurrent_policy import RecurrentPolicy
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
)


class QMixPolicy(RecurrentPolicy):
    def __init__(self, config, policy_config, train=True):
        """
        QMIX/VDN Policy Class to compute Q-values and actions. See parent class for details.
        :param config: (dict) contains information about hyperparameters and algorithm configuration
        :param policy_config: (dict) contains information specific to the policy (obs dim, act dim, etc)
        :param train: (bool) whether the policy will be trained.
        """
        self.args = config["args"]
        self.device = config["device"]
        self.obs_space = policy_config["obs_space"]
        self.obs_dim = get_dim_from_space(self.obs_space)
        self.act_space = policy_config["act_space"]
        self.act_dim = get_dim_from_space(self.act_space)
        self.output_dim = (
            sum(self.act_dim) if isinstance(self.act_dim, np.ndarray) else self.act_dim
        )
        self.hidden_size = self.args.hidden_size
        self.central_obs_dim = policy_config["cent_obs_dim"]
        self.discrete = is_discrete(self.act_space)
        self.multidiscrete = is_multidiscrete(self.act_space)

        if self.args.prev_act_inp:
            # this is only local information so the agent can act decentralized
            self.q_network_input_dim = self.obs_dim + self.act_dim
        else:
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

    def get_q_values(self, obs_batch, prev_action_batch, rnn_states, action_batch=None):
        """
        Computes q values using the given information.
        :param obs: (np.ndarray) agent observations from which to compute q values
        :param prev_actions: (np.ndarray) agent previous actions which are optionally an input to q network
        :param rnn_states: (np.ndarray) RNN states of q network
        :param action_batch: (np.ndarray) if not None, then only return the q values corresponding to actions in action_batch
        :return q_values: (torch.Tensor) computed q values
        :return new_rnn_states: (torch.Tensor) updated RNN states
        """

        # combine previous action with observation for input into q, if specified in args
        if self.args.prev_act_inp:
            prev_action_batch = to_torch(prev_action_batch)
            input_batch = torch.cat((obs_batch, prev_action_batch), dim=-1)
        else:
            input_batch = obs_batch

        q_batch, new_rnn_states = self.q_network(input_batch, rnn_states)

        if action_batch is not None:
            action_batch = to_torch(action_batch).to(self.device)
            q_values = self.q_values_from_actions(q_batch, action_batch)
        else:
            q_values = q_batch
        return q_values, new_rnn_states

    def q_values_from_actions(self, q_batch, action_batch):
        """
        Get q values corresponding to actions.
        :param q_batch: (torch.Tensor) q values corresponding to every action.
        :param action_batch: (torch.Tensor) actions taken by the agent.
        :return q_values: (torch.Tensor) q values in q_batch corresponding to actions in action_batch
        """
        if self.multidiscrete:
            ind = 0
            all_q_values = []
            for i in range(len(self.act_dim)):
                curr_q_batch = q_batch[i]
                curr_action_portion = action_batch[:, :, ind : ind + self.act_dim[i]]
                curr_action_inds = curr_action_portion.max(dim=-1)[1]
                curr_q_values = torch.gather(
                    curr_q_batch, 2, curr_action_inds.unsqueeze(dim=-1)
                )
                all_q_values.append(curr_q_values)
                ind += self.act_dim[i]
            q_values = torch.cat(all_q_values, dim=-1)
        else:
            # convert one-hot action batch to index tensors to gather the q values corresponding to the actions taken
            action_batch = action_batch.max(dim=-1)[1]
            # import pdb; pdb.set_trace()
            q_values = torch.gather(q_batch, 2, action_batch.unsqueeze(dim=-1))
            # q_values is a column vector containing q values for the actions specified by action_batch
        return q_values

    def get_actions(
        self,
        obs,
        prev_actions,
        rnn_states,
        available_actions=None,
        t_env=None,
        explore=False,
    ):
        """See parent class."""
        q_values_out, new_rnn_states = self.get_q_values(obs, prev_actions, rnn_states)
        onehot_actions, greedy_Qs = self.actions_from_q(
            q_values_out,
            available_actions=available_actions,
            explore=explore,
            t_env=t_env,
        )

        return onehot_actions, new_rnn_states, greedy_Qs

    def actions_from_q(
        self, q_values, available_actions=None, explore=False, t_env=None
    ):
        """
        Computes actions to take given q values.
        :param q_values: (torch.Tensor) agent observations from which to compute q values
        :param available_actions: (np.ndarray) actions available to take (None if all actions available)
        :param explore: (bool) whether to use eps-greedy exploration
        :param t_env: (int) env step at which this function was called; used to compute eps for eps-greedy
        :return onehot_actions: (np.ndarray) actions to take (onehot)
        :return greedy_Qs: (torch.Tensor) q values corresponding to greedy actions.
        """
        if self.multidiscrete:
            no_sequence = len(q_values[0].shape) == 2
            batch_size = q_values[0].shape[0] if no_sequence else q_values[0].shape[1]
            seq_len = None if no_sequence else q_values[0].shape[0]
        else:
            no_sequence = len(q_values.shape) == 2
            batch_size = q_values.shape[0] if no_sequence else q_values.shape[1]
            seq_len = None if no_sequence else q_values.shape[0]

        # mask the available actions by giving -inf q values to unavailable actions
        if available_actions is not None:
            q_values = q_values.clone()
            q_values = avail_choose(q_values, available_actions)
        else:
            q_values = q_values

        if self.multidiscrete:
            onehot_actions = []
            greedy_Qs = []
            for i in range(len(self.act_dim)):
                greedy_Q, greedy_action = q_values[i].max(dim=-1)

                if explore:
                    assert no_sequence, "Can only explore on non-sequences"
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
                    if no_sequence:
                        onehot_action = make_onehot(greedy_action, self.act_dim[i])
                    else:
                        onehot_action = make_onehot(
                            greedy_action, self.act_dim[i], seq_len=seq_len
                        )

                onehot_actions.append(onehot_action)
                greedy_Qs.append(greedy_Q)

            onehot_actions = np.concatenate(onehot_actions, axis=-1)
            greedy_Qs = torch.cat(greedy_Qs, dim=-1)
        else:
            greedy_Qs, greedy_actions = q_values.max(dim=-1)
            if explore:
                assert no_sequence, "Can only explore on non-sequences"
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
                if no_sequence:
                    onehot_actions = make_onehot(greedy_actions, self.act_dim)
                else:
                    onehot_actions = make_onehot(
                        greedy_actions, self.act_dim, seq_len=seq_len
                    )

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

    def init_hidden(self, num_agents, batch_size):
        """See parent class."""
        if num_agents == -1:
            return torch.zeros(batch_size, self.hidden_size)
        else:
            return torch.zeros(num_agents, batch_size, self.hidden_size)

    def parameters(self):
        """See parent class."""
        return self.q_network.parameters()

    def load_state(self, source_policy):
        """See parent class."""
        self.q_network.load_state_dict(source_policy.q_network.state_dict())
