import torch
import numpy as np
from argparse import Namespace
from xuance.common import Union, Optional, MeanField_OffPolicyBuffer
from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv
from xuance.torch import Module
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.agents import OffPolicyMARLAgents, BaseCallback


class MFQ_Agents(OffPolicyMARLAgents):
    """The implementation of Mean-Field Q agents.

    Args:
        config: the Namespace variable that provides hyperparameters and other settings.
        envs: the vectorized environments.
        callback: A user-defined callback function object to inject custom logic during training.
    """
    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv],
                 callback: Optional[BaseCallback] = None):
        super(MFQ_Agents).__init__(config, envs, callback)

        self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / config.decay_step_greedy
        self.e_greedy = self.start_greedy

        self.policy = self._build_policy()  # build policy
        self.memory = self._build_memory()  # build memory
        self.learner = self._build_learner(self.config, self.model_keys, self.agent_keys, self.policy, self.callback)

    def _build_memory(self):
        """Build replay buffer for models training
                """
        if self.use_actions_mask:
            avail_actions_shape = {key: (self.action_space[key].n,) for key in self.agent_keys}
        else:
            avail_actions_shape = None
        memory = MeanField_OffPolicyBuffer(self.config.n_agents,
                                           self.config.state_shape,
                                           self.config.obs_shape,
                                           self.config.act_shape,
                                           self.config.act_prob_shape,
                                           self.config.rew_shape,
                                           self.config.done_shape,
                                           self.envs.num_envs,
                                           self.config.buffer_size,
                                           self.config.batch_size)
        return memory

    def _build_policy(self) -> Module:
        """
        Build representation(s) and policy(ies) for agent(s)

        Returns:
            policy (torch.nn.Module): A dict of policies.
        """
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = torch.nn.init.orthogonal_
        activation = ActivationFunctions[self.config.activation]
        device = self.device

        # build representations
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build policies
        if self.config.policy == "MF_Q_network":
            policy = REGISTRY_Policy["MF_Q_network"](
                action_space=self.action_space, n_agents=self.n_agents,
                representation=representation,
                hidden_size=self.config.q_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation, device=device,
                use_distributed_training=self.distributed_training,
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None)
        else:
            raise AttributeError(f"MFQ currently does not support the policy named {self.config.policy}.")

        return policy


def act(self, obs_n, *rnn_hidden, test_mode=False, act_mean=None, agent_mask=None, avail_actions=None):
        batch_size = obs_n.shape[0]
        agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
        obs_in = torch.Tensor(obs_n).to(self.device)
        act_mean = torch.Tensor(act_mean).unsqueeze(dim=-2).repeat(1, self.n_agents, 1).to(self.device)

        if self.use_rnn:  # awaiting to be tested
            batch_agents = batch_size * self.n_agents
            hidden_state, greedy_actions, q_output = self.policy(obs_in.view(batch_agents, 1, -1),
                                                                 act_mean.view(batch_agents, 1, -1),
                                                                 agents_id.view(batch_agents, 1, -1),
                                                                 *rnn_hidden,
                                                                 avail_actions=avail_actions)
        else:
            hidden_state, greedy_actions, q_output = self.policy(obs_in, act_mean, agents_id)
        n_alive = torch.Tensor(agent_mask).sum(dim=-1).unsqueeze(-1).repeat(1, self.dim_act).to(self.device)
        action_n_mask = torch.Tensor(agent_mask).unsqueeze(-1).repeat(1, 1, self.dim_act).to(self.device)
        act_neighbor_sample = self.policy.sample_actions(logits=q_output).to(self.device)
        act_neighbor_onehot = self.learner.onehot_action(act_neighbor_sample, self.dim_act) * action_n_mask
        act_mean_current = act_neighbor_onehot.float().sum(dim=1) / n_alive
        act_mean_current = act_mean_current.cpu().detach().numpy()
        greedy_actions = greedy_actions.cpu().detach().numpy()
        if test_mode:
            return hidden_state, greedy_actions, act_mean_current
        else:
            random_actions = np.random.choice(self.dim_act, [self.nenvs, self.n_agents])
            if np.random.rand() < self.egreedy:
                return hidden_state, random_actions, act_mean_current
            else:
                return hidden_state, greedy_actions, act_mean_current

    def train(self, i_step, n_epochs=1):
        if self.egreedy >= self.end_greedy:
            self.egreedy = self.start_greedy - self.delta_egreedy * i_step
        info_train = {}
        if i_step > self.start_training:
            for i_epoch in range(n_epochs):
                sample = self.memory.sample()
                info_train = self.learner.update(sample)
        info_train["epsilon-greedy"] = self.egreedy
        return info_train
