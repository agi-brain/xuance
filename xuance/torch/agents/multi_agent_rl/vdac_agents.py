import torch
import numpy as np
from argparse import Namespace
from xuance.common import Optional
from xuance.environment import DummyVecMultiAgentEnv
from xuance.torch import Module
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.policies import REGISTRY_Policy, VDN_mixer, QMIX_mixer
from xuance.torch.agents import OnPolicyMARLAgents


class VDAC_Agents(OnPolicyMARLAgents):
    """The implementation of VDAC agents.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecMultiAgentEnv):
        super(VDAC_Agents, self).__init__(config, envs)
        self.state_space = envs.state_space
        self.mixer = config.mixer

        self.gamma = config.gamma
        self.n_envs = envs.num_envs
        self.n_epochs = config.n_epochs
        self.n_minibatch = config.n_minibatch
        if config.state_space is not None:
            config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
        else:
            config.dim_state, state_shape = None, None

        input_representation = get_repre_in(config)
        self.use_rnn = config.use_rnn
        # create representation for actor
        kwargs_rnn = {"N_recurrent_layers": config.N_recurrent_layers,
                      "dropout": config.dropout,
                      "rnn": config.rnn} if self.use_rnn else {}
        representation = REGISTRY_Representation[config.representation](*input_representation, **kwargs_rnn)
        # create policy

        input_policy = get_policy_in_marl(config, representation, mixer=mixer)
        policy = REGISTRY_Policy[config.policy](*input_policy,
                                                use_rnn=config.use_rnn,
                                                rnn=config.rnn,
                                                gain=config.gain)
        optimizer = torch.optim.Adam(policy.parameters(),
                                     lr=config.learning_rate, eps=1e-5,
                                     weight_decay=config.weight_decay)
        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
        self.auxiliary_info_shape = {}

        buffer = MARL_OnPolicyBuffer_RNN if self.use_rnn else MARL_OnPolicyBuffer
        input_buffer = (config.n_agents, config.state_space.shape, config.obs_shape, config.act_shape, config.rew_shape,
                        config.done_shape, envs.num_envs, config.buffer_size,
                        config.use_gae, config.use_advnorm, config.gamma, config.gae_lambda)
        memory = buffer(*input_buffer, max_episode_steps=envs.max_episode_steps, dim_act=config.dim_act)
        self.buffer_size = memory.buffer_size
        self.batch_size = self.buffer_size // self.n_minibatch

        learner = VDAC_Learner(config, policy, optimizer, None, config.device, config.model_dir, config.gamma)
        super(VDAC_Agents, self).__init__(config, envs, policy, memory, learner, device,
                                          config.log_dir, config.model_dir)
        self.share_values = True if config.rew_shape[0] == 1 else False
        self.on_policy = True

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
        agent = self.config.agent
        # build representations
        A_representation = self._build_representation(self.config.representation, self.observation_space, self.config)
        C_representation = self._build_representation(self.config.representation, self.observation_space, self.config)
        # create mixer
        if self.mixer == "VDN":
            mixer = VDN_mixer()
        elif self.mixer == "QMIX":
            dim_state = self.state_space.shape[-1]
            mixer = QMIX_mixer(dim_state, self.config.hidden_dim_mixing_net, self.config.hidden_dim_hyper_net,
                               self.n_agents, self.device)
        elif self.mixer == "Independent":
            mixer = None
        else:
            raise AttributeError(f"Mixer named {self.mixer} is not supported in XuanCe!")
        # build policies
        if self.config.policy == "Categorical_MAAC_Policy":
            policy = REGISTRY_Policy["Categorical_MAAC_Policy"](
                action_space=self.action_space, n_agents=self.n_agents,
                representation_actor=A_representation, representation_critic=C_representation,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                device=device, use_distributed_training=self.distributed_training,
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None)
            self.continuous_control = False
        elif self.config.policy == "Gaussian_MAAC_Policy":
            policy = REGISTRY_Policy["Gaussian_MAAC_Policy"](
                action_space=self.action_space, n_agents=self.n_agents,
                representation_actor=A_representation, representation_critic=C_representation,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                activation_action=ActivationFunctions[self.config.activation_action],
                device=device, use_distributed_training=self.distributed_training,
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None)
            self.continuous_control = True
        else:
            raise AttributeError(f"{agent} currently does not support the policy named {self.config.policy}.")
        return policy


    def act(self, obs_n, *rnn_hidden, avail_actions=None, state=None, test_mode=False):
        batch_size = len(obs_n)
        agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
        obs_in = torch.Tensor(obs_n).reshape([batch_size, self.n_agents, -1]).to(self.device)
        if state is not None:
            state = torch.Tensor(state).to(self.device)
        if avail_actions is not None:
            avail_actions = torch.Tensor(avail_actions).to(self.device)
        if self.use_rnn:
            batch_agents = batch_size * self.n_agents
            hidden_state, dists, values_tot = self.policy(obs_in.reshape(batch_agents, 1, -1),
                                                          agents_id.reshape(batch_agents, 1, -1),
                                                          *rnn_hidden,
                                                          avail_actions=avail_actions.reshape(batch_agents, 1, -1),
                                                          state=state.unsqueeze(2))
            actions = dists.stochastic_sample()
            actions = actions.reshape(batch_size, self.n_agents)
            values_tot = values_tot.reshape([batch_size, self.n_agents, 1])
        else:
            hidden_state, dists, values_tot = self.policy(obs_in, agents_id,
                                                          avail_actions=avail_actions,
                                                          state=state)
            actions = dists.stochastic_sample()
            values_tot = values_tot.reshape([batch_size, self.n_agents, 1])
        return hidden_state, actions.detach().cpu().numpy(), values_tot.detach().cpu().numpy()

    def train(self, i_step, **kwargs):
        if self.memory.full:
            info_train = {}
            indexes = np.arange(self.buffer_size)
            for _ in range(self.n_epochs):
                np.random.shuffle(indexes)
                for start in range(0, self.buffer_size, self.batch_size):
                    end = start + self.batch_size
                    sample_idx = indexes[start:end]
                    sample = self.memory.sample(sample_idx)
                    if self.use_rnn:
                        info_train = self.learner.update_recurrent(sample)
                    else:
                        info_train = self.learner.update(sample)
            self.learner.lr_decay(i_step)
            self.memory.clear()
            return info_train
        else:
            return {}
