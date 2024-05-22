import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from operator import itemgetter
from argparse import Namespace
from xuance.environment import DummyVecMutliAgentEnv
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.representations import REGISTRY_Representation
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.learners import IDDPG_Learner
from xuance.torch.agents import MARLAgents
from xuance.common import MARL_OffPolicyBuffer


class IDDPG_Agents(MARLAgents):
    """The implementation of Independent DDPG agents.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """

    def __init__(self,
                 config: Namespace,
                 envs: DummyVecMutliAgentEnv):
        config.use_parameter_sharing = True
        super(IDDPG_Agents, self).__init__(config, envs)

        self.start_noise = config.start_noise
        self.end_noise = config.end_noise
        self.noise_scale = config.start_noise
        self.delta_noise = (self.start_noise - self.end_noise) / (config.running_steps / self.n_envs)

        # prepare representation and policy
        if self.use_parameter_sharing:  # all agents share a same representation, policy and optimizer.
            if config.representation == "Basic_Identical":
                representation = REGISTRY_Representation["Basic_Identical"](
                    input_shape=self.observation_space[self.agent_keys[0]].shape,
                    device=self.device)
            elif config.representation == "Basic_MLP":
                representation = REGISTRY_Representation["Basic_MLP"](
                    input_shape=self.observation_space[self.agent_keys[0]].shape,
                    hidden_sizes=config.representation_hidden_size,
                    normalize=NormalizeFunctions[config.normalize] if hasattr(config, "normalize") else None,
                    initialize=torch.nn.init.orthogonal_,
                    activation=ActivationFunctions[config.activation],
                    device=self.device)
            else:
                raise f"The {config.agent} currently does not support the representation named {config.representation}."
            if config.policy == "Independent_DDPG_Policy":
                self.policy = REGISTRY_Policy["Independent_DDPG_Policy"](
                    action_space=self.action_space[self.agent_keys[0]],
                    n_agents=self.n_agents,
                    representation=representation,
                    actor_hidden_size=config.actor_hidden_size,
                    critic_hidden_size=config.critic_hidden_size,
                    normalize=NormalizeFunctions[config.normalize] if hasattr(config, "normalize") else None,
                    initialize=torch.nn.init.orthogonal_,
                    activation=ActivationFunctions[config.activation],
                    activation_action=ActivationFunctions[config.activation_action],
                    device=self.device)
            else:
                raise f"The {config.agent} currently does not support the policy named {config.poicy}."
            optimizer = [torch.optim.Adam(self.policy.parameters_actor, config.lr_a, eps=1e-5),
                         torch.optim.Adam(self.policy.parameters_critic, config.lr_c, eps=1e-5)]
            scheduler = [torch.optim.lr_scheduler.LinearLR(optimizer[0], start_factor=1.0, end_factor=0.5,
                                                           total_iters=config.running_steps),
                         torch.optim.lr_scheduler.LinearLR(optimizer[1], start_factor=1.0, end_factor=0.5,
                                                           total_iters=config.running_steps)]
            self.memory = MARL_OffPolicyBuffer(n_agents=config.n_agents,
                                               state_space=None,
                                               obs_space=self.observation_space[self.agent_keys[0]].shape,
                                               act_space=self.action_space[self.agent_keys[0]].shape,
                                               n_envs=envs.num_envs,
                                               buffer_size=config.buffer_size,
                                               batch_size=config.batch_size)
        else:  # agents have individual representations, policies and optimizers.
            representation = {key: None for key in self.agent_keys}
            self.policy = {key: None for key in self.agent_keys}
            optimizer = {key: None for key in self.agent_keys}
            scheduler = {key: None for key in self.agent_keys}
            if config.representation == "Basic_Identical":
                representation = {key: REGISTRY_Representation["Basic_Identical"](
                    input_shape=self.observation_space[self.agent_keys[0]].shape,
                    device=self.device) for key in self.agent_keys}
            elif config.representation == "Basic_MLP":
                for key in self.agent_keys:
                    representation[key] = REGISTRY_Representation["Basic_MLP"](
                        input_shape=self.observation_space[key].shape,
                        hidden_sizes=config.representation_hidden_size,
                        normalize=NormalizeFunctions[config.normalize] if hasattr(config, "normalize") else None,
                        initialize=torch.nn.init.orthogonal_,
                        activation=ActivationFunctions[config.activation],
                        device=self.device)
            else:
                raise f"The {config.agent} currently does not support the representation of {config.representation}."
            if config.policy == "Independent_DDPG_Policy":
                for key in self.agent_keys:
                    self.policy[key] = REGISTRY_Policy["Independent_DDPG_Policy"](
                        action_space=self.action_space[key],
                        n_agents=self.n_agents,
                        representation=representation,
                        actor_hidden_size=config.actor_hidden_size,
                        critic_hidden_size=config.critic_hidden_size,
                        normalize=NormalizeFunctions[config.normalize] if hasattr(config, "normalize") else None,
                        initialize=torch.nn.init.orthogonal_,
                        activation=ActivationFunctions[config.activation],
                        activation_action=ActivationFunctions[config.activation_action],
                        device=self.device)
            else:
                raise f"The {config.agent} currently does not support the policy named {config.poicy}."
            self.memory = MARL_OffPolicyBuffer(n_agents=config.n_agents,
                                               state_space=None,
                                               obs_space=self.observation_space[self.agent_keys[0]].shape,
                                               act_space=self.action_space[self.agent_keys[0]].shape,
                                               rew_space=config.rew_shape,
                                               done_space=config.done_shape,
                                               n_envs=envs.num_envs,
                                               buffer_size=config.buffer_size,
                                               batch_size=config.batch_size)
        self.learner = IDDPG_Learner(config, self.policy, optimizer, scheduler,
                                     config.device, config.model_dir, config.gamma)

    def store_experience(self, obs_dict, actions_dict, obs_next_dict, rewards_dict, terminals_dict, info):
        if self.use_parameter_sharing:
            experience_data = {
                'obs': np.array([itemgetter(*self.agent_keys)(data) for data in obs_dict]),
                'actions': np.array([itemgetter(*self.agent_keys)(data) for data in actions_dict]),
                'obs_next': np.array([itemgetter(*self.agent_keys)(data) for data in obs_next_dict]),
                'rewards': np.array([itemgetter(*self.agent_keys)(data) for data in rewards_dict]),
                'terminals': np.array([itemgetter(*self.agent_keys)(data) for data in terminals_dict]),
                'agent_mask': np.array([itemgetter(*self.agent_keys)(data['agent_mask']) for data in info]),
            }
            self.memory.store(experience_data)

    def action(self, obs_dict, test_mode):
        """
        Returns actions for agents.

        Parameters:
            obs_dict (dict): Observations for each agent in self.agent_keys
            test_mode (bool): True for testing without noises.
        """
        if self.use_parameter_sharing:
            obs_array = np.array([itemgetter(*self.agent_keys)(env_obs) for env_obs in obs_dict])
            batch_size = len(obs_array)
            agents_id = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
            _, actions = self.policy(torch.Tensor(obs_array), agents_id)
        actions = actions.cpu().detach().numpy()
        if not test_mode:
            actions += np.random.normal(0, self.noise_scale, size=actions.shape)
        actions_dict = [{key: actions[e, agt] for agt, key in enumerate(self.agent_keys)} for e in range(self.n_envs)]
        return actions_dict

    def train(self, train_steps):
        obs_dict = self.envs.buf_obs
        for _ in tqdm(range(train_steps)):
            step_info = {}
            if self.current_step < self.start_training:
                actions_dict = [{k: self.action_space[k].sample() for k in self.agent_keys} for _ in range(self.n_envs)]
            else:
                actions_dict = self.action(obs_dict, test_mode=False)
            obs_next_dict, rewards_dict, terminated_dict, truncated, info = self.envs.step(actions_dict)
            self.store_experience(obs_dict, actions_dict, obs_next_dict, rewards_dict, terminated_dict, info)
            if self.current_step >= self.start_training and self.current_step % self.training_frequency == 0:
                sample = self.memory.sample()
                if self.use_parameter_sharing:
                    step_info = self.learner.update_share(sample)
                    step_info["noise_scale"] = self.noise_scale
                else:
                    raise NotImplementedError
            obs_dict = deepcopy(obs_next_dict)

            for i in range(self.n_envs):
                if all(terminated_dict[i].values()) or truncated[i]:
                    obs_dict[i] = info[i]["reset_obs"]
                    if self.use_wandb:
                        step_info["Episode-Steps/env-%d" % i] = info[i]["episode_step"]
                        step_info["Train-Episode-Rewards/env-%d" % i] = info[i]["episode_score"]
                    else:
                        step_info["Episode-Steps"] = {"env-%d" % i: info[i]["episode_step"]}
                        step_info["Train-Episode-Rewards"] = {
                            "env-%d" % i: np.mean(itemgetter(*self.agent_keys)(info[i]["episode_score"]))}
                    self.log_infos(step_info, self.current_step)

            self.current_step += self.n_envs
            if self.noise_scale >= self.end_noise:
                self.noise_scale = self.noise_scale - self.delta_noise

    def test(self, env_fn, test_episodes):
        test_envs = env_fn
        num_envs = test_envs.num_envs
