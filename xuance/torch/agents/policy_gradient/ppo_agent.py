import gymnasium
from tqdm import tqdm
from copy import deepcopy
from argparse import Namespace
from gymnasium.spaces import Space
from xuance.common import Optional, BaseCallback
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.torch import Module
from xuance.torch.utils import ActivationFunctions
from xuance.torch.agents import OnPolicyAgent
from xuance.torch.rl_models.heads import GaussianActorHead, CategoricalActorHead, ValueHead
from xuance.torch.rl_models import CategoricalActor, GaussianActor
from xuance.torch.rl_models import StateValueCritic as Critic
from xuance.torch.rl_models import ActorCritic, SharedActorCritic
from xuance.torch.rl_models.modules import ActionOutput


class PPO_Agent(OnPolicyAgent):
    """The implementation of PPO agent.

    Args:
        config: the Namespace variable that provides hyperparameters and other settings.
        envs: the vectorized environments.
        callback: A user-defined callback function object to inject custom logic during training.
    """

    def __init__(
            self,
            config: Namespace,
            envs: Optional[DummyVecEnv | SubprocVecEnv] = None,
            observation_space: Optional[Space] = None,
            action_space: Optional[Space] = None,
            callback: Optional[BaseCallback] = None
    ):
        super(PPO_Agent, self).__init__(config, envs, observation_space, action_space, callback)
        self.model = self._build_model()  # build RL model
        self.memory = self._build_memory(self.auxiliary_info_shape)  # build memory
        self.learner = self._build_learner(self.config, self.model, self.callback)  # build learner

    def _build_model(self) -> Module:
        shared_representation = getattr(self.config, "shared_representation", True)

        # build representation.
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build actor network
        actor_input = dict(normalizer=self.normalize_fn,
                           initializer=self.initializer,
                           activation=self.activation,
                           device=self.device)
        if shared_representation:
            actor_input.update(dict(feature_dim=representation.output_shapes['state'][0],
                                    hidden_size=self.config.actor_hidden_size))
            if isinstance(self.action_space, gymnasium.spaces.Box):
                actor_input.update(dict(action_dim=self.action_space.shape[0],
                                        activation_action=ActivationFunctions[self.config.activation_action], ))
                actor = GaussianActorHead(**actor_input)
            elif isinstance(self.action_space, gymnasium.spaces.Discrete):
                actor_input.update(dict(action_dim=self.action_space.n))
                actor = CategoricalActorHead(**actor_input)
            else:
                raise NotImplementedError
        else:
            actor_input.update(dict(representation=representation,
                                    actor_hidden_size=self.config.actor_hidden_size,
                                    action_space=self.action_space))
            if isinstance(self.action_space, gymnasium.spaces.Box):
                actor_input.update(dict(activation_action=ActivationFunctions[self.config.activation_action], ))
                actor = GaussianActor(**actor_input)
            elif isinstance(self.action_space, gymnasium.spaces.Discrete):
                actor = CategoricalActor(**actor_input)
            else:
                raise NotImplementedError

        # build critic network and the RL model
        if shared_representation:
            critic = ValueHead(feature_dim=representation.output_shapes['state'][0],
                               hidden_size=self.config.critic_hidden_size,
                               normalizer=self.normalize_fn,
                               initializer=self.initializer,
                               activation=self.activation,
                               device=self.device)
            model = SharedActorCritic(representation=representation, actor=actor, critic=critic)
        else:
            critic = Critic(representation=deepcopy(representation),
                            critic_hidden_size=self.config.critic_hidden_size,
                            normalizer=self.normalize_fn,
                            initializer=self.initializer,
                            activation=self.activation,
                            device=self.device)
            model = ActorCritic(actor=actor, critic=critic)

        return model

    @property
    def auxiliary_info_shape(self):
        return {"old_logp": ()}

    def get_aux_info(self, policy_output: ActionOutput = None):
        """Returns auxiliary information.

        Parameters:
            policy_output (dict): The output information of the policy.

        Returns:
            aux_info (dict): The auxiliary information.
        """
        aux_info = {"old_logp": policy_output.log_probs}
        return aux_info

    def train(self, train_steps):
        train_info = {}
        obs = self.train_envs.buf_obs
        for _ in tqdm(range(train_steps)):
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            policy_out = self.get_actions(obs, return_dists=False, return_logpi=True)
            acts = policy_out.env_actions
            value = policy_out.values
            next_obs, rewards, terminals, truncations, infos = self.train_envs.step(acts)
            aux_info = self.get_aux_info(policy_out)

            self.callback.on_train_step(self.current_step, envs=self.train_envs, policy=self.model,
                                        obs=obs, policy_out=policy_out, acts=acts, vals=value, next_obs=next_obs,
                                        rewards=rewards, terminals=terminals, truncations=truncations,
                                        infos=infos, aux_info=aux_info, train_steps=train_steps)

            self.memory.store(obs, acts, self._process_reward(rewards), value, terminals, aux_info)
            if self.memory.full:
                vals = self.get_terminated_values(next_obs)
                for i in range(self.n_envs):
                    if terminals[i]:
                        self.memory.finish_path(0.0, i)
                    else:
                        self.memory.finish_path(vals[i], i)
                update_info = self.train_epochs(self.n_epochs)
                self.log_infos(update_info, self.current_step)
                train_info.update(update_info)
                self.callback.on_train_epochs_end(self.current_step, policy=self.model, memory=self.memory,
                                                  current_episode=self.current_episode, train_steps=train_steps,
                                                  update_info=update_info)
                self.memory.clear()

            self.returns = self.gamma * self.returns + rewards
            obs = deepcopy(next_obs)
            for i in range(self.n_envs):
                if terminals[i] or truncations[i]:
                    self.ret_rms.update(self.returns[i:i + 1])
                    self.returns[i] = 0.0
                    if self.atari and (not truncations[i]):
                        pass
                    else:
                        if terminals[i]:
                            self.memory.finish_path(0.0, i)
                        else:
                            vals = self.get_terminated_values(next_obs)
                            self.memory.finish_path(vals[i], i)
                        obs[i] = infos[i]["reset_obs"]
                        self.train_envs.buf_obs[i] = obs[i]
                        self.current_episode[i] += 1
                        if self.use_wandb:
                            episode_info = {
                                f"Episode-Steps/rank_{self.rank}/env-{i}": infos[i]["episode_step"],
                                f"Train-Episode-Rewards/rank_{self.rank}/env-{i}": infos[i]["episode_score"]
                            }
                        else:
                            episode_info = {
                                f"Episode-Steps/rank_{self.rank}": {f"env-{i}": infos[i]["episode_step"]},
                                f"Train-Episode-Rewards/rank_{self.rank}": {f"env-{i}": infos[i]["episode_score"]}
                            }
                        self.log_infos(episode_info, self.current_step)
                        train_info.update(episode_info)
                        self.callback.on_train_episode_info(envs=self.train_envs, policy=self.model, env_id=i,
                                                            infos=infos, rank=self.rank, use_wandb=self.use_wandb,
                                                            current_step=self.current_step,
                                                            current_episode=self.current_episode,
                                                            train_steps=train_steps)
            self.current_step += self.n_envs
            self.callback.on_train_step_end(self.current_step, envs=self.train_envs, policy=self.model,
                                            train_steps=train_steps, train_info=train_info)
        return train_info
