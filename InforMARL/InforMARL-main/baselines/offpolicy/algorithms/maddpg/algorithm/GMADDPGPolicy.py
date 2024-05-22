import torch
import numpy as np
from torch.distributions import OneHotCategorical
from baselines.offpolicy.algorithms.base.graph_mlp_policy import GraphMLPPolicy
from baselines.offpolicy.algorithms.maddpg.algorithm.graph_actor_critic import (
    GMADDPG_Actor,
    GMADDPG_Critic,
)
from baselines.offpolicy.algorithms.matd3.algorithm.graph_actor_critic import (
    GMATD3_Actor,
    GMATD3_Critic,
)
from baselines.offpolicy.utils.util import (
    is_discrete,
    is_multidiscrete,
    get_dim_from_space,
    DecayThenFlatSchedule,
    soft_update,
    hard_update,
    gumbel_softmax,
    onehot_from_logits,
    gaussian_noise,
    avail_choose,
    to_numpy,
)
from utils.utils import print_box


class GMADDPGPolicy(GraphMLPPolicy):
    """
    GMADDPG/GMATD3 Policy Class to wrap actor/critic and compute actions. See parent class for details.
    :param config: (dict) contains information about hyperparameters and algorithm configuration
    :param policy_config: (dict) contains information specific to the policy (obs dim, act dim, etc)
    :param target_noise: (int) std of target smoothing noise to add for MATD3 (applies only for continuous actions)
    :param td3: (bool) whether to use MATD3 or MADDPG.
    :param train: (bool) whether the policy will be trained.
    """

    def __init__(self, config, policy_config, target_noise=None, td3=False, train=True):
        self.config = config
        self.device = config["device"]
        self.args = self.config["args"]
        self.tau = self.args.tau
        self.lr = self.args.lr
        self.opti_eps = self.args.opti_eps
        self.weight_decay = self.args.weight_decay

        self.central_obs_dim, self.central_act_dim = (
            policy_config["cent_obs_dim"],
            policy_config["cent_act_dim"],
        )
        self.obs_space = policy_config["obs_space"]
        self.obs_dim = get_dim_from_space(self.obs_space)
        self.node_obs_space = policy_config[
            "node_obs_space"
        ]  # TODO add node_obs_space and edge_obs_space to config
        self.edge_obs_space = policy_config["edge_obs_space"]
        self.act_space = policy_config["act_space"]
        self.discrete = is_discrete(self.act_space)
        self.multidiscrete = is_multidiscrete(self.act_space)

        self.act_dim = get_dim_from_space(self.act_space)
        self.output_dim = (
            sum(self.act_dim) if isinstance(self.act_dim, np.ndarray) else self.act_dim
        )
        self.target_noise = target_noise

        actor_class = GMATD3_Actor if td3 else GMADDPG_Actor
        critic_class = GMATD3_Critic if td3 else GMADDPG_Critic

        self.actor = actor_class(
            args=self.args,
            obs_dim=self.obs_dim,
            node_obs_space=self.node_obs_space,
            edge_obs_space=self.edge_obs_space,
            act_dim=self.act_dim,
            device=self.device,
        )
        self.critic = critic_class(
            args=self.args,
            central_obs_dim=self.central_obs_dim,
            central_act_dim=self.central_act_dim,
            node_obs_space=self.node_obs_space,
            edge_obs_space=self.edge_obs_space,
            device=self.device,
            use_cent_obs=self.args.use_cent_obs,
        )

        self.target_actor = actor_class(
            args=self.args,
            obs_dim=self.obs_dim,
            node_obs_space=self.node_obs_space,
            edge_obs_space=self.edge_obs_space,
            act_dim=self.act_dim,
            device=self.device,
        )
        self.target_critic = critic_class(
            args=self.args,
            central_obs_dim=self.central_obs_dim,
            central_act_dim=self.central_act_dim,
            node_obs_space=self.node_obs_space,
            edge_obs_space=self.edge_obs_space,
            device=self.device,
            use_cent_obs=self.args.use_cent_obs,
        )
        print_box("Actor Network", 80)
        print_box(self.actor, 80)
        print_box("Critic Network", 80)
        print_box(self.critic, 80)

        # sync the target weights
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        if train:
            self.actor_optimizer = torch.optim.Adam(
                self.actor.parameters(),
                lr=self.lr,
                eps=self.opti_eps,
                weight_decay=self.weight_decay,
            )
            self.critic_optimizer = torch.optim.Adam(
                self.critic.parameters(),
                lr=self.lr,
                eps=self.opti_eps,
                weight_decay=self.weight_decay,
            )

            if self.discrete:
                # eps greedy exploration
                self.exploration = DecayThenFlatSchedule(
                    self.args.epsilon_start,
                    self.args.epsilon_finish,
                    self.args.epsilon_anneal_time,
                    decay="linear",
                )

    def get_actions(
        self,
        obs,
        node_obs,
        adj,
        agent_id,
        available_actions=None,
        t_env=None,
        explore=False,
        use_target=False,
        use_gumbel=False,
    ):
        """See parent class."""
        batch_size = obs.shape[0]
        eps = None
        if use_target:
            actor_out = self.target_actor(obs, node_obs, adj, agent_id)
        else:
            actor_out = self.actor(obs, node_obs, adj, agent_id)

        if self.discrete:
            if self.multidiscrete:
                if use_gumbel or (use_target and self.target_noise is not None):
                    onehot_actions = list(
                        map(
                            lambda a: gumbel_softmax(a, hard=True, device=self.device),
                            actor_out,
                        )
                    )
                    actions = torch.cat(onehot_actions, dim=-1)
                elif explore:
                    onehot_actions = list(
                        map(
                            lambda a: gumbel_softmax(a, hard=True, device=self.device),
                            actor_out,
                        )
                    )
                    onehot_actions = torch.cat(onehot_actions, dim=-1)
                    # eps greedy exploration
                    eps = self.exploration.eval(t_env)
                    rand_numbers = np.random.rand(batch_size, 1)
                    take_random = (rand_numbers < eps).astype(int).reshape(-1, 1)
                    # random actions sample uniformly from action space
                    random_actions = [
                        OneHotCategorical(
                            logits=torch.ones(batch_size, self.act_dim[i])
                        ).sample()
                        for i in range(len(self.act_dim))
                    ]
                    random_actions = torch.cat(random_actions, dim=1)
                    actions = (1 - take_random) * to_numpy(
                        onehot_actions
                    ) + take_random * to_numpy(random_actions)
                else:
                    onehot_actions = list(map(onehot_from_logits, actor_out))
                    actions = torch.cat(onehot_actions, dim=-1)

            else:
                if use_gumbel or (use_target and self.target_noise is not None):
                    actions = gumbel_softmax(
                        actor_out, available_actions, hard=True, device=self.device
                    )  # gumbel has a gradient
                elif explore:
                    onehot_actions = gumbel_softmax(
                        actor_out, available_actions, hard=True, device=self.device
                    )  # gumbel has a gradient
                    # eps greedy exploration
                    eps = self.exploration.eval(t_env)
                    rand_numbers = np.random.rand(batch_size, 1)
                    # random actions sample uniformly from action space
                    logits = avail_choose(
                        torch.ones(batch_size, self.act_dim), available_actions
                    )
                    random_actions = OneHotCategorical(logits=logits).sample().numpy()
                    take_random = (rand_numbers < eps).astype(int)
                    actions = (1 - take_random) * to_numpy(
                        onehot_actions
                    ) + take_random * random_actions
                else:
                    actions = onehot_from_logits(
                        actor_out, available_actions
                    )  # no gradient

        else:
            if explore:
                actions = (
                    gaussian_noise(actor_out.shape, self.args.act_noise_std) + actor_out
                )
            elif use_target and self.target_noise is not None:
                assert isinstance(self.target_noise, float)
                actions = gaussian_noise(actor_out.shape, self.target_noise) + actor_out
            else:
                actions = actor_out
            # # clip the actions at the bounds of the action space
            # actions = torch.max(torch.min(actions, torch.from_numpy(self.act_space.high)), torch.from_numpy(self.act_space.low))

        return actions, eps

    def get_random_actions(self, obs, available_actions=None):
        """See parent class."""
        batch_size = obs.shape[0]

        if self.discrete:
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
        else:
            random_actions = np.random.uniform(
                self.act_space.low, self.act_space.high, size=(batch_size, self.act_dim)
            )

        return random_actions

    def soft_target_updates(self):
        """Polyak update the target networks."""
        # polyak updates to target networks
        soft_update(self.target_critic, self.critic, self.args.tau)
        soft_update(self.target_actor, self.actor, self.args.tau)

    def hard_target_updates(self):
        """Copy the live networks into the target networks."""
        # polyak updates to target networks
        hard_update(self.target_critic, self.critic)
        hard_update(self.target_actor, self.actor)
