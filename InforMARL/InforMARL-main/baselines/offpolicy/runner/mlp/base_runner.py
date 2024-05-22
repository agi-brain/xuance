import os
import wandb
import numpy as np
from itertools import chain
from torch.utils.tensorboard import SummaryWriter
import torch
import time

from baselines.offpolicy.utils.mlp_buffer import (
    MlpReplayBuffer,
    PrioritizedMlpReplayBuffer,
)
from baselines.offpolicy.utils.util import (
    is_discrete,
    is_multidiscrete,
    DecayThenFlatSchedule,
)


class MlpRunner(object):
    def __init__(self, config):
        """
        Base class for training MLP policies.

        :param config: (dict) Config dictionary containing parameters for training.
        """
        # non-tunable hyperparameters are in args
        self.args = config["args"]
        self.device = config["device"]
        self.q_learning = ["mqmix", "mvdn"]

        # set tunable hyperparameters
        self.share_policy = self.args.share_policy
        self.algorithm_name = self.args.algorithm_name
        self.env_name = self.args.env_name
        self.num_env_steps = self.args.num_env_steps
        self.use_wandb = self.args.use_wandb
        self.use_reward_normalization = self.args.use_reward_normalization
        self.use_per = self.args.use_per
        self.per_alpha = self.args.per_alpha
        self.per_beta_start = self.args.per_beta_start
        self.buffer_size = self.args.buffer_size
        self.batch_size = self.args.batch_size
        self.hidden_size = self.args.hidden_size
        self.use_soft_update = self.args.use_soft_update
        self.hard_update_interval = self.args.hard_update_interval
        self.train_interval = self.args.train_interval
        self.use_eval = self.args.use_eval
        self.eval_interval = self.args.eval_interval
        self.save_interval = self.args.save_interval
        self.log_interval = self.args.log_interval

        self.total_env_steps = (
            0  # total environment interactions collected during training
        )
        self.num_episodes_collected = 0  # total episodes collected during training
        self.total_train_steps = 0  # number of gradient updates performed
        self.last_train_T = 0
        self.last_eval_T = 0  # last episode after which a eval run was conducted
        self.last_save_T = 0  # last epsiode after which the models were saved
        self.last_log_T = 0
        self.last_hard_update_T = 0

        if config.__contains__("take_turn"):
            self.take_turn = config["take_turn"]
        else:
            self.take_turn = False

        if config.__contains__("use_same_share_obs"):
            self.use_same_share_obs = config["use_same_share_obs"]
        else:
            self.use_same_share_obs = False

        if config.__contains__("use_available_actions"):
            self.use_avail_acts = config["use_available_actions"]
        else:
            self.use_avail_acts = False

        self.episode_length = self.args.episode_length

        self.policy_info = config["policy_info"]
        self.policy_ids = sorted(list(self.policy_info.keys()))
        self.policy_mapping_fn = config["policy_mapping_fn"]

        self.num_agents = config["num_agents"]
        self.agent_ids = [i for i in range(self.num_agents)]

        self.env = config["env"]
        self.eval_env = config["eval_env"]
        self.num_envs = self.env.num_envs
        self.num_eval_envs = self.eval_env.num_envs

        # dir
        self.model_dir = self.args.model_dir
        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / "logs")
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / "models")
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        # initialize all the policies and organize the agents corresponding to each policy
        if self.algorithm_name == "matd3":
            from baselines.offpolicy.algorithms.matd3.algorithm.MATD3Policy import (
                MATD3Policy as Policy,
            )
            from baselines.offpolicy.algorithms.matd3.matd3 import MATD3 as TrainAlgo
        elif self.algorithm_name == "maddpg":
            from baselines.offpolicy.algorithms.maddpg.algorithm.MADDPGPolicy import (
                MADDPGPolicy as Policy,
            )
            from baselines.offpolicy.algorithms.maddpg.maddpg import MADDPG as TrainAlgo
        # elif self.algorithm_name == "masac":
        #     from offpolicy.algorithms.masac.algorithm.MASACPolicy import MASACPolicy as Policy
        #     from offpolicy.algorithms.masac.masac import MASAC as TrainAlgo
        elif self.algorithm_name == "mqmix":
            from baselines.offpolicy.algorithms.mqmix.algorithm.mQMixPolicy import (
                M_QMixPolicy as Policy,
            )
            from baselines.offpolicy.algorithms.mqmix.mqmix import M_QMix as TrainAlgo
        elif self.algorithm_name == "mvdn":
            from baselines.offpolicy.algorithms.mvdn.algorithm.mVDNPolicy import (
                M_VDNPolicy as Policy,
            )
            from baselines.offpolicy.algorithms.mvdn.mvdn import M_VDN as TrainAlgo
        else:
            raise NotImplementedError

        self.collecter = self.collect_rollout
        self.saver = (
            self.save_q if self.algorithm_name in self.q_learning else self.save
        )
        self.train = (
            self.batch_train_q
            if self.algorithm_name in self.q_learning
            else self.batch_train
        )
        self.restorer = (
            self.restore_q if self.algorithm_name in self.q_learning else self.restore
        )

        self.policies = {
            p_id: Policy(config, self.policy_info[p_id]) for p_id in self.policy_ids
        }

        if self.model_dir is not None:
            self.restorer()

        # initialize class for updating policies
        self.trainer = TrainAlgo(
            self.args,
            self.num_agents,
            self.policies,
            self.policy_mapping_fn,
            device=self.device,
        )

        self.policy_agents = {
            policy_id: sorted(
                [
                    agent_id
                    for agent_id in self.agent_ids
                    if self.policy_mapping_fn(agent_id) == policy_id
                ]
            )
            for policy_id in self.policies.keys()
        }

        self.policy_obs_dim = {
            policy_id: self.policies[policy_id].obs_dim for policy_id in self.policy_ids
        }
        self.policy_act_dim = {
            policy_id: self.policies[policy_id].act_dim for policy_id in self.policy_ids
        }
        self.policy_central_obs_dim = {
            policy_id: self.policies[policy_id].central_obs_dim
            for policy_id in self.policy_ids
        }

        num_train_iters = self.num_env_steps / self.train_interval
        self.beta_anneal = DecayThenFlatSchedule(
            self.per_beta_start, 1.0, num_train_iters, decay="linear"
        )

        if self.use_per:
            self.buffer = PrioritizedMlpReplayBuffer(
                self.per_alpha,
                self.policy_info,
                self.policy_agents,
                self.buffer_size,
                self.use_same_share_obs,
                self.use_avail_acts,
                self.use_reward_normalization,
            )
        else:
            self.buffer = MlpReplayBuffer(
                self.policy_info,
                self.policy_agents,
                self.buffer_size,
                self.use_same_share_obs,
                self.use_avail_acts,
                self.use_reward_normalization,
            )

    def run(self):
        """Collect a training episode and perform appropriate training, saving, logging, and evaluation steps."""
        # collect data
        self.trainer.prep_rollout()
        env_info = self.collecter(explore=True, training_episode=True, warmup=False)
        self.env_infos.update(env_info)
        # for k, v in env_info.items():
        #     self.env_infos[k].append(v)

        # save
        if (self.total_env_steps - self.last_save_T) / self.save_interval >= 1:
            self.saver()
            self.last_save_T = self.total_env_steps

        # log
        if ((self.total_env_steps - self.last_log_T) / self.log_interval) >= 1:
            self.log()
            self.last_log_T = self.total_env_steps

        # eval
        if (
            self.use_eval
            and ((self.total_env_steps - self.last_eval_T) / self.eval_interval) >= 1
        ):
            self.eval()
            self.last_eval_T = self.total_env_steps

        return self.total_env_steps

    def batch_train(self):
        """Do a gradient update for all policies."""
        self.trainer.prep_training()
        # gradient updates
        self.train_infos = []
        update_actor = True
        for p_id in self.policy_ids:
            if self.use_per:
                beta = self.beta_anneal.eval(self.total_train_steps)
                sample = self.buffer.sample(self.batch_size, beta, p_id)
            else:
                sample = self.buffer.sample(self.batch_size)

            update = (
                self.trainer.shared_train_policy_on_batch
                if self.use_same_share_obs
                else self.trainer.cent_train_policy_on_batch
            )

            train_info, new_priorities, idxes = update(p_id, sample)
            update_actor = train_info["update_actor"]

            if self.use_per:
                self.buffer.update_priorities(idxes, new_priorities, p_id)

            self.train_infos.append(train_info)

        if self.use_soft_update and update_actor:
            for pid in self.policy_ids:
                self.policies[pid].soft_target_updates()
        else:
            if (
                (self.total_env_steps - self.last_hard_update_T)
                / self.hard_update_interval
            ) >= 1:
                for pid in self.policy_ids:
                    self.policies[pid].hard_target_updates()
                self.last_hard_update_T = self.total_env_steps

    def batch_train_q(self):
        """Do a q-learning update to policy (used for QMix and VDN)."""
        self.trainer.prep_training()
        # gradient updates
        self.train_infos = []
        for p_id in self.policy_ids:
            if self.use_per:
                beta = self.beta_anneal.eval(self.total_train_steps)
                sample = self.buffer.sample(self.batch_size, beta, p_id)
            else:
                sample = self.buffer.sample(self.batch_size)

            train_info, new_priorities, idxes = self.trainer.train_policy_on_batch(
                sample, self.use_same_share_obs
            )

            if self.use_per:
                self.buffer.update_priorities(idxes, new_priorities, p_id)

            self.train_infos.append(train_info)

        if self.use_soft_update:
            self.trainer.soft_target_updates()
        else:
            if (
                self.total_env_steps - self.last_hard_update_T
            ) / self.hard_update_interval >= 1:
                self.trainer.hard_target_updates()
                self.last_hard_update_T = self.total_env_steps

    def save(self):
        """Save all policies to the path specified by the config."""
        for pid in self.policy_ids:
            policy_critic = self.policies[pid].critic
            critic_save_path = self.save_dir + "/" + str(pid)
            if not os.path.exists(critic_save_path):
                os.makedirs(critic_save_path)
            torch.save(policy_critic.state_dict(), critic_save_path + "/critic.pt")

            policy_actor = self.policies[pid].actor
            actor_save_path = self.save_dir + "/" + str(pid)
            if not os.path.exists(actor_save_path):
                os.makedirs(actor_save_path)
            torch.save(policy_actor.state_dict(), actor_save_path + "/actor.pt")

    def save_q(self):
        """Save all policies to the path specified by the config. Used for QMix and VDN."""
        for pid in self.policy_ids:
            policy_Q = self.policies[pid].q_network
            p_save_path = self.save_dir + "/" + str(pid)
            if not os.path.exists(p_save_path):
                os.makedirs(p_save_path)
            torch.save(policy_Q.state_dict(), p_save_path + "/q_network.pt")

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        torch.save(self.trainer.mixer.state_dict(), self.save_dir + "/mixer.pt")

    def restore(self):
        """Load policies policies from pretrained models specified by path in config."""
        for pid in self.policy_ids:
            path = str(self.model_dir) + str(pid)
            print(f"Loading the pretrained model from {path}")
            policy_critic_state_dict = torch.load(
                path + "/critic.pt", map_location=torch.device("cpu")
            )
            policy_actor_state_dict = torch.load(
                path + "/actor.pt", map_location=torch.device("cpu")
            )

            self.policies[pid].critic.load_state_dict(policy_critic_state_dict)
            self.policies[pid].actor.load_state_dict(policy_actor_state_dict)

    def restore_q(self):
        """Load policies policies from pretrained models specified by path in config. Used for QMix and VDN."""
        for pid in self.policy_ids:
            path = str(self.model_dir) + str(pid)
            print(f"Loading the pretrained model from {path}")
            policy_q_state_dict = torch.load(
                path + "/q_network.pt", map_location=torch.device("cpu")
            )
            self.policies[pid].q_network.load_state_dict(policy_q_state_dict)

        policy_mixer_state_dict = torch.load(
            str(self.model_dir) + "/mixer.pt", map_location=torch.device("cpu")
        )
        self.trainer.mixer.load_state_dict(policy_mixer_state_dict)

    @torch.no_grad()
    def warmup(self, num_warmup_episodes):
        """
        Fill replay buffer with enough episodes to begin training.

        :param: num_warmup_episodes (int): number of warmup episodes to collect.
        """
        self.trainer.prep_rollout()
        warmup_rewards = []
        print("Warmup...")
        for _ in range(int(num_warmup_episodes // self.num_envs) + 1):
            env_info = self.collecter(explore=True, training_episode=False, warmup=True)
            warmup_rewards.append(env_info["average_step_rewards"])
        warmup_reward = np.mean(warmup_rewards)
        print(f"Warmup average step rewards: {warmup_reward}")

    def process_infos(self, infos):
        """Process infos returned by environment."""
        env_infos = {}
        for agent_id in range(self.num_agents):
            idv_rews = []
            dist_goals, time_to_goals, min_times_to_goal = [], [], []
            idv_collisions, obst_collisions = [], []
            for info in infos:
                if "individual_reward" in info[agent_id].keys():
                    idv_rews.append(info[agent_id]["individual_reward"])
                if "Dist_to_goal" in info[agent_id].keys():
                    dist_goals.append(info[agent_id]["Dist_to_goal"])
                if "Time_req_to_goal" in info[agent_id].keys():
                    times = info[agent_id]["Time_req_to_goal"]
                    if times == -1:
                        times = (
                            self.all_args.episode_length * self.dt
                        )  # NOTE: Hardcoding `dt`
                    time_to_goals.append(times)
                if "Num_agent_collisions" in info[agent_id].keys():
                    idv_collisions.append(info[agent_id]["Num_agent_collisions"])
                if "Num_obst_collisions" in info[agent_id].keys():
                    obst_collisions.append(info[agent_id]["Num_obst_collisions"])
                if "Min_time_to_goal" in info[agent_id].keys():
                    min_times_to_goal.append(info[agent_id]["Min_time_to_goal"])

            agent_rew = f"agent{agent_id}/individual_rewards"
            times = f"agent{agent_id}/time_to_goal"
            dists = f"agent{agent_id}/dist_to_goal"
            agent_col = f"agent{agent_id}/num_agent_collisions"
            obst_col = f"agent{agent_id}/num_obstacle_collisions"
            min_times = f"agent{agent_id}/min_time_to_goal"

            env_infos[agent_rew] = idv_rews
            env_infos[times] = time_to_goals
            env_infos[min_times] = min_times_to_goal
            env_infos[dists] = dist_goals
            env_infos[agent_col] = idv_collisions
            env_infos[obst_col] = obst_collisions
        return env_infos

    def log(self):
        raise NotImplementedError

    def log_clear(self):
        raise NotImplementedError

    def log_env(self, env_info, suffix=None):
        """
        Log information related to the environment.
        :param env_info: (dict) contains logging information related to the environment.
        :param suffix: (str) optional string to add to end of keys in env_info when logging.
        """
        for k, v in env_info.items():
            if len(v) > 0:
                v = np.mean(v)
                suffix_k = k if suffix is None else suffix + k
                print(suffix_k + " is " + str(v))
                if self.use_wandb:
                    wandb.log({suffix_k: v}, step=self.total_env_steps)
                else:
                    self.writter.add_scalars(
                        suffix_k, {suffix_k: v}, self.total_env_steps
                    )

    def log_train(self, policy_id, train_info):
        """
        Log information related to training.
        :param policy_id: (str) policy id corresponding to the information contained in train_info.
        :param train_info: (dict) contains logging information related to training.
        """
        for k, v in train_info.items():
            policy_k = str(policy_id) + "/" + k
            if self.use_wandb:
                wandb.log({policy_k: v}, step=self.total_env_steps)
            else:
                self.writter.add_scalars(policy_k, {policy_k: v}, self.total_env_steps)

    def collect_rollout(self):
        """Collect a rollout and store the transitions in the buffer."""
        raise NotImplementedError
