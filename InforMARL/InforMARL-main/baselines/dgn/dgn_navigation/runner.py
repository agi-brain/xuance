import wandb
import os
from typing import Dict
import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter  # tensorboardX to work with macos

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

from baselines.dgn.dgn_navigation.model import DGN, ATT
from baselines.dgn.dgn_navigation.buffer import DGNReplayBuffer, ATOCReplayBuffer


def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()


class Runner(object):
    dt = 0.1

    def __init__(self, config: Dict):
        self.args = config["args"]
        self.env = config["env"]
        self.device = config["device"]
        if config.__contains__("render_envs"):
            self.render_envs = config["render_envs"]

        # get parameters
        self.n_ant = self.env.num_agents
        self.observation_space = self.env.observation_space[0].shape[0]
        self.n_actions = self.env.action_space[0].n

        self.episode_length = self.args.episode_length
        self.n_episode = int(self.args.num_env_steps) // self.args.episode_length

        # interval
        self.save_interval = self.args.save_interval
        self.log_interval = self.args.log_interval

        # wandb
        self.use_wandb = self.args.use_wandb

        # dir
        self.model_dir = self.args.model_dir

        # DGN spec parameters
        self.threshold = self.args.threshold
        self.hidden_dim = self.args.hidden_dim
        self.capacity = self.args.capacity
        self.batch_size = self.args.batch_size
        self.n_epoch = self.args.n_epoch
        self.epsilon = self.args.epsilon
        self.comm_flag = self.args.comm_flag
        self.tau = self.args.tau
        self.gamma = self.args.gamma

        # if not testing model
        if not self.args.use_render:
            if self.args.use_wandb:
                self.save_dir = str(wandb.run.dir)
                self.run_dir = str(wandb.run.dir)
            else:
                self.run_dir = config["run_dir"]
                self.log_dir = str(self.run_dir / "logs")
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                self.writer = SummaryWriter(self.log_dir)  # initialize writer
                self.save_dir = str(self.run_dir / "models")
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)

        if self.model_dir is not None:
            print(f"Restoring from checkpoint stored in {self.model_dir}")
            self.restore()
            self.gif_dir = self.model_dir

    def convert_action(self, envs, actions):
        if envs.action_space[0].__class__.__name__ == "MultiDiscrete":
            for i in range(envs.action_space[0].shape):
                uc_actions_env = np.eye(envs.action_space[0].high[i] + 1)[
                    actions[:, :, i]
                ]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif envs.action_space[0].__class__.__name__ == "Discrete":
            actions_env = np.eye(envs.action_space[0].n)[actions]
        return actions_env

    def save(self):
        """Save policy network."""
        torch.save(self.policy.state_dict(), str(self.save_dir) + "/actor.pt")

    def restore(self):
        """Restore policy's networks from a saved model."""
        policy_actor_state_dict = torch.load(
            str(self.model_dir) + "/actor.pt", map_location=torch.device("cpu")
        )
        self.policy.load_state_dict(policy_actor_state_dict)

    def process_infos(self, infos):
        """Process infos returned by environment."""
        env_infos = {}
        for agent_id in range(self.n_ant):
            idv_rews = []
            dist_goals, time_to_goals, min_times_to_goal = [], [], []
            idv_collisions, obst_collisions = [], []
            # for info in infos:
            if "individual_reward" in infos[agent_id].keys():
                idv_rews.append(infos[agent_id]["individual_reward"])
            if "Dist_to_goal" in infos[agent_id].keys():
                dist_goals.append(infos[agent_id]["Dist_to_goal"])
            if "Time_req_to_goal" in infos[agent_id].keys():
                times = infos[agent_id]["Time_req_to_goal"]
                if times == -1:
                    times = self.args.episode_length * self.dt  # NOTE: Hardcoding `dt`
                time_to_goals.append(times)
            if "Num_agent_collisions" in infos[agent_id].keys():
                idv_collisions.append(infos[agent_id]["Num_agent_collisions"])
            if "Num_obst_collisions" in infos[agent_id].keys():
                obst_collisions.append(infos[agent_id]["Num_obst_collisions"])
            if "Min_time_to_goal" in infos[agent_id].keys():
                min_times_to_goal.append(infos[agent_id]["Min_time_to_goal"])

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

    def log_train(self, train_infos: Dict, total_num_steps: int):
        """
        Log training info.
        train_infos: (dict)
            information about training update.
        total_num_steps: (int)
            total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writer.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos: Dict, total_num_steps: int):
        """
        Log env info.
        env_infos: (dict)
            information about env state.
        total_num_steps: (int)
            total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writer.add_scalars(k, {k: np.mean(v)}, total_num_steps)


class ATOCRunner(Runner):
    """
    Class for training DGN+ATOC policies.
    """

    def __init__(self, config: Dict):
        super(ATOCRunner, self).__init__(config)

        # init model
        self.buff = ATOCReplayBuffer(
            self.capacity, self.observation_space, self.n_actions, self.n_ant
        )
        self.model = DGN(
            self.n_ant, self.observation_space, self.hidden_dim, self.n_actions
        )
        self.model_tar = DGN(
            self.n_ant, self.observation_space, self.hidden_dim, self.n_actions
        )

        self.model = self.model.to(self.device)
        self.model_tar = self.model_tar.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

        # attention layer
        self.att = ATT(self.observation_space).to(self.device)
        self.att_tar = ATT(self.observation_space).to(self.device)
        self.att_tar.load_state_dict(self.att.state_dict())
        self.optimizer_att = optim.Adam(self.att.parameters(), lr=0.0001)
        self.criterion = nn.BCELoss()

        self.M_Null = torch.Tensor(np.array([np.eye(self.n_ant)] * self.batch_size)).to(
            self.device
        )
        self.M_ZERO = torch.Tensor(
            np.zeros((self.batch_size, self.n_ant, self.n_ant))
        ).to(self.device)

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""

        cost_all = 0
        cost_comm = 0
        i_episode = 0

        while i_episode < self.n_episode:
            if i_episode > 40:
                self.epsilon -= 0.001
                if self.epsilon < 0.01:
                    self.epsilon = 0.01
            i_episode += 1
            steps = 0
            reward_over_eps = []
            obs, adj = self.env.reset()

            while steps < self.episode_length:
                steps += 1
                action = []
                cost_all += adj.sum()
                v_a = np.array(
                    self.att(torch.Tensor(np.array([obs])).to(self.device))[0]
                    .cpu()
                    .data
                )

                for i in range(self.n_ant):
                    if np.random.rand() < self.epsilon:
                        adj[i] = adj[i] * 0 if np.random.rand() < 0.5 else adj[i] * 1
                    else:
                        adj[i] = (
                            adj[i] * 0 if v_a[i][0] < self.threshold else adj[i] * 1
                        )
                n_adj = adj * self.comm_flag
                cost_comm += n_adj.sum()
                n_adj = n_adj + np.eye(self.n_ant)

                q = self.model(
                    torch.Tensor(np.array([obs])).to(self.device),
                    torch.Tensor(np.array([n_adj])).to(self.device),
                )[0]
                for i in range(self.n_ant):
                    if np.random.rand() < self.epsilon:
                        a = np.random.randint(self.n_actions)
                    else:
                        a = q[i].argmax().item()
                    action.append(a)

                orig_action = action
                action = self.convert_action(self.env, action)

                next_obs, next_adj, reward, terminated, infos = self.env.step(action)
                reward_over_eps.append(reward)

                self.buff.add(
                    np.array(obs),
                    orig_action,
                    reward,
                    np.array(next_obs),
                    n_adj,
                    next_adj,
                    terminated,
                )
                obs = next_obs
                adj = next_adj

            ## compute return and update network
            # train_infos = self.policy.update_policy(self.optimizer)
            train_infos = (
                dict()
            )  # since we don't have policy network in DGN, init to empty dict

            # post process
            total_num_steps = (i_episode + 1) * self.episode_length

            # log information
            if i_episode % self.log_interval == 0:
                env_infos = self.process_infos(infos)

                avg_ep_rew = np.mean(reward_over_eps) * self.episode_length
                train_infos["average_episode_rewards"] = avg_ep_rew
                print(
                    f"Average episode rewards is {avg_ep_rew:.3f} \t"
                    f"Total timesteps: {total_num_steps} \t "
                    f"Percentage complete {total_num_steps / self.args.num_env_steps * 100:.3f}"
                )
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # update network only after we have enough experience
            if self.buff.len > self.batch_size:
                for e in range(self.n_epoch):
                    O, A, R, Next_O, Matrix, Next_Matrix, D = self.buff.getBatch(
                        self.batch_size
                    )
                    O = torch.Tensor(O).to(self.device)
                    Matrix = torch.Tensor(Matrix).to(self.device)
                    Next_O = torch.Tensor(Next_O).to(self.device)
                    Next_Matrix = torch.Tensor(Next_Matrix).to(self.device)

                    label = (
                        self.model(Next_O, Next_Matrix + self.M_Null).max(dim=2)[0]
                        - self.model(Next_O, self.M_Null).max(dim=2)[0]
                    )
                    label = (label - label.mean()) / (label.std() + 0.000001) + 0.5
                    label = torch.clamp(label, 0, 1).unsqueeze(-1).detach()
                    loss = self.criterion(self.att(Next_O), label)
                    self.optimizer_att.zero_grad()
                    loss.backward()
                    self.optimizer_att.step()

                    V_A_D = self.att_tar(Next_O).expand(-1, -1, self.n_ant)
                    Next_Matrix = torch.where(
                        V_A_D > self.threshold, Next_Matrix, self.M_ZERO
                    )
                    Next_Matrix = Next_Matrix * self.comm_flag + self.M_Null

                    q_values = self.model(O, Matrix)
                    target_q_values = self.model_tar(Next_O, Next_Matrix).max(dim=2)[0]
                    target_q_values = np.array(target_q_values.cpu().data)
                    expected_q = np.array(q_values.cpu().data)

                    for j in range(self.batch_size):
                        for i in range(self.n_ant):
                            expected_q[j][i][A[j][i]] = (
                                R[j][i]
                                + (1 - D[j]) * self.gamma * target_q_values[j][i]
                            )

                    loss = (
                        (q_values - torch.Tensor(expected_q).to(self.device))
                        .pow(2)
                        .mean()
                    )
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    with torch.no_grad():
                        for p, p_targ in zip(
                            self.model.parameters(), self.model_tar.parameters()
                        ):
                            p_targ.data.mul_(self.tau)
                            p_targ.data.add_((1 - self.tau) * p.data)
                        for p, p_targ in zip(
                            self.att.parameters(), self.att_tar.parameters()
                        ):
                            p_targ.data.mul_(self.tau)
                            p_targ.data.add_((1 - self.tau) * p.data)


class DGNRunner(Runner):
    """
    Class for training DGN policies.
    """

    dt = 0.1

    def __init__(self, config: Dict):
        super(DGNRunner, self).__init__(config)

        # init model
        self.buff = DGNReplayBuffer(self.capacity)
        self.model = DGN(
            self.n_ant, self.observation_space, self.hidden_dim, self.n_actions
        )
        self.model_tar = DGN(
            self.n_ant, self.observation_space, self.hidden_dim, self.n_actions
        )

        self.model = self.model.to(self.device)
        self.model_tar = self.model_tar.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

        self.O = np.ones((self.batch_size, self.n_ant, self.observation_space))
        self.Next_O = np.ones((self.batch_size, self.n_ant, self.observation_space))
        self.Matrix = np.ones((self.batch_size, self.n_ant, self.n_ant))
        self.Next_Matrix = np.ones((self.batch_size, self.n_ant, self.n_ant))

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""

        cost_all = 0
        cost_comm = 0
        i_episode = 0

        while i_episode < self.n_episode:
            if i_episode > 100:
                self.epsilon -= 0.0004
                if self.epsilon < 0.1:
                    self.epsilon = 0.1
            i_episode += 1
            steps = 0
            reward_over_eps = []
            obs, adj = self.env.reset()

            while steps < self.episode_length:
                steps += 1
                action = []
                q = self.model(
                    torch.Tensor(np.array([obs])).to(self.device),
                    torch.Tensor(adj).to(self.device),
                )[0]
                for i in range(self.n_ant):
                    if np.random.rand() < self.epsilon:
                        a = np.random.randint(self.n_actions)
                    else:
                        a = q[i].argmax().item()
                    action.append(a)

                orig_action = action
                action = self.convert_action(self.env, action)

                next_obs, next_adj, reward, terminated, infos = self.env.step(action)
                reward_over_eps.append(reward)

                self.buff.add(
                    np.array(obs),
                    orig_action,
                    reward,
                    np.array(next_obs),
                    adj,
                    next_adj,
                    terminated,
                )
                obs = next_obs
                adj = next_adj

            ## compute return and update network
            # train_infos = self.policy.update_policy(self.optimizer)
            train_infos = (
                dict()
            )  # since we don't have policy network in DGN, init to empty dict

            # post process
            total_num_steps = (i_episode + 1) * self.episode_length

            # log information
            if i_episode % self.log_interval == 0:
                env_infos = self.process_infos(infos)

                avg_ep_rew = np.mean(reward_over_eps) * self.episode_length
                train_infos["average_episode_rewards"] = avg_ep_rew
                print(
                    f"Average episode rewards is {avg_ep_rew:.3f} \t"
                    f"Total timesteps: {total_num_steps} \t "
                    f"Percentage complete {total_num_steps / self.args.num_env_steps * 100:.3f}"
                )
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            if i_episode < 100:
                continue

            # update network only after we have enough experience
            if self.buff.num_experiences > self.batch_size:
                for e in range(self.n_epoch):
                    batch = self.buff.getBatch(self.batch_size)
                    for j in range(self.batch_size):
                        sample = batch[j]
                        self.O[j] = sample[0]
                        self.Next_O[j] = sample[3]
                        self.Matrix[j] = sample[4]
                        self.Next_Matrix[j] = sample[5]

                    q_values = self.model(
                        torch.Tensor(self.O).to(self.device),
                        torch.Tensor(self.Matrix).to(self.device),
                    )
                    target_q_values = self.model_tar(
                        torch.Tensor(self.Next_O).to(self.device),
                        torch.Tensor(self.Next_Matrix).to(self.device),
                    ).max(dim=2)[0]
                    target_q_values = np.array(target_q_values.cpu().data)
                    expected_q = np.array(q_values.cpu().data)

                    for j in range(self.batch_size):
                        sample = batch[j]
                        for i in range(self.n_ant):
                            expected_q[j][i][sample[1][i]] = (
                                sample[2][i]
                                + (1 - sample[6]) * self.gamma * target_q_values[j][i]
                            )

                    loss = (
                        (q_values - torch.Tensor(expected_q).to(self.device))
                        .pow(2)
                        .mean()
                    )
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                if i_episode % 5 == 0:
                    self.model_tar.load_state_dict(self.model.state_dict())
