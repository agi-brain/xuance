import wandb
import os
from typing import Dict
import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter  # tensorboardX to work with macos

from baselines.gpg.rl_navigation.modules.policy import Policy
from baselines.gpg.rl_navigation.make_g import build_graph


def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()


class Runner(object):
    """
    Class for training GPG policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """

    dt = 0.1

    def __init__(self, config: Dict):
        self.args = config["args"]
        self.env = config["env"]
        self.device = config["device"]
        self.num_agents = self.args.num_agents
        if config.__contains__("render_envs"):
            self.render_envs = config["render_envs"]

        # parameters
        self.graph_type = self.args.graph_type
        self.num_env_steps = self.args.num_env_steps
        self.episode_length = self.args.episode_length
        self.use_wandb = self.args.use_wandb

        # interval
        self.save_interval = self.args.save_interval
        # self.use_eval = self.args.use_eval
        # self.eval_interval = self.args.eval_interval
        self.log_interval = self.args.log_interval

        # dir
        self.model_dir = self.args.model_dir

        # if not testing model
        if not self.args.use_render:
            if self.use_wandb:
                self.save_dir = str(wandb.run.dir)
                self.run_dir = str(wandb.run.dir)
            else:
                self.run_dir = config["run_dir"]
                self.log_dir = str(self.run_dir / "logs")
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                self.writer = SummaryWriter(self.log_dir)
                self.save_dir = str(self.run_dir / "models")
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)

        # policy network
        self.policy = Policy(
            self.args,
            self.env.observation_space[0],
            self.env.action_space[0],
            device=self.device,
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.args.lr)
        if self.model_dir is not None:
            print(f"Restoring from checkpoint stored in {self.model_dir}")
            self.restore()
            self.gif_dir = self.model_dir

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        episodes = int(self.num_env_steps) // self.episode_length

        # This is where the episodes are actually run.
        for episode in range(episodes):
            reward_over_eps = []
            state, adj = self.env.reset()
            g = build_graph(adj)
            done = False

            for step in range(self.episode_length):
                action = self.policy.select_action(
                    state, g
                )  # shape [num_agents, action_dim]

                # Step through environment using chosen action
                state, adj, reward, done, infos = self.env.step(action)
                if self.graph_type == "dynamic":
                    g = build_graph(adj)
                # state.shape = [num_agents, state_dim]

                reward_over_eps.append(reward)
                # Save reward
                self.policy.reward_episode.append(reward)

            # compute return and update network
            train_infos = self.policy.update_policy(self.optimizer)

            # post process
            total_num_steps = (episode + 1) * self.episode_length

            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # log information
            if episode % self.log_interval == 0:
                env_infos = self.process_infos(infos)

                avg_ep_rew = np.mean(reward_over_eps) * self.episode_length
                train_infos["average_episode_rewards"] = avg_ep_rew
                print(
                    f"Average episode rewards is {avg_ep_rew:.3f} \t"
                    f"Total timesteps: {total_num_steps} \t "
                    f"Percentage complete {total_num_steps / self.num_env_steps * 100:.3f}"
                )
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # # eval
            # if episode % self.eval_interval == 0 and self.use_eval:
            #     self.eval(total_num_steps)

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
        for agent_id in range(self.num_agents):
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

    def get_collisions(self, env_infos: Dict):
        """
        Get the collisions from the env_infos
        Example: {'agent0/individual_rewards': [5],
                'agent0/time_to_goal': [0.6000000000000001],
                'agent0/min_time_to_goal': [0.23632679886748278],
                'agent0/dist_to_goal': [0.03768003822249384],
                'agent0/num_agent_collisions': [1.0],
                'agent0/num_obstacle_collisions': [0.0],
                'agent1/individual_rewards': [5],
                'agent1/time_to_goal': [0.6000000000000001],
                'agent1/min_time_to_goal': [0.3067362645187025],
                'agent1/dist_to_goal': [0.0387233764393595],
                'agent1/num_agent_collisions': [1.0],
                'agent1/num_obstacle_collisions': [0.0]}

        """
        collisions = 0
        for k, v in env_infos.items():
            if "collision" in k:
                collisions += v[0]
        return collisions
