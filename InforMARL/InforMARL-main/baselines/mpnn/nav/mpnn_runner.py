import os
import json
import wandb
import datetime
import numpy as np
import torch
from typing import Dict
from tensorboardX import SummaryWriter

import os, sys

sys.path.append(os.path.abspath(os.getcwd()))


from baselines.mpnn.nav.eval import evaluate
from baselines.mpnn.nav.learner import setup_master


np.set_printoptions(suppress=True, precision=4)


class Runner(object):
    dt = 0.1

    def __init__(self, config: Dict):
        self.args = config["args"]
        self.envs = config["envs"]
        self.device = config["device"]
        self.num_agents = config["num_agents"]

        # params
        self.num_env_steps = self.args.num_env_steps

        # interval
        self.save_interval = self.args.save_interval
        self.use_eval = self.args.use_eval
        self.eval_interval = self.args.eval_interval
        self.log_interval = self.args.log_interval

        # dir
        self.model_dir = self.args.model_dir

        self.use_wandb = self.args.use_wandb
        self.use_render = self.args.use_render

        # if not testing model
        if not self.use_render:
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

        # intialize master
        self.master = setup_master(self.args, device=self.device)
        # used during evaluation only
        self.eval_master, self.eval_env = setup_master(
            self.args, return_env=True, device=self.device
        )

    def save(self):
        savedict = {
            "models": [
                agent.actor_critic.state_dict() for agent in self.master.all_agents
            ]
        }
        ob_rms = (
            (None, None)
            if self.envs.ob_rms is None
            else (self.envs.ob_rms[0].mean, self.envs.ob_rms[0].var)
        )
        savedict["ob_rms"] = ob_rms
        torch.save(savedict, str(self.save_dir) + "/model.pt")

    def log_train(self, train_infos: Dict, total_num_steps: int):
        """
        Log training info.
        train_infos: (dict)
            information about training update.
        total_num_steps: (int)
            total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.args.use_wandb:
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
                if self.args.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writer.add_scalars(k, {k: np.mean(v)}, total_num_steps)

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
                            self.args.episode_length * self.dt
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

    def run(self):
        obs = self.envs.reset()  # shape - n_rollout_threads x num_agents x obs_dim
        self.master.initialize_obs(obs)
        n = len(self.master.all_agents)
        episode_rewards = torch.zeros(
            [self.args.n_rollout_threads, n], device=self.device
        )
        final_rewards = torch.zeros(
            [self.args.n_rollout_threads, n], device=self.device
        )
        rewards_over_eps = np.zeros(
            (
                self.args.episode_length,
                self.args.n_rollout_threads,
                self.args.num_agents,
                1,
            ),
            dtype=np.float32,
        )

        # start simulations
        start = datetime.datetime.now()
        episodes = (
            int(self.args.num_env_steps)
            // self.args.episode_length
            // self.args.n_rollout_threads
        )

        # this is where the episodes are actually run
        for episode in range(episodes):
            for step in range(self.args.episode_length):
                with torch.no_grad():
                    actions_list = self.master.act(step)
                agent_actions = np.transpose(np.array(actions_list), (1, 0, 2))
                # since we are using normaliser, we are also getting the actual
                # rewards before normalisation.
                # Refer nav.env_utils.vec_normalize.MultiAgentVecNormalize()
                obs, reward, done, info, orig_reward = self.envs.step(agent_actions)
                reward = torch.from_numpy(np.stack(reward)).float().to(self.device)
                orig_reward = torch.from_numpy(np.stack(orig_reward)).float()
                rewards_over_eps[step] = orig_reward.unsqueeze(2).cpu().numpy()
                episode_rewards += reward
                masks = torch.FloatTensor(1 - 1.0 * done).to(self.device)
                final_rewards *= masks
                final_rewards += (1 - masks) * episode_rewards
                episode_rewards *= masks

                self.master.update_rollout(obs, reward, masks)

            self.master.wrap_horizon()
            return_vals = self.master.update()
            value_loss = return_vals[:, 0]
            action_loss = return_vals[:, 1]
            dist_entropy = return_vals[:, 2]
            self.master.after_update()

            total_num_steps = (
                (episode + 1) * self.args.episode_length * self.args.n_rollout_threads
            )

            if episode % self.args.save_interval == 0 and not self.args.test:
                self.save()

            if episode % self.args.log_interval == 0:
                end = datetime.datetime.now()
                seconds = (end - start).total_seconds()
                avg_ep_rew = np.mean(rewards_over_eps) * self.args.episode_length
                print(
                    f"Average episode rewards is {avg_ep_rew:.3f} \t"
                    f"Total timesteps: {total_num_steps} \t "
                    f"Percentage complete {total_num_steps / self.num_env_steps * 100:.3f}"
                )
                train_infos = {
                    "value_loss": value_loss[0],
                    "action_loss": action_loss[0],
                    "entropy": dist_entropy[0],
                    "average_episode_rewards": avg_ep_rew,
                }
                self.log_train(train_infos, total_num_steps)
                env_infos = self.process_infos(info)
                self.log_env(env_infos, total_num_steps)
                # if not args.test:
                #     for idx in range(n):
                #         writer.add_scalar('agent'+str(idx)+'/training_reward', mean_reward[idx], episode)

            if (
                self.args.eval_interval is not None
                and episode % self.args.eval_interval == 0
            ):
                ob_rms = (
                    (None, None)
                    if self.envs.ob_rms is None
                    else (self.envs.ob_rms[0].mean, self.envs.ob_rms[0].var)
                )
                (
                    _,
                    eval_perstep_rewards,
                    final_min_dists,
                    num_success,
                    eval_episode_len,
                ) = evaluate(
                    self.args,
                    None,
                    self.master.all_policies,
                    ob_rms=ob_rms,
                    env=self.eval_env,
                    master=self.eval_master,
                )
                eval_reward = eval_perstep_rewards.mean() * self.args.episode_length
                eval_info = {}
                eval_info["eval_average_episode_rewards"] = np.array([eval_reward])
                self.log_env(eval_info, total_num_steps)
                # print(f'Evaluation {episode//self.args.eval_interval:d} | '
                #     f'Mean per-step reward {eval_reward:.2f}')
                # print(f'Num success {num_success:d}/{self.args.num_eval_episodes:d} | '
                # f'Episode Length {eval_episode_len:.2f}')
                # if final_min_dists:
                #     print(f'Final_dists_mean {np.stack(final_min_dists).mean(0)}')
                #     print(f'Final_dists_var {np.stack(final_min_dists).var(0)}')
                # print('='*50)
                # print()

                # if not args.test:
                #     writer.add_scalar('all/eval_success', 100.0*num_success/args.num_eval_episodes, episode)
                #     writer.add_scalar('all/episode_length', eval_episode_len, episode)
                #     for idx in range(n):
                #         writer.add_scalar('agent'+str(idx)+'/eval_per_step_reward', eval_perstep_rewards.mean(0)[idx], episode)
                #         if final_min_dists:
                #             writer.add_scalar('agent'+str(idx)+'/eval_min_dist', np.stack(final_min_dists).mean(0)[idx], episode)

                # curriculum_success_thres = 0.9
                # if return_early and num_success*1./args.num_eval_episodes > curriculum_success_thres:
                #     savedict = {'models': [agent.actor_critic.state_dict() for agent in master.all_agents]}
                #     ob_rms = (None, None) if envs.ob_rms is None else (envs.ob_rms[0].mean, envs.ob_rms[0].var)
                #     savedict['ob_rms'] = ob_rms
                #     savedir = args.save_dir+'/ep'+str(episode)+'.pt'
                #     torch.save(savedict, savedir)
                #     print('===========================================================================================\n')
                #     print('{} agents: training complete. Breaking.\n'.format(args.num_agents))
                #     print('===========================================================================================\n')
                #     break
