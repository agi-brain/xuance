import os, sys

sys.path.append(os.path.abspath(os.getcwd()))
import json
import datetime
import numpy as np
import torch
import utils
import random
from copy import deepcopy
from arguments import get_args
from tensorboardX import SummaryWriter
from eval import evaluate
from learner import setup_master
from pprint import pprint


np.set_printoptions(suppress=True, precision=4)


def train(args, return_early=False):
    writer = SummaryWriter(args.log_dir)
    envs = utils.make_parallel_envs(args)
    master = setup_master(args)
    # used during evaluation only
    eval_master, eval_env = setup_master(args, return_env=True)
    obs = envs.reset()  # shape - num_processes x num_agents x obs_dim
    master.initialize_obs(obs)
    n = len(master.all_agents)
    episode_rewards = torch.zeros([args.num_processes, n], device=args.device)
    final_rewards = torch.zeros([args.num_processes, n], device=args.device)

    # start simulations
    start = datetime.datetime.now()
    for j in range(args.num_updates):
        for step in range(args.num_steps):
            with torch.no_grad():
                actions_list = master.act(step)
            agent_actions = np.transpose(np.array(actions_list), (1, 0, 2))
            obs, reward, done, info = envs.step(agent_actions)
            reward = torch.from_numpy(np.stack(reward)).float().to(args.device)
            episode_rewards += reward
            masks = torch.FloatTensor(1 - 1.0 * done).to(args.device)
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            master.update_rollout(obs, reward, masks)

        master.wrap_horizon()
        return_vals = master.update()
        value_loss = return_vals[:, 0]
        action_loss = return_vals[:, 1]
        dist_entropy = return_vals[:, 2]
        master.after_update()

        if j % args.save_interval == 0 and not args.test:
            savedict = {
                "models": [
                    agent.actor_critic.state_dict() for agent in master.all_agents
                ]
            }
            ob_rms = (
                (None, None)
                if envs.ob_rms is None
                else (envs.ob_rms[0].mean, envs.ob_rms[0].var)
            )
            savedict["ob_rms"] = ob_rms
            savedir = args.save_dir + "/ep" + str(j) + ".pt"
            torch.save(savedict, savedir)

        total_num_steps = (j + 1) * args.num_processes * args.num_steps

        if j % args.log_interval == 0:
            end = datetime.datetime.now()
            seconds = (end - start).total_seconds()
            mean_reward = final_rewards.mean(dim=0).cpu().numpy()
            print(
                "Updates {} | Num timesteps {} | Time {} | FPS {}\nMean reward {}\nEntropy {:.4f} Value loss {:.4f} Policy loss {:.4f}\n".format(
                    j,
                    total_num_steps,
                    str(end - start),
                    int(total_num_steps / seconds),
                    mean_reward,
                    dist_entropy[0],
                    value_loss[0],
                    action_loss[0],
                )
            )
            if not args.test:
                for idx in range(n):
                    writer.add_scalar(
                        "agent" + str(idx) + "/training_reward", mean_reward[idx], j
                    )

                writer.add_scalar("all/value_loss", value_loss[0], j)
                writer.add_scalar("all/action_loss", action_loss[0], j)
                writer.add_scalar("all/dist_entropy", dist_entropy[0], j)

        if args.eval_interval is not None and j % args.eval_interval == 0:
            ob_rms = (
                (None, None)
                if envs.ob_rms is None
                else (envs.ob_rms[0].mean, envs.ob_rms[0].var)
            )
            print(
                "==========================================================================================="
            )
            (
                _,
                eval_perstep_rewards,
                final_min_dists,
                num_success,
                eval_episode_len,
            ) = evaluate(
                args,
                None,
                master.all_policies,
                ob_rms=ob_rms,
                env=eval_env,
                master=eval_master,
            )
            print(
                "Evaluation {:d} | Mean per-step reward {:.2f}".format(
                    j // args.eval_interval, eval_perstep_rewards.mean()
                )
            )
            print(
                "Num success {:d}/{:d} | Episode Length {:.2f}".format(
                    num_success, args.num_eval_episodes, eval_episode_len
                )
            )
            if final_min_dists:
                print("Final_dists_mean {}".format(np.stack(final_min_dists).mean(0)))
                print("Final_dists_var {}".format(np.stack(final_min_dists).var(0)))
            print(
                "===========================================================================================\n"
            )

            if not args.test:
                writer.add_scalar(
                    "all/eval_success", 100.0 * num_success / args.num_eval_episodes, j
                )
                writer.add_scalar("all/episode_length", eval_episode_len, j)
                for idx in range(n):
                    writer.add_scalar(
                        "agent" + str(idx) + "/eval_per_step_reward",
                        eval_perstep_rewards.mean(0)[idx],
                        j,
                    )
                    if final_min_dists:
                        writer.add_scalar(
                            "agent" + str(idx) + "/eval_min_dist",
                            np.stack(final_min_dists).mean(0)[idx],
                            j,
                        )

            curriculum_success_thres = 0.9
            if (
                return_early
                and num_success * 1.0 / args.num_eval_episodes
                > curriculum_success_thres
            ):
                savedict = {
                    "models": [
                        agent.actor_critic.state_dict() for agent in master.all_agents
                    ]
                }
                ob_rms = (
                    (None, None)
                    if envs.ob_rms is None
                    else (envs.ob_rms[0].mean, envs.ob_rms[0].var)
                )
                savedict["ob_rms"] = ob_rms
                savedir = args.save_dir + "/ep" + str(j) + ".pt"
                torch.save(savedict, savedir)
                print(
                    "===========================================================================================\n"
                )
                print(
                    "{} agents: training complete. Breaking.\n".format(args.num_agents)
                )
                print(
                    "===========================================================================================\n"
                )
                break

    writer.close()
    if return_early:
        return savedir


if __name__ == "__main__":
    args = get_args()
    if args.seed is None:
        args.seed = random.randint(0, 10000)
    args.num_updates = args.num_frames // args.num_steps // args.num_processes
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    pprint(vars(args))
    if not args.test:
        with open(os.path.join(args.save_dir, "params.json"), "w") as f:
            params = deepcopy(vars(args))
            params.pop("device")
            json.dump(params, f)
    train(args)
