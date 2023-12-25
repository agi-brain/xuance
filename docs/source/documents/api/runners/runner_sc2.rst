Runner_SC2
==============================================

This part constructs a reinforcement learning framework for training and testing agents in a StarCraft II environment.

.. raw:: html

    <br><hr>


PyTorch
------------------------------------------

.. py:class::
    xuance.torch.runners.runner_sc2.SC2_Runner(args)

    :param args: the arguments.
    :type args: Namespace

.. py:function::
    xuance.torch.runners.runner_sc2.SC2_Runner.init_rnn_hidden()

    Initialize the hidden states of recurrent neural networks used in the agent's policy and critic.

    :return: the initialized hidden states for the policy and critic.
    :rtype: tuple

.. py:function::
    xuance.torch.runners.runner_sc2.SC2_Runner.get_agent_num()

    Retrieve the number of agents and enemies in the environment.

    :return: the number of agents and enemies.
    :rtype: tuple

.. py:function::
    xuance.torch.runners.runner_sc2.SC2_Runner.log_infos(info, x_index)

    Log information during training or testing.

    :param info: recorded information.
    :type info: dict
    :param x_index: the current step or index at which the information is being logged.
    :type x_index: int

.. py:function::
    xuance.torch.runners.runner_sc2.SC2_Runner.log_videos(info, fps, x_index)

    Log video data during training or testing.

    :param info: contain video data to be logged.
    :type info: dict
    :param fps: specifies the number of frames displayed per second in the logged video.
    :type fps: int
    :param x_index: the current step or index at which the information is being logged..
    :type x_index: int

.. py:function::
    xuance.torch.runners.runner_sc2.SC2_Runner.get_actions(obs_n, avail_actions, *rnn_hidden, state, test_mode)

    Obtain actions from the agents based on their observations and hidden states.

    :param obs_n: The joint observations of n agents.
    :type obs_n: np.ndarray
    :param avail_actions: The mask varibales for availabel actions.
    :type avail_actions: Tensor
    :param rnn_hidden: The last final hidden states of the sequence.
    :type rnn_hidden: Tensor
    :param state: The state input.
    :type state: Tensor
    :param test_mode: whether the method is being called in a test mode.
    :type test_mode: bool
    :return: the chosen actions by the agent, Log probability of the chosen actions,
             the updated hidden state for the policy's recurrent neural network,
             the updated hidden state for the critic's RNN, One-hot representation of the chosen actions and
             estimated values associated with the chosen actions.
    :rtype: dict

.. py:function::
    xuance.torch.runners.runner_sc2.SC2_Runner.get_battles_info()

    Aggregate information about battles in the environment.

    :return: Total number of battles in the environment, total number of battles won, total number of Allies killed,
            total number of enemies killed.
    :rtype: tuple

.. py:function::
    xuance.torch.runners.runner_sc2.SC2_Runner.get_battles_result(last_battles_info)

    Calculate and return various statistics related to battles based on the provided information.

    :param last_battles_info: the information about battles(total battles, battles won, dead allies, dead enemies).
    :type last_battles_info: tuple
    :return: the ratio of battles won to battles played, the ratio of dead allies to the estimated total count
            of allies in battles, the ratio of dead enemies to the estimated total count of enemies in battles.
    :rtype: tuple

.. py:function::
    xuance.torch.runners.runner_sc2.SC2_Runner.run_episodes(test_mode)

    Execute a series of episodes in the environment, either for training or testing purposes.

    :param test_mode: control the mode in which episodes are executed.
    :type test_mode: bool
    :return: the average episode score achieved during the training or testing process.
    :rtype: float

.. py:function::
    xuance.torch.runners.runner_sc2.SC2_Runner.test_episodes(test_T, n_test_runs)

    Run multiple testing cycles in a testing mode and calculate statistics such as test scores and win rates.

    :param test_T: the time step for recording test results.
    :type test_T: int
    :param n_test_runs: the number of testing cycles to execute.
    :type n_test_runs: int
    :return: the average test score, the standard deviation of test scores and the win rate.
    :rtype: tuple

.. py:function::
    xuance.torch.runners.runner_sc2.SC2_Runner.run()

    Orchestrate the entire training or testing process.

.. py:function::
    xuance.torch.runners.runner_sc2.SC2_Runner.benchmark()

    Perform a benchmarking process, which involves training and evaluating the models over multiple epochs.

.. py:function::
    xuance.torch.runners.runner_sc2.SC2_Runner.time_estimate(start)

    Estimate the time passed and the remaining time based on the elapsed time since a specified start time.

    :param start: the total time passed since the provided start time.
    :type start: int
    :return: the time passed, the estimated time left.
    :rtype: tuple

.. raw:: html

    <br><hr>


Source Code
-----------------

.. tabs::

  .. group-tab:: PyTorch

    .. code-block:: python

        import os
        import socket
        from pathlib import Path
        from .runner_basic import Runner_Base
        from xuance.torch.agents import REGISTRY as REGISTRY_Agent
        import wandb
        from torch.utils.tensorboard import SummaryWriter
        import time
        import numpy as np
        from copy import deepcopy


        class SC2_Runner(Runner_Base):
            def __init__(self, args):
                super(SC2_Runner, self).__init__(args)
                self.fps = args.fps
                self.args = args
                self.render = args.render
                self.test_envs = None

                time_string = time.asctime().replace(" ", "").replace(":", "_")
                seed = f"seed_{self.args.seed}_"
                self.args.model_dir_load = args.model_dir
                self.args.model_dir_save = os.path.join(os.getcwd(), args.model_dir, seed + time_string)
                if (not os.path.exists(self.args.model_dir_save)) and (not args.test_mode):
                    os.makedirs(self.args.model_dir_save)

                if args.logger == "tensorboard":
                    log_dir = os.path.join(os.getcwd(), args.log_dir, seed + time_string)
                    if not os.path.exists(log_dir):
                        os.makedirs(log_dir)
                    self.writer = SummaryWriter(log_dir)
                    self.use_wandb = False
                elif args.logger == "wandb":
                    config_dict = vars(args)
                    wandb_dir = Path(os.path.join(os.getcwd(), args.log_dir))
                    if not wandb_dir.exists():
                        os.makedirs(str(wandb_dir))
                    wandb.init(config=config_dict,
                            project=args.project_name,
                            entity=args.wandb_user_name,
                            notes=socket.gethostname(),
                            dir=wandb_dir,
                            group=args.env_id,
                            job_type=args.agent,
                            name=args.seed,
                            reinit=True)
                    self.use_wandb = True
                else:
                    raise "No logger is implemented."

                self.running_steps = args.running_steps
                self.training_frequency = args.training_frequency
                self.current_step = 0
                self.env_step = 0
                self.current_episode = np.zeros((self.envs.num_envs,), np.int32)
                self.episode_length = self.envs.max_episode_length
                self.num_agents, self.num_enemies = self.get_agent_num()
                args.n_agents = self.num_agents
                self.dim_obs, self.dim_act, self.dim_state = self.envs.dim_obs, self.envs.dim_act, self.envs.dim_state
                args.dim_obs, args.dim_act = self.dim_obs, self.dim_act
                args.obs_shape, args.act_shape = (self.dim_obs,), ()
                args.rew_shape = args.done_shape = (1,)
                args.action_space = self.envs.action_space
                args.state_space = self.envs.state_space

                # environment details, representations, policies, optimizers, and agents.
                self.agents = REGISTRY_Agent[args.agent](args, self.envs, args.device)
                self.on_policy = self.agents.on_policy

            def init_rnn_hidden(self):
                rnn_hidden = self.agents.policy.representation.init_hidden(self.n_envs * self.num_agents)
                if self.on_policy and self.args.agent in ["MAPPO"]:
                    rnn_hidden_critic = self.agents.policy.representation_critic.init_hidden(self.n_envs * self.num_agents)
                else:
                    rnn_hidden_critic = [None, None]
                return rnn_hidden, rnn_hidden_critic

            def get_agent_num(self):
                return self.envs.num_agents, self.envs.num_enemies

            def log_infos(self, info: dict, x_index: int):
                """
                info: (dict) information to be visualized
                n_steps: current step
                """
                if x_index <= self.running_steps:
                    if self.use_wandb:
                        for k, v in info.items():
                            wandb.log({k: v}, step=x_index)
                    else:
                        for k, v in info.items():
                            try:
                                self.writer.add_scalar(k, v, x_index)
                            except:
                                self.writer.add_scalars(k, v, x_index)

            def log_videos(self, info: dict, fps: int, x_index: int = 0):
                if x_index <= self.running_steps:
                    if self.use_wandb:
                        for k, v in info.items():
                            wandb.log({k: wandb.Video(v, fps=fps, format='gif')}, step=x_index)
                    else:
                        for k, v in info.items():
                            self.writer.add_video(k, v, fps=fps, global_step=x_index)

            def get_actions(self, obs_n, avail_actions, *rnn_hidden, state=None, test_mode=False):
                log_pi_n, values_n, actions_n_onehot = None, None, None
                rnn_hidden_policy, rnn_hidden_critic = rnn_hidden[0], rnn_hidden[1]
                if self.on_policy:
                    if self.args.agent == "COMA":
                        rnn_hidden_next, actions_n, actions_n_onehot = self.agents.act(obs_n, *rnn_hidden_policy,
                                                                                    avail_actions=avail_actions,
                                                                                    test_mode=test_mode)
                    elif self.args.agent == "VDAC":
                        rnn_hidden_next, actions_n, values_n = self.agents.act(obs_n, *rnn_hidden_policy,
                                                                            avail_actions=avail_actions,
                                                                            state=state,
                                                                            test_mode=test_mode)
                    else:
                        rnn_hidden_next, actions_n, log_pi_n = self.agents.act(obs_n, *rnn_hidden_policy,
                                                                            avail_actions=avail_actions,
                                                                            test_mode=test_mode)
                    if test_mode:
                        rnn_hidden_critic_next, values_n = None, 0
                    else:
                        if self.args.agent == "VDAC":
                            rnn_hidden_critic_next = [None, None]
                        else:
                            kwargs = {"state": state}
                            if self.args.agent == "COMA":
                                kwargs.update({"actions_n": actions_n, "actions_onehot": actions_n_onehot})
                            rnn_hidden_critic_next, values_n = self.agents.values(obs_n, *rnn_hidden_critic, **kwargs)
                else:
                    rnn_hidden_next, actions_n = self.agents.act(obs_n, *rnn_hidden_policy,
                                                                avail_actions=avail_actions, test_mode=test_mode)
                    rnn_hidden_critic_next = None
                return {'actions_n': actions_n, 'log_pi': log_pi_n,
                        'rnn_hidden': rnn_hidden_next, 'rnn_hidden_critic': rnn_hidden_critic_next,
                        'act_n_onehot': actions_n_onehot, 'values': values_n}

            def get_battles_info(self):
                battles_game, battles_won = self.envs.battles_game.sum(), self.envs.battles_won.sum()
                dead_allies, dead_enemies = self.envs.dead_allies_count.sum(), self.envs.dead_enemies_count.sum()
                return battles_game, battles_won, dead_allies, dead_enemies

            def get_battles_result(self, last_battles_info):
                battles_game, battles_won, dead_allies, dead_enemies = list(last_battles_info)
                incre_battles_game = float(self.envs.battles_game.sum() - battles_game)
                incre_battles_won = float(self.envs.battles_won.sum() - battles_won)
                win_rate = incre_battles_won / incre_battles_game if incre_battles_game > 0 else 0.0
                allies_count, enemies_count = incre_battles_game * self.num_agents, incre_battles_game * self.num_enemies
                incre_allies = float(self.envs.dead_allies_count.sum() - dead_allies)
                incre_enemies = float(self.envs.dead_enemies_count.sum() - dead_enemies)
                allies_dead_ratio = incre_allies / allies_count if allies_count > 0 else 0.0
                enemies_dead_ratio = incre_enemies / enemies_count if enemies_count > 0 else 0.0
                return win_rate, allies_dead_ratio, enemies_dead_ratio

            def run_episodes(self, test_mode=False):
                step_info, train_info = {}, {}
                videos, best_videos = [[] for _ in range(self.n_envs)], []
                episode_score, best_score = [], -np.inf
                # reset the envs and settings
                obs_n, state, info = self.envs.reset()
                envs_done = self.envs.buf_done
                self.env_step = 0
                filled = np.zeros([self.n_envs, self.episode_length, 1], np.int32)
                rnn_hidden, rnn_hidden_critic = self.init_rnn_hidden()

                if test_mode and self.render:
                    images = self.envs.render(self.args.render_mode)
                    if self.args.render_mode == "rgb_array":
                        for idx, img in enumerate(images):
                            videos[idx].append(img)

                while not envs_done.all():  # start episodes
                    available_actions = self.envs.get_avail_actions()
                    actions_dict = self.get_actions(obs_n, available_actions, rnn_hidden, rnn_hidden_critic,
                                                    state=state, test_mode=test_mode)
                    next_obs_n, next_state, rewards, terminated, truncated, info = self.envs.step(actions_dict['actions_n'])
                    envs_done = self.envs.buf_done
                    rnn_hidden, rnn_hidden_critic = actions_dict['rnn_hidden'], actions_dict['rnn_hidden_critic']

                    if test_mode:
                        if self.render:
                            images = self.envs.render(self.args.render_mode)
                            if self.args.render_mode == "rgb_array":
                                for idx, img in enumerate(images):
                                    videos[idx].append(img)
                        for i_env in range(self.n_envs):
                            if terminated[i_env] or truncated[i_env]:  # one env is terminal
                                episode_score.append(info[i_env]["episode_score"])
                                if best_score < episode_score[-1]:
                                    best_score = episode_score[-1]
                                    best_videos = videos[i_env].copy()
                    else:
                        filled[:, self.env_step] = np.ones([self.n_envs, 1])
                        # store transition data
                        transition = (obs_n, actions_dict, state, rewards, terminated, available_actions)
                        self.agents.memory.store_transitions(self.env_step, *transition)
                        for i_env in range(self.n_envs):
                            if envs_done[i_env]:
                                filled[i_env, self.env_step, 0] = 0
                            else:
                                self.current_step += 1
                            if terminated[i_env] or truncated[i_env]:  # one env is terminal
                                episode_score.append(info[i_env]["episode_score"])
                                available_actions = self.envs.get_avail_actions()
                                # log
                                if self.use_wandb:
                                    step_info["Episode-Steps/env-%d" % i_env] = info[i_env]["episode_step"]
                                    step_info["Train-Episode-Rewards/env-%d" % i_env] = info[i_env]["episode_score"]
                                else:
                                    step_info["Train-Results/Episode-Steps"] = {"env-%d" % i_env: info[i_env]["episode_step"]}
                                    step_info["Train-Results/Episode-Rewards"] = {"env-%d" % i_env: info[i_env]["episode_score"]}
                                self.log_infos(step_info, self.current_step)

                                terminal_data = (next_obs_n, next_state, available_actions, filled)
                                if self.on_policy:
                                    if terminated[i_env]:
                                        values_next = np.array([0.0 for _ in range(self.num_agents)])
                                    else:
                                        batch_select = np.arange(i_env * self.num_agents, (i_env + 1) * self.num_agents)
                                        kwargs = {"state": [next_state[i_env]]}
                                        if self.args.agent == "VDAC":
                                            rnn_h_ac_i = self.agents.policy.representation.get_hidden_item(batch_select,
                                                                                                        *rnn_hidden)
                                            kwargs.update({"avail_actions": available_actions[i_env:i_env+1],
                                                        "test_mode": test_mode})
                                            _, _, values_next = self.agents.act(next_obs_n[i_env:i_env+1],
                                                                                *rnn_h_ac_i, **kwargs)
                                        else:
                                            rnn_h_critic_i = self.agents.policy.representation_critic.get_hidden_item(batch_select,
                                                                                                                    *rnn_hidden_critic)
                                            if self.args.agent == "COMA":
                                                kwargs.update({"actions_n": actions_dict["actions_n"],
                                                            "actions_onehot": actions_dict["act_n_onehot"]})
                                            _, values_next = self.agents.values(next_obs_n[i_env:i_env + 1],
                                                                                *rnn_h_critic_i, **kwargs)
                                    self.agents.memory.finish_path(i_env, self.env_step+1, *terminal_data,
                                                                value_next=values_next,
                                                                value_normalizer=self.agents.learner.value_normalizer)
                                else:
                                    self.agents.memory.finish_path(i_env, self.env_step + 1, *terminal_data)
                                self.current_step += 1
                        self.env_step += 1
                    obs_n, state = deepcopy(next_obs_n), deepcopy(next_state)

                if test_mode:
                    if self.render and self.args.render_mode == "rgb_array":
                        # time, height, width, channel -> time, channel, height, width
                        videos_info = {"Videos_Test": np.array([best_videos], dtype=np.uint8).transpose((0, 1, 4, 2, 3))}
                        self.log_videos(info=videos_info, fps=self.fps, x_index=self.current_step)
                else:
                    self.agents.memory.store_episodes()  # store episode data
                    n_epoch = self.agents.n_epoch if self.on_policy else self.n_envs
                    train_info = self.agents.train(self.current_step, n_epoch=n_epoch)  # train
                    self.log_infos(train_info, self.current_step)

                mean_episode_score = np.mean(episode_score)
                return mean_episode_score

            def test_episodes(self, test_T, n_test_runs):
                test_scores = np.zeros(n_test_runs, np.float)
                last_battles_info = self.get_battles_info()
                for i_test in range(n_test_runs):
                    test_scores[i_test] = self.run_episodes(test_mode=True)
                win_rate, allies_dead_ratio, enemies_dead_ratio = self.get_battles_result(last_battles_info)
                mean_test_score = test_scores.mean()
                results_info = {"Test-Results/Mean-Episode-Rewards": mean_test_score,
                                "Test-Results/Win-Rate": win_rate,
                                "Test-Results/Allies-Dead-Ratio": allies_dead_ratio,
                                "Test-Results/Enemies-Dead-Ratio": enemies_dead_ratio}
                self.log_infos(results_info, test_T)
                return mean_test_score, test_scores.std(), win_rate

            def run(self):
                if self.args.test_mode:
                    self.render = True
                    n_test_episodes = self.args.test_episode
                    self.agents.load_model(self.args.model_dir_load)
                    test_score_mean, test_score_std, test_win_rate = self.test_episodes(0, n_test_episodes)
                    agent_info = f"Algo: {self.args.agent}, Map: {self.args.env_id}, seed: {self.args.seed}, "
                    print(agent_info, "Win rate: %.3f, Mean score: %.2f. " % (test_win_rate, test_score_mean))
                    print("Finish testing.")
                else:
                    test_interval = self.args.eval_interval
                    last_test_T = 0
                    episode_scores = []
                    agent_info = f"Algo: {self.args.agent}, Map: {self.args.env_id}, seed: {self.args.seed}, "
                    print(f"Steps: {self.current_step} / {self.running_steps}: ")
                    print(agent_info, "Win rate: %-, Mean score: -.")
                    last_battles_info = self.get_battles_info()
                    time_start = time.time()
                    while self.current_step <= self.running_steps:
                        score = self.run_episodes(test_mode=False)
                        episode_scores.append(score)
                        if (self.current_step - last_test_T) / test_interval >= 1.0:
                            last_test_T += test_interval
                            # log train results before testing.
                            train_win_rate, allies_dead_ratio, enemies_dead_ratio = self.get_battles_result(last_battles_info)
                            results_info = {"Train-Results/Win-Rate": train_win_rate,
                                            "Train-Results/Allies-Dead-Ratio": allies_dead_ratio,
                                            "Train-Results/Enemies-Dead-Ratio": enemies_dead_ratio}
                            self.log_infos(results_info, last_test_T)
                            last_battles_info = self.get_battles_info()
                            time_pass, time_left = self.time_estimate(time_start)
                            print(f"Steps: {self.current_step} / {self.running_steps}: ")
                            print(agent_info, "Win rate: %.3f, Mean score: %.2f. " % (train_win_rate, np.mean(episode_scores)),
                                time_pass, time_left)
                            episode_scores = []

                    print("Finish training.")
                    self.agents.save_model("final_train_model.pth")

                self.envs.close()
                if self.use_wandb:
                    wandb.finish()
                else:
                    self.writer.close()

            def benchmark(self):
                test_interval = self.args.eval_interval
                n_test_runs = self.args.test_episode // self.n_envs
                last_test_T = 0

                # test the mode at step 0
                test_score_mean, test_score_std, test_win_rate = self.test_episodes(last_test_T, n_test_runs)
                best_score = {"mean": test_score_mean,
                            "std": test_score_std,
                            "step": self.current_step}
                best_win_rate = test_win_rate

                agent_info = f"Algo: {self.args.agent}, Map: {self.args.env_id}, seed: {self.args.seed}, "
                print(f"Steps: {self.current_step} / {self.running_steps}: ")
                print(agent_info, "Win rate: %.3f, Mean score: %.2f. " % (test_win_rate, test_score_mean))
                last_battles_info = self.get_battles_info()
                time_start = time.time()
                while self.current_step <= self.running_steps:
                    # train
                    self.run_episodes(test_mode=False)
                    # test
                    if (self.current_step - last_test_T) / test_interval >= 1.0:
                        last_test_T += test_interval
                        # log train results before testing.
                        train_win_rate, allies_dead_ratio, enemies_dead_ratio = self.get_battles_result(last_battles_info)
                        results_info = {"Train-Results/Win-Rate": train_win_rate,
                                        "Train-Results/Allies-Dead-Ratio": allies_dead_ratio,
                                        "Train-Results/Enemies-Dead-Ratio": enemies_dead_ratio}
                        self.log_infos(results_info, last_test_T)

                        # test the model
                        test_score_mean, test_score_std, test_win_rate = self.test_episodes(last_test_T, n_test_runs)

                        if best_score["mean"] < test_score_mean:
                            best_score = {"mean": test_score_mean,
                                        "std": test_score_std,
                                        "step": self.current_step}
                        if best_win_rate < test_win_rate:
                            best_win_rate = test_win_rate
                            self.agents.save_model("best_model.pth")  # save best model

                        last_battles_info = self.get_battles_info()

                        # Estimate the physic running time
                        time_pass, time_left = self.time_estimate(time_start)
                        print(f"Steps: {self.current_step} / {self.running_steps}: ")
                        print(agent_info, "Win rate: %.3f, Mean score: %.2f. " % (test_win_rate, test_score_mean), time_pass, time_left)

                # end benchmarking
                print("Finish benchmarking.")
                print("Best Score: %.4f, Std: %.4f" % (best_score["mean"], best_score["std"]))
                print("Best Win Rate: {}%".format(best_win_rate * 100))

                self.envs.close()
                if self.use_wandb:
                    wandb.finish()
                else:
                    self.writer.close()

            def time_estimate(self, start):
                time_pass = int(time.time() - start)
                time_left = int((self.running_steps - self.current_step) / self.current_step * time_pass)
                if time_left < 0:
                    time_left = 0
                hours_pass, hours_left = time_pass // 3600, time_left // 3600
                min_pass, min_left = np.mod(time_pass, 3600) // 60, np.mod(time_left, 3600) // 60
                sec_pass, sec_left = np.mod(np.mod(time_pass, 3600), 60), np.mod(np.mod(time_left, 3600), 60)
                INFO_time_pass = f"Time pass: {hours_pass}h{min_pass}m{sec_pass}s,"
                INFO_time_left = f"Time left: {hours_left}h{min_left}m{sec_left}s"
                return INFO_time_pass, INFO_time_left
