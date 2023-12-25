Runner_DRL
=========================================

The definition of basic runner.

.. raw:: html

    <br><hr>

PyTorch
------------------------------------------

.. py:class::
    xuance.torch.runners.runner_drl.Runner_DRL(args)

    Facilitate the training and evaluation of various algorithms in different environments.

    :param args: the arguments.
    :type args: Namespace

.. py:function::
    xuance.torch.runners.runner_drl.Runner_DRL.run()

    Orchestrate the execution of the task, handling either training or testing based on the specified mode in the configuration.

.. py:function::
    xuance.torch.runners.runner_drl.Runner_DRL.benchmark()

    Conduct benchmarking by training the agent for multiple epochs, evaluating its performance, and saving the best-performing model.

.. raw:: html

    <br><hr>

TensorFlow
------------------------------------------

.. py:class::
    xuance.tensorflow.runners.runner_drl.Runner_DRL(args)

    This class provides a structured framework for training, testing, and benchmarking TensorFlow-based reinforcement learning agents in different environments

    :param args: the arguments.
    :type args: Namespace

.. py:function::
    xuance.tensorflow.runners.runner_drl.Runner_DRL.run()

    Test and train modes.

.. py:function::
    xuance.tensorflow.runners.runner_drl.Runner_DRL.benchmark()

    Assess the performance of the reinforcement learning agent over multiple training epochs,
    identify the best-performing model, and save it for potential future use.

.. raw:: html

    <br><hr>

MindSpore
------------------------------------------

.. py:class::
    xuance.mindspore.runners.runner_drl.Runner_DRL(args)

    A general-purpose framework for configuring, training, and evaluating agents across different tasks.

    :param args: the arguments.
    :type args: Namespace

.. py:function::
    xuance.mindspore.runners.runner_drl.Runner_DRL.run()

    Encapsulate the logic for both testing and training modes,
    providing a execution flow, including model loading, testing, training, and cleanup

.. py:function::
    xuance.mindspore.runners.runner_drl.Runner_DRL.benchmark()

    A approach to benchmarking the agent over multiple training epochs,
    identifying the best-performing model, and saving it for further use.

.. raw:: html

    <br><hr>

Source Code
-----------------

.. tabs::

  .. group-tab:: PyTorch

    .. code-block:: python

        import wandb
        from .runner_basic import *
        from xuance.torch.agents import get_total_iters
        from xuance.torch.representations import REGISTRY as REGISTRY_Representation
        from xuance.torch.agents import REGISTRY as REGISTRY_Agent
        from xuance.torch.policies import REGISTRY as REGISTRY_Policy
        from xuance.torch.utils.input_reformat import get_repre_in, get_policy_in
        import itertools
        import torch
        import gym.spaces
        import numpy as np
        from copy import deepcopy


        class Runner_DRL(Runner_Base):
            def __init__(self, args):
                self.args = args
                self.agent_name = self.args.agent
                self.env_id = self.args.env_id
                super(Runner_DRL, self).__init__(self.args)

                if self.env_id in ['Platform-v0']:
                    self.args.observation_space = self.envs.observation_space.spaces[0]
                    old_as = self.envs.action_space
                    num_disact = old_as.spaces[0].n
                    self.args.action_space = gym.spaces.Tuple(
                        (old_as.spaces[0], *(gym.spaces.Box(old_as.spaces[1].spaces[i].low,
                                                            old_as.spaces[1].spaces[i].high, dtype=np.float32) for i in
                                            range(0, num_disact))))
                else:
                    self.args.observation_space = self.envs.observation_space
                    self.args.action_space = self.envs.action_space

                input_representation = get_repre_in(self.args)
                representation = REGISTRY_Representation[self.args.representation](*input_representation)

                input_policy = get_policy_in(self.args, representation)
                if self.agent_name == "DRQN":
                    policy = REGISTRY_Policy[self.args.policy](**input_policy)
                else:
                    policy = REGISTRY_Policy[self.args.policy](*input_policy)

                if self.agent_name in ["DDPG", "TD3", "SAC", "SACDIS"]:
                    actor_optimizer = torch.optim.Adam(policy.actor.parameters(), self.args.actor_learning_rate)
                    if self.agent_name == "TD3":
                        critic_optimizer = torch.optim.Adam(
                            itertools.chain(policy.criticA.parameters(), policy.criticB.parameters()),
                            self.args.critic_learning_rate)
                    else:
                        critic_optimizer = torch.optim.Adam(policy.critic.parameters(), self.args.critic_learning_rate)
                    actor_lr_scheduler = torch.optim.lr_scheduler.LinearLR(actor_optimizer, start_factor=1.0, end_factor=0.25,
                                                                        total_iters=get_total_iters(self.agent_name,
                                                                                                    self.args))
                    critic_lr_scheduler = torch.optim.lr_scheduler.LinearLR(critic_optimizer, start_factor=1.0, end_factor=0.25,
                                                                            total_iters=get_total_iters(self.agent_name,
                                                                                                        self.args))
                    self.agent = REGISTRY_Agent[self.agent_name](self.args, self.envs, policy,
                                                                [actor_optimizer, critic_optimizer],
                                                                [actor_lr_scheduler, critic_lr_scheduler], self.args.device)
                elif self.agent_name in ["PDQN", "MPDQN", "SPDQN"]:
                    conactor_optimizer = torch.optim.Adam(policy.conactor.parameters(), self.args.learning_rate)
                    qnetwork_optimizer = torch.optim.Adam(policy.qnetwork.parameters(), self.args.learning_rate)
                    conactor_lr_scheduler = torch.optim.lr_scheduler.LinearLR(conactor_optimizer, start_factor=1.0, end_factor=0.25,
                                                                        total_iters=get_total_iters(self.agent_name, self.args))
                    qnetwork_lr_scheduler = torch.optim.lr_scheduler.LinearLR(qnetwork_optimizer, start_factor=1.0, end_factor=0.25,
                                                                            total_iters=get_total_iters(self.agent_name, self.args))
                    self.agent = REGISTRY_Agent[self.agent_name](self.args, self.envs, policy,
                                                                [conactor_optimizer, qnetwork_optimizer],
                                                                [conactor_lr_scheduler, qnetwork_lr_scheduler], self.args.device)
                else:
                    optimizer = torch.optim.Adam(policy.parameters(), self.args.learning_rate, eps=1e-5)
                    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0,
                                                                    total_iters=get_total_iters(self.agent_name, self.args))
                    self.agent = REGISTRY_Agent[self.agent_name](self.args, self.envs, policy, optimizer, lr_scheduler,
                                                                self.args.device)

            def run(self):
                if self.args.test_mode:
                    def env_fn():
                        args_test = deepcopy(self.args)
                        args_test.parallels = 1
                        return make_envs(args_test)
                    self.agent.render = True
                    self.agent.load_model(self.agent.model_dir_load, self.args.seed)
                    scores = self.agent.test(env_fn, self.args.test_episode)
                    print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
                    print("Finish testing.")
                else:
                    n_train_steps = self.args.running_steps // self.n_envs
                    self.agent.train(n_train_steps)
                    print("Finish training.")
                    self.agent.save_model("final_train_model.pth")

                self.envs.close()
                if self.agent.use_wandb:
                    wandb.finish()
                else:
                    self.agent.writer.close()

            def benchmark(self):
                # test environment
                def env_fn():
                    args_test = deepcopy(self.args)
                    args_test.parallels = args_test.test_episode
                    return make_envs(args_test)
                train_steps = self.args.running_steps // self.n_envs
                eval_interval = self.args.eval_interval // self.n_envs
                test_episode = self.args.test_episode
                num_epoch = int(train_steps / eval_interval)

                test_scores = self.agent.test(env_fn, test_episode)
                best_scores_info = {"mean": np.mean(test_scores),
                                    "std": np.std(test_scores),
                                    "step": self.agent.current_step}
                for i_epoch in range(num_epoch):
                    print("Epoch: %d/%d:" % (i_epoch, num_epoch))
                    self.agent.train(eval_interval)
                    test_scores = self.agent.test(env_fn, test_episode)

                    if np.mean(test_scores) > best_scores_info["mean"]:
                        best_scores_info = {"mean": np.mean(test_scores),
                                            "std": np.std(test_scores),
                                            "step": self.agent.current_step}
                        # save best model
                        self.agent.save_model(model_name="best_model.pth")

                # end benchmarking
                print("Best Model Score: %.2f, std=%.2f" % (best_scores_info["mean"], best_scores_info["std"]))

                self.envs.close()
                if self.agent.use_wandb:
                    wandb.finish()
                else:
                    self.agent.writer.close()


  .. group-tab:: TensorFlow

    .. code-block:: python

        from .runner_basic import *
        from xuance.tensorflow.representations import REGISTRY as REGISTRY_Representation
        from xuance.tensorflow.agents import REGISTRY as REGISTRY_Agent
        from xuance.tensorflow.policies import REGISTRY as REGISTRY_Policy
        from xuance.tensorflow.utils.input_reformat import get_repre_in, get_policy_in
        import tensorflow.keras as tk
        import gym.spaces
        import numpy as np
        from copy import deepcopy


        class Runner_DRL(Runner_Base):
            def __init__(self, args):
                self.args = args
                self.agent_name = self.args.agent
                self.env_id = self.args.env_id
                super(Runner_DRL, self).__init__(self.args)

                if self.env_id in ['Platform-v0']:
                    self.args.observation_space = self.envs.observation_space.spaces[0]
                    old_as = self.envs.action_space
                    num_disact = old_as.spaces[0].n
                    self.args.action_space = gym.spaces.Tuple(
                        (old_as.spaces[0], *(gym.spaces.Box(old_as.spaces[1].spaces[i].low,
                                                            old_as.spaces[1].spaces[i].high, dtype=np.float32) for i in
                                            range(0, num_disact))))
                else:
                    self.args.observation_space = self.envs.observation_space
                    self.args.action_space = self.envs.action_space

                input_representation = get_repre_in(self.args)
                representation = REGISTRY_Representation[self.args.representation](*input_representation)

                input_policy = get_policy_in(self.args, representation)
                if self.agent_name == "DRQN":
                    policy = REGISTRY_Policy[self.args.policy](**input_policy)
                else:
                    policy = REGISTRY_Policy[self.args.policy](*input_policy)

                if self.agent_name in ["DDPG", "TD3", "SAC", "SACDIS"]:
                    # actor_lr_scheduler = MyLinearLR(self.args.actor_learning_rate, start_factor=1.0, end_factor=0.25,
                    #                                 total_iters=get_total_iters(self.agent_name, self.args))
                    actor_lr_scheduler = tk.optimizers.schedules.ExponentialDecay(self.args.actor_learning_rate,
                                                                                decay_steps=1000, decay_rate=0.9)
                    actor_optimizer = tk.optimizers.Adam(actor_lr_scheduler)
                    # critic_lr_scheduler = MyLinearLR(self.args.critic_learning_rate, start_factor=1.0, end_factor=0.25,
                    #                                  total_iters=get_total_iters(self.agent_name, self.args))
                    critic_lr_scheduler = tk.optimizers.schedules.ExponentialDecay(self.args.critic_learning_rate,
                                                                                decay_steps=1000, decay_rate=0.9)
                    critic_optimizer = tk.optimizers.Adam(critic_lr_scheduler)
                    self.agent = REGISTRY_Agent[self.agent_name](self.args, self.envs, policy,
                                                                [actor_optimizer, critic_optimizer], self.args.device)
                elif self.agent_name in ["PDQN", "MPDQN", "SPDQN"]:
                    conactor_lr_scheduler = tk.optimizers.schedules.ExponentialDecay(self.args.learning_rate,
                                                                                    decay_steps=1000, decay_rate=0.9)
                    conactor_optimizer = tk.optimizers.Adam(conactor_lr_scheduler)
                    qnetwork_lr_scheduler = tk.optimizers.schedules.ExponentialDecay(self.args.learning_rate,
                                                                                    decay_steps=1000, decay_rate=0.9)
                    qnetwork_optimizer = tk.optimizers.Adam(qnetwork_lr_scheduler)
                    self.agent = REGISTRY_Agent[self.agent_name](self.args, self.envs, policy,
                                                                [conactor_optimizer, qnetwork_optimizer],
                                                                self.args.device)
                else:
                    # lr_scheduler = MyLinearLR(self.args.learning_rate, start_factor=1.0, end_factor=0.25,
                    #                           total_iters=get_total_iters(self.agent_name, self.args))
                    lr_scheduler = tk.optimizers.schedules.ExponentialDecay(self.args.learning_rate, decay_steps=1000,
                                                                            decay_rate=0.9)
                    optimizer = tk.optimizers.Adam(lr_scheduler)
                    self.agent = REGISTRY_Agent[self.agent_name](self.args, self.envs, policy, optimizer, self.args.device)

            def run(self):
                if self.args.test_mode:
                    def env_fn():
                        args_test = deepcopy(self.args)
                        args_test.parallels = 1
                        return make_envs(args_test)
                    self.agent.render = True
                    self.agent.load_model(self.agent.model_dir_load, self.args.seed)
                    scores = self.agent.test(env_fn, self.args.test_episode)
                    print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
                    print("Finish testing.")
                else:
                    n_train_steps = self.args.running_steps // self.n_envs
                    self.agent.train(n_train_steps)
                    print("Finish training.")
                    self.agent.save_model("final_train_model")

                self.envs.close()
                if self.agent.use_wandb:
                    wandb.finish()
                else:
                    self.agent.writer.close()

            def benchmark(self):
                # test environment
                def env_fn():
                    args_test = deepcopy(self.args)
                    args_test.parallels = args_test.test_episode
                    return make_envs(args_test)
                train_steps = self.args.running_steps // self.n_envs
                eval_interval = self.args.eval_interval // self.n_envs
                test_episode = self.args.test_episode
                num_epoch = int(train_steps / eval_interval)

                test_scores = self.agent.test(env_fn, test_episode)
                best_scores_info = {"mean": np.mean(test_scores),
                                    "std": np.std(test_scores),
                                    "step": self.agent.current_step}
                for i_epoch in range(num_epoch):
                    print("Epoch: %d/%d:" % (i_epoch, num_epoch))
                    self.agent.train(eval_interval)
                    test_scores = self.agent.test(env_fn, test_episode)

                    if np.mean(test_scores) > best_scores_info["mean"]:
                        best_scores_info = {"mean": np.mean(test_scores),
                                            "std": np.std(test_scores),
                                            "step": self.agent.current_step}
                        # save best model
                        self.agent.save_model(model_name="best_model")

                # end benchmarking
                print("Best Model Score: %.2f, std=%.2f" % (best_scores_info["mean"], best_scores_info["std"]))

                self.envs.close()
                if self.agent.use_wandb:
                    wandb.finish()
                else:
                    self.agent.writer.close()



  .. group-tab:: MindSpore

    .. code-block:: python

        import wandb
        from .runner_basic import *
        from xuance.mindspore.agents import get_total_iters
        from xuance.mindspore.representations import REGISTRY as REGISTRY_Representation
        from xuance.mindspore.agents import REGISTRY as REGISTRY_Agent
        from xuance.mindspore.policies import REGISTRY as REGISTRY_Policy
        from xuance.mindspore.utils.input_reformat import get_repre_in, get_policy_in
        import itertools
        from mindspore.nn import Adam
        from mindspore.nn.learning_rate_schedule import ExponentialDecayLR as lr_decay_model
        import gym.spaces
        import numpy as np
        from copy import deepcopy


        class Runner_DRL(Runner_Base):
            def __init__(self, args):
                self.args = args
                self.agent_name = self.args.agent
                self.env_id = self.args.env_id
                super(Runner_DRL, self).__init__(self.args)

                if self.env_id in ['Platform-v0']:
                    self.args.observation_space = self.envs.observation_space.spaces[0]
                    old_as = self.envs.action_space
                    num_disact = old_as.spaces[0].n
                    self.args.action_space = gym.spaces.Tuple(
                        (old_as.spaces[0], *(gym.spaces.Box(old_as.spaces[1].spaces[i].low,
                                                            old_as.spaces[1].spaces[i].high, dtype=np.float32) for i in
                                            range(0, num_disact))))
                else:
                    self.args.observation_space = self.envs.observation_space
                    self.args.action_space = self.envs.action_space

                input_representation = get_repre_in(self.args)
                representation = REGISTRY_Representation[self.args.representation](*input_representation)

                input_policy = get_policy_in(self.args, representation)
                if self.agent_name == "DRQN":
                    policy = REGISTRY_Policy[self.args.policy](**input_policy)
                else:
                    policy = REGISTRY_Policy[self.args.policy](*input_policy)

                if self.agent_name in ["DDPG", "TD3", "SAC", "SACDIS"]:
                    actor_lr_scheduler = lr_decay_model(learning_rate=self.args.actor_learning_rate,
                                                        decay_rate=0.5,
                                                        decay_steps=get_total_iters(self.agent_name, self.args))
                    critic_lr_scheduler = lr_decay_model(learning_rate=self.args.critic_learning_rate,
                                                        decay_rate=0.5,
                                                        decay_steps=get_total_iters(self.agent_name, self.args))
                    actor_optimizer = Adam(policy.actor.trainable_params(), actor_lr_scheduler, eps=1e-5)
                    if self.agent_name == "TD3":
                        critic_optimizer = Adam(itertools.chain(policy.criticA.trainable_params(),
                                                                policy.criticB.trainable_params()),
                                                critic_lr_scheduler, eps=1e-5)
                    else:
                        critic_optimizer = Adam(policy.critic.trainable_params(), critic_lr_scheduler, eps=1e-5)
                    self.agent = REGISTRY_Agent[self.agent_name](self.args, self.envs, policy,
                                                                {'actor': actor_optimizer, 'critic': critic_optimizer},
                                                                {'actor': actor_lr_scheduler, 'critic': critic_lr_scheduler})
                elif self.agent_name in ["PDQN", "MPDQN", "SPDQN"]:
                    conactor_lr_scheduler = lr_decay_model(learning_rate=self.args.learning_rate,
                                                        decay_rate=0.5,
                                                        decay_steps=get_total_iters(self.agent_name, self.args))
                    qnetwork_lr_scheduler = lr_decay_model(learning_rate=self.args.learning_rate,
                                                        decay_rate=0.5,
                                                        decay_steps=get_total_iters(self.agent_name, self.args))
                    conactor_optimizer = Adam(policy.conactor.trainable_params(), conactor_lr_scheduler, eps=1e-5)
                    qnetwork_optimizer = Adam(policy.qnetwork.trainable_params(), qnetwork_lr_scheduler, eps=1e-5)
                    self.agent = REGISTRY_Agent[self.agent_name](self.args, self.envs, policy,
                                                                [conactor_optimizer, qnetwork_optimizer],
                                                                [conactor_lr_scheduler, qnetwork_lr_scheduler])
                else:
                    lr_scheduler = lr_decay_model(learning_rate=self.args.learning_rate,
                                                decay_rate=0.5,
                                                decay_steps=get_total_iters(self.agent_name, self.args)
                                                )
                    optimizer = Adam(policy.trainable_params(), lr_scheduler, eps=1e-5)
                    self.agent = REGISTRY_Agent[self.agent_name](self.args, self.envs, policy, optimizer, lr_scheduler)

            def run(self):
                if self.args.test_mode:
                    def env_fn():
                        args_test = deepcopy(self.args)
                        args_test.parallels = 1
                        return make_envs(args_test)
                    self.agent.render = True
                    self.agent.load_model(self.agent.model_dir_load, self.args.seed)
                    scores = self.agent.test(env_fn, self.args.test_episode)
                    print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
                    print("Finish testing.")
                else:
                    n_train_steps = self.args.running_steps // self.n_envs
                    self.agent.train(n_train_steps)
                    print("Finish training.")
                    self.agent.save_model(model_name="final_train_model.ckpt")

                self.envs.close()
                if self.agent.use_wandb:
                    wandb.finish()
                else:
                    self.agent.writer.close()

            def benchmark(self):
                # test environment
                def env_fn():
                    args_test = deepcopy(self.args)
                    args_test.parallels = args_test.test_episode
                    return make_envs(args_test)
                train_steps = self.args.running_steps // self.n_envs
                eval_interval = self.args.eval_interval // self.n_envs
                test_episode = self.args.test_episode
                num_epoch = int(train_steps / eval_interval)

                test_scores = self.agent.test(env_fn, test_episode)
                best_scores_info = {"mean": np.mean(test_scores),
                                    "std": np.std(test_scores),
                                    "step": self.agent.current_step}
                for i_epoch in range(num_epoch):
                    print("Epoch: %d/%d:" % (i_epoch, num_epoch))
                    self.agent.train(eval_interval)
                    test_scores = self.agent.test(env_fn, test_episode)

                    if np.mean(test_scores) > best_scores_info["mean"]:
                        best_scores_info = {"mean": np.mean(test_scores),
                                            "std": np.std(test_scores),
                                            "step": self.agent.current_step}
                        # save best model
                        self.agent.save_model(model_name="best_model.ckpt")

                # end benchmarking
                print("Best Model Score: %.2f, std=%.2f" % (best_scores_info["mean"], best_scores_info["std"]))

                self.envs.close()
                if self.agent.use_wandb:
                    wandb.finish()
                else:
                    self.agent.writer.close()


