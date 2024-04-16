Professional Usage
================================

The previous page demonstrated how to directly run an algorithm by calling the runner.
In order to help users better understand the internal implementation process of "XuanCe",
and facilitate further algorithm development and implementation of their own reinforcement learning tasks,
this section will take the PPO algorithm training on the MuJoCo environment task as an example,
and provide a detailed introduction on how to call the API from the bottom level to implement reinforcement learning model training.

.. raw:: html

   <br><hr>

Step 1: Create config file
--------------------------------

A config file should contains the necessary arguments of a PPO agent, and should be a YAML file.
Here we show a config file named "mujoco.yaml" for MuJoCo environment in gym.

.. code-block:: yaml

    dl_toolbox: "torch"  # The deep learning toolbox. Choices: "torch", "mindspore", "tensorlayer"
    project_name: "XuanCe_Benchmark"
    logger: "tensorboard"  # Choices: tensorboard, wandb.
    wandb_user_name: "your_user_name"
    render: False
    render_mode: 'rgb_array' # Choices: 'human', 'rgb_array'.
    test_mode: False
    device: "cuda:0"

    agent: "PPO_Clip"  # choice: PPO_Clip, PPO_KL
    env_name: "MuJoCo"
    vectorize: "Dummy_Gym"
    runner: "DRL"

    representation_hidden_size: [256,]
    actor_hidden_size: [256,]
    critic_hidden_size: [256,]
    activation: "LeakyReLU"

    seed: 79811
    parallels: 16
    running_steps: 1000000
    n_steps: 256
    n_epoch: 16
    n_minibatch: 8
    learning_rate: 0.0004

    use_grad_clip: True

    vf_coef: 0.25
    ent_coef: 0.0
    target_kl: 0.001  # for PPO_KL agent
    clip_range: 0.2  # for PPO_Clip agent
    clip_grad_norm: 0.5
    gamma: 0.99
    use_gae: True
    gae_lambda: 0.95
    use_advnorm: True

    use_obsnorm: True
    use_rewnorm: True
    obsnorm_range: 5
    rewnorm_range: 5

    test_steps: 10000
    eval_interval: 5000
    test_episode: 5
    log_dir: "./logs/ppo/"
    model_dir: "./models/ppo/"

.. raw:: html

   <br><hr>

Step 2: Get the attributes of the example
----------------------------------------------

This section mainly includes parameter reading, environment creation, model creation, and model training.
First, create a `ppo_mujoco.py` file. The code writing process can be divided into the following steps:

**Step 2.1 Get the hyper-parameters of command in console**

Define the following function ``parse_args()``,
which uses the Python package `argparse` to read the command line instructions and obtain the instruction parameters.

.. code-block:: python

    import argparse

    def parse_args():
        parser = argparse.ArgumentParser("Example of XuanCe.")
        parser.add_argument("--method", type=str, default="ppo")
        parser.add_argument("--env", type=str, default="mujoco")
        parser.add_argument("--env-id", type=str, default="InvertedPendulum-v4")
        parser.add_argument("--test", type=int, default=0)
        parser.add_argument("--device", type=str, default="cuda:0")
        parser.add_argument("--benchmark", type=int, default=1)
        parser.add_argument("--config", type=str, default="./ppo_mujoco_config.yaml")

        return parser.parse_args()

**Step 2.2 Get all attributes of the example**

First, the ``parse_args()`` function from Step 2.1 is called to read the command line instruction parameters,
and then the configuration parameters from Step 1 are obtained.

.. code-block:: python

    from xuance import get_arguments

    if __name__ == "__main__":
    parser = parse_args()
    args = get_arguments(method=parser.method,
                         env=parser.env,
                         env_id=parser.env_id,
                         config_path=parser.config,
                         parser_args=parser)
    run(args)

In this step, the ``get_arguments()`` function from "XuanCe" is called.
In this function, it first searches for readable parameters based on the combination of the ``env`` and ``env_id`` variables in the `xuance/configs/` directory.
If default parameters already exist, they are all read. Then, the function continues to index the configuration file from Step 1 using the ``config.path`` path and reads all the parameters from the .yaml file.
Finally, it reads all the parameters from the ``parser``.

During the three reading processes, if there are duplicate variable names, the latter parameters will overwrite the former ones.
Ultimately, the ``get_arguments()`` function will return the ``args`` variable, which contains all the parameter information and is input into the ``run()`` function.

.. raw:: html

   <br><hr>

Step 3: Define run(), create and run the model
--------------------------------------------------------

Define the run() function with the input as the args variable obtained in Step 2.
In this function, environment creation is implemented, and modules such as representation, policy, and agent are instantiated to perform the training.

Here is an example definition of the run() function with comments:

.. code-block:: python

    import os
    from copy import deepcopy
    import numpy as np
    import torch.optim

    from xuance.common import space2shape
    from xuance.environment import make_envs
    from xuance.torch.utils.operations import set_seed
    from xuance.torch.utils import ActivationFunctions

    def run(args):
        agent_name = args.agent  # get the name of Agent.
        set_seed(args.seed)  # set random seed.

        # prepare directories for results
        args.model_dir = os.path.join(os.getcwd(), args.model_dir, args.env_id)  # the path for saved model.
        args.log_dir = os.path.join(args.log_dir, args.env_id)  # the path for logger file.

        # build environments
        envs = make_envs(args)  # create simulation environments
        args.observation_space = envs.observation_space  # get observation space
        args.action_space = envs.action_space  # get action space
        n_envs = envs.num_envs  # get the number of vectorized environments.

        # prepare representation
        from xuance.torch.representations import Basic_MLP
        representation = Basic_MLP(input_shape=space2shape(args.observation_space),
                                   hidden_sizes=args.representation_hidden_size,
                                   normalize=None,
                                   initialize=torch.nn.init.orthogonal_,
                                   activation=ActivationFunctions[args.activation],
                                   device=args.device)  # create representation

        # prepare policy
        from xuance.torch.policies import Gaussian_AC_Policy
        policy = Gaussian_AC_Policy(action_space=args.action_space,
                                    representation=representation,
                                    actor_hidden_size=args.actor_hidden_size,
                                    critic_hidden_size=args.critic_hidden_size,
                                    normalize=None,
                                    initialize=torch.nn.init.orthogonal_,
                                    activation=ActivationFunctions[args.activation],
                                    device=args.device)  # create Gaussian policy

        # prepare agent
        from xuance.torch.agents import PPOCLIP_Agent, get_total_iters
        optimizer = torch.optim.Adam(policy.parameters(), args.learning_rate, eps=1e-5)  # create optimizer
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0,
                                                        total_iters=get_total_iters(agent_name, args))  # for learning rate decay
        agent = PPOCLIP_Agent(config=args,
                              envs=envs,
                              policy=policy,
                              optimizer=optimizer,
                              scheduler=lr_scheduler,
                              device=args.device)  # create a PPO agent

        # start running
        envs.reset()  # reset the environments
        if args.benchmark:  # run benchmark
            def env_fn():  # for creating testing environments
                args_test = deepcopy(args)
                args_test.parallels = args_test.test_episode  # set number of testing environments.
                return make_envs(args_test)  # make testing environments.

            train_steps = args.running_steps // n_envs  # calculate the total running steps.
            eval_interval = args.eval_interval // n_envs  # calculate the number of training steps per epoch.
            test_episode = args.test_episode  # calculate the number of testing episodes.
            num_epoch = int(train_steps / eval_interval)  # calculate the number of epochs.

            test_scores = agent.test(env_fn, test_episode)  # first test
            best_scores_info = {"mean": np.mean(test_scores),  # average episode scores.
                                "std": np.std(test_scores),  # the standard deviation of the episode scores.
                                "step": agent.current_step}  # current step
            for i_epoch in range(num_epoch):  # begin benchmarking
                print("Epoch: %d/%d:" % (i_epoch, num_epoch))
                agent.train(eval_interval)  # train the model for some steps
                test_scores = agent.test(env_fn, test_episode)  # test the model for some episodes

                if np.mean(test_scores) > best_scores_info["mean"]:  # if current score is better than history
                    best_scores_info = {"mean": np.mean(test_scores),
                                        "std": np.std(test_scores),
                                        "step": agent.current_step}
                    # save best model
                    agent.save_model(model_name="best_model.pth")
            # end benchmarking
            print("Best Model Score: %.2f, std=%.2f" % (best_scores_info["mean"], best_scores_info["std"]))
        else:
            if not args.test:  # train the model without testing
                n_train_steps = args.running_steps // n_envs  # calculate the total steps of training
                agent.train(n_train_steps)  # train the model directly.
                agent.save_model("final_train_model.pth")  # save the final model file.
                print("Finish training!")
            else:  # test a trained model
                def env_fn():
                    args_test = deepcopy(args)
                    args_test.parallels = 1
                    return make_envs(args_test)

                agent.render = True
                agent.load_model(agent.model_dir_load, args.seed)  # load the model file
                scores = agent.test(env_fn, args.test_episode)  # test the model
                print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
                print("Finish testing.")

        # the end.
        envs.close()  # close the environment
        agent.finish()  # finish the example

After finishing the above three steps, you can run the `python_mujoco.py` file in console and train the model:

.. code-block:: bash

    python ppo_mujoco.py --method ppo --env mujoco --env-id Ant-v4

The source code of this example can be visited at the following link:

`https://github.com/agi-brain/xuance/examples/ppo/ppo_mujoco.py <https://github.com/agi-brain/xuance/examples/ppo/ppo_mujoco.py/>`_


