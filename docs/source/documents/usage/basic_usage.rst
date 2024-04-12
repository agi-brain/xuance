Quick Start
=======================

.. raw:: html

   <br><hr>

Run a DRL example
-----------------------

| In XuanCe, it is easy to build a DRL agent.
| First you need to create a *runner* and specify the ``agent_name``, ``env_name``,
| then a runner that contains agent, policy, and envs, etc., will be built.
| Finally, execute ``runner.run()`` and the agent's model is training.

.. code-block:: python3

    import xuance
    runner = xuance.get_runner(method='dqn',
                               env='classic_control',
                               env_id='CartPole-v1',
                               is_test=False)
    runner.run()

.. tip::

    If you want to modify the hyper-parameters of the above example,
    you can create a python file named, e.g., "example.py".
    In example.py, define and create a parser arguments before creating runner.

    .. code-block:: python

        import xuance argparse

        def parse_args():
            parser = argparse.ArgumentParser("Run a demo.")
            parser.add_argument("--method", type=str, default="dqn")
            parser.add_argument("--env", type=str, default="classic_control")
            parser.add_argument("--env-id", type=str, default="CartPole-v1")
            parser.add_argument("--test", type=int, default=0)
            parser.add_argument("--device", type=str, default="cuda:0")

            return parser.parse_args()

        if __name__ == '__main__':
            parser = parse_args()
            runner = xuance.get_runner(method=parser.method,
                                       env=parser.env,
                                       env_id=parser.env_id,
                                       parser_args=parser,
                                       is_test=parser.test)
            runner.run()

    Then, run the python file in terminal:

    .. code-block:: bash

        python example.py  # or python example.py --device 'cpu'


.. raw:: html

   <br><hr>

Run an MARL example
-----------------------

XuanCe support MARL algorithms with both cooperative and competitive tasks.
Similaly, you can start by:

.. code-block:: python

    import xuance
    runner = xuance.get_runner(method='maddpg',
                               env='mpe',
                               env_id='simple_spread_v3',
                               is_test=False)
    runner.run()

For competitve tasks in which agents can be divided to two or more sides, you can run a demo by:

.. code-block:: python

    import xuance
    runner = xuance.get_runner(method=["maddpg", "iddpg"],
                               env='mpe',
                               env_id='simple_push_v3',
                               is_test=False)
    runner.run()

In this demo, the agents in `mpe/simple_push <https://pettingzoo.farama.org/environments/mpe/simple_push/>`_ environment are divided into two sides, named "adversary_0" and "agent_0".
The "adversary"s are MADDPG agents, and the "agent"s are IDDPG agents.

Test
-----------------------

After completing the algorithm training, XuanCe will save the model files and training log information in the designated directory.
Users can specify "is_test=True" to perform testing.

.. code-block:: python

    import xuance
    runner = xuance.get_runner(method='dqn',
                               env_name='classic_control',
                               env_id='CartPole-v1',
                               is_test=True)
    runner.run()

In the above code, "runner.benchmark()" can also be used instead of "runner.run()" to train benchmark models and obtain benchmark test results.

Logger
-----------------------

You can use the tensorboard or wandb to visualize the training process by specifying the "logger" parameter in the "xuance/configs/basic.yaml".

.. code-block:: yaml

    logger: tensorboard

or

.. code-block:: yaml

    logger: wandb

**1. Tensorboard**

After completing the model training, the log files are stored in the "log" folder in the root directory.
The specific path depends on the user's actual configuration.
Taking the path "./logs/dqn/torch/CartPole-v0" as an example, users can visualize the logs using the following command:

.. code-block:: bash

    tensorboard --logdir ./logs/dqn/torch/CartPole-v1/ --port 6006

Then, we can see the training curves at http://localhost:6006/.

.. image:: ../../figures/log/tensorboard.png

**2. W&B**

If you choose to use the wandb tool for training visualization,
you can create an account according to the official W&B instructions and specify the username "wandb_user_name" in the "xuance/configs/basic.yaml" file.

.. image:: ../../figures/log/wandb.png

For information on using W&B and its local deployment, you can refer to the following link:

| **wandb**: `https://github.com/wandb/wandb.git <https://github.com/wandb/wandb.git/>`_
| **wandb server**: `https://github.com/wandb/server.git <https://github.com/wandb/server.git/>`_