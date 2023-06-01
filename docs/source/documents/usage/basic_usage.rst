Quick Start
=======================

Run a DRL agent
-----------------------

In XuanPolicy, it is easy to build a DRL agent. First you need to create a *runner* 
and specify the ``agent_name``, ``env_name``, then a runner that contains agent, policy, and envs, etc., will be built. 
Finally, execute ``runner.run`` and the agent's model is training.
:: 

    import xuanpolicy as xp
    runner = xp.get_runner(agent_name='dqn', env_name='toy/CartPole-v0', is_test=False)
    runner.run()

After training the agent, you can test and view the model by the following codes:

Run an MARL agent
-----------------------

XuanPolicy support MARL algorithms with both cooperative and competitive tasks. 
Similaly, you can start by:
:: 

    import xuanpolicy as xp
    runner = xp.get_runner(agent_name='maddpg', env_name='mpe/simple_spread', is_test=False)
    runner.run()

For competitve tasks in which agents can be divided to two or more sides, you can run a demo by:

:: 

    import xuanpolicy as xp
    runner = xp.get_runner(agent_name=["maddpg", "iddpg"], env_name='mpe/simple_push', is_test=False)
    runner.run()

In this demo, the agents in `mpe/simple_push <https://pettingzoo.farama.org/environments/mpe/simple_push/>`_ environment are divided into two sides, named "adversary_0" and "agent_0".
The "adversary"s are MADDPG agents, and the "agent"s are IDDPG agents. 

Test
-----------------------

:: 

    import xuanpolicy as xp
    runner_test = xp.get_runner(agent_name='dqn', env_name='toy/CartPole-v0', is_test=True)
    runner_test.run()



Logger
-----------------------

You can use tensorboard to visualize what happened in the training process. After training, the log file will be automatically generated in the directory ".results/" and you should be able to see some training data after running the command.

:: 
    
    tensorboard --logdir ./logs/dqn/torch/CartPole-v0

