Basic Usage
====

#### Train a Model

:: 
    import xuanpolicy as xp

    runner = xp.get_runner(agent_name='dqn', env_name='toy/CartPole-v0', is_test=False)
    runner.run()

#### Test the Model
:: 
    import xuanpolicy as xp
    runner_test = xp.get_runner(agent_name='dqn', env_name='toy/CartPole-v0', is_test=True)
    runner_test.run()

## Logger
You can use tensorboard to visualize what happened in the training process. After training, the log file will be automatically generated in the directory ".results/" and you should be able to see some training data after running the command.

 | $ tensorboard --logdir ./logs/dqn/torch/CartPole-v0
