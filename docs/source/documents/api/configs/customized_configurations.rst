Custom Configurations
--------------------------
Users can also choose not to use the default parameters provided by XuanCe,
or in cases where XuanCe does not include the user's specific task, they can customize their own .yaml parameter configuration file in the same manner.

However, during the process of obtaining the runner, it is necessary to specify the location where the parameter file is stored, as shown below:

.. code-block:: python

    import xuance as xp
    runner = xp.get_runner(method='dqn',
                           env='classic_control',
                           env_id='CartPole-v1',
                           config_path="xxx/xxx.yaml",
                           is_test=False)
    runner.run()
