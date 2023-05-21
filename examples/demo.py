import xuanpolicy

runner = xuanpolicy.get_runner(agent_name='maddpg',
                               env_name="mpe/simple_spread",
                               config_path='',
                               is_test=False)
runner.run()
