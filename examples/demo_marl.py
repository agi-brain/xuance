import xuanpolicy

runner = xuanpolicy.get_runner(agent_name=['random', 'iddpg'],
                               env_name="mpe/simple_adversary",
                               is_test=False)
runner.run()
