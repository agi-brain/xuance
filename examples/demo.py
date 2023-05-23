import xuanpolicy

runner = xuanpolicy.get_runner(agent_name='dqn',
                               env_name="toy/CartPole-v0",
                               is_test=False)
runner.run()
