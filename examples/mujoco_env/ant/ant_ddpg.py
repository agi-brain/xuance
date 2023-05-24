import xuanpolicy

runner = xuanpolicy.get_runner(agent_name='ddpg',
                               env_name="mujoco/Ant-v2",
                               is_test=False)
runner.run()
