import xuanpolicy

runner = xuanpolicy.get_runner(agent_name='sac',
                               env_name="mujoco/Ant-v3",
                               is_test=False)
runner.run()
