import xuanpolicy

runner = xuanpolicy.get_runner(agent_name='a2c',
                               env_name="mujoco/InvertedPendulum-v2",
                               is_test=False)
runner.run()
