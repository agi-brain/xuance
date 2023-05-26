import xuanpolicy

runner = xuanpolicy.get_runner(agent_name='sac',
                               env_name="mujoco/HalfCheetah-v3",
                               is_test=True)
runner.run()
