import xuanpolicy

runner = xuanpolicy.get_runner(agent_name='a2c',
                               # env_name="mujoco/InvertedPendulum-v2",
                               env_name="toy/CartPole",
                               is_test=True)
runner.run()
