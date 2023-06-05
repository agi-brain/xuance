import xuanpolicy

runner = xuanpolicy.get_runner(agent_name='ddpg',
                               env_name="mujoco/Ant",
                               # env_name="toy/CartPole",
                               is_test=False)
runner.run()
