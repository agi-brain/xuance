from xuanpolicy import get_runner
runner = get_runner(agent_name='dqn',
                    env_name="toy_env/CartPole-v0",
                    is_test=False)
runner.run()
