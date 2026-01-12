import xuance

runner = xuance.get_runner(algo='ppo',
                           env='classic_control',
                           env_id='CartPole-v1')
runner.run(mode='test')