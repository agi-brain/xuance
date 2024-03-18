import xuance
runner = xuance.get_runner(method='dqn',
                           env='Box2D',
                           env_id='LunarLander-v2',
                           is_test=False,
                           config_path="./configs/best_Lunar_C51.yaml")

runner.run()
