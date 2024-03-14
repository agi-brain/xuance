import xuance
runner = xuance.get_runner(method='sac',
                           env='Box2D',
                           env_id='LunarLander-v2',
                           is_test=True,
                           config_path="./configs/basic.yaml")
runner.run()