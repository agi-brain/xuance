import xuance
runner = xuance.get_runner(method='mappo',
                           env='mpe',
                           env_id='simple_spread_v3',
                           config_path='./configs/mappo.yaml',
                           is_test=True)
runner.run()