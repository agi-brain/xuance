import xuance
runner = xuance.get_runner(method='mappo',
                           env='mpe',
                           env_id='simple_spread_v3',
                           is_test=True,
                           config_path="./configs/mappo.yaml")
runner.run()