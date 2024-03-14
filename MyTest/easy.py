import xuance
runner = xuance.get_runner(method='MPE',
                           env='mpe',
                           env_id='simple_adversary_v3',
                           is_test=False,
                           config_path="./configs/mpe_test.yaml")
print(type(runner))
runner.run()