Tuning Tools
--------------

HyperParameters Tuning for A Single Objective
'''''''''''''''''''''''''''''''''''''''''''''''

We support users in automatically tuning hyperparameters using our core APIs: ``HyperParameterTuner`` and ``set_hyperparameters``.

.. code-block:: python

    from xuance.common import HyperParameterTuner, set_hyperparameters

    tuner = HyperParameterTuner(method='dqn',
                                config_path='./examples/dqn/dqn_configs/dqn_cartpole.yaml',
                                running_steps=1000,
                                test_episodes=2)

Next, users should select the hyperparameters they wish to tune by calling the ``select_hyperparameter`` method.

.. code-block:: python

    selected_hyperparameters = tuner.select_hyperparameter(['learning_rate'])

If you want to override the default parameter ranges, you can use the ``set_hyperparameters`` method.

.. code-block:: python

    overrides = {
        'learning_rate': [0.001, 0.0001, 0.00001]
    }
    selected_hyperparameters = set_hyperparameters(selected_hyperparameters, overrides)

Finally, you can initiate the hyperparameter tuning process by using the ``tune`` method.

.. code-block:: python

    tuner.tune(selected_hyperparameters, n_trails=5, pruner=None)

Full code: `tune_dqn.py <https://github.com/agi-brain/xuance/blob/master/examples/hyperparameter_tuning/tune_dqn.py>`_

Hyper-Parameters Tuning for Multi-Objective
'''''''''''''''''''''''''''''''''''''''''''''

We also support users in automatically tuning hyperparameters by optimizing multiple objectives.
The core APIs: ``MultiObjectiveTuner`` and ``set_hyperparameters``.

.. code-block::python

    from xuance.common import MultiObjectiveTuner, set_hyperparameters
    from optuna.visualization import plot_pareto_front

    tuner = MultiObjectiveTuner(method='dqn',
                                config_path='./examples/dqn/dqn_configs/dqn_cartpole.yaml',
                                running_steps=10000,
                                test_episodes=2)

Then, select the hyperparameters by calling the ``select_hyperparameter`` method.

.. code-block:: python

    selected_hyperparameters = tuner.select_hyperparameter(['learning_rate', 'gamma'])

Similarly, if you want to override the default parameter ranges, you can use the ``set_hyperparameters`` method.

.. code-block:: python

    overrides = {
        'learning_rate': [0.001, 0.0001, 0.00001]
    }
    selected_hyperparameters = set_hyperparameters(selected_hyperparameters, overrides)

Finally, you can initiate the hyperparameter tuning process by using the ``tune`` method,
and select the objectives to be optimized.

.. code-block:: python

    study = tuner.tune(selected_hyperparameters, n_trials=30, pruner=None, directions=['maximize', 'maximize'],
                       selected_objectives=['test_score', 'Qloss'])

Full code: `tune_dqn_multiobjective.py <https://github.com/agi-brain/xuance/blob/master/examples/hyperparameter_tuning/tune_dqn_multiobjective.py>`_

APIs
'''''''''

.. automodule:: xuance.common.tuning_tools
    :members:
    :undoc-members:
    :show-inheritance:
