MetaDrive
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MetaDrive is an autonomous driving simulator that supports generating infinite scenes with various road maps and traffic settings for research of generalizable RL.

| **Official link**: `https://metadriverse.github.io/metadrive/ <https://metadriverse.github.io/metadrive/>`_
| **Paper link**: `https://arxiv.org/pdf/2109.12674.pdf <https://arxiv.org/pdf/2109.12674.pdf>`_

Installation
'''''''''''''''''''''''''''''''''''''''''

Open the terminal and create your conda environment.
Then, you can choose one of the listed methods to finish the installation of MetaDrive.

**Method 1**: From PyPI.

.. code-block:: bash

    pip install metadrive

**Method 2**: From GitHub.

.. code-block:: bash

    git clone https://github.com/metadriverse/metadrive.git
    cd metadrive
    pip install -e .

Try an Example
'''''''''''''''''''''''''''''''''''''''''

.. attention::

    Please note that each process should only have one single MetaDrive instance due to the limit of the underlying simulation engine.
    Thus the parallelization of training environment should be in process-level instead of thread-level.

Create a python file named, e.g., "demo_metadrive.py"

.. code-block:: python

    import argparse
    from xuance import get_runner

    def parse_args():
        parser = argparse.ArgumentParser("Run a demo.")
        parser.add_argument("--method", type=str, default="ppo")
        parser.add_argument("--env", type=str, default="metadrive")
        parser.add_argument("--env-id", type=str, default="your_map")
        parser.add_argument("--test", type=int, default=0)
        parser.add_argument("--device", type=str, default="cuda:0")
        parser.add_argument("--parallels", type=int, default=10)
        parser.add_argument("--benchmark", type=int, default=1)
        parser.add_argument("--test-episode", type=int, default=5)

        return parser.parse_args()


    if __name__ == '__main__':
        parser = parse_args()
        runner = get_runner(method=parser.method,
                            env=parser.env,
                            env_id=parser.env_id,
                            parser_args=parser,
                            is_test=parser.test)
        if parser.benchmark:
            runner.benchmark()
        else:
            runner.run()

Open the terminal the type the python command:

.. code-block:: bash

    python demo_metadrive.py

| Then, let your GPU and CPU work and wait for the training process to finish.
| Finally, you can test the trained model and view the effectiveness.

.. code-block:: bash

    python demo_metadrive.py --benchmark 0 --test 1

.. tip::

    When you successfully trained a model and visualize the MetaDrive simulator,
    you might find that the fps is too low to watch the effectiveness.

    **Solution**: You can hold on the F key to accelerate the simulation.

APIs
'''''''''''''''''''''''''''''''''''''''''

.. automodule:: xuance.environment.single_agent_env.metadrive
    :members:
    :undoc-members:
    :show-inheritance:

