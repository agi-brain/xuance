Runner_Base
======================================

The basic class named Runner_Base, which is designed to handle running environments using the Xuance library. 
The class has an __init__ method to initialize some attributes, including setting random seeds and building environments. 
Additionally, there's a run method that can be customzied by users.

.. raw:: html

    <br><hr>

PyTorch
------------------------------------------

.. py:class::
  xuance.torch.runners.runner_basic.Runner_Base(args)

  :param args: the arguments.
  :type args: Namespace

.. py:function::
  xuance.torch.runners.runner_basic.Runner_Base.run()


.. raw:: html

    <br><hr>


TensorFlow
------------------------------------------

.. py:class::
  xuance.tensorflow.runners.runner_basic.Runner_Base(args)

  :param args: the arguments.
  :type args: Namespace

.. py:function::
  xuance.tensorflow.runners.runner_basic.Runner_Base.run()


.. py:class::
  xuance.tensorflow.runners.runner_basic.MyLinearLR(initial_learning_rate, start_factor, end_factor, total_iters)

  An implementation for learning rate decay.

  :param initial_learning_rate: The initialized learning rate.
  :type initial_learning_rate: float
  :param start_factor: The start factor of learning rate.
  :type start_factor: float
  :param end_factor: Factor for the minimum learning rate.
  :type end_factor: float
  :param total_iters: The number of iterations for decaying learning rate.
  :type total_iters: int

.. raw:: html

    <br><hr>

MindSpore
------------------------------------------

.. py:class::
  xuance.mindspore.runners.runner_basic.Runner_Base(args)

  :param args: the arguments.
  :type args: Namespace

.. py:function::
  xuance.mindspore.runners.runner_basic.Runner_Base.run()

.. raw:: html

    <br><hr>

Source Code
-----------------

.. tabs::

  .. group-tab:: PyTorch

    .. code-block:: python

      from xuance.environment import make_envs
      from xuance.torch.utils.operations import set_seed


      class Runner_Base(object):
          def __init__(self, args):
              # set random seeds
              set_seed(args.seed)

              # build environments
              self.envs = make_envs(args)
              self.envs.reset()
              self.n_envs = self.envs.num_envs

          def run(self):
              pass


  .. group-tab:: TensorFlow

    .. code-block:: python

        from xuance.environment import make_envs
        from xuance.tensorflow.utils.operations import set_seed
        import tensorflow.keras as tk


        class Runner_Base(object):
            def __init__(self, args):
                # set random seeds
                set_seed(args.seed)

                # build environments
                self.envs = make_envs(args)
                self.envs.reset()
                self.n_envs = self.envs.num_envs

            def run(self):
                pass


        class MyLinearLR(tk.optimizers.schedules.LearningRateSchedule):
            def __init__(self, initial_learning_rate, start_factor, end_factor, total_iters):
                self.initial_learning_rate = initial_learning_rate
                self.start_factor = start_factor
                self.end_factor = end_factor
                self.total_iters = total_iters
                self.learning_rate = self.initial_learning_rate
                self.delta_factor = (end_factor - start_factor) * self.initial_learning_rate / self.total_iters

            def __call__(self, step):
                self.learning_rate += self.delta_factor
                return self.learning_rate


  .. group-tab:: MindSpore

    .. code-block:: python

        from xuance.environment import make_envs
        from xuance.mindspore.utils.operations import set_seed


        class Runner_Base(object):
            def __init__(self, args):
                # set random seeds
                set_seed(args.seed)

                # build environments
                self.envs = make_envs(args)
                self.envs.reset()
                self.n_envs = self.envs.num_envs

            def run(self):
                pass


