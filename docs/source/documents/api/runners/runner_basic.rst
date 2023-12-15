Runner_Base
======================================

xxxxxx.

.. raw:: html

    <br><hr>

**PyTorch:**

.. py:class::
  xuance.torch.runners.runner_basic.Runner_Base(args)

  :param args: xxxxxx.
  :type args: xxxxxx

.. py:function::
  xuance.torch.runners.runner_basic.Runner_Base.run()

  xxxxxx.

.. raw:: html

    <br><hr>

**TensorFlow:**

.. raw:: html

    <br><hr>

**MindSpore:**

.. py:class::
  xuance.mindspore.runners.runner_basic.Runner_Base(args)

  :param args: xxxxxx.
  :type args: xxxxxx

.. py:function::
  xuance.mindspore.runners.runner_basic.Runner_Base.run()

  xxxxxx.

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

