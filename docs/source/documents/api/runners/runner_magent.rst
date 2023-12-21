Runner_MAgent
==============================================

xxxxxx.

.. raw:: html

    <br><hr>

**PyTorch:**

.. py:class::
  xuance.torch.runners.runner_magent.MAgent_Runner(args)

  :param args: the arguments.
  :type args: Namespace

.. raw:: html

    <br><hr>


Source Code
-----------------

.. tabs::

  .. group-tab:: PyTorch

    .. code-block:: python

        from .runner_pettingzoo import Pettingzoo_Runner


        class MAgent_Runner(Pettingzoo_Runner):
            def __init__(self, args):
                super(MAgent_Runner, self).__init__(args)
                self.fps = 50

