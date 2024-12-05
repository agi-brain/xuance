Runner_MAgent
==============================================

Run a multi-agent environment, leveraging functionalities from the RunnerPettingzoo parent class.
RunnerMAgent makes some extensions to it.

.. raw:: html

    <br><hr>

PyTorch
------------------------------------------

.. py:class::
  xuance.torch.runners.runner_magent.RunnerMAgent(args)

  :param args: the arguments.
  :type args: Namespace

.. raw:: html

    <br><hr>


Source Code
-----------------

.. tabs::

  .. group-tab:: PyTorch

    .. code-block:: python

        from .runner_pettingzoo import RunnerPettingzoo


        class RunnerMAgent(RunnerPettingzoo):
            def __init__(self, args):
                super(RunnerMAgent, self).__init__(args)
                self.fps = 50

