Running GPG requires install deep graph library (https://www.dgl.ai/) and compatible version of PyTorch.


This folder contains code for training GPG as well as reloading parameters for the navigation environment in the `multiagent` folder.


If you wish to train GPG from scratch, run main.py. 
If you wish to watch formations, run parameter_reload.py. parameter_reload.py loads model from ./logs. 
