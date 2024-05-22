Running GPG requires install deep graph library (https://www.dgl.ai/) and compatible version of PyTorch.


This folder contains code for training GPG as well as reloading parameters for larger swarms.

To run this you must first install gym_formation according to the instructions inside gym_formation


If you wish to train GPG from scratch, run main.py. 
If you wish to watch formations, run parameter_reload.py. parameter_reload.py loads model from ./logs. 

A simple arrowhead formation is specified in constr_formation_flying.py (inside gym_formation/gym_flock/envs/)
To increase the number of agents simply change the parameter self.n_agents in constr_formation_flying.py, line 39.


Lastly, if you wish to use the VPG baseline, uncomment line 23 and comment out line 22 in main.py. 

Thank You
