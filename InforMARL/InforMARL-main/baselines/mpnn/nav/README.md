# Learning Transferable Cooperative Behavior in Multi-Agent Teams

[Akshat Agarwal](https://agakshat.github.io)\*, [Sumit Kumar](https://sumitsk.github.io)\*, [Katia Sycara](http://www.cs.cmu.edu/~sycara/)

Robotics Institute, Carnegie Mellon University

This is the official repository of the 'Learning Transferable Cooperative Behavior in Multi-Agent Teams' paper, available at https://arxiv.org/abs/1906.01202

This work has been presented at the [Learning and Reasoning with Graph-Structured Reprsentations](https://graphreason.github.io/) workshop (https://graphreason.github.io/papers/29.pdf) at ICML, 2019 held in Long Beach, USA. 

## Installation
See `requirements.txt` file for the list of dependencies. Create a virtualenv with python 3.5 and setup everything by executing `pip install -r requirements.txt`. 

## Examples
See `arguments.py` file for the list of various command line arguments one can set while running scripts. 

### Normal Training
Training on **Coverage Control** (`simple_spread`) environment can be started by running:

`python main.py --env-name simple_spread --num-agents 3 --entity-mp --save-dir 0`

Similarly scripts for **Formation Control** (`simple_formation`) and **Line Control** (`simple_line`) can be launched as:

`python main.py --env-name simple_formation --num-agents 3  --save-dir 0`

`python main.py --env-name simple_line --num-agents 3  --save-dir 0`

Specify the flag `--test` if you do not want to save anything. 

### Curriculum Training
To start curriculum training, specify the number of agents in `automate.py` file and execute:

`python automate.py --env-name simple_spread --entity-mp --save-dir 0`

## Results
The models trained via curriculum learning on the three environments can be found in `models` subdirectory.

The corresponding results obtained from the trained policies are located in `videos` subdirectory.

### Transfer 
You can also continue training from a saved model. For example, for training a team of 5 agents in `simple_spread` task from a policy trained with 3 agents, execute:

`python main.py --env-name simple_spread --entity-mp --continue-training --load-dir models/ss/na3_uc.pt --num-agents 5`


## Contact
For any queries, feel free to raise an issue or contact the authors at sumit.sks4@gmail.com or agarwalaks30@gmail.com.

## License
This project is licensed under the MIT License.
