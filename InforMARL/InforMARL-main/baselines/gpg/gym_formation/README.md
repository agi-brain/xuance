# Gym Flock

## Dependencies
- OpenAI [Gym](https://github.com/openai/gym) 0.11.0
- Python 3 (Python 2 doesn't work)

## To install
1) Clone this repository
2) `pip3 install -e . `

## To use

Include the following code in your Python script:
~~~~
import gym  
import gym_flock 
env = gym.make("FormationFlying-v3")` 
~~~~
and then use the `env.reset()` and `env.step()` for interfacing with the environment as you would with other OpenAI Gym environments. 
These implementations also include a `env.controller()` function that gives the best current set of actions to be used for imitation learning.




