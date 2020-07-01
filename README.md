## Unreal Engine
For problems in dynamic collision avoidance, the following map with walking human characters can be used to gather data to train a RL agent. The human characters are set to walk on designed spline paths which can be customised in the .uproj file in the Unreal Editor. The compiled Unreal project is available at this gdrive link: [here](https://drive.google.com/drive/folders/1ZYeQIaZDmKPVhS7VEsjHF7_miHy66nuY?usp=sharing). The project uses the Microsoft AirSIM plugin for the quadcopter and all the asset files can be accessed in the Contents folder.

| ![Environment Sample](imgs/1.png) | | ![Env sample-II ](imgs/2.png) |

## Installation
* To use the custom map, do a standard install of Microsoft AirSim and Unreal Engine and then open the project by locating the '.uproj' file in the above link. That should load the project into the Unreal editor, one can then modify or add more static/dynamic objects accordingly in the map.

* A python wrapper class over AirSim API can be found in the drone_airsim.py which facilates out of the box training(in parallel) for a Quadcopter agent in this environment. Run `python setup.py install` to install the package. 

## Description 
**State space**: 4 RGB/Grayscale stacked frames(from previous timesteps) from a monocular camera on the quadcopter.
**Action Space**: 4 discrete set of actions: Go straight, Yaw Left, Yaw right, Reverse. All of these actions maintain the vertical hover height of the quadcopter. The hover height is maintained so as the drone doesn't fly over obstacles but rather learns to avoid them at the same height. One can add modify or add more actions easily, if required.
**Reward function**:  The agent incurs a negative reward proportional to the distance between its current position and its goal position at each timestep. The episode ends with either a collision or when goal state is reached corresponding to a high negative and positive reward respectively.
