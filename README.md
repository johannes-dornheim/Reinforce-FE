# Reinforce-FE
A collection of Reinforcement Learning environments based on physical simulations and supporting the "openAI Gym" environment API.
```
@article{dornheim2019model,
  title={Model-Free Adaptive Optimal Control of Episodic Fixed-Horizon Manufacturing Processes using Reinforcement Learning},
  author={Dornheim, Johannes and Link, Norbert and Gumbsch, Peter},
  journal={International Journal of Control, Automation and Systems},
  pages={1--12},
  year={2019},
  publisher={Springer}
}
```
## Setup
### prerequisites
- Linux OS (tested on Ubuntu, Mint)
- Python 3
- For the Deep-Drawing Environments: Abaqus (Version 6.14)

### Installation

```
git clone https://github.com/johannes-dornheim/Reinforce-FE
cd Reinforce-FE
pip install -e .
```
Test the installation of Reinforce-FE and Abaqus modules:
```
cd examples/agents
python random_agent.py
```
should run the random-agent on the 5 time-step Deep-Drawing environment.


## Features
## Usage
## Publications based on the code
- Dornheim, Johannes, Norbert Link, and Peter Gumbsch. "Model-Free Adaptive Optimal Control of Episodic Fixed-Horizon Manufacturing Processes using Reinforcement Learning." 
- Dornheim, Johannes, and Norbert Link. "Multiobjective Reinforcement Learning for Reconfigurable Adaptive Optimal Control of Manufacturing Processes."

## About
This repository is a result of ongoing research, carried out by the [Intelligent Systems Research Group](http://www.iwi.hs-karlsruhe.de/ResearchGroups/ISRG/) at the Karlsruhe University of Applied Sciences. The according research projects are funded by the DFG (within the Research Training Group 1483) and the German Federal Ministry of Education and Research (under grant \#03FH061PX5). The Abaqus deep drawing simulation code is based on working material of the [Chair for Continuum Mechanics in Engineering Mechanics](https://www.itm.kit.edu/english/cm/index.php), Karlsruhe Institute of Technology (KIT).
