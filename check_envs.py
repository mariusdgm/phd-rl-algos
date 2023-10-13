import rl_envs_forge

from rl_envs_forge.envs.labyrinth.labyrinth import Labyrinth

print(rl_envs_forge.__version__)

env = Labyrinth(10, 10, seed=0)
env.action_space.n