
import socket
from absl import flags
from .SimEnv import SimulationEnvironment_Unmaze,SimulationEnvironment_CA,SimulationEnvironment_CubeCollection,SimulationEnvironment_Warehouse
# from .env_core import EnvCore
# from .env_corew import EnvCorew
# from .env_continuous import ContinuousActionEnv

FLAGS = flags.FLAGS
FLAGS(['train_sc.py'])


