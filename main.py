import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(PROJECT_ROOT)
from Librairies import ETN
from Librairies.utils import Raw_to_binned,Setup_Plot,Get_versions,Get_savename,Savename_to_name,Draw_simat,Cosim
import Librairies.Temp_net as tp
from Librairies import atn
from Librairies.settings import LIST_MARKER,LIST_COLOR,Cosim_triple

import numpy as np
import random as rd
import math
import matplotlib.pyplot as plt
import networkx as nx

if __name__ == '__main__':
	name = "conf17"

	#load empirical dataset
	pass
