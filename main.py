import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(PROJECT_ROOT)

import Librairies.temp_net as tp
import ADM_metrics as metrics

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
	ref_name = "conf17"

	#load ADM dataset
	#adm_tij = tp.ADM_name(10,ref_name).load_TN()
	#print(np.shape(adm_tij))

	#load xp dataset
	xp_tij = tp.XP_name(ref_name).load_TN()
	print(np.shape(xp_tij))

	#metrics for network comparison
	
