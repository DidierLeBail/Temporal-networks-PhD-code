import os
import sys
PROJECT_ROOT = os.path.dirname(__file__)
sys.path.append(PROJECT_ROOT)

import Librairies.temp_net as tp
import ADM_metrics as metrics

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
	'''
	ref_name = "conf17"
	#load ADM dataset
	adm_tij = tp.ADM_name(10,ref_name).load_TN()
	print(np.shape(adm_tij))

	#load xp dataset
	xp_tij = tp.XP_name(ref_name).load_TN()
	print(np.shape(xp_tij))
	'''

	#metrics for network comparison
	list_models = []
	for version_nb in range(1,15):
		name = tp.ADM_name(version_nb,"")
		is_tuned = True
		folder_name = 'ADM_class_V'+str(version_nb)
		list_models.append((name,is_tuned,folder_name))

	list_ref_name = ['conf16','conf17','highschool3','utah','work2']
	list_refs = [tp.XP_name(ref_name) for ref_name in list_ref_name]

	tensor = metrics.Distance_tensor(list_models,list_refs)
	'''
	#compute and save the reference observables
	tensor.save_obs_real()
	#compute and save the distance tensor
	tensor.compute_tensor()
	'''
	tensor.load_tensor()
	'''
	#compute the scores of each dataset wrt each observable
	tensor.compute_scores()
	#compute the various rankings
	tensor.get_rankings()
	#save score and rankings
	tensor.save_rank_score()
	'''
	tensor.load_rank_score()
	#compute and save the Kendall similarity matrix btw observables
	#the save format is .gexf so the matrix should be visualized with Gephi
	tensor.sim_obs()

	#rename models and refs so that the names can be placed on the figure
	renamed_data = {name:name for name in list_ref_name}
	renamed_data['highschool3'] = 'HS3'
	for i in range(1,15):
		renamed_data['ADM_class_V'+str(i)] = str(i)
	#choose what color should the renamed data appear
	rename_to_color = {}
	for key in list_ref_name:
		rename_to_color[renamed_data[key]] = 'green'
	for i in range(2,14):
		rename_to_color[str(i)] = 'black'
	rename_to_color['1'] = 'red'
	rename_to_color['9'] = 'red'
	rename_to_color['14'] = 'blue'

	renamed_obs = {obs:obs for obs in metrics.OBS_TO_TYPE}
	renamed_obs['edge_events_activity'] = 'nb of\nevents'
	renamed_obs['edge_newborn_activity'] = 'edge\nNBA'
	renamed_obs['clustering_coeff'] = 'clustering\ncoeff'
	renamed_obs['cc_size'] = 'cc size'

	#display the rankings (one per observable)
	tensor.display_rankings(renamed_obs=renamed_obs,rename_to_color=rename_to_color,renamed_data=renamed_data)

	#separate the conferences and the schools from the workplace
	dic_xp = {('conferences','g'):[],('schools','k'):[],('workplace','yellow'):[]}
	for name in list_ref_name:
		if 'conf' in name:
			dic_xp[('conferences','g')].append(name)
		elif 'work' in name:
			dic_xp[('workplace','yellow')].append(name)
		else:
			dic_xp[('schools','k')].append(name)

	#display the global rankings (aggregated on observables)
	for option in ['mean','sim']:
		tensor.display_glob_rank(option=option,focused_model='ADM_class_V14',renamed_data=renamed_data,dic_xp=dic_xp)

	#these groups have been defined from the analysis of sim_obs.gexf made with Gephi

	#blue group
	group1 = ['ETN3','edge_weight','node_interactivity','cc_size','ETN2_weight']
	#red group
	group2 = ['edge_newborn_activity','deg_assortativity','edge_activity','ETN3_weight','node_activity']
	#green group
	group3 = ['edge_events_activity','clustering_coeff','edge_interactivity']
	dic_group = {('blue','group I'):group1,('red','group II'):group2,('green','group III'):group3}

	#visualize how each group of observable contribute to the variations in score for our models
	tensor.score_variation(dic_group)
