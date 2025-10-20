import os
import sys
ROOT_DIR = os.path.dirname(__file__)
sys.path.append(ROOT_DIR)

import Librairies.Temp_net as tp
import ADM_metrics as metrics
import libs.centered_motifs as motifs_lib
from Librairies import utils

import numpy as np
import matplotlib.pyplot as plt

# FOR PRODUCING THE FIGURES OF THE ADM CLASS PAPER
def ADM_paper():
	#metrics for network comparison
	list_models = []
	for version_nb in range(1, 15):
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
	tensor.compute_sim_obs()

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

if __name__ == '__main__':
	
	#example how to load an ADM dataset and an XP dataset
	######################################################
	'''
	ref_name = "conf17"
	
	#load ADM dataset
	adm_tij = tp.ADM_name(10,ref_name).load_TN()
	print(np.shape(adm_tij))

	#load xp dataset
	xp_tij = tp.XP_name(ref_name).load_TN()
	print(np.shape(xp_tij))
	'''

	#to compute the NCTN and ECTN histograms as well as the NCTN autosimilarity (compare btw models and XP!!)
	######################################################
	#choose the dataset and load its t_ij data
	xp_name = "conf16"
	xp_tij = tp.XP_name(xp_name).load_TN()
	#convert the t_ij data into a temporal network object (a temporal network with useful methods like e.g. aggregation methods)
	temp_net = tp.Temp_net(xp_tij)
	#convert the t_ij data into a sequence of graphs (temporal network), stored in temp_net.TN
	temp_net.get_TN()
	print("TN loaded")

	#compute and display the NCTN (depth 3, not truncated) similarity under time aggregation btw 2 data sets (usually low btw models and empirical data)
	#choose another dataset to compare with ref_name
	other_name = "conf17"
	other_net = tp.Temp_net(tp.XP_name(other_name).load_TN())
	other_net.get_data_time()
	print("other TN loaded")
	
	choice = "NCTN"
	xlabel = "time aggregation"; ylabel = choice+" similarity"; title = choice+" sim vs agg btw "+xp_name+" and "+other_name
	fontsize = 14
	X,Y = motifs_lib.compute_CTN_sim(temp_net,other_net,agg_max=50,step=2,choice="NCTN",depth=3,trunc=None)
	fig,ax = utils.setup_plot(xlabel,ylabel,title=title,fontsize=fontsize)
	ax.plot(X,Y,'.',label=xp_name+'/'+other_name)
	ax.legend(fontsize=fontsize)
	plt.show()

	exit()

	#compute NCTN histo: NCTN_to_weight[seq] = nb of occurrences of the NCTN of string representation seq
	#(see the first part of the paper https://arxiv.org/pdf/2501.16070 to have a thorough description of NCTN and ECTN)
	#consider NCTN of depth 3
	depth = 3
	NCTN_to_weight = motifs_lib.get_NCTN_to_weight(temp_net.TN,depth)
	print("NCTN computed")

	#compute ECTN histo
	ECTN_to_weight = motifs_lib.get_ECTN_to_weight(temp_net.TN,depth)
	print("ECTN computed")

	#keep only the 20 most frequent ECTN (just for demonstration purpose)
	truncated_ECTN_to_weight = utils.truncate_histo(ECTN_to_weight,20)
	print()
	print("ECTN string representation, nb of occurrences in the network "+xp_name+" at aggregation level 1")
	for item in truncated_ECTN_to_weight.items():
		print(*item)

	#display some specific NCTN and ECTN
	NCTN_seq = '111010101'
	ECTN_seq = '101010202033'
	
	motifs_lib.draw_CTN(NCTN_seq,depth,"NCTN of string "+NCTN_seq,choice="NCTN")
	motifs_lib.draw_CTN(ECTN_seq,depth,"ECTN of string "+ECTN_seq,choice="ECTN")

	#display the 10 most frequent NCTN in xp_name
	motifs_lib.draw_10_CTN(NCTN_to_weight,depth,starting_rank=0,normalize=True,choice="NCTN")

	#display the 10 most frequent ECTN in xp_name starting from rank 5 (so ranks 5 to 15)
	motifs_lib.draw_10_CTN(ECTN_to_weight,depth,starting_rank=5,normalize=False,choice="ECTN")

	plt.show()

	#compute and display the NCTN (depth 3, not truncated) autosimilarity under time aggregation (this usually differs much btw models and empirical data)
	choice = "NCTN"
	X,Y = motifs_lib.compute_CTN_autosim(temp_net,agg_max=50,step=2,choice=choice,depth=3,trunc=None)
	xlabel = "time aggregation"; ylabel = choice+" autosimilarity"; title = choice+" autosim vs agg"
	fontsize = 14
	fig,ax = utils.setup_plot(xlabel,ylabel,title=title,fontsize=fontsize)
	ax.plot(X,Y,'.',label=xp_name)
	ax.legend(fontsize=fontsize)
	plt.show()
