import Librairies.Observables as obs_lib
import Librairies.Temp_net as tp
from Librairies.Temp_net import ADM_DIR,ROOT_DIR
from scipy.stats import kendalltau
import networkx as nx
import os
import numpy as np
import matplotlib.pyplot as plt

#SAVE OBS METHODS
####################################################################################################################################
#save the realization val of the observable of name name_obs in the dataset name
def Save_point(path,name_obs,val):
	savepath = os.path.join(path,'/point/'+name_obs+'.txt')
	np.savetxt(savepath,[val])
def Save_distribution(path,name_obs,val):
	savepath = os.path.join(path,'/distribution/'+name_obs+'.txt')
	np.savetxt(savepath,np.array(list(zip(*val.items()))),fmt='%d')
def Save_vector(path,name_obs,val):
	for agg,dic in val.items():
		savepath = os.path.join(path,'/vector/'+name_obs+str(agg)+'.txt')
		np.savetxt(savepath,np.array(list(zip(*dic.items()))),fmt='%s')

#LOAD OBS METHODS
####################################################################################################################################
#load observable realization and put it in a form suitable for distance comptutation
def Load_point(path,name_obs):
	savepath = os.path.join(path,'/point/'+name_obs+'.txt')
	return np.loadtxt(savepath)
def Load_distribution(path,name_obs):
	savepath = os.path.join(path,'/distribution/'+name_obs+'.txt')
	tab = np.loadtxt(savepath,dtype=int)
	norm = float(np.sum(tab[1,:]))
	return {tab[0,i]:float(tab[1,i])/norm for i in range(np.size(tab,1))}
def Load_vector(path,name_obs,agg_max=10):
	res = {}
	for agg in range(1,agg_max+1):
		savepath = os.path.join(path,'/vector/'+name_obs+str(agg)+'.txt')
		tab = np.loadtxt(savepath,dtype=str)
		res[agg] = {tab[0,i]:float(tab[1,i]) for i in range(np.size(tab,1))}
	return res

#DISTANCE OBS METHODS
####################################################################################################################################
#return the cosine similarity btw the two dict of ETN etn1 and etn2
def cosim(etn1,etn2):
	norm1 = sum([val**2 for val in etn1.values()])
	norm2 = sum([val**2 for val in etn2.values()])
	s = 0
	for key in set(etn1.keys()).intersection(set(etn2.keys())):
		s += etn1[key]*etn2[key]
	return s/sqrt(norm1*norm2)

#compute distance between two realizations obs1,obs2 of the same observable
#note that all these distances are renormalized between 0 and 1
def point_distance(obs1,obs2):
	return abs(obs2-obs1)/(2*max(abs(obs2),abs(obs1)))

def vector_distance(obs1,obs2):
	#we actually have one sub-vector per aggregation level
	#the similarity btw the two vectors is the product of the similarity of their sub-vectors
	#the distance is 1-similarity
	tot_sim = 1
	for agg in obs1.keys():
		tot_sim *= cosim(obs1[agg],obs2[agg])
	return 1-tot_sim

def distribution_distance(obs1,obs2):
	#first use log-binning to enhance the quality of the raw power-laws
	X1,Y1 = Raw_to_binned(obs1)
	X2,Y2 = Raw_to_binned(obs2)
	nb = max(len(Y1),len(Y2))
	T1 = np.zeros(nb); T2 = np.zeros(nb)
	T1[:len(Y1)] = np.power(10,np.asarray(Y1))[:]
	T2[:len(Y2)] = np.power(10,np.asarray(Y2))[:]
	T = (T1+T2)/2
	return (np.sum(T1*np.log2(T1/T+1e-12)) + np.sum(T2*np.log2(T2/T+1e-12)))/2

#GLOBAL VARIABLES
####################################################################################################################################

#define the observables that will be sampled: these are the reference observables
TYPE_TO_OBS = {'point':['clustering_coeff','deg_assortativity'],'vector':['ETN3']}
TYPE_TO_OBS['distribution'] = ['cc_size']
TYPE_TO_OBS['distribution'] += ['edge_activity','edge_newborn_activity','edge_events_activity','node_activity']
TYPE_TO_OBS['distribution'] += ['edge_interactivity','node_interactivity']
TYPE_TO_OBS['distribution'] += ['edge_weight','ETN2_weight','ETN3_weight']
OBS_TO_TYPE = {}
for type_obs,val in TYPE_TO_OBS.items():
	for name_obs in val:
		OBS_TO_TYPE[name_obs] = type_obs

SAVE_OBS = {'point':Save_point,'distribution':Save_distribution,'vector':Save_vector}
LOAD_OBS = {'point':Load_point,'distribution':Load_distribution,'vector':Load_vector}
DISTANCE_OBS = {'point':point_distance,'distribution':distribution_distance,'vector':vector_distance}

####################################################################################################################################

#return the reference observables for a given temporal network TN (instance of tp.Temp_net)
def sample_ref_obs(TN):
	node_train = obs_lib.get_node_train(TN.TN)
	edge_train = obs_lib.get_edge_train(TN.TN)
	#compute the weighted fully aggregated network
	agg_TN = TN.fully_agg(is_weighted=True)

	#observable realizations
	obs_real = {key:{} for key in TYPE_TO_OBS}

	#distribution observables
	obs_real['distribution']['edge_activity'] = obs_lib.duration(edge_train)
	obs_real['distribution']['edge_newborn_activity'] = obs_lib.newborn_duration(edge_train)
	obs_real['distribution']['edge_events_activity'] = obs_lib.weak_event_duration(edge_train)
	obs_real['distribution']['edge_interactivity'] = obs_lib.interduration(edge_train)
	obs_real['distribution']['edge_weight'] = obs_lib.sample_values([agg_TN[el[0]][el[1]]['weight'] for el in agg_TN.edges()])
	#would also work: obs_real['distribution']['edge_time_weight'] = obs_lib.natural_weight(obj_train=edge_train)
	obs_real['distribution']['node_activity'] = obs_lib.duration(node_train)
	obs_real['distribution']['node_interactivity'] = obs_lib.interduration(node_train)
	obs_real['distribution']['cc_size'] = obs_lib.cc_size(TN.TN)
	for depth in [2,3]:
		obs_real['distribution']['ETN'+str(depth)+'_weight'] = obs_lib.natural_weight(obj_to_weight=obs_lib.NCTN_to_weight(TN.TN,depth))

	#vector observable: compute the NCTN_to_weight for depth 3 and aggregation levels from 1 to 10
	#but retain only the 20 most frequent NCTN at each level of aggregation
	giant_vec = {}; depth = 3
	for agg in range(1,11):
		#aggregate the network
		TN.sliding_time_aggregation(agg)
		dic = obs_lib.NCTN_to_weight(TN.TN,depth)

		#retain only the 20 most frequent NCTN
		most_freq = sorted(dic.keys(),key=lambda seq:dic[seq],reverse=True)[:20]
		giant_vec[agg] = {seq:dic[seq] for seq in most_freq}
	obs_real['vector']['ETN3'] = dict(**giant_vec)

	#point observables
	obs_real['point']['clustering_coeff'] = nx.average_clustering(agg_TN)
	obs_real['point']['deg_assortativity'] = nx.degree_pearson_correlation_coefficient(agg_TN,weight='weight')

	return obs_real

#collect descriptive information about TN (instance of tp.Temp_net):
#nb of nodes, number of timestamps, number of temporal edges
#minimum node activity, maximum node activity
def collect_info(TN):
	X,Y = zip(*TN.get_info.items())
	return [list(X),[str(y) for y in Y]]

#DISTANCE_TENSOR
####################################################################################################################################

class Distance_tensor:
	"""
	list_models = list of triples (tp.TN_name,boolean,folder_name)
	the first element is the name of the models we want to compare
	the second is the name of the folder in which to store the ampled observables
	the third element is True if the model is tuned (different parameters for each reference) and False else

	list_refs = list of tp.TN_name
	"""
	def __init__(self,list_models,list_refs):
		self.list_refs = list_refs
		self.list_models = list_models
		#where is stored the distance tensor
		self.tensor_folder = os.path.join(ROOT_DIR,'results/distance_tensor')

		#model with the lowest or highest global rank
		self.mean_best_model = ''
		self.sim_best_model = ''
		self.mean_worst_model = ''
		self.sim_worst_model = ''

		#name_to_int[name] = integer identifier associated to dataset name
		self.name_to_int = {name.savename:i for i,name in enumerate(list_refs)}
		for i,el in enumerate(list_models):
			self.name_to_int[el[2]] = i + len(list_refs)
		self.int_to_name = {i:name for name,i in self.name_to_int.items()}

		#obs_to_int[name_obs] = integer identifier associated to observable name_obs
		self.obs_to_int = {name_obs:i for i,name_obs in enumerate(OBS_TO_TYPE.keys())}
		self.int_to_obs = {i:name for name,i in self.obs_to_int.items()}
		self.nb_obs = len(self.int_to_obs)
		self.nb_data = len(self.name_to_int)
		self.nb_XP = len(list_refs)
		self.nb_mod = len(list_models)

		#dic_tensor[name_obs][name1][name2] = distance from name1 to name2 wrt the observable name_obs
		self.tensor = np.zeros((self.nb_obs,self.nb_data,self.nb_data))
		#score[name_obs][name] = score of the dataset name wrt the observable name_obs
		self.score = np.zeros((self.nb_obs,self.nb_data))
		#stat_XP[name_obs] = [Q1,Q2,Q3] for the set of empirical datasets (list_refs)
		self.stat_XP = np.zeros((self.nb_obs,3))
		#ranking[name_obs][k] = [i_k,r_k], where r_k is the rank of the dataset i_k
		#ranking[name_obs] is a 2D numpy array
		self.ranking = np.zeros((self.nb_obs,self.nb_data,2),dtype=int)
		#glob_rank[i] = rank of dataset i averaged over all observables
		self.mean_glob_rank = np.zeros(self.nb_data)
		self.sim_glob_rank = np.zeros(self.nb_data)
		#ranking Kendall similarity matrix btw observables
		self.sim_obs = np.eye(self.nb_obs)

		#obs_to_xlabel[obs] = xlabel associated to the observable obs when its distribution is plotted
		self.obs_to_xlabel = {}
		for obs in TYPE_TO_OBS['distribution']:
			if 'events' in obs:
				self.obs_to_xlabel[obs] = r"$\log_{10}(n)$"
			elif 'duration' in obs:
				self.obs_to_xlabel[obs] = r"$\log_{10}(\Delta t)$"
			elif 'weight' in obs:
				self.obs_to_xlabel[obs] = r"$\log_{10}(w)$"
			elif 'degree' in obs:
				self.obs_to_xlabel[obs] = r"$\log_{10}(k)$"
			else:
				self.obs_to_xlabel[obs] = r"$\log_{10}(n)$"

	#intermediate in self.save_obs_real
	def save_obs_real_from_t_ij(self,t_ij,path_chunk):
		#convert it into a tp.Temp_net instance
		TN = tp.Temp_net(t_ij)

		#compute the list of interaction graphs at aggregation level 1
		TN.sliding_time_aggregation(1)

		#collect global info
		global_info = collect_info(TN)
		np.savetxt(os.path.join(path_chunk,'global_info.txt'),np.asarray(global_info,dtype=str),fmt='%s',delimiter=',')

		#sample the reference observables
		obs_real = sample_ref_obs(TN)

		#save the obs realizations
		for type_obs,obs_to_real in obs_real.items():
			for name_obs,real in obs_to_real.items():
				SAVE_OBS[type_obs](path_chunk,name_obs,real)

	#compute and store the realizations of each reference observable for every dataset
	def save_obs_real(self):
		for name in self.list_refs:
			#load the t_ij
			t_ij = name.load_TN()

			#compute and store the realizations of the reference observables at the correct location
			path_chunk = os.path.join(ADM_DIR,'ref_data/'+name.savename)
			self.save_obs_real_from_t_ij(t_ij,path_chunk)

		for name,is_tuned,folder_name in self.list_models:
			if is_tuned:
				for ref in self.list_refs:
					name.ref_name = ref.savename
					name.get_savename()
					t_ij = name.load_TN()

					path_chunk = os.path.join(ADM_DIR,'models/'+folder_name+'/'+ref.savename)
					self.save_obs_real_from_t_ij(t_ij,path_chunk)
			else:
				t_ij = name.load_TN()
				path_chunk = os.path.join(ADM_DIR,'models/'+folder_name)
				self.save_obs_real_from_t_ij(t_ij,path_chunk)

	#compute the distance tensor from the stored realizations of the ref observables
	#then save the tensor.
	#note that we do not compute the distance from model to model, but only ref to ref and ref to model
	#res[name_obs][d1][d2] = distance from d1 to d2 relatively to the observable name_obs
	def compute_tensor(self):
		#compute and save the distance tensor
		res = {name_obs:{} for name_obs in OBS_TO_TYPE.keys()}

		#compute the block ref-ref
		print('block ref-ref begins')
		for name_obs,type_obs in OBS_TO_TYPE.items():
			print(name_obs)
			for ref in self.list_refs:
				res[name_obs][ref.savename] = {}
			for i,ref1 in enumerate(self.list_refs):
				name1 = ref1.savename
				path_chunk = os.path.join(ADM_DIR,'ref_data/'+name1)
				obs1 = LOAD_OBS[type_obs](path_chunk,name_obs)
				for ref2 in self.list_refs[i+1:]:
					name2 = ref2.savename
					path_chunk = os.path.join(ADM_DIR,'ref_data/'+name2)
					obs2 = LOAD_OBS[type_obs](path_chunk,name_obs)
					res[name_obs][name1][name2] = DISTANCE_OBS[type_obs](obs1,obs2)
					res[name_obs][name2][name1] = res[name_obs][name1][name2]
				res[name_obs][name1][name1] = 0
		#save the block ref-ref
		for name_obs,dic in res.items():
			first_row = ['&']+[ref.savename for ref in self.list_refs]
			tab = [first_row]
			for name1 in first_row[1:]:
				row = [name1]
				for name2 in first_row[1:]:
					row.append(str(dic[name1][name2]))
				tab.append(row)
			savepath = os.path.join(self.tensor_folder,name_obs+'XP_XP.txt')
			np.savetxt(savepath,np.asarray(tab,dtype=str),fmt='%s')
		print("tensor block ref-ref stored in "+self.tensor_folder)

		#compute the block ref-model
		res = {name_obs:{} for name_obs in OBS_TO_TYPE}
		print('block ref-model begins')
		for name_obs,type_obs in OBS_TO_TYPE.items():
			print(name_obs)
			#ref_obs_real[ref_name] = realization of name_obs in the dataset ref_name
			ref_obs_real = {}
			for ref in self.list_refs:
				ref_name = ref.savename
				path_chunk = os.path.join(ADM_DIR,'ref_data/'+ref_name)
				ref_obs_real[ref_name] = LOAD_OBS[type_obs](path_chunk,name_obs)

			for _,is_tuned,model_name in self.list_models:
				res[name_obs][model_name] = {}
				if is_tuned:
					for ref_name,ref_real in ref_obs_real.items():
						path_chunk = os.path.join(ADM_DIR,'models/'+model_name+'/'+ref_name)
						model_real = LOAD_OBS[type_obs](path_chunk,name_obs)

						res[name_obs][model_name][ref_name] = DISTANCE_OBS[type_obs](model_real,ref_real)
						res[name_obs][ref_name][model_name] = res[name_obs][model_name][ref_name]
				else:
					path_chunk = os.path.join(ADM_DIR,'models/'+model_name)
					model_real = LOAD_OBS[type_obs](path_chunk,name_obs)
					for ref_name,ref_real in ref_obs_real.items():
						res[name_obs][model_name][ref_name] = DISTANCE_OBS[type_obs](model_real,ref_real)
						res[name_obs][ref_name][model_name] = res[name_obs][model_name][ref_name]
		#save the block ref-model
		for name_obs,dic in res.items():
			first_row = ['&']+[el[2] for el in self.list_models]
			tab = [first_row]
			for ref_name in ref_obs_real:
				row = [ref_name]
				for model_name in first_row[1:]:
					row.append(str(dic[ref_name][model_name]))
				tab.append(row)
			savepath = os.path.join(self.tensor_folder,name_obs+'XP_model.txt')
			np.savetxt(savepath,np.asarray(tab,dtype=str),fmt='%s')
		print("tensor block ref-model stored in "+self.tensor_folder)

	#load the distance tensor
	def load_tensor(self):
		for n,name_obs in self.int_to_obs.items():
			path1 = os.path.join(self.tensor_folder,name_obs+'XP_XP.txt')
			path2 = os.path.join(self.tensor_folder,name_obs+'XP_model.txt')
			tab = np.loadtxt(path1,dtype=str)
			tab2 = np.loadtxt(path2,dtype=str)
			for i in range(self.nb_XP):
				#load the XP_XP block
				k1 = self.name_to_int[tab[i+1,0]]
				for j in range(self.nb_XP):
					k2 = self.name_to_int[tab[0,j+1]]
					self.tensor[n,k1,k2] = float(tab[i+1,j+1])
				#load the XP_model block
				k1 = self.name_to_int[tab2[i+1,0]]
				for j in range(self.nb_mod):
					k2 = self.name_to_int[tab2[0,j+1]]
					self.tensor[n,k1,k2] = float(tab2[i+1,j+1])
					self.tensor[n,k2,k1] = self.tensor[n,k1,k2]

	#compute statistical properties of the block XP-XP of the distance tensor
	#median, first and third quartiles
	def compute_stat_XP(self):
		for t in range(self.nb_obs):
			list_val = []
			for i,ref1 in enumerate(self.list_refs[:-1]):
				ind1 = self.name_to_int[ref1.savename]
				for ref2 in self.list_refs[i+1:]:
					ind2 = self.name_to_int[ref2.savename]
					list_val.append(self.tensor[t,ind1,ind2])
			self.stat_XP[t,:] = np.quantile(list_val,[0.25,0.5,0.75])
			if self.stat_XP[t,2]==self.stat_XP[t,0]:
				self.stat_XP[t,2] = self.stat_XP[t,0]+1

	#compute wrt the observable of obs_to_int obs_num the score of the dataset with name_to_int data_num
	def score_dataset(self,obs_num,data_num):
		list_val = []
		for ref in self.list_refs:
			ref_num = self.name_to_int[ref.savename]
			if ref_num!=data_num:
				list_val.append(self.tensor[obs_num,ref_num,data_num])
		med = np.quantile(list_val,0.5)
		self.score[obs_num,data_num] = (self.stat_XP[obs_num,1]-med)/(self.stat_XP[obs_num,2]-self.stat_XP[obs_num,0])

	#compute the scores of every dataset wrt every observable
	def compute_scores(self):
		self.compute_stat_XP()
		for obs_num in self.int_to_obs:
			for data_num in self.int_to_name:
				self.score_dataset(obs_num,data_num)

	#compute one ranking per observable
	#ranking[name_obs][k] = [i_k,r_k], where r_k is the rank of the dataset i_k
	#ranking[name_obs] is a 2D numpy array
	#then compute the global ranking :
	#glob_rank[i] = rank of dataset i averaged over all observables
	#finally determine the best model, defined as the model with the lowest global rank
	def get_rankings(self):
		list_datasets = list(range(self.nb_data))
		for t in range(self.nb_obs):
			list_datasets.sort(key=lambda i_mod:self.score[t,i_mod],reverse=True)
			rank = 1; score = self.score[t,list_datasets[0]]
			for k,i_mod in enumerate(list_datasets):
				if self.score[t,i_mod]<score:
					rank = k+1
					score = self.score[t,i_mod]
				self.ranking[t,k,0] = i_mod
				self.ranking[t,k,1] = rank

		#compute the mean global ranking
		for t in range(self.nb_obs):
			for k in range(self.nb_data):
				i,r = self.ranking[t,k,:]
				self.mean_glob_rank[i] += r
		self.mean_glob_rank /= self.nb_obs

		#compute the sim global ranking
		glob_score = np.sum(self.score,axis=0)/self.nb_data
		score_max = np.max(glob_score)
		for k in range(self.nb_data):
			self.sim_glob_rank[k] = score_max-glob_score[k]

	#save self.score, self.ranking, self.mean_glob_rank and self.sim_glob_rank
	def save_rank_score(self):
		path = os.path.join(ROOT_DIR,'results/rank_score')
		np.savetxt(os.path.join(path,'score.txt'),self.score)

		for name,tab in zip(['mean_glob_rank','sim_glob_rank'],[self.mean_glob_rank,self.sim_glob_rank]):
			savepath = os.path.join(path,name+str(self.nb_mod)+'.txt')
			np.savetxt(savepath,tab)
		for t in range(self.nb_obs):
			savepath = os.path.join(path,'ranking'+str(t)+'_'+str(self.nb_mod)+'.txt')
			np.savetxt(savepath,self.ranking[t,:,:],fmt='%d')
		print('score and rankings saved in '+path)

	#load self.score, self.ranking, self.mean_glob_rank and self.sim_glob_rank
	#then deduce self.mean_best_model and self.sim_best_model
	def load_rank_score(self):
		path = os.path.join(ROOT_DIR,'results/rank_score')
		self.score = np.loadtxt(os.path.join(path,'score.txt'))

		savepath = os.path.join(path,'sim_glob_rank'+str(self.nb_mod)+'.txt')
		self.sim_glob_rank = np.loadtxt(savepath)
		savepath = os.path.join(path,'mean_glob_rank'+str(self.nb_mod)+'.txt')
		self.mean_glob_rank = np.loadtxt(savepath)

		for t in range(self.nb_obs):
			savepath = os.path.join(path,'ranking'+str(t)+'_'+str(self.nb_mod)+'.txt')
			self.ranking[t,:,:] = np.loadtxt(savepath,dtype=int)[:,:]
		models = [folder_name for _,_,folder_name in self.list_models]
		key1 = lambda name:self.mean_glob_rank[self.name_to_int[name]]
		key2 = lambda name:self.sim_glob_rank[self.name_to_int[name]]
		self.mean_best_model = min(models,key=key1)
		self.sim_best_model = min(models,key=key2)
		self.mean_worst_model = max(models,key=key1)
		self.sim_worst_model = max(models,key=key2)

	#compute the ranking Kendall similarity matrix btw observables
	#convert this matrix into a weighted network and save it in gephi format
	#if ADM, list_models should only contain versions 1 to 13 because we are interested in trade-offs within
	#the ADM class:
	#what observables can we independently obtain by moving in the hypotheses space?
	def compute_sim_obs(self):
		models_num = [self.name_to_int[model_name] for _,_,model_name in self.list_models]
		for t in range(self.nb_obs-1):
			tab = sorted(models_num,key=lambda k:self.ranking[t,k,0])
			x = [self.ranking[t,k,1] for k in tab]
			for u in range(t+1,self.nb_obs):
				tab2 = sorted(models_num,key=lambda k:self.ranking[u,k,0])
				y = [self.ranking[u,k,1] for k in tab2]
				self.sim_obs[t,u] = kendalltau(x,y)[0]
				self.sim_obs[u,t] = self.sim_obs[t,u]

		#build and save the weighted similarity network
		network = nx.Graph()
		for t in range(self.nb_obs-1):
			obs1 = self.int_to_obs[t]
			for u in range(t+1,self.nb_obs):
				obs2 = self.int_to_obs[u]
				weight = abs(self.sim_obs[t,u])
				if weight>0:
					network.add_edge(obs1,obs2,weight=weight)
		savepath = os.path.join(ROOT_DIR,'results/sim_obs.gexf')
		nx.write_gexf(network,savepath)
		print('Kendall similarity btw observables saved at '+savepath)

	#for each model version, compute the difference btw its score and the score of the basis model
	#(the first element of self.list_models) for each observable
	#then sum up these variations for each group of observables
	def score_variation(self,dic_group):
		list_model_name = [model_name for _,_,model_name in self.list_models]
		fontsize = 16; markersize = 12
		variations = {}; i_basis = self.name_to_int[self.list_models[0][2]]
		for model_name in list_model_name:
			i = self.name_to_int[model_name]
			variations[i] = {key:{} for key in dic_group.keys()}
			for key,group in dic_group.items():
				for obs in group:
					t = self.obs_to_int[obs]
					val = self.score[t,i]-self.score[t,i_basis]
					variations[i][key][obs] = val

		#display the total score variations for each group of observables
		fig,ax = plt.subplots(1,1,constrained_layout=True)
		X = []; dic_Y = {key:[] for key in dic_group.keys()}
		for i,val in variations.items():
			#compute the model version
			mod_version = int(self.int_to_name[i].split('_V')[1])
			X.append(mod_version)
			for key,value in val.items():
				dic_Y[key].append(sum(list(value.values())))
		group_to_label = {}
		for num in ['I'*(i+1) for i in range(len(dic_group))]:
			group_to_label['group '+num] = r"$\Delta s^{"+num+r"}$"
		for key,Y in dic_Y.items():
			ax.plot(X,Y,'.',color=key[0],label=group_to_label[key[1]],markersize=markersize)
		ax.plot([1,self.nb_mod],[0,0],'--')
		ax.set_xlabel('model identifier',fontsize=fontsize)
		ax.set_ylabel(r"$\Delta s$",fontsize=fontsize)
		ax.legend(fontsize=fontsize)
		xticks = [_ for _ in X]
		ax.set_xticks(xticks)
		for label in ax.get_xticklabels()+ax.get_yticklabels():
			label.set_fontsize(fontsize)
		#display the figure
		plt.show()

		#for each group, display the contribution of each observable
		fontsize = 14
		fig,ax = plt.subplots(len(dic_group),1,constrained_layout=True)
		for i,key in enumerate(dic_group.keys()):
			ax[i].set_title(key[1],fontsize=fontsize)
		for k in range(len(ax)-1):
			ax[k].tick_params(axis='x',which='both',bottom=False,labelbottom=False)
		ax[-1].set_xlabel('model identifier',fontsize=fontsize)
		ax[-1].set_xticks(xticks)
		for label in ax[-1].get_xticklabels():
			label.set_fontsize(fontsize)
		for mod_version,val in zip(xticks,variations.values()):
			for k,key in enumerate(dic_group.keys()):
				ax[k].plot([mod_version]*2,[0,len(dic_group[key])],'--',color='b')
				space = max([abs(score) for score in val[key].values()])*1.5
				if space==0:
					space = 1
				for t,obs in enumerate(dic_group[key]):
					score = val[key][obs]
					ax[k].annotate('',xy=(mod_version+score/space,t),xytext=(mod_version,t),arrowprops=dict(arrowstyle="->"),annotation_clip=False)
		for k,key in enumerate(dic_group.keys()):
			ax[k].set_yticks(range(len(dic_group[key])))
			ax[k].set_yticklabels(dic_group[key],fontsize=fontsize)
		#display the figure
		plt.show()

	#display the rankings of all datasets w.r.t. every observable on the same figure
	def display_rankings(self,renamed_obs={},rename_to_color={},renamed_data={}):
		fontsize = 14
		list_obs = list(OBS_TO_TYPE)
		obs_labels = [renamed_obs[obs] for obs in list_obs]
		fig,ax = plt.subplots(1,1,constrained_layout=True)
		ax.set_xlim(0,self.nb_obs)
		ax.set_ylim(-1,self.nb_data)
		ax.set_axis_off()
		for x in range(1,self.nb_obs):
			ax.plot([x,x],[-1,self.nb_data],color='k')
		ax.plot([0,self.nb_obs],[self.nb_data-1,self.nb_data-1],color='k')
		for i,obs in enumerate(obs_labels):
			x = i+0.5; y = self.nb_data-0.5
			ax.annotate(obs,(x,y),fontsize=fontsize,ha='center')
		#display the rankings
		for i,obs in enumerate(list_obs):
			x_pos = i+0.5; t = self.obs_to_int[obs]
			#we will write in the same case (same y) the numbers of models with same rank
			list_chunks = []
			k = 0; rank = 1; chunk = []
			while k<np.size(self.ranking,1):
				i,r = self.ranking[t,k,:]
				if r>rank:
					list_chunks.append(chunk)
					rank = r
					chunk = []
				name = self.int_to_name[i]
				chunk.append(renamed_data[name])
				k += 1
			#add the eventual last chunk
			if chunk:
				list_chunks.append(chunk)
			#place the chunks
			for j,chunk in enumerate(list_chunks):
				y_pos = self.nb_data-1.75-j
				text = chunk[0]
				for name in chunk[1:]:
					text += ', '+name
				if text in rename_to_color:
					color = rename_to_color[text]
				else:
					color = 'black'
				ax.annotate(text,(x_pos,y_pos),fontsize=fontsize,ha='center',color=color)
		plt.show()

	#display the global ranking of all datasets
	def display_glob_rank(self,option='mean',renamed_data={},dic_xp={},focused_model=None):
		if option=='mean':
			glob_rank = self.mean_glob_rank
			best_model = self.mean_best_model
		elif option=='sim':
			glob_rank = self.sim_glob_rank
			best_model = self.sim_best_model
		X0 = []; Y0 = []; models = [model_name for _,_,model_name in self.list_models]

		plt.figure()
		y = 1
		for name in set(models).difference({focused_model,best_model}):
			x = glob_rank[self.name_to_int[name]]
			X0.append(x); Y0.append(y)
		plt.scatter(X0,Y0,c='b',label='models')
		for name,color,ytext in zip([focused_model,best_model],['b','b'],[0.1,-0.15]):
			x = glob_rank[self.name_to_int[name]]
			plt.scatter(x,y,c=color)
			plt.annotate(renamed_data[name],(x,y),xytext=(x,1+ytext))
		y = 0

		for (label,color),val in dic_xp.items():
			X0 = []; Y0 = []
			for name in val:
				x = glob_rank[self.name_to_int[name]]
				X0.append(x); Y0.append(y)
			plt.scatter(X0,Y0,c=color,label=label)
		#emphasize the recovering between models and XP data
		#first compute the global rank of the best model (lowest global rank among the models)
		rank_best_model = glob_rank[self.name_to_int[best_model]]
		plt.plot([rank_best_model]*2,[-1,2],'--',c='b')
		#then compute the global rank of the worst XP dataset (highest global rank among the XP data)
		worst_XP = max([self.name_to_int[name.savename] for name in self.list_refs],key=lambda i:glob_rank[i])
		plt.plot([glob_rank[worst_XP]]*2,[-1,2],'--',c='b')
		plt.xlabel('global rank')
		plt.ylim(-1,2)
		plt.yticks([])
		plt.legend()
		plt.show()
