import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(PROJECT_ROOT)
import Librairies.Temp_net as tp
from Librairies.settings import Raw_to_binned,XP_data,Setup_Plot,Vector_distance,Get_versions,Load_instance_param,obs_to_type,type_to_obs,Load_obs
from Librairies.atn import ADM_class

import numpy as np
import math
import networkx as nx
import random as rd
from sklearn import linear_model
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.markers import MarkerStyle
from scipy.stats import kendalltau,kstest
from scipy.optimize import curve_fit

def f(x,alpha,beta,shift):
	return -alpha*x - beta*np.power(10,x) + shift

#estimate the power-laws exponent and exponential cutoff of the distribution data
def Param(X,Y):
	return curve_fit(f,X,Y,p0=[1,0,0],bounds=([0,0,-np.inf],np.inf))[0]

#return the distance btw the distributions of respective parameters param1 and param2
def Param_dist(param1,param2):
	d1 = abs(np.arctan(param1[0])-np.arctan(param2[0]))
	return (1-np.exp(-d1*16/np.pi))/(1-np.exp(-8))

#return the cosine similarity btw the two lists of ETN etn1 and etn2
def Cosim(etn1,etn2):
	norm1 = sum([val**2 for val in etn1.values()])
	norm2 = sum([val**2 for val in etn2.values()])
	s = 0
	for key in set(etn1.keys()).intersection(set(etn2.keys())):
		s += etn1[key]*etn2[key]
	return s/sqrt(norm1*norm2)

#compute the distance btw empirical distributions obs1 and obs2 exploiting the fact they are power-like
def Power_dist(obs1,obs2):
	#first use log-binning to enhance the quality of the raw power-laws
	X1,Y1 = Raw_to_binned(obs1)
	X2,Y2 = Raw_to_binned(obs2)
	#then use regression to evaluate the power-law exponent as well as the exponential cutoff
	param1 = Param(X1,Y1)
	param2 = Param(X2,Y2)
	#finally compute the normalized distance
	return Param_dist(param1,param2)

#draw the fitness accross generations of the model version_nb adapting to the dataset name
def Fitness(name,version_nb):
	path = 'analysis/ADM_class_V'+str(version_nb)+'/'+name+'/'
	best_fit = np.loadtxt(path+'best_fit.txt')
	avg_fit = np.loadtxt(path+'avg_fit.txt')
	fontsize = 16
	fig,ax = plt.subplots(1,1,constrained_layout=True)
	#ax.set_title('fitness accross generations\n'+'reference : '+name+', model : '+str(version_nb))
	ax.plot(best_fit,'.',label='best')
	ax.plot(avg_fit,'--',label='avg')
	ax.legend(fontsize=fontsize)
	ax.set_xlabel('generation',fontsize=fontsize)
	ax.set_ylabel('product of ETN3 distances',fontsize=fontsize)
	for label in ax.get_xticklabels()+ax.get_yticklabels():
		label.set_fontsize(fontsize)
	#store the results
	path = 'figures/fitness/ADM_class_V'+str(version_nb)+'/'+name+'_fitness.png'
	plt.savefig(path)
	plt.close()

#compare the reference observables between the model and the reference dataset
def Compare_mod_ref(name,version_nb):
	model_name = 'ADM_class_V'+str(version_nb)
	model_path = 'results/Analyze/'+model_name
	xp_path = 'results/Analyze/'+name

	#observables of reference
	dic_obs = {'point':[],'distribution':[],'vector':[]}
	dic_obs['vector'] = ['ETN3']
	dic_obs['point'] = ['clustering_coeff','deg_assortativity']
	dic_obs['distribution'].append('edge_weight')
	dic_obs['distribution'].append('static_degree')
	dic_obs['distribution'].append('edge_activity')
	dic_obs['distribution'].append('node_activity')
	dic_obs['distribution'].append('edge_interactivity')
	dic_obs['distribution'].append('node_interactivity')
	dic_obs['distribution'].append('edge_newborn_activity')
	dic_obs['distribution'].append('edge_events_activity')
	dic_obs['distribution'].append('cc_size')
	dic_obs['distribution'] += ['ETN2_weight','ETN3_weight']

	#distance[name_obs] = distance between the xp dataset and the model instance
	#with respect to the observable name_obs
	distance = {}
	for type_obs,list_obs in dic_obs.items():
		for name_obs in list_obs:
			obs_model = Load_obs[type_obs](model_path,name_obs)
			obs_xp = Load_obs[type_obs](xp_path,name_obs)
			distance[name_obs] = Distance_obs[type_obs](obs_model,obs_xp)

	#save the results
	path = 'results/Compare/'+model_name+'/'+name+'/'
	tab = np.array(list(zip(*dic.items())),dtype=str)
	np.savetxt(path+'distance.txt',tab,fmt='%s')

#draw the fitness for all versions
def Draw_fitness():
	for version_nb in [1,3,9,12,19]:
		print(str(version_nb)+' begins')
		for name in XP_data:
			Fitness(name,version_nb)

#for each distribution observable, visualize its distribution for the model version_nb
#and compare it with all its XP references
def Compare_model_to_all(version_nb,option='tuned'):
	if option=='random':
		endpath = '/random'
	else:
		endpath = ''
	list_color = ['red','blue','green','purple','black']
	for name_obs in type_to_obs['distribution']:
		fig,ax = plt.subplots(1,1,constrained_layout=True)
		ax.set_title('comparison of the ADM version '+str(version_nb)+'\nagainst its XP references ; observable : '+name_obs)
		ax.set_xlabel(r'$\log_{10}(n)$')
		ax.set_ylabel(r'$\log_{10}(P)$')
		list_patch = []
		for color,name in zip(list_color,XP_data):
			#load the XP data and plot it
			dic = Load_distribution('analysis/'+name,name_obs)
			ax.plot(*Raw_to_binned(dic),'v',color=color)
			#load the model data and plot it
			dic = Load_distribution('analysis/ADM_class_V'+str(version_nb)+'/'+name+endpath,name_obs)
			ax.plot(*Raw_to_binned(dic),'.',color=color)
			list_patch.append(mpatches.Patch(color=color,label=name))
		list_patch.append(mlines.Line2D([],[],color='orange',marker='v',markersize=15,label='empirical reference'))
		list_patch.append(mlines.Line2D([],[],color='orange',marker='.',markersize=15,label='best model instance'))
		ax.legend(handles=list_patch)
		plt.savefig('figures/obs_distr/ADM_class_V'+str(version_nb)+'/'+option+'/'+name_obs+'.png')
		plt.close()

#res[n] = ETN3 autosim at level n
def Get_autosim(path):
	dic_ETN = Load_vector(path,'ETN3',agg_max=10)
	res = {1:1}
	for n in range(2,11):
		res[n] = Cosim(dic_ETN[n],dic_ETN[1])
	return res

class Proximity_tensor:
	"""
	docstring for Proximity_tensor
	"""
	def __init__(self,n2=19,instance_path='tuned'):
		#decide wether the instances have been tuned or are random
		self.instance_path = instance_path+'/'
		#model with the lowest global rank
		self.mean_best_model = ''
		self.sim_best_model = ''
		self.mean_worst_model = ''
		self.sim_worst_model = ''
		#name_to_int[name] = integer identifier associated to dataset name
		self.name_to_int = {name:i for i,name in enumerate(XP_data)}
		for version_nb in range(1,n2+1):
			self.name_to_int['ADM_class_V'+str(version_nb)] = len(XP_data)+version_nb-1
		self.int_to_name = {i:name for name,i in self.name_to_int.items()}
		#obs_to_int[name_obs] = integer identifier associated to observable name_obs
		self.obs_to_int = {name_obs:i for i,name_obs in enumerate(obs_to_type.keys())}
		self.int_to_obs = {i:name for name,i in self.obs_to_int.items()}
		self.nb_obs = len(self.int_to_obs)
		self.nb_data = len(self.name_to_int)
		self.nb_XP = len(XP_data)
		self.nb_mod = n2
		#dic_tensor[name_obs][name1][name2] = distance from name1 to name2 wrt the observable name_obs
		self.tensor = np.zeros((self.nb_obs,self.nb_data,self.nb_data))
		#score[name_obs][name] = score of the dataset name wrt the observable name_obs
		self.score = np.zeros((self.nb_obs,self.nb_data))
		#stat_XP[name_obs] = [Q1,Q2,Q3] for the set of empirical datasets
		self.stat_XP = np.zeros((self.nb_obs,3))
		#ranking[name_obs][k] = [i_k,r_k], where r_k is the rank of the dataset i_k
		#ranking[name_obs] is a 2D numpy array
		self.ranking = np.zeros((self.nb_obs,self.nb_data,2),dtype=int)
		#ranking Kendall similarity matrix btw observables
		self.sim_obs = np.eye(self.nb_obs)
		#obs_to_xlabel[obs] = xlabel associated to the observable obs when its distribution is plotted
		self.obs_to_xlabel = {}
		for obs in type_to_obs['distribution']:
			if 'events' in obs:
				self.obs_to_xlabel[obs] = r"$\log_{10}(n)$"
			elif 'activity' in obs:
				self.obs_to_xlabel[obs] = r"$\log_{10}(\Delta t)$"
			elif 'weight' in obs:
				self.obs_to_xlabel[obs] = r"$\log_{10}(w)$"
			elif 'degree' in obs:
				self.obs_to_xlabel[obs] = r"$\log_{10}(k)$"
			else:
				self.obs_to_xlabel[obs] = r"$\log_{10}(n)$"
		#glob_rank[i] = rank of dataset i averaged over all observables
		self.mean_glob_rank = np.zeros(self.nb_data)
		self.sim_glob_rank = np.zeros(self.nb_data)

	#load the proximity tensors
	def Load_tensor(self):
		path_XP = 'analysis/distance_tensor/'
		path_model = path_XP+self.instance_path
		for n,name_obs in self.int_to_obs.items():
			tab = np.loadtxt(path_XP+name_obs+'XP_XP.txt',dtype=str)
			tab2 = np.loadtxt(path_model+name_obs+'XP_model.txt',dtype=str)
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
	def Stat_XP(self):
		for t in range(self.nb_obs):
			list_val = []
			for i in range(self.nb_XP-1):
				ind1 = self.name_to_int[XP_data[i]]
				for j in range(i+1,self.nb_XP):
					ind2 = self.name_to_int[XP_data[j]]
					list_val.append(self.tensor[t,ind1,ind2])
			self.stat_XP[t,:] = np.quantile(list_val,[0.25,0.5,0.75])
			if self.stat_XP[t,2]==self.stat_XP[t,0]:
				self.stat_XP[t,2] = self.stat_XP[t,0]+1

	#compute the score of the model version_nb wrt the observable name_obs
	def Score_model(self,name_obs,version_nb):
		i_mod = self.name_to_int['ADM_class_V'+str(version_nb)]
		i_obs = self.obs_to_int[name_obs]
		list_val = []
		for name in XP_data:
			list_val.append(self.tensor[i_obs,self.name_to_int[name],i_mod])
		model_med = np.quantile(list_val,0.5)
		self.score[i_obs,i_mod] = (self.stat_XP[i_obs,1]-model_med)/(self.stat_XP[i_obs,2]-self.stat_XP[i_obs,0])
	
	#compute the score of the XP dataset name wrt the observable name_obs
	def Score_XP(self,name_obs,name):
		i_mod = self.name_to_int[name]
		i_obs = self.obs_to_int[name_obs]
		list_val = []
		for name2 in XP_data:
			ind2 = self.name_to_int[name2]
			if ind2!=i_mod:
				list_val.append(self.tensor[i_obs,i_mod,ind2])
		med = np.quantile(list_val,0.5)
		self.score[i_obs,i_mod] = (self.stat_XP[i_obs,1]-med)/(self.stat_XP[i_obs,2]-self.stat_XP[i_obs,0])

	#compute the scores of every dataset wrt every observable
	def Compute_scores(self):
		self.Stat_XP()
		for obs in self.obs_to_int.keys():
			for name in XP_data:
				self.Score_XP(obs,name)
			for version_nb in range(1,self.nb_mod+1):
				self.Score_model(obs,version_nb)

	#compute one ranking per observable
	#ranking[name_obs][k] = [i_k,r_k], where r_k is the rank of the dataset i_k
	#ranking[name_obs] is a 2D numpy array
	#then compute the global ranking :
	#glob_rank[i] = rank of dataset i averaged over all observables
	#finally determine the best model, defined as the model with the lowest global rank
	def Get_rankings(self):
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
	def Save_rank_score(self):
		path = 'analysis/rank_score/'+self.instance_path
		if self.nb_mod==19:
			np.savetxt(path+'score.txt',self.score)
		for name,tab in zip(['mean_glob_rank','sim_glob_rank'],[self.mean_glob_rank,self.sim_glob_rank]):
			np.savetxt(path+name+str(self.nb_mod)+'.txt',tab)
		for t in range(self.nb_obs):
			np.savetxt(path+'ranking'+str(t)+'_'+str(self.nb_mod)+'.txt',self.ranking[t,:,:],fmt='%d')

	#load self.score, self.ranking, self.mean_glob_rank and self.sim_glob_rank
	#then deduce self.mean_best_model and self.sim_best_model
	def Load_rank_score(self):
		path = 'analysis/rank_score/'+self.instance_path
		if self.nb_mod<19:
			self.score = np.loadtxt(path+'score.txt')[:,:self.nb_data]
		else:
			self.score = np.loadtxt(path+'score.txt')
		self.sim_glob_rank = np.loadtxt(path+'sim_glob_rank'+str(self.nb_mod)+'.txt')
		self.mean_glob_rank = np.loadtxt(path+'mean_glob_rank'+str(self.nb_mod)+'.txt')
		for t in range(self.nb_obs):
			self.ranking[t,:,:] = np.loadtxt(path+'ranking'+str(t)+'_'+str(self.nb_mod)+'.txt',dtype=int)[:,:]
		models = ['ADM_class_V'+str(i) for i in range(1,self.nb_mod+1)]
		self.mean_best_model = min(models,key=lambda name:self.mean_glob_rank[self.name_to_int[name]])
		self.sim_best_model = min(models,key=lambda name:self.sim_glob_rank[self.name_to_int[name]])
		self.mean_worst_model = max(models,key=lambda name:self.mean_glob_rank[self.name_to_int[name]])
		self.sim_worst_model = max(models,key=lambda name:self.sim_glob_rank[self.name_to_int[name]])

	def Load_data_reg_fitness_score(self,n2,option='mean',list_obs='all'):
		if option=='mean':
			func = np.mean
		elif option=='median':
			func = np.median
		avg_fitness = np.zeros(n2) #n_samples = n2
		for i in range(1,n2+1):
			model = 'ADM_class_V'+str(i); tab = []
			for name in XP_data:
				best_fit = np.loadtxt('analysis/'+model+'/'+name+'/best_fit.txt')[-1]
				tab.append(best_fit)
			avg_fitness[i-1] = func(tab)
		#n_features = n_obs
		if list_obs=='all':
			newt_to_t = range(self.nb_obs)
			X = np.zeros((n2,self.nb_obs))
			for i in range(1,n2+1):
				i_mod = self.name_to_int['ADM_class_V'+str(i)]
				X[i-1,:] = self.score[:,i_mod]
		else:
			X = np.zeros((n2,len(list_obs))); new_t = 0; newt_to_t = []
			for obs in list_obs:
				t = self.obs_to_int[obs]; newt_to_t.append(t)
				for i in range(1,n2+1):
					i_mod = self.name_to_int['ADM_class_V'+str(i)]
					X[i-1,new_t] = self.score[t,i_mod]
				new_t += 1
		return avg_fitness,X,newt_to_t

	#multiple linear regression of the model fitness
	#with respect to the scores relative to all observables
	#sort the observables by decreasing absolute value of the slope and display the results
	def Reg_fitness_score(self,n2,option='mean',list_obs='all'):
		avg_fitness,X,newt_to_t = self.Load_data_reg_fitness_score(n2,option=option,list_obs=list_obs)
		if list_obs=='all':
			fig_title = '_all_'
		else:
			fig_title = '_not_all_'
		regr = linear_model.LinearRegression()
		regr.fit(X,avg_fitness)
		reg_quality = regr.score(X,avg_fitness)
		slopes = regr.coef_
		#plot results
		fontsize = 12
		fig,ax = plt.subplots(1,1,constrained_layout=True)
		title = 'observables ranked by decreasing absolute value\n'
		title += 'of the slope obtained by multiple linear regression\n'
		title += 'of the models fitness (regression score: '+str(reg_quality)+')'
		ax.set_title(title,fontsize=fontsize)
		ax.set_xlabel('observable rank',fontsize=fontsize)
		ax.set_ylabel('observable slope',fontsize=fontsize)
		ax.set_xticks(range(np.size(X,1)))
		sorted_newt = sorted(range(len(newt_to_t)),key=lambda new_t: abs(slopes[new_t]),reverse=True)
		xlabels = []
		for new_t in sorted_newt:
			xlabels.append(self.int_to_obs[newt_to_t[new_t]])
		ax.set_xticklabels(xlabels,rotation=90,fontsize=fontsize)
		Y = [slopes[new_t] for new_t in sorted_newt]
		ax.plot(Y,'.'); ax.plot([0,len(sorted_newt)-1],[0]*2,'--')
		plt.savefig('figures/fitness_score/'+option+fig_title+'obs_slopes.png')

	def Get_obs_slopes(self,n2,option='mean',list_obs='all'):
		avg_fitness,X,newt_to_t = self.Load_data_reg_fitness_score(n2,option=option,list_obs=list_obs)
		regr = linear_model.LinearRegression()
		regr.fit(X,avg_fitness)
		slopes = regr.coef_
		intercept = regr.intercept_
		res = {}
		for t,slope in zip(newt_to_t,slopes):
			obs = self.int_to_obs[t]
			res[obs] = slope
		return res,intercept
	
	def Get_reg_score(self,n2,slopes,intercept,option='mean',list_obs='all'):
		avg_fitness,X,newt_to_t = self.Load_data_reg_fitness_score(n2,option=option,list_obs=list_obs)
		regr = linear_model.LinearRegression()
		regr.intercept_ = intercept
		regr.coef_ = np.zeros(len(slopes))
		for new_t,t in enumerate(newt_to_t):
			regr.coef_[new_t] = slopes[self.int_to_obs[t]]
		return regr.score(X,avg_fitness)

	#use the first nb_mod versions to compute the Pearson correlation coefficient
	#btw fitness and observable score (use aggregator to assign a fitness to each version)
	def Pearson_fitness_score(self,nb_mod,aggregator='mean'):
		if aggregator=='mean':
			func = np.mean
		elif aggregator=='median':
			func = np.median
		truc = 'ADM_class_V'
		#assign a fitness to each version
		version_to_fitness = [0]*nb_mod
		for i in range(1,nb_mod+1):
			model = truc+str(i)
			tab = []
			for name in XP_data:
				best_fit = np.loadtxt('analysis/'+model+'/'+name+'/best_fit.txt')[-1]
				tab.append(best_fit)
			version_to_fitness[i-1] = func(tab)
		fontsize = 14
		fig,ax = plt.subplots(1,1,constrained_layout=True)
		list_obs = list(self.obs_to_int.keys())
		ax.set_xticks(range(self.nb_obs))
		ax.set_xticklabels(list_obs,fontsize=fontsize,rotation=90)
		ax.set_ylabel('Pearson correlation',fontsize=fontsize)
		X = []; Y = []
		for i,obs in enumerate(list_obs):
			t = self.obs_to_int[obs]
			tab = [0]*nb_mod
			for j in range(1,nb_mod+1):
				model = truc+str(j); num = self.name_to_int[model]
				tab[j-1] = self.score[t,num]
			X.append(i)
			Y.append(np.corrcoef(tab,version_to_fitness)[0,1])
		ax.scatter(X,Y)
		ax.plot([0,self.nb_obs-1],[0,0],'--')
		for label in ax.get_yticklabels():
			label.set_fontsize(fontsize)
		plt.savefig('figures/fitness_score/correlation_'+aggregator+str(nb_mod)+'.png')

	#compute the ranking Kendall similarity matrix btw observables
	#convert this matrix into a weighted network and save it in gephi format
	#only versions 1 to 13 are taken into account because we are interested in trade-offs within
	#the ADM class :
	#what observables can we independently obtain by moving in hypothesis space ?
	def Sim_obs(self,save=True):
		models_ind = [self.name_to_int['ADM_class_V'+str(i)] for i in range(1,14)]
		for t in range(self.nb_obs-1):
			tab = sorted(models_ind,key=lambda k:self.ranking[t,k,0])
			x = [self.ranking[t,k,1] for k in tab]
			for u in range(t+1,self.nb_obs):
				tab2 = sorted(models_ind,key=lambda k:self.ranking[u,k,0])
				y = [self.ranking[u,k,1] for k in tab2]
				self.sim_obs[t,u] = kendalltau(x,y)[0]
				self.sim_obs[u,t] = self.sim_obs[t,u]
		#build and save the weighted similarity network
		if save:
			network = nx.Graph()
			for t in range(self.nb_obs-1):
				obs1 = self.int_to_obs[t]
				for u in range(t+1,self.nb_obs):
					obs2 = self.int_to_obs[u]
					weight = abs(self.sim_obs[t,u])
					if weight>0:
						network.add_edge(obs1,obs2,weight=weight)
			nx.write_gexf(network,'analysis/sim_obs.gexf')

	#visualize the similarity matrix btw observables and save the figure
	def Visu_sim_obs(self,dic_group):
		grouped_obs = []; list_color = []
		for info,group in dic_group.items():
			grouped_obs += group
			list_color += [info[0]]*len(group)
		new_mat = np.eye(self.nb_obs)
		for i,obs1 in enumerate(grouped_obs):
			i_obs = self.obs_to_int[obs1]
			for j,obs2 in enumerate(grouped_obs):
				j_obs = self.obs_to_int[obs2]
				new_mat[i,j] = self.sim_obs[i_obs,j_obs]
		fig,ax = plt.subplots(1,1,constrained_layout=True)
		fontsize = 12
		labels = grouped_obs
		ax.set_xticks(range(self.nb_obs))
		ax.set_yticks(range(self.nb_obs))
		ax.set_xticklabels(labels,rotation=90,fontsize=fontsize)
		ax.set_yticklabels(labels,fontsize=fontsize)
		for xtick,color,ytick in zip(ax.get_xticklabels(),list_color,ax.get_yticklabels()):
			xtick.set_color(color)
			ytick.set_color(color)
		im = ax.imshow(new_mat,cmap='gnuplot2')
		plt.colorbar(im)
		plt.savefig('figures/simat_obs.png')

	#draw the figure for the aps paper
	#showing the agreement between qualitative appreciation of the distributions
	#and the score computation
	#for each chosen observable, we display the distributions of the worst, average and best datasets,
	#as well as their scores
	def Verif_score(self,chosen_obs,option='mean'):
		if option=='mean':
			best_model = self.mean_best_model
			glob_rank = self.mean_glob_rank
		else:
			best_model = self.sim_best_model
			glob_rank = self.sim_glob_rank
		#we associate one color and one marker to one dataset
		line_dic = {}; list_marker = ['.','x','*','v','s']
		list_color = ['b','k','g','r','orange']
		chosen_XP = ['conf16','utah']
		#we choose the best and worst model according to self.sim_glob_rank, as well as the original ADM
		chosen_model = [best_model,'ADM_class_V14']
		worst_model = max(['ADM_class_V'+str(i) for i in range(1,self.nb_mod+1)],key=lambda name:glob_rank[self.name_to_int[name]])
		chosen_model.append(worst_model)
		model_to_label = {'ADM_class_V'+str(i):'V'+str(i) for i in range(1,self.nb_mod+1)}
		for name in chosen_XP:
			model_to_label[name] = name
		#we choose the realisation of the model closest to the conferences
		dic_anapath = {name:'analysis/'+name for name in chosen_XP}
		for model in chosen_model:
			dic_anapath[model] = 'analysis/'+model+'/conf16'
		tot_chosen = chosen_XP+chosen_model
		for color,marker,name in zip(list_color,list_marker,tot_chosen):
			line_dic[name] = [color,marker]
		fontsize = 16
		for obs in chosen_obs:
			t = self.obs_to_int[obs]
			fig,ax = plt.subplots(1,2,constrained_layout=True,gridspec_kw={'width_ratios': [1,3]})
			#compute and display the scores
			for name,val in line_dic.items():
				score = self.score[t,self.name_to_int[name]]
				ax[0].scatter([0],[score],c=val[0],marker=val[1])
			ax[0].set_title("score",fontsize=fontsize)
			ax[0].tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
			for label in ax[0].get_yticklabels():
				label.set_fontsize(fontsize)
			#display the distributions
			for name,val in line_dic.items():
				data = Load_distribution(dic_anapath[name],obs)
				ax[1].plot(*Raw_to_binned(data),val[1],color=val[0],label=model_to_label[name])
			ax[1].set_title(obs,fontsize=fontsize)
			ax[1].set_xlabel(self.obs_to_xlabel[obs],fontsize=fontsize)
			ax[1].set_ylabel(r"$\log_{10}(P)$",fontsize=fontsize)
			ax[1].legend(fontsize=fontsize)
			for label in ax[1].get_xticklabels()+ax[1].get_yticklabels():
				label.set_fontsize(fontsize)
			fig.savefig('figures/score/verif_score/random_obs/verif_'+obs+'_'+option+'.png')
		plt.close('all')

	def Verif_autosim(self,option='mean'):
		if option=='mean':
			best_model = self.mean_best_model
			glob_rank = self.mean_glob_rank
		else:
			best_model = self.sim_best_model
			glob_rank = self.sim_glob_rank
		#we associate one color and one marker to one dataset
		line_dic = {}; list_marker = ['.','x','*','v','s']
		list_color = ['b','k','g','r','orange']
		chosen_XP = ['conf16','utah']
		#we choose the best and worst model according to self.sim_glob_rank, as well as the original ADM
		chosen_model = [best_model,'ADM_class_V14']
		worst_model = max(['ADM_class_V'+str(i) for i in range(1,self.nb_mod+1)],key=lambda name:glob_rank[self.name_to_int[name]])
		model_to_label = {'ADM_class_V'+str(i):'V'+str(i) for i in range(1,self.nb_mod+1)}
		chosen_model.append(worst_model)
		for name in chosen_XP:
			model_to_label[name] = name
		#we choose the realisation of the model closest to the utah, because the utah is highly autosimilar
		dic_anapath = {name:'analysis/'+name for name in chosen_XP}
		for model in chosen_model:
			dic_anapath[model] = 'analysis/'+model+'/utah'
		tot_chosen = chosen_XP+chosen_model
		for color,marker,name in zip(list_color,list_marker,tot_chosen):
			line_dic[name] = [color,marker]
		fontsize = 16
		#load data
		dic_data = {}
		for name,path in dic_anapath.items():
			dic_data[name] = Get_autosim(path)
		#plot the figure
		fig,ax = plt.subplots(1,2,constrained_layout=True,gridspec_kw={'width_ratios': [1,3]})
		#compute and display the scores
		t = self.obs_to_int['ETN3']
		for name,val in line_dic.items():
			score = self.score[t,self.name_to_int[name]]
			ax[0].scatter([0],[score],c=val[0],marker=val[1])
		ax[0].set_title("score",fontsize=fontsize)
		ax[0].tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
		for label in ax[0].get_yticklabels():
			label.set_fontsize(fontsize)
		#display the autosim
		for name,val in line_dic.items():
			ax[1].plot(*zip(*dic_data[name].items()),val[1],color=val[0],label=model_to_label[name])
		ax[1].set_title('autosimilarity for ETN of depth 3',fontsize=fontsize)
		ax[1].set_xlabel('aggregation level',fontsize=fontsize)
		ax[1].set_ylabel("cosine similarity",fontsize=fontsize)
		ax[1].legend(fontsize=fontsize)
		for label in ax[1].get_xticklabels()+ax[1].get_yticklabels():
			label.set_fontsize(fontsize)
		fig.savefig('figures/score/verif_score/ETN_obs/verif_autosim_'+option+'.png')
		plt.close('all')

	#For each point observable, we plot the score as a function of the observable value.
	#We also indicate as orange vertical lines the values observed in the five empirical references.
	def Verif_point_score(self,list_models,option='median'):
		if option=='median':
			func = np.median
		elif option=='averaged':
			func = np.mean
		fontsize = 16
		#gather data
		for obs in type_to_obs['point']:
			fig,ax = plt.subplots(1,1,constrained_layout=True)
			ax.set_xlabel(option+' '+obs,fontsize=fontsize)
			ax.set_ylabel('model score',fontsize=fontsize)
			#load model data
			X = []; Y = []
			for nb in range(1,self.nb_mod+1):
				model = 'ADM_class_V'+str(nb)
				Y.append(self.score[self.obs_to_int[obs],self.name_to_int[model]])
				X.append(func([Load_point('analysis/'+model+'/'+name,obs) for name in XP_data]))
			ax.plot(X,Y,'.',label='models')
			#make the models in list_models visible
			for i in list_models:
				x = X[i-1]; y = Y[i-1]
				ax.scatter(x,y,c='r')
				ax.annotate('V'+str(i),(x,y),xytext=(x,y))
			#load XP data
			m = np.min(Y); M = np.max(Y)
			for name in XP_data[1:]:
				x = Load_point('analysis/'+name,obs)
				ax.plot([x,x],[m,M],'--',color='orange')
			x = Load_point('analysis/'+XP_data[0],obs)
			ax.plot([x,x],[m,M],'--',color='orange',label='references')
			ax.legend(fontsize=fontsize)
			for label in ax.get_xticklabels()+ax.get_yticklabels():
				label.set_fontsize(fontsize)
			plt.savefig('figures/score/verif_score/point_obs/verif_'+obs+'_'+option+'.png')

	#display the 5 most frequent motifs at aggregation level 1
	#of the 'ref' data set and the best and worst adjacent versions wrt the chosen option
	def Verif_vector_score(self,ref,option='mean',agg_level=1):
		if option=='mean':
			best_model = self.mean_best_model
			worst_model = max(['ADM_class_V'+str(i) for i in range(1,self.nb_mod+1)],key=lambda name:self.mean_glob_rank[self.name_to_int[name]])
		elif option=='sim':
			best_model = self.sim_best_model
			worst_model = max(['ADM_class_V'+str(i) for i in range(1,self.nb_mod+1)],key=lambda name:self.sim_glob_rank[self.name_to_int[name]])
		#load data
		dic = {ref:[],best_model:[],worst_model:[]}
		vector = Load_vector('analysis/'+ref,'ETN3',agg_max=agg_level)[agg_level]
		dic[ref] = sorted(vector.keys(),key=lambda seq:vector[seq],reverse=True)[:5]
		for model in [worst_model,best_model]:
			vector = Load_vector('analysis/'+model+'/'+ref,'ETN3',agg_max=agg_level)[agg_level]
			dic[model] = sorted(vector.keys(),key=lambda seq:vector[seq],reverse=True)[:5]
		#plot motifs
		fontsize = 18
		for name,val in dic.items():
			fig,ax = plt.subplots(1,5,figsize=(12,3),constrained_layout=True)
			for k in range(5):
				ax[k].set_axis_off()
				ax[k].set_title(Ordinal(k),fontsize=fontsize)
				Plot_motif(val[k],3,ax[k])
			if name==ref:
				name_fig = 'ref'
			elif name==best_model:
				name_fig = 'best'
			else:
				name_fig = 'worst'
			plt.savefig('figures/score/verif_score/ETN_obs/verif_ETN3_'+name_fig+'_'+option+'_level'+str(agg_level)+'.png')

	def Get_renamed_obs(self):
		min_to_maj = {'i':'I','w':'W','a':'A','s':'S','n':'NB','c':'C','e':'E'}
		renamed_obs = {}
		for obs in obs_to_type.keys():
			ind_ = [i for i in range(len(obs)) if obs[i]=='_']
			if ind_:
				newname = obs[:ind_[0]]
				for i in range(len(ind_)):
					newname += ' '+min_to_maj[obs[ind_[i]+1]]+'.'
			else:
				newname = obs
			renamed_obs[obs] = newname
		renamed_obs['edge_events_activity'] = 'nb of events'
		return renamed_obs

	#for each ADM version, compute the difference btw its score and the score of the basis version
	#ADM_class_V1 for each observable
	#then sum up these variations for each group of observables
	def Score_variation(self,dic_group,n2):
		fontsize = 16; markersize = 12
		variations = {}; i_basis = self.name_to_int['ADM_class_V1']
		for i in range(1,n2+1):
			variations[i] = {key:{} for key in dic_group.keys()}
			name = 'ADM_class_V'+str(i)
			for key,group in dic_group.items():
				for obs in group:
					t = self.obs_to_int[obs]
					val = self.score[t,self.name_to_int[name]]-self.score[t,i_basis]
					variations[i][key][obs] = val
		#display the total score variations for the three groups
		fig,ax = plt.subplots(1,1,constrained_layout=True)
		X = []; dic_Y = {key:[] for key in dic_group.keys()}
		for i,val in variations.items():
			X.append(i)
			for key,value in val.items():
				dic_Y[key].append(sum(list(value.values())))
		group_to_label = {}
		for num in ['I','II','III']:
			group_to_label['group '+num] = r"$\Delta s^{"+num+r"}$"
		for key,Y in dic_Y.items():
			ax.plot(X,Y,'.',color=key[0],label=group_to_label[key[1]],markersize=markersize)
		ax.plot([1,n2],[0,0],'--')
		ax.set_xlabel('version number',fontsize=fontsize)
		ax.set_ylabel(r"$\Delta s$",fontsize=fontsize)
		ax.legend(fontsize=fontsize)
		ax.set_xticks(range(1,n2+1))
		for label in ax.get_xticklabels()+ax.get_yticklabels():
			label.set_fontsize(fontsize)
		plt.savefig('figures/score/variation_per_group_'+str(n2)+'.png')
		fontsize = 14
		#for each group, display the contribution of each observable
		renamed_obs = self.Get_renamed_obs()
		fig,ax = plt.subplots(len(dic_group),1,constrained_layout=True)
		for i,key in enumerate(dic_group.keys()):
			ax[i].set_title(key[1],fontsize=fontsize)
		for k in range(len(ax)-1):
			ax[k].tick_params(axis='x',which='both',bottom=False,labelbottom=False)
		ax[-1].set_xlabel('version number',fontsize=fontsize)
		ax[-1].set_xticks(range(1,n2+1))
		for label in ax[-1].get_xticklabels():
			label.set_fontsize(fontsize)
		for i,val in variations.items():
			for k,key in enumerate(dic_group.keys()):
				ax[k].plot([i]*2,[0,len(dic_group[key])],'--',color='b')
				space = max([abs(score) for score in val[key].values()])*1.5
				if space==0:
					space = 1
				for t,obs in enumerate(dic_group[key]):
					score = val[key][obs]
					ax[k].annotate('',xy=(i+score/space,t),xytext=(i,t),arrowprops=dict(arrowstyle="->"),annotation_clip=False)
		for k,key in enumerate(dic_group.keys()):
			ax[k].set_yticks(range(len(dic_group[key])))
			labels = [renamed_obs[obs] for obs in dic_group[key]]
			ax[k].set_yticklabels(labels,fontsize=fontsize)
		#save the figure
		plt.savefig('figures/score/variation_per_obs_'+str(n2)+'.png')
		plt.close()

	#check whether composite versions inherit from the score of their adjacent components combined
	def Visu_compo_prop(self,dic_group):
		truc = 'ADM_class_V'; markersize = 12; fontsize = 16; fontsize2 = 14
		variations = {}; i_basis = self.name_to_int['ADM_class_V1']
		for i in range(1,self.nb_mod+1):
			variations[i] = {key:0 for key in dic_group.keys()}
			name = truc+str(i)
			for key,group in dic_group.items():
				for obs in group:
					t = self.obs_to_int[obs]
					val = self.score[t,self.name_to_int[name]]-self.score[t,i_basis]
					variations[i][key] += val
				variations[i][key] /= len(group)
		#composition table : dic_compo[n] = set of adjacent versions composing the version n
		dic_compo = {}
		for version_nb,val in zip(range(14,self.nb_mod+1),[{2,3,4,5,8,13},{2,5,8,13},{5,8,11,13},{3,5,8,9,12,13},{2,3,5,8,12,13},{7,9,13}]):
			dic_compo[version_nb] = val
		#predic_score[group][n] = predicted score of the composite version n for group
		predic_score = {key:{} for key in dic_group.keys()}
		for group in predic_score.keys():
			for n in range(14,self.nb_mod+1):
				predic_score[group][n] = np.mean([variations[i][group] for i in dic_compo[n]])
		#plot the predicted score vs the observed score for each group
		for group in dic_group.keys():
			fig,ax = plt.subplots(1,1,constrained_layout=True)
			ax.set_xlabel('observed score variation',fontsize=fontsize)
			ax.set_ylabel('predicted variation',fontsize=fontsize)
			#ax.set_title(group[1]+' score prediction from adjacent components')
			X = []; Y = []
			for n in range(14,self.nb_mod):
				X.append(variations[n][group])
				Y.append(predic_score[group][n])
			ax.plot(X,Y,'.',markersize=markersize)
			#highlight the version 19
			ax.scatter(variations[19][group],predic_score[group][19],s=markersize*3,color='r')
			for label in ax.get_xticklabels()+ax.get_yticklabels():
				label.set_fontsize(fontsize)
			#compare the predicted sign with the real sign
			#These are in unitless percentages of the figure size. (0,0 is bottom left)
			left, bottom, width, height = [0.35, 0.6, 0.3, 0.3]
			ax2 = fig.add_axes([left, bottom, width, height])
			ax2.set_xlabel('sign accordance',fontsize=fontsize2)
			ax2.set_ylabel('nb of composite\nversions',fontsize=fontsize2)
			#ax.set_title(group[1]+' score sign accordance btw\ncomposite version and adjacent components')
			tab = []
			for n in range(14,self.nb_mod+1):
				if np.sign(variations[n][group])==np.sign(predic_score[group][n]):
					tab.append('match')
				else:
					tab.append('mismatch')
			ax2.hist(tab,histtype='bar')
			for label in ax2.get_xticklabels()+ax2.get_yticklabels():
				label.set_fontsize(fontsize2)
			plt.savefig('figures/score/compo_predic_score_'+group[1].replace(' ','_')+'.png')

	#display the global ranking of all datasets
	def Glob_rank_slide(self,option='mean'):
		if option=='mean':
			glob_rank = self.mean_glob_rank
			best_model = self.mean_best_model
		else:
			glob_rank = self.sim_glob_rank
			best_model = self.sim_best_model
		renamed = {'ADM_class_V'+str(i):'V'+str(i) for i in range(1,self.nb_mod+1)}
		X0 = []; Y0 = []; models = ['ADM_class_V'+str(i) for i in range(1,self.nb_mod+1)]
		plt.figure()
		y = 1
		for name in set(models).difference({'ADM_class_V14',best_model}):
			x = glob_rank[self.name_to_int[name]]
			X0.append(x); Y0.append(y)
		plt.scatter(X0,Y0,c='b',label='models')
		for name,color,ytext in zip(['ADM_class_V14',best_model],['b','b'],[0.1,-0.15]):
			x = glob_rank[self.name_to_int[name]]
			plt.scatter(x,y,c=color)
			plt.annotate(renamed[name],(x,y),xytext=(x,1+ytext))
		y = 0
		#separate the conferences and the schools from the workplace
		dic_xp = {'g':[],'k':[],'yellow':[]}
		for name in XP_data:
			if 'conf' in name:
				dic_xp['g'].append(name)
			elif 'work' in name:
				dic_xp['yellow'].append(name)
			else:
				dic_xp['k'].append(name)
		for label,color in zip(['conferences','schools','workplace'],['g','k','yellow']):
			X0 = []; Y0 = []
			for name in dic_xp[color]:
				x = glob_rank[self.name_to_int[name]]
				X0.append(x); Y0.append(y)
			plt.scatter(X0,Y0,c=color,label=label)
		#emphasize the recovering between models and XP data
		#first compute the global rank of the best model (lowest global rank among the models)
		rank_best_model = glob_rank[self.name_to_int[best_model]]
		plt.plot([rank_best_model,rank_best_model],[-1,2],'--',c='b')
		#then compute the global rank of the worst XP dataset (highest global rank among the XP data)
		worst_XP = max([self.name_to_int[name] for name in XP_data],key=lambda i:glob_rank[i])
		plt.plot([glob_rank[worst_XP],glob_rank[worst_XP]],[-1,2],'--',c='b')
		plt.xlabel('global rank')
		plt.ylim(-1,2)
		plt.yticks([])
		plt.legend()
		plt.savefig('figures/score/'+option+'_global_ranking_slide.png')
		plt.close()

	#display the global ranking of all datasets
	def Glob_rank(self,version=None,option='mean',other=True):
		fontsize = 16; markersize = 60
		if option=='mean':
			glob_rank = self.mean_glob_rank
			best_model = self.mean_best_model
			other_model = self.sim_best_model
			other_ytext = 1+0.03; dx = 0.5
			final_label = 'inverse global score resulting from averaged rank'
		else:
			glob_rank = self.sim_glob_rank
			best_model = self.sim_best_model
			other_model = self.mean_best_model
			other_ytext = 1.05; dx = 0.1
			final_label = 'inverse global score resulting from averaged score'
		X0 = []; Y0 = []; models = ['ADM_class_V'+str(i) for i in range(1,self.nb_mod+1)]
		affich = {models[i-1]:'V'+str(i) for i in range(1,self.nb_mod+1)}
		fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(6,3))
		y = 1
		for name in set(models).difference({'ADM_class_V14',best_model,other_model}):
			x = glob_rank[self.name_to_int[name]]
			X0.append(x); Y0.append(y)
		ax.scatter(X0,Y0,c='b',label='models',s=markersize)
		if version is not None:
			if type(version)==int:
				name = 'ADM_class_V'+str(version)
				x = glob_rank[self.name_to_int[name]]
				ax.scatter(x,y,c='r',s=markersize)
				ax.annotate(affich[name],(x,y),xytext=(x,y+0.1),fontsize=fontsize)
			else:
				version.sort(key=lambda nb:glob_rank[self.name_to_int['ADM_class_V'+str(nb)]])
				for i,nb in enumerate(version):
					name = 'ADM_class_V'+str(nb)
					x = glob_rank[self.name_to_int[name]]
					ax.scatter(x,y,c='r',s=markersize)
					if i%2==0:
						ytext = y+0.05
					else:
						ytext = y-0.15
					ax.annotate(affich[name],(x,y),xytext=(x,ytext),fontsize=fontsize)
		for name,color,ytext in zip(['ADM_class_V14',best_model],['r','r'],[0.05,-0.15]):
			x = glob_rank[self.name_to_int[name]]
			ax.scatter(x,y,c=color,s=markersize)
			ax.annotate(affich[name],(x,y),xytext=(x,y+ytext),fontsize=fontsize)
		#if we display the best model according to the other strategy
		if other:
			x = glob_rank[self.name_to_int[other_model]]
			#check whether best_model and other_models have the same global rank
			if x==glob_rank[self.name_to_int[best_model]]:
				marker = MarkerStyle('o',fillstyle='left')
				other_xtext = x - dx
			else:
				marker = 'o'
				other_xtext = x
			ax.scatter(x,y,c='y',marker=marker,s=markersize)
			ax.annotate(affich[other_model],(x,y),xytext=(other_xtext,other_ytext),fontsize=fontsize)
		y = 0
		#separate the conferences from the schools and workplace
		dic_xp = {'g':[],'k':[],'purple':[]}
		for name in XP_data:
			if 'conf' in name:
				dic_xp['g'].append(name)
			elif 'work' in name:
				dic_xp['purple'].append(name)
			else:
				dic_xp['k'].append(name)
		for label,color in zip(['conferences','schools','workplace'],['g','k','purple']):
			X0 = []; Y0 = []
			for name in dic_xp[color]:
				x = glob_rank[self.name_to_int[name]]
				X0.append(x); Y0.append(y)
			ax.scatter(X0,Y0,c=color,label=label,s=markersize)
		#emphasize the recovering between models and XP data
		#first compute the global rank of the best model (lowest global rank among the models)
		rank_best_model = glob_rank[self.name_to_int[best_model]]
		ax.plot([rank_best_model,rank_best_model],[-1,2],'--',c='b')
		#then compute the global rank of the worst XP dataset (highest global rank among the XP data)
		worst_XP = max([self.name_to_int[name] for name in XP_data],key=lambda i:glob_rank[i])
		ax.plot([glob_rank[worst_XP],glob_rank[worst_XP]],[-1,2],'--',c='b')
		ax.set_xlabel(final_label,fontsize=fontsize)
		for label in ax.get_xticklabels():
			label.set_fontsize(fontsize)
		ax.set_ylim(-0.2,1.2)
		ax.set_yticks([])
		ax.legend(fontsize=fontsize)
		plt.savefig('figures/score/'+option+str(self.nb_mod)+'_global_ranking.png')

	def Display_top_stab(self):
		fontsize = 16
		X = []; Y = []
		num_obs = []
		for obs in type_to_obs['vector']+type_to_obs['distribution']:
			num_obs.append(self.obs_to_int[obs])
		for u in range(1,self.nb_data+5):
			#dic_Q[i] = nb of observables st the rank of ADM version i is <= u
			dic_Q = {i:1 for i in range(1,20)}
			for t in num_obs:
				for k in range(self.nb_data):
					i_mod,rank = self.ranking[t,k,:]
					model = self.int_to_name[i_mod]
					if 'V' in model:
						i = int(model[model.find('V')+1:])
						dic_Q[i] *= int(rank<=u)
			X.append(u)
			Y.append(sum(list(dic_Q.values())))
		#display the results
		fig,ax = plt.subplots(1,1,constrained_layout=True)
		ax.set_xlabel('u',fontsize=fontsize)
		ax.set_ylabel('Q',fontsize=fontsize)
		ax.set_yticks(range(0,20,2))
		X = np.asarray(X); Y = np.asarray(Y)
		ax.plot(X,Y,'.')
		#add the upper bound for Q(u)
		ind = 0
		while Y[ind]==0:
			ind += 1
		u = X[ind]; q = Y[ind]
		u_lim = u + (self.nb_mod-q)/(self.nb_obs-2)
		X1 = np.asarray([u,u_lim])
		Y1 = (self.nb_obs-2)*(X1-u) + q
		#ax.plot(X1,Y1,'--')
		for label in ax.get_yticklabels()+ax.get_xticklabels():
			label.set_fontsize(fontsize)
		plt.savefig('figures/score/top_stability_ranking.png')

		#among all the models with rank <= u, what is the proportion of recurrent models,
		#i.e. i st Q_{i} = 1

	#display the rankings of all datasets w.r.t. every observable on the same figure
	def Display_rankings(self):
		renamed_XP_data = {name:name for name in XP_data}
		renamed_XP_data['highschool3'] = 'HS3'
		fontsize = 14
		renamed_obs = self.Get_renamed_obs()
		for obs,val in renamed_obs.items():
			renamed_obs[obs] = val.replace('.','')
		renamed_obs['edge_events_activity'] = 'nb of\nevents'
		renamed_obs['edge_newborn_activity'] = 'edge\nNBA'
		renamed_obs['clustering_coeff'] = 'clustering\ncoeff'
		renamed_obs['cc_size'] = 'cc size'
		list_obs = list(obs_to_type.keys())
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
				if 'V' in name:
					version_nb = name[name.find('V')+1:]
					chunk.append(version_nb)
				else:
					chunk.append(renamed_XP_data[name])
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
				if text in renamed_XP_data.values():
					color = 'green'
				elif text in ['1','9']:
					color = 'red'
				elif text=='14':
					color = 'blue'
				else:
					color = 'black'
				ax.annotate(text,(x_pos,y_pos),fontsize=fontsize,ha='center',color=color)
		plt.show()

def Setup(nb_mod):
	#figures
	for path in ['fitness','score','score/verif_score']:
		if not os.path.isdir('figures/'+path):
			os.mkdir('figures/'+path)
	for version_nb in range(1,nb_mod+1):
		model = '/ADM_class_V'+str(version_nb)
		for path in ['fitness','obs_distr']:
			if not os.path.isdir('figures/'+path+model):
				os.mkdir('figures/'+path+model)
	#analysis
	for path in ['distance_tensor','rank_score']:
		if not os.path.isdir('analysis/'+path):
			os.mkdir('analysis/'+path)
	#random
	for version_nb in range(1,nb_mod+1):
		model = '/ADM_class_V'+str(version_nb)
		for name in XP_data:
			path = 'analysis'+model+'/'+name+'/random'
			if not os.path.isdir(path):
				os.mkdir(path)
				for type_obs in type_to_obs.keys():
					os.mkdir(path+'/'+type_obs)
		path = 'figures/obs_distr'+model+'/random_tuned'
		if not os.path.isdir(path):
			os.mkdir(path)

#compute the distance tensor
#for models, we only know the distance model-to-XP (the diagonal block model-to-model is missing)
#res[name_obs][d1][d2] = distance from d1 to d2 relatively to the observable name_obs
def Dic_tensor(n2,block_XP=True,option='tuned'):
	res = {name_obs:{} for name_obs in obs_to_type.keys()}
	path = 'analysis/'
	savepath = 'analysis/distance_tensor/'+option+'/'
	if option=='random':
		endpath = '/random'
	else:
		endpath = ''
	#compute the block XP-XP
	if block_XP:
		print('block XP-XP begins')
		for name_obs,type_obs in obs_to_type.items():
			print(name_obs)
			for name1 in XP_data:
				res[name_obs][name1] = {}
			for i in range(len(XP_data)):
				name1 = XP_data[i]
				obs1 = Load_obs[type_obs](path+name1,name_obs)
				for j in range(i+1,len(XP_data)):
					name2 = XP_data[j]
					obs2 = Load_obs[type_obs](path+name2,name_obs)
					res[name_obs][name1][name2] = Distance_obs[type_obs](obs1,obs2)
					res[name_obs][name2][name1] = res[name_obs][name1][name2]
				res[name_obs][name1][name1] = 0
		#save the block XP_XP
		for name_obs,dic in res.items():
			first_row = ['&']+XP_data
			tab = [first_row]
			for name1 in first_row[1:]:
				row = [name1]
				for name2 in first_row[1:]:
					row.append(str(dic[name1][name2]))
				tab.append(row)
			np.savetxt(savepath+name_obs+'XP_XP.txt',np.asarray(tab,dtype=str),fmt='%s')
	else:
		for name_obs,type_obs in obs_to_type.items():
			for name in XP_data:
				res[name_obs][name] = {}
	#compute the block XP_model
	print('block XP-model begins')
	for name_obs,type_obs in obs_to_type.items():
		print(name_obs)
		for version_nb in range(1,n2+1):
			model_name = 'ADM_class_V'+str(version_nb)
			res[name_obs][model_name] = {}
			for name in XP_data:
				obs1 = Load_obs[type_obs](path+model_name+'/'+name+endpath,name_obs)
				obs2 = Load_obs[type_obs](path+name,name_obs)
				res[name_obs][model_name][name] = Distance_obs[type_obs](obs1,obs2)
				res[name_obs][name][model_name] = res[name_obs][model_name][name]
	#save the block XP_model
	for name_obs,dic in res.items(): 
		first_row = ['&']+['ADM_class_V'+str(version_nb) for version_nb in range(1,n2+1)]
		tab = [first_row]
		for name1 in XP_data:
			row = [name1]
			for name2 in first_row[1:]:
				row.append(str(dic[name1][name2]))
			tab.append(row)
		np.savetxt(savepath+name_obs+'XP_model.txt',np.asarray(tab,dtype=str),fmt='%s')

#print all versions that did not run
def Get_failures():
	failure = []
	for version_nb in range(1,15):
		for name in XP_data:
			if not os.path.exists('analysis/ADM_class_V'+str(version_nb)+'/'+name+'/distribution/cc_size.txt'):
				failure.append((version_nb,name))
	for el in failure:
		print(el)

#compare distributions of models with references
def Compare_models_XP(n1,n2,option='tuned'):
	#compare distributions of models with references
	for version_nb in range(n1,n2+1):
		print(str(version_nb)+' begins')
		Compare_model_to_all(version_nb,option=option)
	print('all models have been compared to their references')

#compute scores, rankings, proximity tensor and observables similarity matrix
def Ana():
	if not os.path.exists('analysis/rank_score/score.txt'):
		#Compare_models_XP()
		#compute and save the proximity tensor
		Dic_tensor()
		print('proximity tensor computed')
		#compute the score for each observable and dataset
		tensor = Proximity_tensor()
		#initialize the tensor
		tensor.Load_tensor()
		print('proximity tensor loaded')
		#compute the three first quartiles of the distances btw XP data
		tensor.Stat_XP()
		print('XP stat properties computed')
		#compute the score of each model and each XP dataset wrt each observable
		for name_obs in obs_to_type.keys():
			for name in XP_data:
				tensor.Score_XP(name_obs,name)
			for version_nb in range(1,15):
				tensor.Score_model(name_obs,version_nb)
		print('scores computed')
		#compute a ranking per observable
		#then deduce a unique global ranking. However, two possibilities :
		# - global rank = rank averaged over observables
		# - global rank = rank according to a global distance, defined as 1-global similarity,
		#   global similarity = product of similarities for each observable
		tensor.Get_rankings()
		print('rankings computed')
		#display the global ranking of all datasets according to their nature (conference, school or model)
		tensor.Glob_rank('mean')
		tensor.Glob_rank('sim')
		#save the scores of all datasets for all observables as well as the two global scores and rankings
		tensor.Save_rank_score()
	else:
		tensor = Proximity_tensor()
		#load rankings and scores
		tensor.Load_rank_score()
		print('rankings and scores loaded')

		#compute the ranking Kendall similarity matrix btw observables
		#convert this matrix into a weighted network and save it in gephi format
		tensor.Sim_obs()
		print('observables similarity matrix computed')
		#compute the groups of observables with Gephi
		#group1 is red, group2 is green, group3 is blue
		group1 = ['node_interactivity','edge_weight','cc_size','ETN3']
		group2 = ['edge_interactivity','clustering_coeff','deg_assortativity']
		group3 = ['ETN2_weight','ETN3_weight','edge_activity','edge_events_activity','edge_newborn_activity','node_activity']

		#visualize the similarity matrix
		#group1 is red, group2 is green, group3 is blue
		tensor.Visu_sim_obs(group1,group2,group3)

		#for each ADM version, compute the difference btw its score and the score of the basis version
		#for each observable. Then sum up these variations for each group of observables.
		tensor.Score_variation(dic_group)

#visualize point observables and associated scores
def Visu_point_obs():
	#tensor = Proximity_tensor()
	#load rankings and scores
	#tensor.Load_rank_score()
	point_obs = type_to_obs['point']
	for name_obs in point_obs:
		fig,ax = plt.subplots(1,1,constrained_layout=True)
		X = []
		for version_nb in range(1,15):
			for name in XP_data:
				X.append(Load_point('analysis/ADM_class_V'+str(version_nb)+'/'+name,name_obs))
		ax.hist(X,bins=20,color='blue',label='models',histtype='step')
		X = []; x_max = ax.get_ylim()[1]
		for name in XP_data[:-1]:
			x = Load_point('analysis/'+name,name_obs)
			ax.plot([x,x],[0,x_max],color='orange')
		name = XP_data[-1]
		x = Load_point('analysis/'+name,name_obs)
		ax.plot([x,x],[0,x_max],color='orange',label='XP data')
		ax.legend()
		ax.set_xlabel(name_obs)
		ax.set_ylabel('number of occurrences')
		plt.show()

#compare the lnVS15 dataset to other XP datasets
def Compare_new_XP_to_old():
	list_color = ['red','blue','green','purple','black','yellow']
	for name_obs in type_to_obs['distribution']:
		fig,ax = plt.subplots(1,1,constrained_layout=True)
		ax.set_title('comparison of the lnVS15 against other\nXP references ; observable : '+name_obs)
		ax.set_xlabel(r'$\log_{10}(n)$')
		ax.set_ylabel(r'$\log_{10}(P)$')
		list_patch = []
		for color,name in zip(list_color,XP_data):
			#load the XP data and plot it
			dic = Load_distribution('analysis/'+name,name_obs)
			X,Y = zip(*dic.items())
			ax.plot(np.log10(X),np.log10(Y),'v',label=name,color=color)
			list_patch.append(mpatches.Patch(color=color,label=name))
		#load the lnVS15 data and plot it
		dic = Load_distribution('analysis/work2',name_obs)
		X,Y = zip(*dic.items())
		ax.plot(np.log10(X),np.log10(Y),'.',label='lnVS15',color='saddlebrown')
		list_patch.append(mlines.Line2D([],[],color='orange',marker='v',markersize=15,label='empirical reference'))
		list_patch.append(mlines.Line2D([],[],color='orange',marker='.',markersize=15,label='lnVS15'))
		ax.legend(handles=list_patch)
		plt.savefig('figures/lnVS15/'+name_obs+'.png')
		plt.close()

#compare the XP datasets with each other for the three types of observables
def Compare_XP_XP(XP_data):
	#comparison of distribution observables
	#we associate one marker and one color to each XP dataset
	list_marker = ['.','x','s','<','>']
	list_color = ['b','k','orange','purple','saddlebrown']
	obs_to_xlabel = {}
	for obs in type_to_obs['distribution']:
		if 'events' in obs:
			obs_to_xlabel[obs] = r"$\log_{10}(n)$"
		elif 'activity' in obs:
			obs_to_xlabel[obs] = r"$\log_{10}(\Delta t)$"
		elif 'weight' in obs:
			obs_to_xlabel[obs] = r"$\log_{10}(w)$"
		else:
			obs_to_xlabel[obs] = r"$\log_{10}(n)$"
	fontsize = 16
	for name_obs in type_to_obs['distribution']:
		print(name_obs)
		fig,ax = plt.subplots(1,1,constrained_layout=True)
		ax.set_ylabel(r"$\log_{10}(P)$",fontsize=fontsize)
		ax.set_xlabel(obs_to_xlabel[name_obs],fontsize=fontsize)
		for marker,name,color in zip(list_marker,XP_data,list_color):
			data = Load_distribution('analysis/'+name,name_obs)
			ax.plot(*Raw_to_binned(data),marker,color=color,label=name)
		ax.legend(fontsize=fontsize)
		for label in ax.get_xticklabels()+ax.get_yticklabels():
			label.set_fontsize(fontsize)
		plt.savefig('figures/XP_data/'+name_obs+'_XP.png')
	'''
	#comparison of point observables :
	#we associate one color to each group of datasets : blue for conferences, black for schools
	#and green for the workplace
	#then for each dataset, we draw one vertical line of the corresponding color at the position
	#corresponding to the value of the point observable
	name_to_color = {}; fontsize = 14
	for name in XP_data:
		if 'conf' in name:
			name_to_color[name] = 'blue'
		elif 'work' in name:
			name_to_color[name] = 'green'
		else:
			name_to_color[name] = 'black'
	for name_obs in type_to_obs['point']:
		fig,ax = plt.subplots(1,1,constrained_layout=True)
		ax.set_xlabel(name_obs,fontsize=fontsize)
		ax.tick_params(axis='y',which='both',left=False,labelleft=False)
		for name in XP_data:
			x = Load_point('analysis/'+name,name_obs)
			ax.plot([x,x],[0,1],color=name_to_color[name])
		list_patch = []
		list_patch.append(mlines.Line2D([],[],color='blue',label='conferences'))
		list_patch.append(mlines.Line2D([],[],color='black',label='schools'))
		list_patch.append(mlines.Line2D([],[],color='green',label='workplace'))
		ax.legend(handles=list_patch,fontsize=fontsize)
		plt.savefig('figures/XP_data/'+name_obs+'_XP.png')
	'''
	#comparison of vector observables :
	#we compute the vector ETN similarity matrix between all XP datasets and visualize it
	#we plot also the histogram of the similarity values
	#load the ETN3 distribution
	dic_ETN = {}
	for name in XP_data:
		dic_ETN[name] = Load_vector('analysis/'+name,'ETN3')
	#compute the similarity matrix
	n = len(XP_data); mat = np.eye(n)
	for i in range(n-1):
		name1 = XP_data[i]
		for j in range(i+1,n):
			name2 = XP_data[j]
			mat[i,j] = 1-Vector_distance(dic_ETN[name1],dic_ETN[name2])
			mat[j,i] = mat[i,j]
	fig,ax = plt.subplots(1,1,constrained_layout=True)
	fontsize = 16
	ax.set_xticks(range(n))
	ax.set_yticks(range(n))
	ax.set_xticklabels(XP_data,rotation=90,fontsize=fontsize)
	ax.set_yticklabels(XP_data,fontsize=fontsize)
	im = ax.imshow(mat,cmap='gnuplot2')
	plt.colorbar(im)
	plt.savefig('figures/XP_data/ETN_simat.png')

	#global comparison
	#compute a similarity matrix between XP datasets where we define :
	#Sim(D,D') = product of similarities btw D and D' wrt each observable
	all_data = {}
	#for obs,type_obs in obs_to_type.items():
	for type_obs in type_to_obs.keys():
		for obs in type_to_obs[type_obs]:
			all_data[obs] = {}
			for name in XP_data:
				all_data[obs][name] = Load_obs[type_obs]('analysis/'+name,obs)
		n = len(XP_data)
		mat = np.ones((n,n))
		#for obs,type_obs in obs_to_type.items():
		for obs in type_to_obs[type_obs]:
			for i in range(n-1):
				data1 = all_data[obs][XP_data[i]]
				for j in range(i+1,n):
					data2 = all_data[obs][XP_data[j]]
					mat[i,j] *= 1-Distance_obs[type_obs](data1,data2)
		for i in range(n-1):
			for j in range(i+1,n):
				mat[j,i] = mat[i,j]
		fig,ax = plt.subplots(1,1,constrained_layout=True)
		fontsize = 16
		ax.set_xticks(range(n))
		ax.set_yticks(range(n))
		ax.set_xticklabels(XP_data,rotation=90,fontsize=fontsize)
		ax.set_yticklabels(XP_data,fontsize=fontsize)
		#ax.set_title('XP similarity matrix wrt '+type_obs+' observables',fontsize=fontsize)
		im = ax.imshow(mat,cmap='gnuplot2')
		plt.colorbar(im)
		plt.savefig('figures/XP_data/'+type_obs+'_simat_XP.png')

#similarity histogram btw XP datasets
def Histo_sim(load=True,chosen_obs='all',low_sim=0.5,vertical_line=False):
	if not load:
		all_data = {}
		if chosen_obs=='all':
			list_type = list(type_to_obs.keys())
		else:
			list_type = [chosen_obs]
		for type_obs in list_type:
			for obs in type_to_obs[type_obs]:
				all_data[obs] = {}
				for name in XP_data:
					all_data[obs][name] = Load_obs[type_obs]('analysis/'+name,obs)
		all_sim = []; n = len(XP_data); list_obs = set(())
		for type_obs in list_type:
			for obs in type_to_obs[type_obs]:
				for i in range(n-1):
					data1 = all_data[obs][XP_data[i]]
					for j in range(i+1,n):
						data2 = all_data[obs][XP_data[j]]
						sim = 1-Distance_obs[type_obs](data1,data2)
						all_sim.append(sim)
						if sim<low_sim:
							list_obs.add(obs)
		np.savetxt('figures/XP_data/histo_sim/obs_low_sim_'+chosen_obs+'.txt',list(list_obs),fmt='%s')
		np.savetxt('figures/XP_data/histo_sim/XP_sim_'+chosen_obs+'.txt',all_sim)
	all_sim = np.loadtxt('figures/XP_data/histo_sim/XP_sim_'+chosen_obs+'.txt')
	fontsize = 16
	fix,ax = plt.subplots(1,1,constrained_layout=True)
	ax.hist(all_sim,bins=20,density=True,stacked=True)
	ax.set_xlabel('similarity',fontsize=fontsize)
	ax.set_ylabel('density probability',fontsize=fontsize)
	for label in ax.get_xticklabels()+ax.get_yticklabels():
		label.set_fontsize(fontsize)
	if vertical_line:
		ax.plot([low_sim,low_sim],list(ax.get_ylim()),'--',c='r')
	plt.savefig('figures/XP_data/histo_sim/histo_sim_'+chosen_obs+'.png')

#conclusion : we need a new distance designed specifically for comparing large distributions
#this distance has to be one for a vertical distribution and a horizontal distribution
#this distance has to be invariant under rotation
#so this distance is the angular gap of the distributions in log-log scale :
#D(d1,d2) = abs(theta1 - theta2)/(pi/2)
#with tan(theta(d)) = absolute value of the exponent of the power-law sampled from the dataset d
def Test_binned():
	# something random to plot
	name = 'conf16'; type_obs = 'distribution'
	#for obs in type_to_obs[type_obs]:
	for obs in ['edge_events_activity']:
		data = Load_obs[type_obs]('analysis/'+name,obs)
		XY = list(data.items())
		XY.sort(key=lambda el:el[0])
		X,Y = zip(*XY)
		X = np.asarray(X,dtype=float); Y = np.asarray(Y,dtype=float)

		nb = 50; ind = 1
		cond = (abs(np.log10(X[ind]/X[ind-1]))>0.01)
		while cond:
			ind += 1
			if ind<len(X):
				cond = (abs(np.log10(X[ind]/X[ind-1]))>0.01)
			else:
				cond = False
		if ind==len(X):
			bins = np.zeros(ind)
			bins[:ind] = X[:ind]-0.5
		else:
			bins = np.zeros(ind+nb)
			bins[:ind] = X[:ind]-0.5; bins[ind:] = np.logspace(np.log10(X[ind]-0.5),np.log10(X[-1]+0.5),num=nb)
		widths = bins[1:]-bins[:-1]

		# Calculate histogram
		hist = np.histogram(X,bins=bins,weights=Y)
		new_Y = hist[0]/widths

		# plot it!
		fig,ax = plt.subplots(1,1,constrained_layout=True)
		ax.set_title('plot of '+obs+' with or without log-binning')
		ax.plot(np.log10(X),np.log10(Y),'.',label='without')
		new_X = np.log10((bins[:-1]+bins[1:])/2)
		new_Y = np.log10(new_Y)
		ax.plot(new_X,new_Y,'.',label='with')
		ax.plot([np.log10(X[ind-1]),np.log10(X[ind-1])],[new_Y[-1],0],'--')
		ax.legend()
		ax.set_xlabel(r"$\log_{10}(\Delta t)$")
		ax.set_ylabel(r"$\log_{10}(P)$")
		plt.show()

#genetic visualization :
#build a multigraph where nodes are couples (reference,version)
#and we draw an edge of type k btw i and j if i and j have the same value for the parameter k
#this allows to visualize variation accross integer parameters

#other visualization :
#for each parameter, make a histogram bar : plot the number of couples vs a given parameter value
#possibly plot the number of couples for the most frequent value vs the integer parameter
def Visu_int_param(option='histogram'):
	fontsize = 14
	set_int_param = {'m_max','m','c'}
	#dic_couples[num] = couple (ref,version_nb) of identifier num
	dic_couples = {}; num = 0
	for version_nb in range(1,20):
		for name in XP_data:
			dic_couples[num] = (name,str(version_nb))
			num += 1
	#collect integer parameters
	int_param = {param:{} for param in set_int_param}
	for num,couple in dic_couples.items():
		model = 'ADM_class_V'+couple[1]
		name = couple[0]
		#load best parameters
		best_param = np.loadtxt('analysis/'+model+'/'+name+'/best_param.txt',dtype=str,delimiter=',')
		#collect integer parameters
		for i,param in enumerate(best_param[0,:]):
			if param in set_int_param:
				int_param[param][num] = int(best_param[1,i])
	if option=='histogram':
		for param,dic in int_param.items():
			fig,ax = plt.subplots(1,1,constrained_layout=True)
			tab = list(dic.values())
			m = np.min(tab); M = np.max(tab)
			nb_bins = 20
			h = (M-m)/nb_bins
			bins = ax.hist(tab,bins=nb_bins,histtype='bar',alpha=0.7)[1]
			if h>0:
				#dic_histo[k] = set of couples in the kth bin
				dic_histo = {}
				for key,val in dic.items():
					k = min(floor((val-m)/h),nb_bins-1)
					if k in dic_histo:
						dic_histo[k].add(dic_couples[key])
					else:
						dic_histo[k] = {dic_couples[key]}
				fitness = []; X = []
				for k,val in dic_histo.items():
					x = 0; X.append((bins[k]+bins[k+1])/2)
					for couple in val:
						model = 'ADM_class_V'+couple[1]
						name = couple[0]
						best_fit = np.loadtxt('analysis/'+model+'/'+name+'/best_fit.txt')[-1]
						x += best_fit
					fitness.append(x/len(val))
				fitness = np.asarray(fitness)*(ax.get_ylim()[1]/np.max(fitness))
				ax.plot(X,fitness,'.')
			ax.set_xlabel(param,fontsize=fontsize)
			ax.set_ylabel('number of couples',fontsize=fontsize)
			#ax.set_title('nb of couples (reference,version)\nper value of parameter '+param)
			for label in ax.get_xticklabels()+ax.get_yticklabels():
				label.set_fontsize(fontsize)
			plt.savefig('figures/best_param/param_histo_'+param+'.png')
	elif option=='matrix':
		#associate one color to each possible integer value, which gives 4 possible colors
		matrix = np.zeros((len(XP_data),19),dtype=int)
		name_to_int = {name:i for i,name in enumerate(XP_data)}
		for param,dic in int_param.items():
			for num,val in dic.items():
				i,j = name_to_int[dic_couples[num][0]],int(dic_couples[num][1])-1
				matrix[i,j] = val
			fig,ax = plt.subplots(1,1,constrained_layout=True)
			cax = ax.imshow(matrix,cmap='gnuplot2')
			max_val = np.max(matrix)
			cbar = fig.colorbar(cax,ticks=list(range(max_val+1)),location='top')
			for label in cbar.ax.get_xticklabels():
				label.set_fontsize(fontsize)
			ax.set_xticks(range(19))
			ax.set_xticklabels(range(1,20),fontsize=fontsize)
			ax.set_yticks(range(len(XP_data)))
			ax.set_yticklabels(XP_data,fontsize=fontsize)
			ax.set_xlabel('version number',fontsize=fontsize)
			plt.savefig('figures/best_param/param_matrix_'+param+'.png')
			matrix[:,:] = 0

#perform a Kolmogorov-Simrnov test to check whether the distribution of float parameters has been
#altered by the genetic tuning
def KS_float_param():
	XP_info = {name:{} for name in XP_data}
	for name in XP_data:
		global_info = np.loadtxt('analysis/'+name+'/global_info.txt',dtype=str,delimiter=',')
		for i in range(len(global_info[0,:])):
			x = global_info[0,i]; y = global_info[1,i]
			if x in {'N','T','nb of edges'}:
				XP_info[name][x] = int(y)
			else:
				XP_info[name][x] = float(y)
	#set_param = set of float parameters
	set_param = {'a','lambda','alpha','a_min','a_max','p_c','p_u','p_g','p_d'}
	#dic_bounds[param] = (lower bound,upper bound) for param
	dic_bounds = {'lambda':{name:(0.01,10) for name in XP_data}}
	for param in ['alpha','p_c','p_d','p_g','p_u']:
		dic_bounds[param] = {name:(0.001,1) for name in XP_data}
	for param in ['a','a_min']:
		dic_bounds[param] = {}
		for name in XP_data:
			dic_bounds[param][name] = (XP_info[name]['a_min'],XP_info[name]['a_max'])
	dic_bounds['a_max'] = {}
	for name in XP_data:
		dic_bounds['a_max'][name] = (XP_info[name]['a_min'],1)
	#load float parameters
	dic_param = {}
	for version_nb in range(1,20):
		model = 'ADM_class_V'+str(version_nb)
		for name in XP_data:
			tab = np.loadtxt('analysis/'+model+'/'+name+'/best_param.txt',dtype=str,delimiter=',')
			for i,param in enumerate(tab[0,:]):
				if param in set_param:
					if param in dic_param.keys():
						dic_param[param][(model,name)] = float(tab[1,i])
					else:
						dic_param[param] = {(model,name):float(tab[1,i])}
	#perform a KS test for each parameter: dic_KS[param] = p value for the KS test associated to param
	dic_KS = {}
	for param,dic in dic_param.items():
		#gather the parameter values
		tab = []
		for couple,val in dic.items():
			name = couple[1]; bounds = dic_bounds[param][name]
			tab.append((val-bounds[0])/(bounds[1]-bounds[0]))
		dic_KS[param] = kstest(tab,lambda x:x)[1]
	#display the results
	fontsize = 16
	fig,ax = plt.subplots(1,1,constrained_layout=True)
	X,Y = zip(*dic_KS.items())
	ax.plot(Y,'.')
	ax.set_xticks(range(len(Y)))
	ax.plot(list(ax.get_xlim()),[0.05,0.05],'--')

	y_max = np.max(Y)

	ax.add_patch(mpatches.Rectangle((0,0),len(Y)-1,0.05,alpha=0.4,color='green'))
	ax.annotate('not uniform',(len(Y)/2,0.025),xytext=(len(Y)/2,0.025),fontsize=fontsize)
	ax.add_patch(mpatches.Rectangle((0,0.05),len(Y)-1,y_max-0.05,alpha=0.4,color='red'))
	ax.annotate('uniform',(len(Y)/2,(y_max+0.05)/2),xytext=(len(Y)/2,(y_max+0.05)/2),fontsize=fontsize)

	X_to_xlabels = {'a':r"$a$",'p_g':r"$p_{g}$",'p_d':r"$p_{d}$",'p_u':r"$p_{u}$",'p_c':r"$p_{c}$",'lambda':r"$\lambda$"}
	X_to_xlabels['a_min'] = r"$a^{min}$"; X_to_xlabels['a_max'] = r"$a^{max}$"; X_to_xlabels['alpha'] = r"$\alpha$"
	xlabels = [X_to_xlabels[x] for x in X]
	ax.set_xticklabels(xlabels,fontsize=fontsize)
	ax.set_xlabel('float parameter',fontsize=fontsize)
	ax.set_ylabel('p value',fontsize=fontsize)
	for label in ax.get_yticklabels():
		label.set_fontsize(fontsize)
	plt.savefig('figures/best_param/float_param/KS_test.png')	

#for float parameters, we compute the distribution for each parameter over all references and versions
def Visu_float_param():
	X_to_xlabels = {'a':r"$a$",'p_g':r"$p_{g}$",'p_d':r"$p_{d}$",'p_u':r"$p_{u}$",'p_c':r"$p_{c}$",'lambda':r"$\lambda$"}
	X_to_xlabels['a_min'] = r"$a^{min}$"; X_to_xlabels['a_max'] = r"$a^{max}$"; X_to_xlabels['alpha'] = r"$\alpha$"
	set_param = {'a','lambda','alpha','a_min','a_max','p_c','p_u','p_g','p_d'}
	dic_param = {}
	#load float parameters
	for version_nb in range(1,20):
		model = 'ADM_class_V'+str(version_nb)
		for name in XP_data:
			tab = np.loadtxt('analysis/'+model+'/'+name+'/best_param.txt',dtype=str,delimiter=',')
			for i,param in enumerate(tab[0,:]):
				if param in set_param:
					if param in dic_param.keys():
						dic_param[param][(model,name)] = float(tab[1,i])
					else:
						dic_param[param] = {(model,name):float(tab[1,i])}
	fontsize = 16
	for param,dic in dic_param.items():
		fig,ax = plt.subplots(1,1,constrained_layout=True)
		tab = list(dic.values())
		m = np.min(tab); M = np.max(tab)
		nb_bins = 20
		h = (M-m)/nb_bins
		#x_i = m + i*h
		#i_x = floor((x-m)/h)
		bins = ax.hist(tab,bins=nb_bins,alpha=1)[1]
		#dic_histo[k] = set of couples in the kth bin
		dic_histo = {}
		for key,val in dic.items():
			k = min(floor((val-m)/h),nb_bins-1)
			if k in dic_histo:
				dic_histo[k].add(key)
			else:
				dic_histo[k] = {key}
		fitness = []; X = []
		for k,val in dic_histo.items():
			x = 0; X.append((bins[k]+bins[k+1])/2)
			for couple in val:
				best_fit = np.loadtxt('analysis/'+couple[0]+'/'+couple[1]+'/best_fit.txt')[-1]
				x += best_fit
			fitness.append(x/len(val))
		ax.set_xlabel(X_to_xlabels[param],fontsize=fontsize)
		ax.set_ylabel('number of instances',fontsize=fontsize)
		#ax.set_title('distribution for '+param+' over all references and versions')
		fitness = np.asarray(fitness)*(ax.get_ylim()[1]/np.max(fitness))
		#ax.plot(X,fitness,'.')
		for label in ax.get_xticklabels()+ax.get_yticklabels():
			label.set_fontsize(fontsize)
		plt.savefig('figures/best_param/float_param/param_histo_'+param+'.png')

#plot the genetic parameters vs XP datasets and versions
def Plot_genetic():
	#load the genetic parameters
	#dic_param[version_nb][param][name] = value of parameter of name param for the XP dataset name and model
	#version_nb
	dic_param = {}
	for version_nb in range(1,15):
		model_name = 'ADM_class_V'+str(version_nb)
		dic_param[version_nb] = {}
		for j,name in enumerate(XP_data):
			tab = np.loadtxt('analysis/'+model_name+'/'+name+'/best_param.txt',delimiter=',',dtype=str)
			for i,param in enumerate(tab[0,:]):
				if param in {'c','m'}:
					val = int(tab[1,i])
				else:
					val = float(tab[1,i])
				if param in dic_param[version_nb]:
					dic_param[version_nb][param][j] = val
				else:
					dic_param[version_nb][param] = {j:val}
	#visualize dic_param
	set_param = set(())
	for dic in dic_param.values():
		set_param = set_param.union(set(dic.keys()))
	fontsize = 14; nb_data = len(XP_data)
	for chosen_param in set_param.difference({'m','c'}):
		fig,ax = plt.subplots(1,1,constrained_layout=True)
		for version_nb in range(1,16):
			ax.plot([0,nb_data],[version_nb-0.5,version_nb-0.5],'--',color='black')
		for version_nb in range(1,15):
			if chosen_param in dic_param[version_nb]:
				X,Y = zip(*dic_param[version_nb][chosen_param].items())
				#y is rescaled, for probability parameters,
				#the rescaling is linear and sends 0 to version_nb-0.5 and 1 to version_nb+0.5
				Y = np.asarray(Y) + version_nb-0.5
				ax.plot(X,Y,'.')
		ax.set_ylabel(chosen_param,fontsize=fontsize)
		ax.set_xlabel('empirical dataset',fontsize=fontsize)
		ax.set_xticks(range(nb_data))
		ax.set_xticklabels(XP_data,rotation=90,fontsize=fontsize)
		ax.set_yticks(range(1,15))
		ax.set_xlim(0,nb_data-0.5)
		for label in ax.get_yticklabels():
			label.set_fontsize(fontsize)
		plt.show()

#compare a random instance with the instance returned by the genetic tuning
#for the model version_nb, each dataset and each distribution observable
def Compare_tuned_to_random(version_nb):
	list_color = ['red','blue','green','purple','black']
	for name_obs in type_to_obs['distribution']:
		fig,ax = plt.subplots(1,1,constrained_layout=True)
		ax.set_title('comparison of the tuned ADM version '+str(version_nb)+'\nagainst a random instance ; observable : '+name_obs)
		ax.set_xlabel(r'$\log_{10}(n)$')
		ax.set_ylabel(r'$\log_{10}(P)$')
		list_patch = []
		for color,name in zip(list_color,XP_data):
			#load the random instance and plot it
			dic = Load_distribution('analysis/ADM_class_V'+str(version_nb)+'/'+name+'/random',name_obs)
			ax.plot(*Raw_to_binned(dic),'v',color=color)
			#load the tuned instance and plot it
			dic = Load_distribution('analysis/ADM_class_V'+str(version_nb)+'/'+name,name_obs)
			ax.plot(*Raw_to_binned(dic),'.',color=color)
			list_patch.append(mpatches.Patch(color=color,label=name))
		list_patch.append(mlines.Line2D([],[],color='orange',marker='v',markersize=15,label='random instance'))
		list_patch.append(mlines.Line2D([],[],color='orange',marker='.',markersize=15,label='tuned instance'))
		ax.legend(handles=list_patch)
		plt.savefig('figures/obs_distr/ADM_class_V'+str(version_nb)+'/random_tuned/'+name_obs+'.png')
		plt.close()

#plot the instantaneous degree in reference datasets
def XP_deg():
	dic_histo = {}
	for name in XP_data:
		print(name)
		temp_net = Temp_net.Temp_net(np.loadtxt('data/'+name+'.txt',dtype=int))
		temp_net.Get_data_time()
		temp_net.Get_TN(1)
		temp_net.Get_info()
		#histo[d] = nb of occurrences of the fact 'a node has instantaneous degree d in G'
		histo = {}
		for events in temp_net.TN.values():
			for node in events.nodes:
				d = events.degree(node)
				if d in histo:
					histo[d] += 1
				else:
					histo[d] = 1
		#rescale the histo
		X,Y = zip(*histo.items())
		Y = np.asarray(Y,dtype=float)
		norm = np.sum(Y)
		for i in range(len(Y)):
			Y[i] /= norm
		dic_histo[name] = (X,np.log10(Y))
	#plot the histo
	fig,ax = plt.subplots(1,1,constrained_layout=True)
	for name,val in dic_histo.items():
		ax.plot(*val,'.',label=name)
	ax.legend()
	ax.set_xlabel(r"$d$")
	ax.set_ylabel(r"$\log_{10}(P)$")
	ax.set_title('frequency of instantaneous degree\nin reference datasets')
	plt.savefig('figures/XP_data/inst_degree.png')
	plt.show()

def Pre_init_tensor(nb_mod,block_XP=True,option='tuned'):
	Dic_tensor(nb_mod,block_XP=block_XP,option=option)
	print('proximity tensor computed')
	#compute the score for each observable and dataset
	tensor = Proximity_tensor(n2=nb_mod,instance_path=option)
	#initialize the tensor
	tensor.Load_tensor()
	print('proximity tensor loaded')
	#compute the three first quartiles of the distances btw XP data
	tensor.Stat_XP()
	print('XP stat properties computed')
	#compute the score of each model and each XP dataset wrt each observable
	for name_obs in obs_to_type.keys():
		for name in XP_data:
			tensor.Score_XP(name_obs,name)
		for version_nb in range(1,nb_mod+1):
			tensor.Score_model(name_obs,version_nb)
	print('scores computed')
	#compute a ranking per observable
	#then deduce a unique global ranking. However, two possibilities :
	# - global rank = rank averaged over observables
	# - global rank = score averaged over observables
	tensor.Get_rankings()
	print('rankings computed')
	#display the global ranking of all datasets according to their nature (conference, school or model)
	#tensor.Glob_rank('mean')
	#tensor.Glob_rank('sim')
	#save the scores of all datasets for all observables as well as the two global scores and rankings
	tensor.Save_rank_score()
	print('rankings saved')
	#tensor.Sim_obs()
	#print('observables similarity matrix computed')

def Init_tensor(nb_mod,instance_path):
	tensor = Proximity_tensor(n2=nb_mod,instance_path=instance_path)
	tensor.Load_tensor()
	tensor.Load_rank_score()
	tensor.Sim_obs(save=False)
	return tensor

#check whether a given version has a similar distance to each dataset
#relatively to a given observable
def Check_variability():
	tensor = Init_tensor()
	#compute for each version the standard deviation of its distances to references
	#then normalize this std to the std of distances btw references
	#then plot this std vs the version score
	#tot_std gathers normalized std over all versions and all observables
	tot_std = []
	fig,ax = plt.subplots(1,1,constrained_layout=True)
	for t,name_obs in tensor.int_to_obs.items():
		#X = version score, Y = normalized std of version distance to references
		X = []; Y = []
		for version_nb in range(1,19):
			model = 'ADM_class_V'+str(version_nb)
			#std of version distance to references
			tab = [tensor.tensor[t,tensor.name_to_int[model],tensor.name_to_int[name]] for name in XP_data]
			version_std = np.std(tab)
			#std of distances btw references
			tab = []
			for i,name1 in enumerate(XP_data[:-1]):
				k1 = tensor.name_to_int[name1]
				for name2 in XP_data[i+1:]:
					k2 = tensor.name_to_int[name2]
					tab.append(tensor.tensor[t,k1,k2])
			XP_std = np.std(tab)
			#update X and Y
			X.append(tensor.score[t,tensor.name_to_int[model]])
			Y.append(version_std/XP_std)
		tot_std += Y
		#plot
		ax.plot(X,Y,'.')
	ax.set_xlabel('version score')
	ax.set_ylabel('version variability')
	plt.show()

	#compute the distribution of normalized std (over all versions and all observables)
	fig,ax = plt.subplots(1,1,constrained_layout=True)
	ax.hist(tot_std,bins=20)
	ax.set_xlabel('version variability')
	ax.set_ylabel('probability')
	ax.set_title('distribution of model variability over all versions and all observables')
	plt.show()

#plot the two global rankings vs fitness
def Score_fitness():
	tensor = Init_tensor()
	list_models = [tensor.name_to_int['ADM_class_V'+str(version_nb)] for version_nb in range(1,19)]
	#avg_fit[n] = fitness of model n averaged over the references
	avg_fit = {n:0 for n in list_models}
	for n in list_models:
		model = tensor.int_to_name[n]
		for name in XP_data:
			best_fit = np.loadtxt('analysis/'+model+'/'+name+'/best_fit.txt')[-1]
			avg_fit[n] += best_fit
		avg_fit[n] /= len(XP_data)
	X,Y1 = zip(*avg_fit.items())
	for option,title,tab in zip(['mean','sim'],['averaged rank','averaged score'],[tensor.mean_glob_rank,tensor.sim_glob_rank]):
		Y2 = [tab[x] for x in X]
		fig,ax = plt.subplots(1,1,constrained_layout=True)
		ax.set_title('fitness vs '+title)
		ax.set_xlabel('model '+title)
		ax.set_ylabel('model fitness')
		ax.plot(Y2,Y1,'.')
		plt.savefig('figures/score/'+option+'_fitness_global_score.png')

#plot the free parameter param2 vs the readable parameter param1
def Visu_param(param1,param2):
	set_int_param = {'m_max','m','c'}
	#load the readable parameter
	dic_readable = {}
	for name in XP_data:
		global_info = np.loadtxt('analysis/'+name+'/global_info.txt',dtype=str,delimiter=',')
		for i,param in enumerate(global_info[0,:]):
				if param==param1:
					dic_readable[name] = float(global_info[1,i])
	#load the data
	X = []; Y = []
	set_couples = set(())
	for version_nb in range(1,19):
		model = 'ADM_class_V'+str(version_nb)
		for name in XP_data:
			best_param = np.loadtxt('analysis/'+model+'/'+name+'/best_param.txt',dtype=str,delimiter=',')
			dic_param = {}
			for i,param in enumerate(best_param[0,:]):
				if param in set_int_param:
					dic_param[param] = int(best_param[1,i])
				else:
					dic_param[param] = float(best_param[1,i])
			#collect the free parameter and the readable parameter
			if param2 in dic_param:
				if dic_param[param2]>1:
					set_couples.add((version_nb,name))
				Y.append(dic_param[param2])
				X.append(dic_readable[name])
	for couple in set_couples:
		print(couple)
	exit()
	#plot the figure
	fig,ax = plt.subplots(1,1,constrained_layout=True)
	ax.set_xlabel(param1)
	ax.set_ylabel(param2)
	ax.set_title(param2+' vs '+param1)
	ax.plot(X,Y,'.')
	plt.savefig('figures/best_param/'+param2+'_VS_'+param1+'.png')

def Verif_score_all(nb_mod,instance_path,option='sim'):
	#decide of observables to check that the score accounts faithfully for the distance btw two distributions
	chosen_obs = ['cc_size','edge_activity','edge_interactivity','node_interactivity']
	tensor = Proximity_tensor(n2=nb_mod,instance_path=instance_path)
	tensor.Load_rank_score()
	tensor.Verif_score(chosen_obs,option=option)

#For each point observable, we plot the score as a function of the observable value.
#We also indicate as orange vertical lines the values observed in the five empirical references.
def Verif_point_score_tot(nb_mod,instance_path):
	tensor = Proximity_tensor(n2=nb_mod,instance_path=instance_path)
	tensor.Load_rank_score()
	best_mod = tensor.name_to_int[tensor.sim_best_model]-tensor.nb_XP+1
	worst_mod = tensor.name_to_int[tensor.sim_worst_model]-tensor.nb_XP+1
	for option in ['median','averaged']:
		tensor.Verif_point_score([best_mod,14,worst_mod],option=option)

#5 most frequent motifs at aggregation level 1
#of the 'conf16' data set and the best and worst adjacent versions.
def Verif_vector_score_tot(nb_mod,instance_path):
	tensor = Proximity_tensor(n2=nb_mod,instance_path=instance_path)
	tensor.Load_rank_score()
	ref = 'utah'
	for option in ['mean','sim']:
		tensor.Verif_vector_score(ref,option=option,agg_level=5)

#ETN autosimilarity vs ETN score : a higher score should yield a similar autosimilarity curve
#to the empirical case
def Verif_autosim_tot(nb_mod,instance_path):
	tensor = Proximity_tensor(n2=nb_mod,instance_path=instance_path)
	tensor.Load_rank_score()
	for option in ['mean','sim']:
		tensor.Verif_autosim(option=option)

#multiple linear regression of the model fitness
#with respect to the scores relative to all observables
def Reg_fitness():
	tensor = Proximity_tensor()
	tensor.Load_rank_score()
	list_obs = ['ETN2_weight','ETN3_weight','ETN3']
	#list_obs += ['clustering_coeff']
	list_obs += ['node_interactivity','cc_size']
	list_obs += ['node_activity','edge_events_activity']
	#tensor.Reg_fitness_score(14,option='mean',list_obs='all')
	tensor.Reg_fitness_score(14,option='mean',list_obs=list_obs)
	#tensor.Reg_fitness_score(14,option='median',type_obs='all')

#begin with all observables and the 14 first versions to compute
#the obs slopes via reg of the fitness, and compute the regression score.
#Then remove the observable with the smallest absolute slope value from
#the list of observables and repeat. Plot the score vs number of iterations.
def Get_min_listobs_fitness_reg(set_nb=set(()),ETN3=False):
	tensor = Proximity_tensor()
	tensor.Load_rank_score()
	list_obs = [obs for obs in tensor.obs_to_int.keys()]
	if not ETN3:
		list_obs.remove('ETN3')
		fig_title = '_without_ETN3'
	else:
		fig_title = '_with_ETN3'
	tot_obs = len(list_obs)
	X = []; Y1 = []; Y2 = []; fontsize = 16
	fig,ax = plt.subplots(1,1,constrained_layout=True)
	#ax.set_title('regression score of fitness vs observable score\nfor decreasing set of observables',fontsize=fontsize)
	ax.set_xlabel('number of observables',fontsize=fontsize)
	ax.set_ylabel('regression quality',fontsize=fontsize)
	for k in range(tot_obs):
		X.append(tot_obs-k)
		slopes,intercept = tensor.Get_obs_slopes(14,option='mean',list_obs=list_obs)
		Y1.append(tensor.Get_reg_score(18,slopes,intercept,option='mean',list_obs=list_obs))
		Y2.append(tensor.Get_reg_score(14,slopes,intercept,option='mean',list_obs=list_obs))
		if tot_obs-k in set_nb:
			print('for '+str(tot_obs-k)+' observables, list_obs:')
			print(list_obs)
		list_obs.sort(key=lambda obs:abs(slopes[obs]),reverse=True)
		del list_obs[-1]
	ax.plot(X,Y1,'.',label='test set of models')
	ax.plot(X,Y2,'.',label='training set of models')
	ax.legend(fontsize=fontsize)
	for label in ax.get_xticklabels()+ax.get_yticklabels():
		label.set_fontsize(fontsize)
	plt.savefig('figures/fitness_score/reg_quality'+fig_title+'.png')

#Get_min_listobs_fitness_reg(ETN3=True)
#Get_min_listobs_fitness_reg()
#> results of Get_min_listobs_fitness_reg(ETN3=True)
#list of nb most relevant observables:
#nb = 6; list_obs = ['ETN3_weight', 'ETN3', 'node_interactivity', 'edge_events_activity', 'cc_size', 'node_activity']
#nb = 5; list_obs = ['ETN3_weight', 'ETN3', 'node_interactivity', 'edge_events_activity', 'cc_size']
#nb = 1; list_obs = ['ETN3']
#> results of Get_min_listobs_fitness_reg(ETN3=False)
#list of nb most relevant observables:
#nb = 7; list_obs = ['node_interactivity', 'clustering_coeff', 'ETN3_weight', 'cc_size', 'edge_weight', 'deg_assortativity', 'edge_interactivity']
#nb = 5; list_obs = ['node_interactivity', 'clustering_coeff', 'ETN3_weight', 'cc_size', 'edge_weight']
#nb = 2; list_obs = ['node_interactivity', 'clustering_coeff']
#nb = 1; list_obs = ['node_interactivity']

#compute the score variation btw random and tuned instances
def Score_random_tuned():
	'''
	Dic_tensor(14,block_XP=False,option='random')
	#compute the score of random instances
	tensor_rand.Stat_XP()
	for name_obs in obs_to_type.keys():
		for name in XP_data:
			tensor_rand.Score_XP(name_obs,name)
		for version_nb in range(1,15):
			tensor_rand.Score_model(name_obs,version_nb)
	tensor_rand.Get_rankings()
	tensor_rand.Save_rank_score()
	'''
	tensor_rand = Proximity_tensor(n2=14,instance_path='random')
	tensor_rand.Load_rank_score()
	#load the score of tuned instances
	tensor_tuned = Proximity_tensor()
	tensor_tuned.Load_rank_score()
	#for each observable compare the score of random with tuned instances
	#dic_score[obs] = list; dic_score[obs][i-1] = score(tuned version i) - score (random version i)
	dic_score = {}
	for obs in obs_to_type.keys():
		tab = []; t_rand = tensor_rand.obs_to_int[obs]; t_tuned = tensor_tuned.obs_to_int[obs]
		for i in range(1,15):
			model = 'ADM_class_V'+str(i)
			j1 = tensor_tuned.name_to_int[model]; j2 = tensor_rand.name_to_int[model]
			tab.append(tensor_tuned.score[t_tuned,j1]-tensor_rand.score[t_rand,j2])
		dic_score[obs] = tab
	#plot the results
	fig,ax = plt.subplots(1,1,constrained_layout=True)
	fontsize = 13
	ax.set_title('comparison of the score btw random and tuned instances',fontsize=fontsize)
	ax.set_xticks(range(len(dic_score)))
	xlabels = list(dic_score.keys())
	ax.set_xticklabels(xlabels,rotation=90,fontsize=fontsize)
	ax.set_ylabel('score(tuned)-score(random)',fontsize=fontsize)
	for x,obs in enumerate(xlabels):
		ax.plot([x]*14,dic_score[obs],'.',c='b')
	plt.savefig('figures/score/random_tuned/score_variation.png')

#comparison of some random observables btw random and tuned instances
#specific figure for the aps paper
def APS_random_tuned():
	tensor = Init_tensor(19,'tuned')
	list_obs = ['cc_size','edge_activity','ETN3_weight']
	list_ref = ['conf17','utah']
	list_versions = [1,9,12]
	#load the data (dic_data[(k,l)] is used to build the panel (k,l))
	dic_data = {}
	for k,obs in enumerate(list_obs):
		for l,ref in enumerate(list_ref):
			for m,nb in enumerate(list_versions):
				model = 'ADM_class_V'+str(nb)
				dic_data[(k,l,m)] = {}
				dic_data[(k,l,m)][ref] = Load_obs[obs_to_type[obs]]('analysis/'+ref,obs)
				dic_data[(k,l,m)]['tuned'] = Load_obs[obs_to_type[obs]]('analysis/'+model+'/'+ref,obs)
				dic_data[(k,l,m)]['random'] = Load_obs[obs_to_type[obs]]('analysis/'+model+'/'+ref+'/random',obs)
	#build the panels
	fontsize = 16; markersize = 10
	#info_plot[label] = (color,marker) for the associated label (ref, random or tuned)
	info_plot = {'random':('green','^'),'tuned':('purple','s')}
	for ref in list_ref:
		info_plot[ref] = ('blue','.')
	for key,val in dic_data.items():
		fig,ax = plt.subplots(1,1,constrained_layout=True)
		ax.set_xlabel(tensor.obs_to_xlabel[list_obs[key[0]]],fontsize=fontsize)
		ax.set_ylabel(r"$\log_{10}(P)$",fontsize=fontsize)
		for label,data in val.items():
			ax.plot(*Raw_to_binned(data),info_plot[label][1],color=info_plot[label][0],label=label,markersize=markersize)
		ax.legend(fontsize=fontsize)
		for label in ax.get_xticklabels()+ax.get_yticklabels():
			label.set_fontsize(fontsize)
		plt.savefig('figures/obs_distr/random_tuned/APS_'+str(key[0])+str(key[1])+str(key[2])+'.png')
#APS_random_tuned()

#compute and display on the same figure the histograms for the best fitness
#before and after genetic tuning, for all versions and references
#also plot the fitness improvement vs the initial fitness
def Fitness_before_after(nb_before=10,nb_after=10):
	fitness_before = []; fitness_after = []; improvement = []
	for i in range(1,20):
		model = 'ADM_class_V'+str(i)
		for name in XP_data:
			fitness = np.loadtxt('analysis/'+model+'/'+name+'/best_fit.txt')
			fitness_before.append(fitness[0])
			fitness_after.append(fitness[-1])
			improvement.append(fitness[-1]-fitness[0])
	fontsize = 16
	fig,ax = plt.subplots(1,1,constrained_layout=True)
	ax.hist(fitness_before,bins=nb_before,density=True,histtype='bar',alpha=0.7,label='before')
	ax.hist(fitness_after,bins=nb_after,density=True,histtype='step',alpha=1,label='after')
	#ax.hist(improvement,bins=20,histtype='bar',alpha=1,label='improvement')
	ax.set_xlabel('fitness',fontsize=fontsize)
	ax.set_ylabel('probability density',fontsize=fontsize)
	ax.legend(fontsize=fontsize,loc='upper center')
	for label in ax.get_xticklabels()+ax.get_yticklabels():
		label.set_fontsize(fontsize)
	plt.savefig('figures/fitness/fitness_before_after.png')
	#plot the fitness improvement vs the initial fitness
	fig,ax = plt.subplots(1,1,constrained_layout=True)
	ax.plot(fitness_before,improvement,'.')
	ax.set_xlabel('fitness before tuning',fontsize=fontsize)
	ax.set_ylabel('fitness improvement under tuning',fontsize=fontsize)
	for label in ax.get_xticklabels()+ax.get_yticklabels():
		label.set_fontsize(fontsize)
	plt.savefig('figures/fitness/fitness_improvement.png')

#figure for aps paper appendix
def Rank_stability():
	tensor = Init_tensor(19,'tuned')
	tensor.Display_top_stab()

#display the rankings of all datasets w.r.t. every observable on the same figure
def Display_all_rankings():
	tensor = Init_tensor(19,'tuned')
	tensor.Display_rankings()
#Display_all_rankings()
#Rank_stability()
#Pre_init_tensor(19,block_XP=False,option='tuned')
#Fitness_before_after(nb_before=20,nb_after=20)
#Draw_fitness()

'''
build a directed weighted network with hypotheses and observables as nodes
and w_{i,j} is how much the hypothesis i impacts the observable j
we keep only weights with absolute value high enough
'''
pass

#Visu_param('N','m_max')
#the only couple for which m = 4 is (14,highschool3) and this is accompanied by a loss in fitness
# --> m should be frozen to 1
#the only references for which we observe m_max > 1 are conf16 and conf17, but this is not
#accompanied by a loss in fitness : m_max seems to be relevant for some datasets
# --> while the statistical properties are universal, the best models describing the different datasets
#are different so we have to be careful when we analyze the score variation : one parameter value or social
#mechanism could be relevant for one reference but not the other, e.g. the hypothesis 'm' is better for
#all datasets except conf16 and conf17. Hence the distributions are really universal : even with
#different mechanisms and parameter values, we get the same power-law distributions

#for chosen_obs,low_sim in zip(['all','point','distribution','vector'],[0.5,0.6,0.87,1]):
#	Histo_sim(load=False,chosen_obs=chosen_obs,low_sim=low_sim)
#Histo_sim(load=False,chosen_obs='distribution',low_sim=0.87,vertical_line=True)

#Visu_int_param(option='matrix')
#Visu_int_param(option='histogram')
#Visu_float_param()
#KS_float_param()
#Score_fitness()
'''
nb_mod = 14; instance_path = 'tuned'
Verif_point_score_tot(nb_mod,instance_path)
Verif_score_all(nb_mod,instance_path,option='sim')
Verif_vector_score_tot(nb_mod,instance_path)
Verif_autosim_tot(nb_mod,instance_path)
'''
#blue group
group1 = ['ETN3','edge_weight','node_interactivity','cc_size','ETN2_weight']
#red group
group2 = ['edge_newborn_activity','deg_assortativity','edge_activity','ETN3_weight','node_activity']
#green group
group3 = ['edge_events_activity','clustering_coeff','edge_interactivity']
dic_group = {('blue','group I'):group1,('red','group II'):group2,('green','group III'):group3}
#tensor = Init_tensor(19,'tuned')
#tensor.Score_variation(dic_group,13)
#for option in ['mean','sim']:
#	tensor.Glob_rank(option=option,version=[1,19],other=False)
#tensor.Visu_compo_prop(dic_group)

#tensor = Init_tensor(19,'tuned')
#nb_mod = 19
#tensor.Pearson_fitness_score(nb_mod,aggregator='mean')

#new model
#the node wanders randomly, he has interest, or a priori A_i,j, when he encounters j with high a priori,
#he tries to interact with him and then update A_i,j. If he observes something that contradicts
#his estimation, e.g. two partners of very different interest for him are discussing together,
#he tries to join the conversation so that A_i,j matches his observations
#(i believes that A_i,j satisfies homophily).
#I_i,j is the real interest i experiences when discussing with j, based on I_i,j i will update A_i,j, taking
#into account the homophily.
#Q_i,j is the quantity of information i can communicate to j. Once i has emitted all possible information
#to j, he has nothing left to say to him. Q_i,j depends on the level of confidentiality of the info.
#types of information : personal info : public info (work subject, labo, workplace, ...), private info
#(family, hobbies, personal tastes, ...) ;
#info about the environement : subject of the conference, architecture of the building, weather, ...
#public info about the others : number of people, crowd flows, language or habbits observed, ...
#private info about the others : friendships among the others, appreciation btw others,
#family of others, hobbies, ...
#when i and j encounter, they first exchange public info and then they decide whether or not they share
#more confidential information (threshold process). If yes, their friendship strengthens, if not, they
#just interact until all information of the chosen level of confidentiality has been exchanged.
#i can be inactive because he needs time to wander through the space and inspect the other agents :
#i inspects agents one by one, checks that A_i,j is compatible with observations and switches to another
#node, if he has not decided to interact.
#also, agents receive information from the environement that affects A_i,j (e.g. i assisted to a conference
#given by j and concludes that he should speak to j later), and brings it closer to the real value I_i,j

#each node has an a priori on other nodes --> matrix A_i,j, which is a belief about how interesting it would
#be for i to discuss with j. A_i,j is updated through homophily and interaction i--j
#I_i,j is the matrix of real interest, i.e. how interesting it really is for i to discuss with j
#when i and j first encounter, they exchange information at a high rate, but then they have nothing left to
#say, so the information flow is lower --> either this flow vanishes completely (i and j stop interacting)
#or it stabilizes (i and j become long-term partners)

#fitness, or objective function : what is the objective of an individual ?
#a priori each individual has its own goal, but what parameters can he take into account ?