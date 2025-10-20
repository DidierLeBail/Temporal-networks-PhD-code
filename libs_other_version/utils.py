#!!!do not use mutable objects as default values!!!
import os
#os.path.dirname(__file__) returns the absolute path to the parent directory of __file__ (current file)
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
ADM_DIR = os.path.join(PROJECT_ROOT,'Articles/ADM_class/analysis')
import numpy as np
from math import *
import matplotlib.pyplot as plt
import Librairies.atn as atn

#global variables
LIST_COLOR = ['red','black','blue','orange','green','purple','brown','pink']
LIST_MARKER = ['.','<','s','x','p','^','>','*','v','1','2','3','4','P']

#normalize an histogram written as a dictionary of arbitrary depth:
#for depth 2, after normalization, histo[key1][key2] = Proba(key2|key1)
def norm_dic_histo(histo):
	bottom_found = False; norm = 0
	for key,val in histo.items():
		if type(val)!=dict:
			bottom_found = True
			norm += val
	if bottom_found:
		return {key:val/norm for key,val in histo.items()}
	return {key:norm_dic_histo(histo[key]) for key in histo.keys()}

#if tuple_keys = (k0,k1,...,kn) then initialize to 1 the value dic[kn]...[k0]
def initialize_dic(dic,tuple_keys):
	if len(tuple_keys)==1:
		key = tuple_keys[0]
		dic[key] = 1
		return None
	key = tuple_keys[-1]
	dic[key] = {}
	return initialize_dic(dic[key],tuple_keys[:-1])

#if tuple_keys = (k0,k1,...,kn) then increase by 1 the value dic[kn]...[k0]
#and add the non-existing keys
def increase_dic(dic,tuple_keys):
	if len(tuple_keys)==1:
		key = tuple_keys[0]
		if key in dic:
			dic[key] += 1
		else:
			dic[key] = 1
		return None
	key = tuple_keys[-1]
	if key in dic:
		return increase_dic(dic[key],tuple_keys[:-1])
	return initialize_dic(dic,tuple_keys)


#anything below this line should be checked
########################################################################





XP_data = list(np.loadtxt(os.path.join(CURRENT_DIR,'XP_data.txt'),dtype=str))

type_to_obs = {'point':['clustering_coeff','deg_assortativity'],'vector':['ETN3']}
type_to_obs['distribution'] = ['cc_size']
type_to_obs['distribution'] += ['edge_activity','edge_events_activity','edge_newborn_activity','node_activity']
type_to_obs['distribution'] += ['edge_interactivity','node_interactivity']
type_to_obs['distribution'] += ['edge_weight','ETN2_weight','ETN3_weight']
#type_to_obs['distribution'] += ['static_degree']
obs_to_type = {}
for type_obs,val in type_to_obs.items():
	for name_obs in val:
		obs_to_type[name_obs] = type_obs

def Draw_simat(simat,tick_labels,fontsize,savepath):
	fig,ax = plt.subplots(constrained_layout=True)
	img = ax.imshow(simat,cmap='gnuplot2')
	im_ratio = simat.shape[0]/simat.shape[1]
	fig.colorbar(img,ax=ax,fraction=0.05*im_ratio)
	tick_loc = list(range(len(tick_labels)))
	ax.set_xticks(tick_loc)
	ax.set_xticklabels(tick_labels,rotation=90,fontsize=fontsize)
	ax.set_yticks(tick_loc)
	ax.set_yticklabels(tick_labels,fontsize=fontsize)
	plt.savefig(savepath)
	plt.close()

#draw samples from tab with specified normalized weights
def Draw_from_tab(tab,weights):
	x = rd.random()
	s = 0; i = 0
	while s<x:
		s += weights[i]
		i += 1
	return tab[i-1]

#return a number drawn from the power law with fixed bounds and exponent -gamma (P(x)=x^{-gamma})
def Power_law(bounds,gamma,size=1):
	if gamma==1:
		return bounds[0]*(bounds[1]/bounds[0])**np.random.random(size)
	return bounds[0]*(1 + np.random.random(size)*((bounds[1]/bounds[0])**(1-gamma) - 1))**(1/(1-gamma))


#save the realization val of the observable of name name_obs in the dataset name
def Save_point(path,name_obs,val):
	np.savetxt(path+'/point/'+name_obs+'.txt',[val])
def Save_distribution(path,name_obs,val):
	np.savetxt(path+'/distribution/'+name_obs+'.txt',np.array(list(zip(*val.items()))),fmt='%d')
def Save_vector(path,name_obs,val):
	anapath = path+'/vector/'+name_obs
	for agg in val.keys():
		np.savetxt(anapath+str(agg)+'.txt',np.array(list(zip(*val[agg].items()))),fmt='%s')
Save_obs = {'point':Save_point,'distribution':Save_distribution,'vector':Save_vector}

#load observable realization and put it in a form suitable for distance comptutation
def Load_point(anapath,name_obs):
	return np.loadtxt(anapath+'/point/'+name_obs+'.txt')
def Load_distribution(anapath,name_obs):
	tab = np.loadtxt(anapath+'/distribution/'+name_obs+'.txt',dtype=int)
	norm = float(np.sum(tab[1,:]))
	return {tab[0,i]:float(tab[1,i])/norm for i in range(np.size(tab,1))}
def Load_vector(anapath,name_obs,agg_max=10):
	res = {}
	for agg in range(1,agg_max+1):
		tab = np.loadtxt(anapath+'/vector/'+name_obs+str(agg)+'.txt',dtype=str)
		res[agg] = {tab[0,i]:float(tab[1,i]) for i in range(np.size(tab,1))}
	return res
Load_obs = {'point':Load_point,'distribution':Load_distribution,'vector':Load_vector}

#compute distance between two realizations obs1,obs2 of the same observable
#note that all these distances are renormalized between 0 and 1
def Point_distance(obs1,obs2):
	return abs(obs2-obs1)/(2*max(abs(obs2),abs(obs1)))
def Cosim(obs1,obs2,trunc=None):
	if trunc is not None:
		most_freq1 = sorted(obs1.keys(),key=lambda seq:obs1[seq],reverse=True)[:trunc]
		newobs1 = {seq:obs1[seq] for seq in most_freq1}
		most_freq2 = sorted(obs2.keys(),key=lambda seq:obs2[seq],reverse=True)[:trunc]
		newobs2 = {seq:obs1[seq] for seq in most_freq2}
	else:
		newobs1 = obs1; newobs2 = obs2
	norm1 = sum([val**2 for val in newobs1.values()])
	norm2 = sum([val**2 for val in newobs2.values()])
	s = 0
	for key in set(newobs1.keys()).intersection(set(newobs2.keys())):
		s += newobs1[key]*newobs2[key]
	return s/sqrt(norm1*norm2)
def Cosim_triple(obs1,obs2,nb_parts=3):
	#separate obs1 and obs2 in nb_parts parts of equal size
	list_argdic = []
	for obs in (obs1,obs2):
		sorted_key = sorted(obs.keys(),key=lambda key:obs[key],reverse=True)
		size = len(sorted_key)//nb_parts
		list_dic = [{key:obs[key] for key in sorted_key[k*size:(k+1)*size]} for k in range(nb_parts)]
		for key in sorted_key[nb_parts*size:]:
			list_dic[-1][key] = obs[key]
		list_argdic.append(list_dic)
	sim = 1
	for k in range(nb_parts):
		sim *= Cosim(list_argdic[0][k],list_argdic[1][k])
	return sim
def Vector_distance(obs1,obs2):
	#we actually have one sub-vector per aggregation level
	#the similarity btw the two vectors is the product of the similarity of their sub-vectors
	#the distance is 1-similarity
	tot_sim = 1
	for agg in obs1.keys():
		norm1 = sum([val**2 for val in obs1[agg].values()])
		norm2 = sum([val**2 for val in obs2[agg].values()])
		s = 0
		for key in set(obs1[agg].keys()).intersection(set(obs2[agg].keys())):
			s += obs1[agg][key]*obs2[agg][key]
		tot_sim *= s/sqrt(norm1*norm2)
	return 1-tot_sim
def Distribution_distance(obs1,obs2):
	#first use log-binning to enhance the quality of the raw power-laws
	X1,Y1 = Raw_to_binned(obs1)
	X2,Y2 = Raw_to_binned(obs2)
	nb = max(len(Y1),len(Y2))
	T1 = np.zeros(nb); T2 = np.zeros(nb)
	T1[:len(Y1)] = np.power(10,np.asarray(Y1))[:]
	T2[:len(Y2)] = np.power(10,np.asarray(Y2))[:]
	T = (T1+T2)/2
	return (np.sum(T1*np.log2(T1/T+1e-12)) + np.sum(T2*np.log2(T2/T+1e-12)))/2

#compute the JSD btw the raw data obs1 and obs2 (keys as well as values are integers)
def JSD(obs1,obs2):
	#obs[key] = probability of key
	keys1 = set(obs1.keys())
	keys2 = set(obs2.keys())
	mixture = {}
	for key in keys1.union(keys2):
		if key not in obs1:
			mixture[key] = obs2[key]/2
		elif key not in obs2:
			mixture[key] = obs1[key]/2
		else:
			mixture[key] = (obs1[key] + obs2[key])/2
	#compute the JSD
	res = 0
	for key in keys1:
		res += obs1[key]*log2(obs1[key]/mixture[key])
	for key in keys2:
		res += obs2[key]*log2(obs2[key]/mixture[key])
	return res/2

#####################################################################################################
Distance_obs = {'point':Point_distance,'distribution':Distribution_distance,'vector':Vector_distance}

#return log-binned data to enhance the quality of power-law distributions
#if a central point is specified, the smallest bins are around it and the space of the bins
#increases with the distance to the central point
def Raw_to_binned(data,nb=50,central_point=None):
	if len(data)==1:
		return np.log10(list(data.keys())[0]),0
	XY = list(data.items())
	XY.sort(key=lambda el:el[0])
	X,Y = zip(*XY)
	X = np.asarray(X,dtype=float); Y = np.asarray(Y,dtype=float)

	ind = 1
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
	res_X = []; res_Y = []
	for x,y in zip((bins[:-1]+bins[1:])/2,hist[0]/widths):
		if y>0:
			res_X.append(np.log10(x))
			res_Y.append(np.log10(y))
	return res_X,res_Y

#return a figure with the correct fontsize and x,y labels
def Setup_Plot(xlabel,ylabel,fontsize=14,title=''):
	fig,ax = plt.subplots(1,1,constrained_layout=True)
	ax.set_xlabel(xlabel,fontsize=fontsize)
	ax.set_ylabel(ylabel,fontsize=fontsize)
	if title:
		ax.set_title(title,fontsize=fontsize)
	for label in ax.get_xticklabels()+ax.get_yticklabels():
		label.set_fontsize(fontsize)
	return fig,ax

#create the correct files and folders when we start a new project
def Setup_init_project():
	for folder in ['codata','figures']:
		os.mkdir(folder)
	np.savetxt('ideas.txt',['0'],fmt='%s')

#what is written here will not run at settings.py importation as a modules
if __name__=='__main__':
	pass
