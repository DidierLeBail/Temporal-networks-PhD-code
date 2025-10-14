#!!!do not use mutable objects as default values!!!
import os
#os.path.dirname(__file__) returns the absolute path to the parent directory of __file__ (current file)
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
ADM_DIR = os.path.join(PROJECT_ROOT,'Articles/ADM_class/analysis')
import numpy as np
from math import *
import matplotlib.pyplot as plt
import libs.atn as atn

#global variables
LIST_COLOR = ['red','black','blue','orange','green','purple','brown','pink']
LIST_MARKER = ['.','<','s','x','p','^','>','*','v','1','2','3','4','P']

XP_data = list(np.loadtxt(os.path.join(CURRENT_DIR,'XP_data.txt'),dtype=str))

type_to_obs = {'point': ['clustering_coeff', 'deg_assortativity'], 'vector': ['ETN3']}
type_to_obs['distribution'] = ['cc_size']
type_to_obs['distribution'] += ['edge_activity', 'edge_events_activity', 'edge_newborn_activity', 'node_activity']
type_to_obs['distribution'] += ['edge_interactivity', 'node_interactivity']
type_to_obs['distribution'] += ['edge_weight', 'ETN2_weight', 'ETN3_weight']
#type_to_obs['distribution'] += ['static_degree']
obs_to_type = {}
for type_obs,val in type_to_obs.items():
	for name_obs in val:
		obs_to_type[name_obs] = type_obs

def Draw_simat(simat, tick_labels, fontsize, savepath):
	fig, ax = plt.subplots(constrained_layout=True)
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

#return how we should call the dataset corresponding to name in a figure or a data file
def Get_savename(name):
	if type(name)==tuple:
		if type(name[-1])==str:
			if 'randomized' in name[-1]:
				if len(name)==2:
					return Get_savename_no_random(name[0])+'_'+name[-1]
				else:
					return Get_savename_no_random(name[:-1])+'_'+name[-1]
			return Get_savename_no_random(name)
		return Get_savename_no_random(name)
	return Get_savename_no_random(name)

#return how we should call the dataset corresponding to name in a figure or a data file
#given the dataset is not a randomization of another
def Get_savename_no_random(name):
	if type(name)==tuple:
		if type(name[0])==str:
			if name[0]=='ADM':
				savename = name[0]+str(name[1])+name[2]
			elif name[0]=='min_EW':
				savename = name[0]+str(name[1])
			elif name[0]=='min_ADM' or name[0]=='min_V7':
				savename = 'min_ADM'+str(name[1])
		elif type(name[0])==np.ndarray:
			savename = name[1]
	else:
		savename = name
	return savename

#for not randomized datasets
def Savename_to_name_no_random(savename):
	if savename[:7]=='min_ADM':
		name = ('min_ADM',int(savename[7:]))
	elif savename[:6]=='min_EW':
		name = ('min_EW',int(savename[6:]))
	elif savename[:3]=='ADM':
		ind = 0; set_int = {str(i) for i in range(10)}
		while savename[ind+3] in set_int:
			ind += 1
		name = ('ADM',int(savename[3:3+ind]),savename[3+ind:])
	else:
		name = savename
	return name

#include randomized datasets
def Savename_to_name(savename):
	if '_randomized' in savename:
		ind = 0
		while savename[ind:ind+11]!='_randomized':
			ind += 1
		name_no_random = Savename_to_name_no_random(savename[:ind])
		if type(name_no_random)==tuple:
			return name_no_random + (savename[ind+1:],)
		return (name_no_random,savename[ind+1:])
	return Savename_to_name_no_random(savename)

def Load_TN_Min_EW(version_nb):
	return atn.Min_EW(138,3635,0.79,**Get_versions(choice="min_EW")[version_nb]).Evolve()

def Load_TN_min_ADM(version_nb):
	version = Get_versions(choice='ADM')[7]
	#load XP info of conf16
	XP_info = {'N':138,'T':3635,'nb of edges':153371,'sigma':0.34,'mu':-0.56}
	#modifies the version
	version['m'] = 'cst'
	version['a'] = 'cst'
	version['c_ij'] = False
	version['update'] = 'linear'
	version['context'] = None
	version['remove'] = 'node'
	#decide of the parameters
	dic_param = {}
	dic_param['a'] = 0.3
	dic_param['m'] = 1
	if version_nb==2:
		p_d = 0.1
	elif version_nb==1:
		p_d = 0.02
	dic_param['p_d'] = p_d
	dic_param['p_u'] = 1
	dic_param['p_g'] = 0.08498
	#generate the model instance
	model = atn.ADM_class(XP_info,**version)
	for param in model.free_param.keys():
		model.free_param[param] = dic_param[param]
	model.Refresh()
	return model.Evolve()

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

def Get_versions(choice='ADM'):
	versions = {}
	if choice=='ADM':
		#version 1 = basis version
		#versions 2 to 13 are adjacent to 1
		#versions 14 and beyond are combinations of multiple adjacent versions

		#basis version :
		#m_{i}, a_{i}, alpha_{i}, contextual interactions are neutral, c_{i,j} used, constant egonet growth,
		#removal of edges depending on their weight, Alpha update process
		versions[1] = {'m':'random','a':'power','update':'alpha,i','context':'neutral','c_ij':True,'egonet_growth':'cst','remove':'edge'}

		#2nd version : linear reinforcement process and no gradual decay
		versions[2] = {'m':'random','a':'power','update':'linear','context':'neutral','c_ij':True,'egonet_growth':'cst','remove':'edge'}

		#3rd version : random removal of nodes
		versions[3] = {'m':'random','a':'power','update':'alpha,i','context':'neutral','c_ij':True,'egonet_growth':'cst','remove':'node'}

		#4th version : varying egonet growth
		versions[4] = {'m':'random','a':'power','update':'alpha,i','context':'neutral','c_ij':True,'egonet_growth':'var','remove':'edge'}

		#5th version : c_{i,j} = 1
		versions[5] = {'m':'random','a':'power','update':'alpha,i','context':'neutral','c_ij':False,'egonet_growth':'cst','remove':'edge'}

		#6th version : intentional and contextual interactions are equivalent
		versions[6] = {'m':'random','a':'power','update':'alpha,i','context':'equivalent','c_ij':True,'egonet_growth':'cst','remove':'edge'}

		#7th version : contextual interactions are pure noise
		versions[7] = {'m':'random','a':'power','update':'alpha,i','context':'noise','c_ij':True,'egonet_growth':'cst','remove':'edge'}

		#8th version : no contextual interactions
		versions[8] = {'m':'random','a':'power','update':'alpha,i','context':None,'c_ij':True,'egonet_growth':'cst','remove':'edge'}

		#9th version : alpha
		versions[9] = {'m':'random','a':'power','update':'alpha','context':'neutral','c_ij':True,'egonet_growth':'cst','remove':'edge'}

		#10th version : alpha_{i}, beta_{i}
		versions[10] = {'m':'random','a':'power','update':'alpha,beta,i','context':'neutral','c_ij':True,'egonet_growth':'cst','remove':'edge'}

		#11th version : alpha_{i,j}, beta_{i,j}
		versions[11] = {'m':'random','a':'power','update':'alpha,beta,ij','context':'neutral','c_ij':True,'egonet_growth':'cst','remove':'edge'}

		#12th version : a
		versions[12] = {'m':'random','a':'cst','update':'alpha,i','context':'neutral','c_ij':True,'egonet_growth':'cst','remove':'edge'}

		#13th version : m
		versions[13] = {'m':'cst','a':'power','update':'alpha,i','context':'neutral','c_ij':True,'egonet_growth':'cst','remove':'edge'}

		#14th version : ori_ADM, i.e. the following combination : versions 2+3+4+5+8+13
		versions[14] = {'m':'cst','a':'power','update':'linear','context':None,'c_ij':False,'egonet_growth':'var','remove':'node'}

		#15th version : 2+5+8+13
		versions[15] = {'m':'cst','a':'power','update':'linear','context':None,'c_ij':False,'egonet_growth':'cst','remove':'edge'}

		#16th version : 5+8+11+13
		versions[16] = {'m':'cst','a':'power','update':'alpha,beta,ij','context':None,'c_ij':False,'egonet_growth':'cst','remove':'edge'}

		#17th version : 3+5+8+9+12+13 (simplest version with exponential Hebbian process)
		versions[17] = {'m':'cst','a':'cst','update':'alpha','context':None,'c_ij':False,'egonet_growth':'cst','remove':'node'}

		#18th version : 2+3+5+8+12+13 (simplest version with linear Hebbian process)
		versions[18] = {'m':'cst','a':'cst','update':'linear','context':None,'c_ij':False,'egonet_growth':'cst','remove':'node'}

		#19th version : 7+9+13 (best expected version)
		versions[19] = {'m':'cst','a':'power','update':'alpha','context':'noise','c_ij':True,'egonet_growth':'cst','remove':'edge'}
		#15th version : linear reinforcement process and linear decay process
		#16th version : transitive initialization for weights of social ties
	elif choice=='min_EW':
		#if shift=True and removal'=None then
		#the number of temporal edges is O(duree**2) instead of O(duree)
		#so the analysis is barely doable. One way to do it would be to parallelize the computation of
		#observables by using the CPT clusters
		#On the contrary, if shift=False and 'removal'!=None then
		#the data set will only contain newborn activations in the stationary state
		versions[1] = {'shift':False,'removal':None,'newborn':'random'}
		versions[2] = {'shift':True,'removal':'node_unif','newborn':'random'}
		versions[3] = {'shift':True,'removal':'edge_unif','newborn':'random'}
	return versions

def Load_instance_param(version_nb,name):
	model = 'ADM_class_V'+str(version_nb)
	set_int_param = {'m_max','m','c'}
	dic_param = {}
	#load parameters
	best_param = np.loadtxt(os.path.join(ADM_DIR,model+'/'+name+'/best_param.txt'),dtype=str,delimiter=',')
	for i,param in enumerate(best_param[0,:]):
		if param in set_int_param:
			dic_param[param] = int(best_param[1,i])
		else:
			dic_param[param] = float(best_param[1,i])
	return dic_param

def Load_XP_info(name):
	global_info = np.loadtxt(os.path.join(ADM_DIR,name+'/global_info.txt'),dtype=str,delimiter=',')
	XP_info = {}
	for i in range(len(global_info[0,:])):
		x = global_info[0,i]; y = global_info[1,i]
		if x in {'N','T','nb of edges'}:
			XP_info[x] = int(y)
		else:
			XP_info[x] = float(y)
	return XP_info

#load TN data from the ADM class
def Load_TN_ADM(version_nb,name):
	versions = Get_versions(choice='ADM')
	#load instance parameters
	dic_param = Load_instance_param(version_nb,name)
	#load XP info
	XP_info = Load_XP_info(name)
	#generate the model instance
	Model = atn.ADM_class(XP_info,**versions[version_nb])
	for param in Model.free_param.keys():
		Model.free_param[param] = dic_param[param]
	Model.Refresh()
	return Model.Evolve()

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
