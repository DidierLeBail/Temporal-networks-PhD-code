#study reversibility and causality in temporal networks
import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(PROJECT_ROOT)
import Librairies.Temp_net as tp
from Librairies.settings import XP_data,Setup_Plot,Vector_distance,Cosim,Get_versions,Load_instance_param,Get_savename,Raw_to_binned,Load_TN_ADM
from Librairies.atn import ADM_class,Min_EW,Life_game
import Librairies.ETN as etn_lib

import numpy as np
import math
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import networkx as nx
import random as rd
from scipy.optimize import curve_fit
from scipy.stats import ks_2samp, shapiro, kendalltau, permutation_test, spearmanr
from scipy.stats import t as t_test_func
from scipy.interpolate import UnivariateSpline,splrep,splev
#import piecewise_regression as pr
#import alphashape as ash
#from descartes import PolygonPatch
#from dcor.independence import distance_correlation_t_test
from tsmoothie.smoother import *
from tsmoothie.utils_func import sim_randomwalk
#turn off warnings from polynomial fitting
import warnings
warnings.simplefilter('ignore',np.RankWarning)
print('begin')

#returns 'scalar'/'distr' if obs is a scalar/distribution observable
def Get_nature(obs):
	scalar_signature = {'ICC','nb','error','entropy','sim','avg','frac'}
	for seq in scalar_signature:
		if seq in obs:
			return 'scalar'
	return 'distr'

#res[i] = integer name of the folder corresponding to the (obs,arg) pair obs_with_arg[i]
#record[arg] = name of the folder corresponding to arg (one record per observable)
def Check_folder_flow(obs_with_arg):
	res = [0]*len(obs_with_arg)
	for i,(obs,arg) in enumerate(obs_with_arg):
		if not os.path.isdir('codata/'+obs):
			os.mkdir('codata/'+obs)
			res[i] = 0
			os.mkdir('codata/'+obs+'/0')
			np.savetxt('codata/'+obs+'/record.txt',[str(arg),'0'],fmt='%s')
		else:
			tab = np.loadtxt('codata/'+obs+'/record.txt',dtype=str)
			if tab.ndim==1:
				record = {tab[0]:tab[1]}
			else:
				record = {tab[0,i]:tab[1,i] for i in range(np.size(tab,1))}
			key = str(arg)
			if key in record:
				res[i] = int(record[key])
			else:
				res[i] = len(record)
				record[key] = str(res[i])
				os.mkdir('codata/'+obs+'/'+str(res[i]))
			np.savetxt('codata/'+obs+'/record.txt',np.array(list(zip(*record.items())),dtype=str),fmt='%s')
	return res

#res = integer name of the folder corresponding to the (obs,arg) pair
#record[arg] = name of the folder corresponding to arg (one record per observable)
def Find_folder(obs,arg):
	tab = np.loadtxt('codata/'+obs+'/record.txt',dtype=str)
	if tab.ndim==1:
		record = {tab[0]:tab[1]}
	else:
		record = {tab[0,i]:tab[1,i] for i in range(np.size(tab,1))}
	return int(record[str(arg)])

def Check_dir(obs):
	num = Find_folder(*obs)
	if not os.path.isdir('figures/'+obs[0]):
		os.mkdir('figures/'+obs[0])
	if not os.path.isdir('figures/'+obs[0]+'/'+str(num)):
		os.mkdir('figures/'+obs[0]+'/'+str(num))

#return X,np.log10(Y) after normalization of the histo (X,Y)
def Raw_to_exp(histo):
	X,Y = zip(*histo.items())
	Y = np.asarray(Y,dtype=float)
	#normalization
	norm = np.sum(Y)
	Y[:] /= norm
	#transfo
	return X,np.log10(Y)

#return the xlabel and ylabel associated to obs in Plot_distr_nb_flow
#as well as the function taking as input the raw histogram and returning the curve to be plotted
#in Plot_distr_nb_flow
def Obs_to_labels(name_obs):
	if 'duration' in name_obs:
		xlabel = r'$\log_{10}(\Delta t)$'; ylabel = r'$\log_{10}(P)$'
		func = Raw_to_binned
	elif 'inst_deg' in name_obs:
		xlabel = r'$k$'; ylabel = r'$\log_{10}(P)$'
		func = Raw_to_exp
	else:
		xlabel = r'$\log_{10}(n)$'; ylabel = r'$\log_{10}(P)$'
		func = Raw_to_binned
	return xlabel,ylabel,func

def Rewrite_listobs(list_obs):
	res = []
	for obs in list_obs:
		if type(obs)!=tuple:
			res.append((obs,()))
		elif type(obs[1])!=tuple:
			res.append((obs[0],tuple([obs[1]])))
		else:
			res.append(obs)
	return res

def Rewrite_obs(obs):
	if type(obs)!=tuple:
		return (obs,())
	elif type(obs[1])!=tuple:
		return (obs[0],tuple([obs[1]]))
	else:
		return obs

#observable data are stored in 'codata/obs/arg_num/Flow_'+savename+'_n'+str(agg)+'_b'+str(b)+'.txt'
#compute the flow of internal observables (i.e. observables that can be computed
#with only the info of one single state (n,b)) under TS and temporal aggregation
#obs_with_arg = list containing tuples (obs,arg)
def Internalobs_nb_flow(name,obs_with_arg,list_b,list_agg):
	#check the format is correct
	obs_with_arg = Rewrite_listobs(obs_with_arg)
	if list_agg==0 or list_agg==[]:
		return None
	if type(list_agg)==int:
		list_agg = [list_agg]
	elif type(list_agg)!=list:
		raise ValueError('list_agg should be either an integer or a list of integers')
	#check the folder arborescence is correct
	obsarg_to_num = Check_folder_flow(obs_with_arg)
	#separate the scalar from distr observables
	distr_obs = []; scalar_obs = []
	for i,el in enumerate(obs_with_arg):
		if Get_nature(el[0])=='scalar':
			scalar_obs.append(i)
		else:
			distr_obs.append(i)
	savename = Get_savename(name)
	net = tp.Temp_net(tp.Load_TN(name))
	net.Init(1)
	for agg in list_agg:
		print('level '+str(agg)+' begins')
		res = {i:[] for i in scalar_obs}
		for b in list_b:
			print(str(b)+' begins')
			net.Local_TS(b,agg=agg)
			for i in scalar_obs:
				obs,arg = obs_with_arg[i]
				res[i].append(net.Get_obs(obs,arg))
			for i in distr_obs:
				arg_num = str(obsarg_to_num[i])
				obs,arg = obs_with_arg[i]
				histo = net.Get_obs(obs,arg)
				#save distr data
				np.savetxt('codata/'+obs+'/'+arg_num+'/Flow_'+savename+'_n'+str(agg)+'_b'+str(b)+'.txt',np.array(list(zip(*histo.items()))),fmt='%d')	
			net.Clear_memory()
		#save scalar data
		for i,val in res.items():
			obs = obs_with_arg[i][0]
			arg_num = str(obsarg_to_num[i])
			np.savetxt('codata/'+obs+'/'+arg_num+'/Flow_'+savename+'_n'+str(agg)+'.txt',np.array([list_b,val]))

#observable data are stored in 'codata/obs/arg_num/Flow_'+savename+'_n'+str(agg)+'_b'+str(b)+'.txt'
#compute the flow of NCTN and ECTN similarities observables, that are observables using (n,b) and (n,1)
#to be computed; types of observables:
#('NCTN0sim',depth): cosine similarity of NCTN btw (n,b) and (n,1)
#(note: why not btw (n,b) and (1,b) or (1,1)?)
#('NCTN0sim_trunc',depth): truncated cosine similarity
#obs_with_arg = list containing tuples (obs,arg)
def Vector_sim_nb_flow(name,list_b,list_agg):
	obs_with_arg = []; list_obj = []
	for depth in [2,3]:
		for el in ['NCTN','ECTN']:
			list_obj.append((el,depth))
			for suffix in ['','_trunc']:
				obs_with_arg.append((el+'0'+'sim'+suffix,depth))
	if list_agg==0 or list_agg==[]:
		return None
	if type(list_agg)==int:
		list_agg = [list_agg]
	elif type(list_agg)!=list:
		raise ValueError('list_agg should be either an integer or a list of integers')
	#check the folder arborescence is correct
	obsarg_to_num = Check_folder_flow(obs_with_arg)
	obs_to_i = {obs:i for i,obs in enumerate(obs_with_arg)}
	savename = Get_savename(name)
	net = tp.Temp_net(tp.Load_TN(name))
	net.Init(1)
	for agg in list_agg:
		print('level '+str(agg)+' begins')
		res = {i:[] for i in range(len(obs_with_arg))}
		#initialize the external observables
		ext_memory = {}
		for el,depth in list_obj:
			if el=='NCTN':
				ext_memory[(el,depth)] = self.ETN_histo(depth)
			elif el=='ECTN':
				ext_memory[(el,depth)] = self.EdgeTN_histo(depth)
		for b in list_b:
			print(str(b)+' begins')
			net.Local_TS(b,agg=agg)
			for el,depth in list_obj:
				if el=='NCTN':
					new_histo = self.ETN_histo(depth)
				elif el=='ECTN':
					new_histo = self.EdgeTN_histo(depth)
				for suffix,threshold in zip(['','_trunc'],[None,20]):
					i = obs_to_i[(el+'0'+'sim'+suffix,depth)]
					res[i].append(Cosim(ext_memory[(el,depth)],new_histo,trunc=threshold))
		#save scalar data
		for i,val in res.items():
			obs = obs_with_arg[i][0]
			arg_num = str(obsarg_to_num[i])
			np.savetxt('codata/'+obs+'/'+arg_num+'/Flow_'+savename+'_n'+str(agg)+'.txt',np.array([list_b,val]))

#return either <x> or <x**2>/<x>
def Histo_to_scalar(histo,scalar):
	if scalar=='avg':
		func = lambda X,Y:np.sum(X*Y)
	elif scalar=='frac':
		func = lambda X,Y:np.sum(X**2*Y)/np.sum(X*Y)
	else:
		raise ValueError('scalar should be either avg or frac')
	if len(histo)==1:
		if 0.1 in histo:
			return -1
		return func(list(histo.keys())[0],1)
	else:
		X,Y = zip(*histo.items())
		X = np.array(X); Y = np.array(Y)
		norm = np.sum(Y)
		return func(X,Y/norm)

#extract scalar info from distr obs, namely <x> and <x**2>/<x>, named (obs0avg,arg) and (obs0frac,arg)
def Scalar_from_distr(name,obs,list_b,list_agg):
	savename = Get_savename(name)
	#nb of the folder where the realization of the distr obs is stored
	num = Find_folder(*obs)
	#compute the scalar obs: (obs0avg,arg) for <x> and (obs0frac,arg) for <x**2>/<x>
	#then save their realizations
	list_scalar = ['avg','frac']
	obs_with_arg = [(obs[0]+'0'+scalar,obs[1]) for scalar in list_scalar]
	obsarg_to_num = Check_folder_flow(obs_with_arg)
	for agg in list_agg:
		res = {i:[] for i in range(len(list_scalar))}
		for b in list_b:
			tab = np.loadtxt('codata/'+obs[0]+'/'+str(num)+'/Flow_'+savename+'_n'+str(agg)+'_b'+str(b)+'.txt',dtype=float)
			if tab.ndim==2:
				data = {tab[0,i]:tab[1,i] for i in range(np.size(tab,1))}
			elif len(tab)==0:
				data = {0.1:1}
			else:
				data = {tab[0]:tab[1]}
			for i in res.keys():
				res[i].append(Histo_to_scalar(data,list_scalar[i]))
		#save scalar obs
		for i,val in res.items():
			new_obs = obs_with_arg[i][0]
			arg_num = str(obsarg_to_num[i])
			np.savetxt('codata/'+new_obs+'/'+arg_num+'/Flow_'+savename+'_n'+str(agg)+'.txt',np.array([list_b,val]))

#display the results of Internalobs_nb_flow for a scalar observable
def Plot_scalar_nb_flow(name,obs,list_b,list_agg):
	if 30 in list_agg:
		whole_agg = [5,10,20,30,50,100]
	else:
		whole_agg = [5,10,20,50,100]
	early_agg = [1,2,3,4,5]
	list_marker = ['s','^','<','>','*','v','1','2','3','4','P','p']
	list_color = ['blue','green','red','black','brown','purple','gray']
	savename = Get_savename(name)
	#nb of the folder where the realization of obs is stored
	num = Find_folder(*obs)
	savepath = 'figures/'+obs[0]+'/'+str(num)+'/'+savename
	if not os.path.isdir('figures/'+obs[0]):
		os.mkdir('figures/'+obs[0])
	if not os.path.isdir('figures/'+obs[0]+'/'+str(num)):
		os.mkdir('figures/'+obs[0]+'/'+str(num))
	if not os.path.isdir(savepath):
		os.mkdir(savepath)
	#load the data (obs realization)
	data = {}
	for agg in list_agg:
		data[agg] = np.loadtxt('codata/'+obs[0]+'/'+str(num)+'/Flow_'+savename+'_n'+str(agg)+'.txt')
	#display the results
	#first plot: realization vs b for several values of n
	for chosen_agg,savefig in zip([early_agg,whole_agg],['early','whole']):
		fig,ax = Setup_Plot(r'$b$',obs[0],fontsize=14)
		for agg,marker,color in zip(chosen_agg,list_marker,list_color):
			val = data[agg]
			ax.plot(val[0,:],val[1,:],marker,color=color,markersize=7,label=r'$n = $'+str(agg))
		ax.legend(fontsize=14,ncol=min(3,len(chosen_agg)),bbox_to_anchor=(0.5,1.23),loc="upper center")
		plt.savefig(savepath+'/'+savefig+'_vs_b.png')
		plt.close('all')

#exchange the 1 and 2 in L
def Sym1_2(L):
	new_L = []
	for i in L:
		if i==1:
			j = 2
		elif i==2:
			j = 1
		else:
			j = i
		new_L.append(j)
	return new_L

#display the results of Internalobs_nb_flow for a motif profile observable
def Plot_ECTNprofile_nb_flow(name,obs,list_b,list_agg):
	list_marker = ['s','^','<','>','*','v','1','2','3','4','P','p']
	list_color = ['blue','green','red','black','brown','purple','gray','orange','yellow','pink','cyan']
	savename = Get_savename(name)
	#nb of the folder where the realization of obs is stored
	num = Find_folder(*obs)
	savepath = 'figures/'+obs[0]+'/'+str(num)+'/'+savename
	if not os.path.isdir(savepath):
		os.mkdir(savepath)
	#determine the appropriate way of displaying the observable distribution
	xlabel = r'$b$'; ylabel = 'abundancy'
	profile_to_int = tp.All_ECTN_profiles(obs[1][0])[0]
	int_to_profile = {val:key for key,val in profile_to_int.items()}
	#use symmetries in profile abundancies to reduce the number of labels
	#representative of equivalence classes of profiles
	class_rep = {}
	for profile in profile_to_int.keys():
		L = [int(letter) for letter in profile]
		#1/2 and TR symmetries
		dic_nb = {}; list_func = [lambda L:L,lambda L:L[::-1],Sym1_2]
		for func1 in list_func:
			for func2 in list_func:
				new_L = func2(func1(L))
				dic_nb[sum([i*4**ind for ind,i in enumerate(new_L)])] = new_L.copy()
		min_nb = min(dic_nb.keys())
		rep = ''.join([str(i) for i in dic_nb[min_nb]])
		if rep in class_rep:
			class_rep[rep].add(profile)
		else:
			class_rep[rep] = {profile}
	profile_to_rep = {}
	for rep,val in class_rep.items():
		for profile in val:
			profile_to_rep[profile] = rep
	#load the data (obs realization)
	data = {rep:{agg:{} for agg in list_agg} for rep in class_rep.keys()}
	for agg in list_agg:
		for b in list_b:
			tab = np.loadtxt('codata/'+obs[0]+'/'+str(num)+'/Flow_'+savename+'_n'+str(agg)+'_b'+str(b)+'.txt',dtype=int)
			for ind in range(np.size(tab,1)):
				i = tab[0,ind]
				if i in int_to_profile:
					rep = profile_to_rep[int_to_profile[i]]
					if b in data[rep][agg]:
						data[rep][agg][b] += tab[1,ind]/len(class_rep[rep])
					else:
						data[rep][agg][b] = tab[1,ind]/len(class_rep[rep])
	#separate the frequent from rare profiles and identify the ghost profiles
	#choose the threshold so that we have as many frequent as rare observables
	#among the non ghost profiles
	rep_type = {agg:{'rare':set(()),'frequent':set(())} for agg in list_agg}
	#rep_abundancy[agg][rep] = abundancy of rep at level agg
	rep_abundancy = {agg:{} for agg in list_agg}
	#ghost_rep gives the profiles that never appear
	ghost_rep = {agg:set(()) for agg in list_agg}
	for rep,val in data.items():
		for agg,dic_ in val.items():
			m = max(list(dic_.values()))
			if m==0:
				ghost_rep[agg].add(rep)
			else:
				rep_abundancy[agg][rep] = m
	#dic_threshold[agg] = threshold for the level agg
	dic_threshold = {}
	for agg,dic_ in rep_abundancy.items():
		dic_threshold[agg] = np.percentile(list(dic_.values()),[100/3,50,200/3])
		for rep,m in dic_.items():
			if m<=dic_threshold[agg][1]:
				rep_type[agg]['rare'].add(rep)
			else:
				rep_type[agg]['frequent'].add(rep)
	redo = False
	for agg in list_agg:
		for set_rep in rep_type[agg].values():
			if len(set_rep)>len(list_color):
				redo = True
	if redo:
		rep_type = {agg:{'rare':set(()),'medium':set(()),'frequent':set(())} for agg in list_agg}
		for agg,dic_ in rep_abundancy.items():
			for rep,m in dic_.items():
				if m<=dic_threshold[agg][0]:
					rep_type[agg]['rare'].add(rep)
				elif m<=dic_threshold[agg][2]:
					rep_type[agg]['medium'].add(rep)
				else:
					rep_type[agg]['frequent'].add(rep)
	#save the ghost representatives
	ghost_rep = {agg:''.join(sorted(list(val))) for agg,val in ghost_rep.items()}
	max_len = max([len(el) for el in ghost_rep.values()])
	for agg,val in ghost_rep.items():
		ghost_rep[agg] += '4'*(max_len-len(val))
	np.savetxt(savepath+'/ghost_rep.txt',np.array(list(zip(*ghost_rep.items())),dtype=str).T,fmt='%s')
	#display the results
	#first family of plots: realization of each profile vs b on the same plot for several values of n
	for agg in list_agg:
		for savefig,set_rep in rep_type[agg].items():
			fig,ax = Setup_Plot(xlabel,ylabel,fontsize=14)
			for rep,marker,color in zip(list(set_rep),list_marker,list_color):
				ax.plot(*zip(*data[rep][agg].items()),marker,color=color,markersize=7,label=rep)
			ax.legend(fontsize=14,ncol=5,bbox_to_anchor=(0.5,1.23),loc="upper center")
			plt.savefig(savepath+'/n'+str(agg)+savefig+'.png')
			plt.close('all')
	return None

#display the results of Internalobs_nb_flow for the NCTN motif profile observable
def Plot_NCTNprofile_nb_flow(name,obs,list_b,list_agg):
	list_marker = ['s','^','<','>','*','v','1','2','3','4','P','p']
	list_color = ['blue','green','red','black','brown','purple','gray']
	savename = Get_savename(name)
	#nb of the folder where the realization of obs is stored
	num = Find_folder(*obs)
	savepath = 'figures/'+obs[0]+'/'+str(num)+'/'+savename
	if not os.path.isdir(savepath):
		os.mkdir(savepath)
	#determine the appropriate way of displaying the observable distribution
	xlabel = r'$b$'; ylabel = 'abundancy'
	dic_ = tp.All_NCTN_profiles(obs[1][0])
	dic_label = {val:key for key,val in dic_.items()}
	#load the data (obs realization)
	data = {i:{agg:{} for agg in list_agg} for i in dic_label.keys()}
	for agg in list_agg:
		for b in list_b:
			tab = np.loadtxt('codata/'+obs[0]+'/'+str(num)+'/Flow_'+savename+'_n'+str(agg)+'_b'+str(b)+'.txt',dtype=int)
			for ind in range(np.size(tab,1)):
				data[tab[0,ind]][agg][b] = tab[1,ind]
	#display the results
	#first family of plots: realization of each profile vs b on the same plot for several values of n
	for agg in list_agg:
		fig,ax = Setup_Plot(xlabel,ylabel,fontsize=14)
		for (i,label),marker,color in zip(dic_label.items(),list_marker,list_color):
			ax.plot(*zip(*data[i][agg].items()),marker,color=color,markersize=7,label=label)
		ax.legend(fontsize=14,ncol=4,bbox_to_anchor=(0.5,1.23),loc="upper center")
		plt.savefig(savepath+'/n'+str(agg)+'.png')
		plt.close('all')
	return None

#display the results of Internalobs_nb_flow for a distribution observable which is not a motif profile
def Plot_distr_nb_flow(name,obs,list_b,list_agg,nb_visub=6):
	if 'profile' in obs[0]:
		if 'NCTN' in obs[0]:
			return Plot_NCTNprofile_nb_flow(name,obs,list_b,list_agg)
		elif 'ECTN' in obs[0]:
			return Plot_ECTNprofile_nb_flow(name,obs,list_b,list_agg)
	if 30 in list_agg:
		whole_agg = [5,10,20,30,50,100]
	else:
		whole_agg = [5,10,20,50,100]
	early_agg = [1,2,3,4,5]
	list_marker = ['s','^','<','>','*','v','1','2','3','4','P','p']
	list_color = ['blue','green','red','black','brown','purple','gray']
	savename = Get_savename(name)
	#nb of the folder where the realization of obs is stored
	num = Find_folder(*obs)
	savepath = 'figures/'+obs[0]+'/'+str(num)+'/'+savename
	if not os.path.isdir(savepath):
		os.mkdir(savepath)
	#determine the appropriate way of displaying the observable distribution
	xlabel,ylabel,func = Obs_to_labels(obs[0])
	#load the data (obs realization)
	chosen_ind = [(k*(len(list_b)-1))//(nb_visub-1) for k in range(nb_visub)]
	chosen_b = [list_b[ind] for ind in chosen_ind]
	data = {}
	for agg in list_agg:
		data[agg] = {}
		for b in chosen_b:
			tab = np.loadtxt('codata/'+obs[0]+'/'+str(num)+'/Flow_'+savename+'_n'+str(agg)+'_b'+str(b)+'.txt',dtype=int)
			if tab.ndim==2:
				data[agg][b] = {tab[0,i]:tab[1,i] for i in range(np.size(tab,1))}
			elif len(tab)==0:
				#no realization were observed so to keep track of this info, we put an impossible realization
				#obs less than 1, which will result in a negative value in the log plane
				data[agg][b] = {0.1:1}
			else:
				data[agg][b] = {tab[0]:tab[1]}
	#display the results
	#first family of plots: realization for several b on the same plot for several values of n
	for agg in list_agg:
		fig,ax = Setup_Plot(xlabel,ylabel,fontsize=14)
		for b,marker,color in zip(chosen_b,list_marker,list_color):
			ax.plot(*func(data[agg][b]),marker,color=color,markersize=7,label=r'$b = $'+str(b))
		ax.legend(fontsize=14,ncol=min(3,len(chosen_b)),bbox_to_anchor=(0.5,1.23),loc="upper center")
		plt.savefig(savepath+'/n'+str(agg)+'.png')
		plt.close('all')
	#second family of plots: realization for several n on the same plot for several values of b
	for b in chosen_b:
		for chosen_agg,savefig in zip([early_agg,whole_agg],['early','whole']):
			fig,ax = Setup_Plot(xlabel,ylabel,fontsize=14)
			for agg,marker,color in zip(chosen_agg,list_marker,list_color):
				ax.plot(*func(data[agg][b]),marker,color=color,markersize=7,label=r'$n = $'+str(agg))
			ax.legend(fontsize=14,ncol=min(3,len(chosen_agg)),bbox_to_anchor=(0.5,1.23),loc="upper center")
			plt.savefig(savepath+'/'+savefig+'_b'+str(b)+'.png')
			plt.close('all')
	#third family of plots: KS test outcome vs n and b
	#more precisely we compute KS_0(n,b) = KS btw (n+1,b) and (n,b)
	#and KS_1(n,b) = KS btw (n,b+1) and (n,b) then we visualize the log of the associated p-value matrix
	#this allows to identify the (n,b) at which the observable changes the most
	pass

#compare the distribution obs_type (duration, interduration, etc.) of motifs with different abundancies
#motif_type = ECTN or NCTN; we visualize 6 motifs
def Comp_motif_abundancy(obs_type,motif_type,name,agg=1,b=1,nb_visu=6):
	list_marker = ['s','^','<','>','*','v','1','2','3','4','P','p']
	list_color = ['blue','green','red','black','brown','purple','gray']
	savename = Get_savename(name)
	obs = (motif_type+'0'+obs_type,3)
	#nb of the folder where the realization of obs is stored
	num = Find_folder(*obs)
	savepath = 'figures/'+obs[0]+'/'+str(num)+'/'+savename
	if not os.path.isdir(savepath):
		os.mkdir(savepath)
	#determine the appropriate way of displaying the observable distribution
	xlabel,ylabel,func = Obs_to_labels(obs[0])
	#load the data (obs realization)
	chosen_ind = [(k*(len(list_b)-1))//(nb_visub-1) for k in range(nb_visub)]
	data = {}
	pass
	for file in folder:
		pass
		tab = np.loadtxt('codata/'+obs[0]+'/'+str(num)+'/Flow_'+savename+'_n'+str(agg)+'_b'+str(b)+'.txt',dtype=int)
		if tab.ndim==2:
			data[agg][b] = {tab[0,i]:tab[1,i] for i in range(np.size(tab,1))}
		elif len(tab)==0:
			#no realization was observed so to keep track of this info, we put an impossible realization
			#obs less than 1, which will result in a negative value in the log plane
			data[agg][b] = {0.1:1}
		else:
			data[agg][b] = {tab[0]:tab[1]}
	pass

#load the data corresponding to a scalar observable vs b for a given level of aggregation
def Load_vs_b(name,agg,obs):
	obs = Rewrite_obs(obs)
	return np.loadtxt('codata/'+obs[0]+'/'+str(Find_folder(*obs))+'/Flow_'+Get_savename(name)+'_n'+str(agg)+'.txt')

#return the symbolic description of a curve X,Y = tab[0,:],tab[1,:]
def Tab_to_string(tab):
	if tab.ndim==2:
		Y = tab[1,:]
	else:
		Y = tab
	der_tab = Y[1:]-Y[:-1]
	#check for the hypothesis of Gaussian white noise (in this case the curve is described as flat)
	if shapiro(der_tab)[1]>0.05:
		return '0'*len(der_tab)
	digit_to_letter = {-1:'-',1:'+'}
	sign_der = np.sign(der_tab)
	#a domain is defined as a connected component of the inverse image of np.sign(der_tab)
	#first identify the domains
	string = ''.join([digit_to_letter[digit] for digit in sign_der])
	#second check the validity of each domain:
	#a domain can either or not absorb its successor, to decide this, we may need to consider
	#the successor of the successor, etc.
	#we seek to extend at most as possible the domain of the current letter
	new_string = ''; current_letter = string[0]; ind = 0
	while ind<len(string):
		if string[ind]==current_letter:
			new_string += current_letter
			ind += 1
		else:
			next_letter = string[ind]; end = ind; ok = True
			while ok:
				if end<len(string):
					if string[end]==next_letter:
						end += 1
					else:
						ok = False
				else:
					ok = False
			#test whether we can absorb the next domain into the current one
			max_ind = min(end+1,len(string))
			if np.sign(der_tab[ind-1])==np.sign(Y[max_ind]-Y[ind-1]):
				new_string += current_letter*(end-ind)
				ind = end
			else:
				new_string += next_letter*(end-ind)
				current_letter = next_letter
				ind = end
	#check the coherence of the extended domains
	domains = []
	letter = new_string[0]; start = 0; end = 0; sign = sign_der[0]
	while end<len(new_string):
		if new_string[end]==letter:
			end += 1
		else:
			domains.append((start,end-1,sign))
			start = end; sign *= -1
			letter = digit_to_letter[sign]
	domains.append((start,end-1,sign))
	new_new_string = ''
	for (start,end,sign) in domains:
		if np.sign(Y[end+1]-Y[max(start-1,0)])!=sign:
			new_new_string += digit_to_letter[-sign]*(end-start+1)
		else:
			new_new_string += digit_to_letter[sign]*(end-start+1)
	return new_new_string

#return the symbolic description of a curve X,Y = tab[0,:],tab[1,:]
def New_string_V2(Y,string):
	digit_to_letter = {-1:'-',1:'+'}
	#we seek to extend at most as possible the domain of the current letter
	new_string = ''; current_letter = string[0]; ind = 0
	while ind<len(string):
		if string[ind]==current_letter:
			new_string += current_letter
			ind += 1
		else:
			next_letter = string[ind]; end = ind; ok = True
			while ok:
				if end<len(string):
					if string[end]==next_letter:
						end += 1
					else:
						ok = False
				else:
					ok = False
			#test whether we can absorb the next domain into the current one
			max_ind = min(end+1,len(string))
			if np.sign(Y[ind]-Y[ind-1])==np.sign(Y[max_ind]-Y[ind-1]):
				new_string += current_letter*(end-ind)
				ind = end
			else:
				new_string += next_letter*(end-ind)
				current_letter = next_letter
				ind = end
	#check the coherence of the extended domains
	domains = []
	letter = new_string[0]; start = 0; end = 0; sign = np.sign(Y[1]-Y[0])
	while end<len(new_string):
		if new_string[end]==letter:
			end += 1
		else:
			domains.append((start,end-1,sign))
			start = end; sign *= -1
			letter = digit_to_letter[sign]
	domains.append((start,end-1,sign))
	new_new_string = ''
	for (start,end,sign) in domains:
		if np.sign(Y[end+1]-Y[max(start-1,0)])!=sign:
			new_new_string += digit_to_letter[-sign]*(end-start+1)
		else:
			new_new_string += digit_to_letter[sign]*(end-start+1)
	return new_new_string

def New_string_V3(Y,string):
	digit_to_letter = {-1:'-',1:'+'}
	#we seek to extend at most as possible the domain of the current letter: what is the cost of absorbing
	#or keeping a domain?
	#the question is at which point a deviation within an existing domain becomes significant?
	domains = Find_domains(string)
	#for each domain
	start,end,sign = domains[0]
	next_dom = domains[1]
	exit()
	new_string = ''; current_letter = string[0]; ind = 0
	while ind<len(string):
		if string[ind]==current_letter:
			new_string += current_letter
			ind += 1
		else:
			next_letter = string[ind]; end = ind; ok = True
			while ok:
				if end<len(string):
					if string[end]==next_letter:
						end += 1
					else:
						ok = False
				else:
					ok = False
			#test whether we can absorb the next domain into the current one
			max_ind = min(end+1,len(string))
			if np.sign(Y[ind]-Y[ind-1])==np.sign(Y[max_ind]-Y[ind-1]):
				new_string += current_letter*(end-ind)
				ind = end
			else:
				new_string += next_letter*(end-ind)
				current_letter = next_letter
				ind = end
	#check the coherence of the extended domains
	domains = []
	letter = new_string[0]; start = 0; end = 0; sign = np.sign(Y[1]-Y[0])
	while end<len(new_string):
		if new_string[end]==letter:
			end += 1
		else:
			domains.append((start,end-1,sign))
			start = end; sign *= -1
			letter = digit_to_letter[sign]
	domains.append((start,end-1,sign))
	new_new_string = ''
	for (start,end,sign) in domains:
		if np.sign(Y[end+1]-Y[max(start-1,0)])!=sign:
			new_new_string += digit_to_letter[-sign]*(end-start+1)
		else:
			new_new_string += digit_to_letter[sign]*(end-start+1)
	return new_new_string

def Tab_to_string_V2(tab):
	if tab.ndim==2:
		Y = tab[1,:]
	else:
		Y = tab
	der_tab = Y[1:]-Y[:-1]
	#first check for monotonicity
	sign_der = np.sign(der_tab)
	if (sign_der<0).all():
		return '-'*len(der_tab)
	elif (sign_der>0).all():
		return '+'*len(der_tab)
	elif (sign_der==0).all():
		return '0'*len(der_tab)
	#check for the hypothesis of Gaussian white noise (in this case the curve is described as flat)
	if shapiro(der_tab)[1]>0.05:
		return '0'*len(der_tab)
	digit_to_letter = {-1:'-',1:'+'}
	first_string = ''.join([digit_to_letter[digit] for digit in sign_der])
	ok = True
	while ok:
		string = New_string_V2(Y,first_string)
		if string==first_string:
			ok = False
		else:
			first_string = string
	return string

def Find_domains(string):
	digit_to_letter = {-1:'-',1:'+'}
	domains = []; letter = string[0]; start = 0; end = 0
	if letter=='-':
		sign = -1
	elif letter=='+':
		sign = 1
	while end<len(string):
		if string[end]==letter:
			end += 1
		else:
			domains.append((start,end-1,sign))
			start = end; sign *= -1
			letter = digit_to_letter[sign]
	domains.append((start,end-1,sign))
	return domains

def Check_zero(Y,string):
	#identify the sensible borders, near which the slope is potentially zero: they are the transitions
	#'-+' or '+-' which are included in a '-' domain of the absolute value of the curve derivative
	der_string = Tab_to_string_V2(abs(Y[1:]-Y[:-1]))
	if not '-' in der_string:
		return string
	new_string = ''; string_with_zeros = string
	domains = Find_domains(string)
	for (start,end,sign) in domains:
		#check wether the border is sensible
		values = {der_string[max(end-2,0)],der_string[end-1],der_string[min(end,len(der_string)-1)]}
		if '-' in values and end<len(string)-1:
			#the border is sensible:
			#start from the border and extend symetrically until reaching new borders
			#if doing so, we encounter no zero domain, we keep the border
			#if we encounter a zero domain, we extend it at most as possible
			if end==0:
				left = 0; right = min(2,len(string)-1)
			else:
				left = end-1; right = min(end+1,len(string)-1)
			ok = True; zero_begin = False; extend_left = True; extend_right = True
			left_zero = end; right_zero = end
			while ok:
				if extend_left:
					if left==0:
						extend_left = False
					else:
						left -= 1
					if shapiro(Y[left:right+1])[1]>0.05:
						left_zero = left
						if not zero_begin:
							zero_begin = True
					else:
						if zero_begin:
							extend_left = False
				if extend_right:
					if right==len(string)-1:
						extend_right = False
					else:
						right += 1
					if shapiro(Y[left:right+1])[1]>0.05:
						right_zero = right
						if not zero_begin:
							zero_begin = True
					else:
						if zero_begin:
							extend_right = False
				if not extend_left and not extend_right:
					ok = False
			if zero_begin:
				string_with_zeros = string_with_zeros[:left_zero]+'0'*(right_zero-left_zero+1)+string_with_zeros[right_zero+1:]
	#build the new string: read string_with_zeros assuming a domain absorbs the '0' at its right border
	first_letter = string_with_zeros[0]
	for letter in string_with_zeros:
		if letter==first_letter:
			new_string += letter
		elif letter=='0':
			new_string += first_letter
		else:
			first_letter = letter
			new_string += letter
	return new_string

def Tab_to_string_V3(tab):
	if tab.ndim==2:
		Y = tab[1,:]
	else:
		Y = tab
	der_tab = Y[1:]-Y[:-1]
	#first check for monotonicity
	sign_der = np.sign(der_tab)
	if (sign_der<0).all():
		return '-'*len(der_tab)
	elif (sign_der>0).all():
		return '+'*len(der_tab)
	elif (sign_der==0).all():
		return '0'*len(der_tab)
	#check for the hypothesis of Gaussian white noise (in this case the curve is described as flat)
	if shapiro(der_tab)[1]>0.05 and shapiro(Y)[1]>0.05:
		#check that the tab_derivative is centered around zero
		nb_test = 50; nb_match = 0
		for _ in range(nb_test):
			nb_match += int(ks_2samp(der_tab/np.std(der_tab),np.random.normal(loc=0,scale=1,size=len(der_tab)))[1]>0.05)
		if nb_match>nb_test/2:
			return '0'*len(der_tab)
	digit_to_letter = {-1:'-',1:'+'}
	first_string = ''.join([digit_to_letter[digit] for digit in sign_der])
	ok = True
	while ok:
		string = New_string_V2(Y,first_string)
		if string==first_string:
			ok = False
		else:
			first_string = string
	#check whether the '-+' or '+-' borders are not part of a zero domain
	return Check_zero(Y,string)

def Tab_to_string_V4(tab):
	if tab.ndim==2:
		Y = tab[1,:]
	else:
		Y = tab
	der_tab = Y[1:]-Y[:-1]
	#first check for monotonicity
	sign_der = np.sign(der_tab)
	if (sign_der<0).all():
		return '-'*len(der_tab)
	elif (sign_der>0).all():
		return '+'*len(der_tab)
	elif (sign_der==0).all():
		return '0'*len(der_tab)
	#check for the hypothesis of Gaussian white noise (in this case the curve is described as flat)
	if shapiro(der_tab)[1]>0.05 and shapiro(Y)[1]>0.05:
		#check that the tab_derivative is centered around zero
		nb_test = 50; nb_match = 0
		for _ in range(nb_test):
			nb_match += int(ks_2samp(der_tab/np.std(der_tab),np.random.normal(loc=0,scale=1,size=len(der_tab)))[1]>0.05)
		if nb_match>nb_test/2:
			return '0'*len(der_tab)
	digit_to_letter = {-1:'-',1:'+'}
	first_string = ''.join([digit_to_letter[digit] for digit in sign_der])
	ok = True
	while ok:
		string = New_string_V3(Y,first_string)
		if string==first_string:
			ok = False
		else:
			first_string = string
	#check whether the '-+' or '+-' borders are not part of a zero domain
	return Check_zero(Y,string)

#subroutine of Stoch_iter_string
def Func_stochiter(Y,first_string,string,ind):
	#collect the motifs of the form '-+-' and '+-+': we collect the position of the deviation
	letter = first_string[ind]
	if letter=='+':
		if first_string[ind-1]=='-' and first_string[ind+1]=='-':
			if Y[ind+2]<Y[ind] or Y[ind+1]<Y[ind-1]:
				string[ind] = '-'
	elif letter=='-':
		if first_string[ind-1]=='+' and first_string[ind+1]=='+':
			if Y[ind+2]>Y[ind] or Y[ind+1]>Y[ind-1]:
				string[ind] = '+'
	return string

def Stoch_iter_string(Y,first_string,start_pos):
	string = list(first_string)
	left_border = start_pos; right_border = start_pos
	while left_border>0 and right_border<len(first_string)-1:
		#we extend by the left of right with equal probability if both extensions are possible
		if rd.random()<0.5:
			string = Func_stochiter(Y,first_string,string,left_border)
			left_border -= 1
		else:
			string = Func_stochiter(Y,first_string,string,right_border)
			right_border += 1
	for ind in range(left_border,0,-1):
		string = Func_stochiter(Y,first_string,string,ind)
	for ind in range(right_border,len(first_string)-1):
		string = Func_stochiter(Y,first_string,string,ind)
	return ''.join(string)

def Iter_string(Y,first_string):
	string = list(first_string)
	#collect the motifs of the form '-+-' and '+-+': we collect the position of the deviation
	for ind in range(1,len(first_string)-1):
		letter = first_string[ind]
		if letter=='+':
			if first_string[ind-1]=='-' and first_string[ind+1]=='-':
				if Y[ind+2]<Y[ind] or Y[ind+1]<Y[ind-1]:
					string[ind] = '-'
		elif letter=='-':
			if first_string[ind-1]=='+' and first_string[ind+1]=='+':
				if Y[ind+2]>Y[ind] or Y[ind+1]>Y[ind-1]:
					string[ind] = '+'
	return ''.join(string)

#return the deviation values in a domain
def Get_dev(Y,domain):
	res = []
	len_dom = domain[1]-domain[0]+1
	for i in range(len_dom):
		ind = domain[0]+i
		if (Y[ind+1]-Y[ind])*domain[2]<0:
			res.append(abs(Y[ind+1]-Y[ind]))
	return res

def Iter_string2(Y,first_string):
	string = list(first_string)
	#collect the motif domain '---+++---' (up) or '+++---+++' (down) and try to absorb the central droplet
	#begin by the smallest droplets
	domains = Find_domains(first_string)
	nb_dom = len(domains)
	#we identify a motif by the position of the droplet into the list of domains 
	motifs = []
	for ind in range(1,nb_dom-1):
		motifs.append((ind,domains[ind][2]))
	#sort the motifs by increasing droplet size
	motifs.sort(key=lambda el:domains[el[0]][1]-domains[el[0]][0]+1)
	#we collect all the accepted deviations to deduce an acceptance criterion for a new deviation
	list_dev = []
	for domain in domains:
		list_dev += Get_dev(Y,domain)
	dev_max = max(list_dev)
	#try to absorb the smallest droplet, if not, try to absorb the next smallest droplet, etc.
	num = 0
	while num<len(motifs):
		motif = motifs[num]; sign = motif[1]
		#acceptance criterion: the new deviations are less than the maximum already accepted ones
		merged_domain = (domains[motif[0]-1][0],domains[motif[0]+1][1],-sign)
		new_list_dev = Get_dev(Y,merged_domain)
		new_max = max(new_list_dev)
		if new_max<=dev_max:
			#absorb the droplet
			if sign<0:
				new_letter = '+'
			elif sign>0:
				new_letter = '-'
			for ind in range(domains[motif[0]][0],domains[motif[0]][1]+1):
				string[ind] = new_letter
			num = len(motifs)
		else:
			num += 1
	return ''.join(string)

#take a domain and count the nb of times the order relation is preserved btw the elements
#the merging with another domain must preserve this order relation, so e.g. we can use Kendall similarity
#to quantify how much the order relation has been modified by the absorption of a new domain
#to quantify the level of order we use the Kendall similarity btw the domain and its perfectly ordered
#version
def Iter_string2_V2(Y,first_string):
	string = list(first_string)
	#collect the motif domain '---+++---' (up) or '+++---+++' (down) and try to absorb the central droplet
	#begin by the smallest droplets
	domains = Find_domains(first_string)
	nb_dom = len(domains)
	#we identify a motif by the position of the droplet into the list of domains 
	motifs = []
	for ind in range(1,nb_dom-1):
		motifs.append((ind,domains[ind][2]))
	#sort the motifs by increasing droplet size
	motifs.sort(key=lambda el:domains[el[0]][1]-domains[el[0]][0]+1)
	'''
	#Kend_dom[ind] = Kendall similarity btw domain of index ind in domains and its perfectly ordered version
	Kend_dom = []
	for (start,end,sign) in domains:
		if sign>0:
			Kend_dom.append(kendalltau(Y[start:end+2],range(start,end+2),alternative='greater'))
		elif sign<0:
			Kend_dom.append(kendalltau(Y[start:end+2],range(end+1,start-1,-1),alternative='greater'))
	'''
	#try to absorb the smallest droplet, if not, try to absorb the next smallest droplet, etc.
	num = 0
	while num<len(motifs):
		motif = motifs[num]; sign = motif[1]; start = domains[motif[0]-1][0]; end = domains[motif[0]+1][1]
		#acceptance criterion: absorbing the droplet preserves the global order
		#merged_domain = (start,end,-sign)
		if sign<0:
			p_value = kendalltau(Y[start:end+2],range(start,end+2),alternative='greater')[1]
		elif sign>0:
			p_value = kendalltau(Y[start:end+2],range(end+1,start-1,-1),alternative='greater')[1]
		if p_value<=0.05:
			#absorb the droplet
			if sign<0:
				new_letter = '+'
			elif sign>0:
				new_letter = '-'
			for ind in range(domains[motif[0]][0],domains[motif[0]][1]+1):
				string[ind] = new_letter
			num = len(motifs)
		else:
			num += 1
	return ''.join(string)

def Get_droplet(Y,invert_pos=None):
	sign_der = np.sign(Y[1:]-Y[:-1])
	#invert the bits at positions invert_pos and len(Y)-invert_pos-2
	if invert_pos is not None:
		sign_der[invert_pos] *= -1
		sign_der[len(Y)-invert_pos-2] *= -1
	#absorb the zeros by the domain at its left
	digit_to_letter = {-1:'-',1:'+'}; string = ''; letter = ''
	for digit in sign_der:
		if digit==0:
			string += letter
		else:
			letter = digit_to_letter[digit]
			string += letter
	keep = True
	while keep:
		new_string = Iter_string(Y,string)
		if new_string==string:
			keep = False
		else:
			string = new_string
	#the behaviour cannot change at the last point
	if string[-1]!=string[-2]:
		string = string[:-1]+string[-2]
	keep = True
	while keep:
		new_string = Iter_string2_V2(Y,string)
		if new_string==string:
			keep = False
		else:
			string = new_string
	return string

def Tab_to_string_V5(tab):
	if tab.ndim==2:
		Y = tab[1,:]
	else:
		Y = tab
	der_tab = Y[1:]-Y[:-1]
	#first check for monotonicity
	sign_der = np.sign(der_tab)
	if (sign_der<0).all():
		return '-'*len(der_tab)
	elif (sign_der>0).all():
		return '+'*len(der_tab)
	elif (sign_der==0).all():
		return '0'*len(der_tab)
	#if the derivative has accidental zeros, remove them
	pos_zeros = []
	for ind,sign in enumerate(sign_der):
		if sign==0:
			pos_zeros.append(ind)
	if pos_zeros:
		#add a random noise on data points with amplitude equal to the minimum of the non-zero absolute value
		#of the raw derivative
		m = max(min([abs(el) for el in der_tab if el!=0]),1e-10)
		for ind in pos_zeros:
			Y[ind] += (2*rd.random()-1)*m
		der_tab = Y[1:]-Y[:-1]; sign_der = np.sign(der_tab)
	digit_to_letter = {-1:'-',1:'+'}
	string = ''.join([digit_to_letter[digit] for digit in sign_der])
	keep = True
	while keep:
		new_string = Iter_string(Y,string)
		if new_string==string:
			keep = False
		else:
			string = new_string
	#the behaviour cannot change at the last point
	if string[-1]!=string[-2]:
		string = string[:-1]+string[-2]
	keep = True
	while keep:
		new_string = Iter_string2_V2(Y,string)
		if new_string==string:
			keep = False
		else:
			string = new_string
	#if there is only one domain, no further processing is needed
	if len(Find_domains(string))==1:
		return string
	#take care of the curve limits: we assume that the domain at the left is correct
	#(because here the x left limit is actually the inf of the domain of definition of the curve)
	#only the domain at the right may be wrong: to test this, we extend the curve by gluing it with its
	#image under symmetry wrt Oy axis:
	Y_extended = np.concatenate((Y,Y[::-1]))
	#now the right limit domain has become a droplet so we can apply the same analysis as before:
	string_extended = Get_droplet(Y_extended)
	#if the same letter is attributed to the right domain and its full image, then it should be absorbed
	#however if the right domain is extended below its length (new_length<2*length) then it is confirmed
	#length of the right domain in the original string
	left_length = 0
	while string[-1-left_length]==string[-1]:
		left_length += 1
	#length of the image of the right domain
	right_length = 0; keep = True 
	while keep:
		if string_extended[len(string)+right_length]==string[-1]:
			right_length += 1
			if right_length==len(string):
				keep = False
		else:
			keep = False
	if right_length>=left_length:
		#absorb the right domain
		letter = string[-1]; ind = len(string)-1
		while string[ind]==letter:
			ind -= 1
		if letter=='-':
			new_letter = '+'
		elif letter=='+':
			new_letter = '-'
		return string[:ind+1]+new_letter*(len(string)-ind-1)
	return string

#same as Tab_to_string_V5 but inverts the derivative sign at position invert_pos
#before converting into a string
def Tab_to_string_V6(tab,invert_pos,start_string=None):
	if tab.ndim==2:
		Y = tab[1,:]
	else:
		Y = tab
	der_tab = Y[1:]-Y[:-1]
	#first check for monotonicity
	sign_der = np.sign(der_tab)
	if (sign_der<0).all():
		return '-'*len(der_tab)
	elif (sign_der>0).all():
		return '+'*len(der_tab)
	elif (sign_der==0).all():
		return '0'*len(der_tab)
	#if the derivative has accidental zeros, remove them
	pos_zeros = []
	for ind,sign in enumerate(sign_der):
		if sign==0:
			pos_zeros.append(ind)
	if pos_zeros:
		#add a random noise on data points with amplitude equal to the minimum of the non-zero absolute value
		#of the raw derivative
		m = max(min([abs(el) for el in der_tab if el!=0]),1e-10)
		for ind in pos_zeros:
			Y[ind] += (2*rd.random()-1)*m
		sign_der = np.sign(Y[1:]-Y[:-1])
	digit_to_letter = {-1:'-',1:'+'}
	if start_string is None:
		#inverts the bit at position ind
		sign_der[invert_pos] *= -1
		string = ''.join([digit_to_letter[digit] for digit in sign_der])
	else:
		#inverts the bit at position ind
		string_list = list(start_string)
		if string_list[ind]=='-':
			string_list[ind] = '+'
		elif string_list[ind]=='+':
			string_list[ind] = '-'
		string = ''.join(string_list)
	keep = True
	while keep:
		new_string = Iter_string(Y,string)
		if new_string==string:
			keep = False
		else:
			string = new_string
	#the behaviour cannot change at the last point
	if string[-1]!=string[-2]:
		string = string[:-1]+string[-2]
	keep = True
	while keep:
		new_string = Iter_string2_V2(Y,string)
		if new_string==string:
			keep = False
		else:
			string = new_string
	#if there is only one domain, no further processing is needed
	if len(Find_domains(string))==1:
		return string
	#take care of the curve limits: we assume that the domain at the left is correct
	#(because here the x left limit is actually the inf of the domain of definition of the curve)
	#only the domain at the right may be wrong: to test this, we extend the curve by gluing it with its
	#image under symmetry wrt Oy axis:
	Y_extended = np.concatenate((Y,Y[::-1]))
	#now the right limit domain has become a droplet so we can apply the same analysis as before:
	string_extended = Get_droplet(Y_extended,invert_pos=invert_pos)
	'''
	print('begin')
	print(string)
	print(string_extended)
	print('end')
	Display_symbol(string_extended,np.array([list(range(len(Y_extended))),Y_extended]))
	'''
	#if the right domain disappears in the extended string or if the same letter is attributed to the
	#right domain and its image, and that image has higher length than the original domain, then
	#the original domain is absorbed
	left_length = 0
	while string[-1-left_length]==string[-1]:
		left_length += 1
	#length of the image of the right domain
	right_length = 0; keep = True 
	while keep:
		if string_extended[len(string)+right_length]==string[-1]:
			right_length += 1
			if right_length==len(string):
				keep = False
		else:
			keep = False
	if right_length>=left_length:
		#absorb the right domain
		letter = string[-1]; ind = len(string)-1
		while string[ind]==letter:
			ind -= 1
		if letter=='-':
			new_letter = '+'
		elif letter=='+':
			new_letter = '-'
		return string[:ind+1]+new_letter*(len(string)-ind-1)
	return string

#subroutine of Tab_to_string_V7
def Sub_V7(Y,string):
	keep = True
	while keep:
		new_string = Iter_string(Y,string)
		if new_string==string:
			keep = False
		else:
			string = new_string
	#the behaviour cannot change at the last point
	if string[-1]!=string[-2]:
		string = string[:-1]+string[-2]
	keep = True
	while keep:
		new_string = Iter_string2_V2(Y,string)
		if new_string==string:
			keep = False
		else:
			string = new_string
	#if there is only one domain, no further processing is needed
	if len(Find_domains(string))==1:
		return string
	#take care of the curve limits: we assume that the domain at the left is correct
	#(because here the x left limit is actually the inf of the domain of definition of the curve)
	#only the domain at the right may be wrong: to test this, we extend the curve by gluing it with its
	#image under symmetry wrt Oy axis:
	Y_extended = np.concatenate((Y,Y[::-1]))
	#now the right limit domain has become a droplet so we can apply the same analysis as before:
	string_extended = Get_droplet(Y_extended)
	#if the same letter is attributed to the right domain and its full image, then it should be absorbed
	#however if the right domain is extended below its length (new_length<2*length) then it is confirmed
	#length of the right domain in the original string
	left_length = 0
	while string[-1-left_length]==string[-1]:
		left_length += 1
	#length of the image of the right domain
	right_length = 0
	while string_extended[len(string)+right_length]==string[-1]:
		right_length += 1
	if right_length>=left_length:
		#absorb the right domain
		letter = string[-1]; ind = len(string)-1
		while string[ind]==letter:
			ind -= 1
		if letter=='-':
			new_letter = '+'
		elif letter=='+':
			new_letter = '-'
		return string[:ind+1]+new_letter*(len(string)-ind-1)
	return string

#apply Tab_to_string_V5 to tab until convergence
def Tab_to_string_V7(tab):
	if tab.ndim==2:
		Y = tab[1,:]
	else:
		Y = tab
	der_tab = Y[1:]-Y[:-1]
	#first check for monotonicity
	sign_der = np.sign(der_tab)
	if (sign_der<0).all():
		return '-'*len(der_tab)
	elif (sign_der>0).all():
		return '+'*len(der_tab)
	elif (sign_der==0).all():
		return '0'*len(der_tab)
	#if the derivative has accidental zeros, remove them
	pos_zeros = []
	for ind,sign in enumerate(sign_der):
		if sign==0:
			pos_zeros.append(ind)
	if pos_zeros:
		#add a random noise on data points with amplitude equal to the minimum of the non-zero absolute value
		#of the raw derivative
		m = max(min([abs(el) for el in der_tab if el!=0]),1e-10)
		for ind in pos_zeros:
			Y[ind] += (2*rd.random()-1)*m
		der_tab = Y[1:]-Y[:-1]; sign_der = np.sign(der_tab)
	digit_to_letter = {-1:'-',1:'+'}
	string = ''.join([digit_to_letter[digit] for digit in sign_der])
	keep = True
	while keep:
		new_string = Sub_V7(Y,string)
		if new_string==string:
			keep = False
		else:
			string = new_string
	return string

#repeat the whole analysis but with inversions of the derivative sign at each data point
#and then merge the results: the inversions that cause many changes are significant and if this leads
#to a simplified description, this description is retained
def Stoch_tab_to_string(tab,original=None):
	if original is None:
		original = Tab_to_string_V5(tab)
		arg = None
	else:
		arg = original
	if tab.ndim==2:
		Y = tab[1,:]
	else:
		Y = tab
	der_tab = Y[1:]-Y[:-1]
	#first check for monotonicity
	sign_der = np.sign(der_tab)
	if (sign_der<0).all():
		return '-'*len(der_tab)
	elif (sign_der>0).all():
		return '+'*len(der_tab)
	elif (sign_der==0).all():
		return '0'*len(der_tab)
	#if the derivative has accidental zeros, remove them
	pos_zeros = []
	for ind,sign in enumerate(sign_der):
		if sign==0:
			pos_zeros.append(ind)
	if pos_zeros:
		#add a random noise on data points with amplitude equal to the minimum of the non-zero absolute value
		#of the raw derivative
		m = max(min([abs(el) for el in der_tab if el!=0]),1e-10)
		for ind in pos_zeros:
			Y[ind] += (2*rd.random()-1)*m
	###
	'''
	#test for the curve being flat: we combine different estimators and use a majority rule
	is_flat = 0
	#test of statistical independence btw x and y values
	X = np.asarray(range(len(Y)),dtype=float)
	if distance_correlation_t_test(X,Y).pvalue>0.05:
		is_flat += 1
	#check for the hypothesis of Gaussian white noise (in this case the curve is described as flat)
	if shapiro(der_tab)[1]>0.05 and shapiro(Y)[1]>0.05:
		#check that the tab_derivative is centered around zero and has std_der = sqrt(2)*std_tab
		nb_test = 50; nb_match = 0; scale = math.sqrt(2)*np.std(Y)
		for _ in range(nb_test):
			nb_match += int(ks_2samp(der_tab,np.random.normal(loc=0,scale=scale,size=len(der_tab)))[1]>0.05)
		if nb_match>nb_test/2:
			is_flat += 1
	#test whether there is no rank correlation btw x and y values
	p_value = kendalltau(Y,range(len(Y)),alternative='two-sided')[1]
	if p_value>0.05:
		is_flat += 1
	#test whether there is no rank correlation btw x and y values
	X = list(range(len(Y)))
	if permutation_test((X,),lambda x: spearmanr(x,Y).statistic,permutation_type='pairings').pvalue>0.05:
		is_flat += 1
	if is_flat>=2:
		return '0'*len(der_tab)
	###
	'''
	if tab.ndim==2:
		nb_test = len(tab[0,:])-1
	else:
		nb_test = len(tab)-1
	#do as many trials as data points in der_tab
	#dic_nb[s] = nb of trials yielding the string s; note that we include only the strings that preserve
	#the first domain
	dic_nb = {}; letter_to_digit = {'0':0,'-':-1,'+':1}
	for ind in range(nb_test):
		string = Tab_to_string_V6(tab,ind,start_string=arg)
		if string[0]==original[0]:
			if string in dic_nb:
				dic_nb[string] += 1
			else:
				dic_nb[string] = 1
	#we expect the original string to be among the most frequent
	#if the original is the only most frequent, we take the simplest string (least nb of letters)
	#among the 2nd most frequent and keep it if it is strictly simpler than the original
	#if no, we take the simplest string among the most frequent
	#strings grouped and sorted by frequency
	freq_to_s = {}
	for string,freq in dic_nb.items():
		if freq in freq_to_s:
			freq_to_s[freq].append(string)
		else:
			freq_to_s[freq] = [string]
	#the nb of most frequent strings is len(freq_to_s[list_freq[0]])
	list_freq = sorted(freq_to_s.keys(),reverse=True)
	if freq_to_s[list_freq[0]]==[original]:
		#take the simplest string among the 2nd most frequent but check its existence first
		if len(list_freq)>1:
			#the 2nd most frequent exists
			string = min(freq_to_s[list_freq[1]],key=lambda s:len(String_to_letter(s)))
			#check whether the string is strictly simpler than the original
			if len(String_to_letter(string))<len(String_to_letter(original)):
				pass
			else:
				string = original
		else:
			#the 2nd most frequent does not exist
			string = original
	else:
		#there are several most frequent strings or the original is not the most frequent
		string = min(freq_to_s[list_freq[0]],key=lambda s:len(String_to_letter(s)))
	#add a check step:
	#it may be that the right domain is wrong: to check for its absorption, we collect all the sub-domains
	#that are deviations in the left domain and of same size as the right domain. Then if the right domain
	#constitutes a regular deviation of the left domain, we absorb it. By regular deviation, we mean similar
	#to the existing deviations of the same size in terms of order relation
	domains = Find_domains(string)
	if len(domains)==1:
		return string
	der_tab = Y[1:]-Y[:-1]; digit_to_letter = {-1:'-',1:'+'}
	#size of the domain at the right border
	start,end = domains[-1][:2]
	size_border = end-start+1
	#we determine the extension and shape of the right domain: we collect its true values in derivative sign
	#then its start and end values, as well as its min and max values
	raw_string = ''.join([digit_to_letter[el] for el in np.sign(der_tab)])
	substring = raw_string[start:]
	start_val,end_val = Y[start],Y[-1]
	min_val,max_val = min(Y[start:]),max(Y[start:])
	#then we collect all substrings in the left domain that are identical to substring
	#dic_sub[pos] = substring at position pos in string
	dic_sub = {}
	for pos in range(domains[-2][0],domains[-2][1]+1):
		if raw_string[pos:pos+size_border]==substring:
			sub_tab = Y[pos:pos+size_border+1]
			min_sub,max_sub = min(sub_tab),max(sub_tab)
			start_sub,end_sub = sub_tab[0],sub_tab[-1]
			dic_sub[pos] = (min_sub,max_sub,start_sub,end_sub)
	if not dic_sub:
		return string
	#now we see how are characterized the deviations, and whether the right domain is similar
	#first characteristic, quantifying how noisy is a substring:
	#quality = abs(start_sub-end_sub)/abs(max_sub-min_sub); if no noise, quality = 1
	list_quality = [abs(el[2]-el[3])/abs(el[1]-el[0]) for el in dic_sub.values()]
	quality = abs(start_val-end_val)/abs(max_val-min_val)
	if quality>max(list_quality):
		return string
	#second characteristic, how are ordered the deviations wrt each other?
	#if there is only one deviation, the domain must respect at least partially
	#the order relation of the left domain to be considered as a deviation
	list_pos = sorted(dic_sub.keys()); sign = domains[-2][2]
	if len(list_pos)==1:
		is_dev = False
		if (end_val-dic_sub[list_pos[0]][2])*sign<0:
			is_dev = True
		elif (end_val-dic_sub[list_pos[0]][3])*sign>0 or (start_val-dic_sub[list_pos[0]][2])*sign>0:
			is_dev = True
		if not is_dev:
			return string
	else:
		#four cases: (1) the consecutive deviations are perfectly ordered (2) they intersect but are
		#correctly ordered (3) they intersect but they are not correctly ordered (4) they are completely
		#ordered with the wrong sign (opposite to the sign if the left domain)
		strongest_case = 0
		#strongest_case indicates the strongest deviation to the order relation in the left domain
		for ind in range(len(list_pos)-1):
			#determines the deviation strength
			if (dic_sub[list_pos[ind+1]][3]-dic_sub[list_pos[ind]][2])*sign>0:
				case = 1
			elif (dic_sub[list_pos[ind]][3]-dic_sub[list_pos[ind+1]][2])*sign>0:
				case = 4
			elif (dic_sub[list_pos[ind+1]][3]-dic_sub[list_pos[ind]][3])*sign>0 or (dic_sub[list_pos[ind+1]][2]-dic_sub[list_pos[ind]][2])*sign>0:
				case = 2
			else:
				case = 3
			strongest_case = max(strongest_case,case)
		#determines the deviation strength of the right domain
		if (end_val-dic_sub[list_pos[-1]][2])*sign>0:
			case = 1
		elif (dic_sub[list_pos[-1]][3]-start_val)*sign>0:
			case = 4
		elif (end_val-dic_sub[list_pos[-1]][3])*sign>0 or (start_val-dic_sub[list_pos[-1]][2])*sign>0:
			case = 2
		else:
			case = 3
		if case>strongest_case:
			return string
	#the right domain can be considered as a deviation of the left domain so it is absorbed
	letter = string[-1]; ind = domains[-1][0]
	if letter=='-':
		new_letter = '+'
	elif letter=='+':
		new_letter = '-'
	return string[:ind]+new_letter*(len(string)-ind)

#apply Stoch_tab_to_string to tab until convergence
def Tab_to_string_V8(tab):
	string = Stoch_tab_to_string(tab,original=None)
	keep = True
	while keep:
		new_string = Stoch_tab_to_string(tab,original=string)
		if new_string==string:
			keep = False
	return string

#display the data X,Y = tab[0,:],tab[1,:] with a color code allowing to check the symbolic expression for tab
def Display_symbol(string,tab):
	#display the curve together with the sign of its derivative
	sign_to_color = {'-':'red','0':'green','+':'blue'}
	sign_to_label = {'-':'decreasing','0':'constant','+':'increasing'}
	split_data = {key:[] for key in sign_to_color.keys()}
	for ind,val in enumerate(tab[1,:-1]):
		split_data[string[ind]].append(list(tab[:,ind]))
	split_data[string[-1]].append(list(tab[:,-1]))
	key_to_remove = {key for key,val in split_data.items() if not val}
	for key in key_to_remove:
		del split_data[key]
	for key,val in split_data.items():
		split_data[key] = np.asarray(val)
	fig,ax = Setup_Plot('x','y',fontsize=14)
	for key,val in split_data.items():
		ax.plot(val[:,0],val[:,1],'.',color=sign_to_color[key],markersize=7,label=sign_to_label[key])
	ax.legend(fontsize=14)
	plt.show()

#convert a string describing a curve into a single letter
#a letter is a string composed with symbols '+', '-' and '0' but two consecutive symbols cannot be identical
def String_to_letter(string):
	letter = string[0]; ind = 0
	while ind<len(string):
		if string[ind]!=letter[-1]:
			letter += string[ind]
		ind += 1
	return letter

#convert a single curve into a single letter
def Curve_to_letter(tab,agg):
	return String_to_letter(Stoch_tab_to_string_V5(tab,agg))

#piecwise linear regression (too slow!)
def Curve_to_letter_V2(tab):
	if tab.ndim==2:
		X = tab[0,:]; Y = tab[1,:]
	else:
		X = np.asarray(range(len(tab)),dtype=float); Y = tab
	der_tab = Y[1:]-Y[:-1]
	#first check for monotonicity
	sign_der = np.sign(der_tab)
	if (sign_der<0).all():
		return '-'
	elif (sign_der>0).all():
		return '+'
	elif (sign_der==0).all():
		return '0'
	#check for the hypothesis of Gaussian white noise (in this case the curve is described as flat)
	if shapiro(der_tab)[1]>0.05 and shapiro(Y)[1]>0.05:
		#check that the tab_derivative is centered around zero
		nb_test = 50; nb_match = 0
		for _ in range(nb_test):
			nb_match += int(ks_2samp(der_tab/np.std(der_tab),np.random.normal(loc=0,scale=1,size=len(der_tab)))[1]>0.05)
		if nb_match>nb_test/2:
			return '0'
	return String_to_letter(pr.ModelSelection(X,Y,max_breakpoints=3,verbose=False).results)

#build the word corresponding to the variation wrt the aggregation level ("n-word")
def Get_n_string(Y):
	digit_to_letter = {-1:'-',1:'+'}
	return ''.join([digit_to_letter[digit] for digit in np.sign(Y[1:]-Y[:-1])])

def Get_bslope_string(Y,b_word):
	digit_to_letter = {-1:'-',1:'+',0:'0'}
	der_tab = Y[1:]-Y[:-1]; sign_der = np.sign(Y[1:]-Y[:-1])
	if (sign_der<0).all():
		return '-'*len(sign_der)
	elif (sign_der>0).all():
		return '+'*len(sign_der)
	elif (sign_der==0).all():
		return '0'*len(sign_der)
	#if the curve vs b is flat for every agg, then the b-slope is flat vs agg
	if b_word==['0']*len(b_word):
		return '0'*len(sign_der)
	string = ['']*len(sign_der)
	#compute first what is certain about the evolution of the b-slope, e.g. if the curve vs b goes from
	#'+' to '-', then it is sure that the b-slope has decreased
	for ind,letter in enumerate(b_word[:-1]):
		next_letter = b_word[ind+1]
		#compute the difference btw the nb of '+' and the nb of '-' in letter and next_letter
		nb = 0
		for el in letter:
			if el=='+':
				nb += 1
			elif el=='-':
				nb -= 1
		nb_next = 0
		for el in next_letter:
			if el=='+':
				nb_next += 1
			elif el=='-':
				nb_next -= 1
		if nb_next<nb:
			string[ind] = '-'
		elif nb_next>nb:
			string[ind] = '+'
		else:
			string[ind] = digit_to_letter[sign_der[ind]]
	string = ''.join(string)
	keep = True
	while keep:
		new_string = Iter_string(Y,string)
		if new_string==string:
			keep = False
		else:
			string = new_string
	return string

#describe the observable behaviour in dataset name with a sentence made up with three words
def Describe_scalar(name,obs,list_agg):
	#load the curves vs b for several aggregation levels
	data = {agg:Load_vs_b(name,agg,obs) for agg in list_agg}
	sentence = ['']*3
	#first word: vs b
	letter = ''; b_word = []
	for agg in list_agg:
		print(str(agg)+' begins')
		new_letter = Curve_to_letter(data[agg],agg)
		if letter!=new_letter:
			sentence[0] += '|'+new_letter
			letter = new_letter
		b_word.append(new_letter)
	sentence[0] = sentence[0][1:]
	return sentence
	#second word: vs agg
	nb_b = len(data[list_agg[0]][0,:]); letter = ''
	for j in range(nb_b):
		tab = np.array([list_agg,[data[agg][1,j] for agg in list_agg]])
		#new_letter = Curve_to_letter(tab)
		new_letter = String_to_letter(Get_n_string(tab[1,:]))
		if letter!=new_letter:
			sentence[1] += '|'+new_letter
			letter = new_letter
	sentence[1] = sentence[1][1:]
	#third word: slope wrt b vs agg
	tab = np.array([list_agg,[data[agg][1,-1]-data[agg][1,0] for agg in list_agg]])
	#sentence[2] = Curve_to_letter(tab)
	sentence[2] = String_to_letter(Get_bslope_string(tab[1,:],b_word))
	return sentence

#given a description of a list of datasets wrt a given observable, find the preorder structure ranking the
#datasets from most simple to complex; dic_sentence[num] = sentence describing the dataset of integer
#identifier num
def Sentences_to_preorder(dic_sentence):
	sort_num = list(dic_sentence.keys())
	#dic_nb[num] = total nb of letters needed to describe the dataset indexed by num
	dic_nb = {}
	for num,sentence in dic_sentence.items():
		nb = 0
		for word in sentence:
			for letter in word:
				if letter=='|':
					nb += 1
		dic_nb[num] = nb + len(sentence)
	#group the datasets described by the same sentence
	groups = []; set_num = set(dic_sentence.keys())
	while set_num:
		el = next(iter(set_num))
		group = {num for num in set_num if dic_sentence[num]==dic_sentence[el]}
		set_num = set_num.difference(group)
		groups.append(list(group))
	num_to_group = {num:ind for ind,group in enumerate(groups) for num in group}
	#sort the datasets by increasing nb of letters, if they have the same nb of letters,
	#put the datasets belonging to the same group next to each other
	sort_num.sort(key=lambda num:(dic_nb[num],num_to_group[num]))
	#group the datasets with the same nb of letters
	old_nb = dic_nb[sort_num[0]]; preorder = [sort_num[0]]; old_group = num_to_group[sort_num[0]]
	for num in sort_num[1:]:
		if num_to_group[num]==old_group:
			preorder += [num]
		else:
			old_group = num_to_group[num]
			if dic_nb[num]==old_nb:
				preorder += [',',num]
			else:
				preorder += ['\n',num]
				old_nb = dic_nb[num]
	return preorder

#return a sequence of operations of minimum length to be applied to sentence1 to obtain sentence2
#for this we have to define elementary operations generating a group action on the space of sentences
#first possibility: string edit distance, considering one letter as one symbol and a sentence as a string
def Arrow_sentence(sentence1,sentence2,choice='GED'):
	#arrow[i] = edit sequence for turning the ith word of sentence1 into the ith word of sentence2
	arrow = []
	if choice=='GED':
		for word1,word2 in zip(sentence1,sentence2):
			arrow.append(Operations_from_GED(word1.split('|'),word2.split('|')))
	return arrow

#a diagram is the data of nodes organized into slices and a minimal set of labelled arrows
def Sentences_to_diag(dic_sentence):
	#determine the preorder on nodes
	preorder = Sentences_to_preorder(dic_sentence)
	#collect the nodes and organize them into slices and determine the sentences associated to each node
	node_to_sentence = {}
	#list_nb[k] = nb of nodes on the horizontal line of depth k
	list_nb = []; list_nodes = []; node = ''
	nb = 1
	for el in preorder:
		if el==',':
			nb += 1
			list_nodes.append(node)
			node = ''
		elif el=='\n':
			list_nb.append(nb)
			nb = 1
			list_nodes.append(node)
			node = ''
		else:
			if len(node)==0:
				node += str(el)
			else:
				node += '_'+str(el)
	list_nb.append(nb)
	list_nodes.append(node)
	for node in list_nodes:
		if '_' in node:
			node_to_sentence[node] = dic_sentence[int(node.split('_')[0])]
		else:
			node_to_sentence[node] = dic_sentence[int(node)]
	#determine a set of labelled arrows
	#edge_to_arrow[(node1,node2)] = label of the arrow from node1 to node2
	edge_to_arrow = {}; tot_nb = 0
	#first we draw only arrows btw successive node slices, we move from top to bottom
	for k,nb1 in enumerate(list_nb[:-1]):
		nb2 = list_nb[k+1]
		list_1 = list_nodes[tot_nb:tot_nb+nb1]
		list_2 = list_nodes[tot_nb+nb1:tot_nb+nb1+nb2]
		#has_slot[ind2] = 1 if the node list_nodes[tot_nb+nb1+ind2] has still no incoming arrow
		#nb_rem = nb of available slots
		ind1 = 0; ind2 = 0; has_slot = [1]*nb2; nb_rem = nb2
		while nb_rem>0:
			if has_slot[ind2]:
				node1 = list_nodes[tot_nb+ind1]; node2 = list_nodes[tot_nb+nb1+ind2]
				edge_to_arrow[(node1,node2)] = Arrow_sentence(node_to_sentence[node1],node_to_sentence[node2])
				has_slot[ind2] = 0; nb_rem -= 1
			ind1 = (ind1+1)%nb1
			ind2 = (ind2+1)%nb2
		tot_nb += nb1
	return list_nodes,list_nb,edge_to_arrow

#display a diagram as a network with directed labelled arrows and nodes organized as a preorder
def Display_diag(diagram,savefig=''):
	if savefig:
		if savefig[0]!='_':
			savefig = '_'+savefig
	list_nodes,list_nb,edge_to_arrow = diagram
	fig,ax = plt.subplots(1,1)
	ax.set_axis_off()
	width = max(list_nb)
	ax.set_xlim(-width/2,width/2)
	ax.set_ylim(-len(list_nb),0)
	#display the nodes line by line, from top to bottom
	node_to_position = {}
	tot_nb = 0
	for k,nb in enumerate(list_nb):
		y = -k
		if nb%2==0:
			offset = -nb//2 + 0.5
		else:
			offset = -(nb-1)//2
		for i in range(nb):
			node_name = list_nodes[tot_nb+i]
			x = offset + i
			ax.annotate(node_name,(x,y),fontsize=14,ha='center')
			node_to_position[node_name] = (x,y)
		tot_nb += nb
	#draw arrows btw nodes
	for key,val in edge_to_arrow.items():
		pos1 = node_to_position[key[0]]; pos2 = node_to_position[key[1]]
		ax.annotate('',pos2,xytext=pos1,arrowprops=dict(arrowstyle='->'))
		#add the edge label
		x = (pos1[0]+pos2[0])/2; y = (pos1[1]+pos2[1])/2
		label = []
		for word in val:
			res = ''
			for el in word:
				if el[0]=='ins':
					res += r'$I_{'+str(el[2])+r'}^{'+el[1]+r'}$'
				elif el[0]=='del':
					res += r'$D_{'+str(el[1])+r'}$'
				else:
					res += r'$S_{'+str(el[2])+r'}^{'+el[1]+r'}$'
			if not res:
				res = '.'
			label.append(res)
		label = '|'.join(label)
		ax.annotate(label,(x,y),fontsize=13,ha='center')
	plt.savefig('figures/sym_diag/diag'+savefig+'.png')

#deduce the edit sequence to be applied to word1 from the aligned sequences neword1 and neword2
def Aligned_words_to_operations(neword1,neword2):
	#an insertion at position i means in the new string, the inserted letter will be at position i
	#only the sub-string at the right is modified
	#collect first the insertions, second the substitutions and third the deletions
	insertions = []; substitutions = []; deletions = []; nb_del = 0
	for pos,(letter1,letter2) in enumerate(zip(neword1,neword2)):
		if letter1=='#':
			insertions = [('ins',letter2,pos)] + insertions
		elif letter2=='#':
			deletions = [('del',pos-nb_del)] + deletions
			nb_del += 1
		elif letter1!=letter2:
			substitutions = [('sub',letter2,pos)] + substitutions
	return deletions+substitutions+insertions

#minimum sequence of operations for turning word1 into word2 wrt the string edit distance
def Operations_from_GED(word1,word2):
	n = len(word1); p = len(word2)
	#M[i,j] = edit distance btw word1[:i] and word2[:j]
	M = np.zeros((n+1,p+1),dtype=int)
	for i in range(1,n+1):
		M[i,0] = i
	for j in range(1,p+1):
		M[0,j] = j
	for i in range(1,n+1):
		for j in range(1,p+1):
			M[i,j] = min([
				M[i-1,j]+1, #deletion
				M[i,j-1]+1, #insertion
				M[i-1,j-1]+int(word1[i-1]!=word2[j-1]) #substitution
				])
	#determine the minimum edit sequence to go from word1 to word2
	i = n; j = p; neword1 = []; neword2 = []; sign = np.sign(i-j)
	while i>0 and j>0:
		list_option = [("sub",M[i-1,j-1])]
		if np.sign(i-1-j)*sign>=0:
			list_option.append(("del",M[i-1,j]))
		if np.sign(i-j+1)*sign>=0:
			list_option.append(("ins",M[i,j-1]))
		chosen_option = min(list_option,key=lambda el:el[1])[0]
		if chosen_option=="sub":
			neword1 = [word1[i-1]] + neword1
			neword2 = [word2[j-1]] + neword2
			i -= 1; j -= 1
		elif chosen_option=='del':
			neword1 = [word1[i-1]] + neword1
			neword2 = ['#'] + neword2
			i -= 1
		elif chosen_option=='ins':
			neword1 = ['#'] + neword1
			neword2 = [word2[j-1]] + neword2
			j -= 1
	if i==0:
		for ind in range(j-1,-1,-1):
			neword1 = ['#'] + neword1
			neword2 = [word2[ind]] + neword2
		return Aligned_words_to_operations(neword1,neword2)
	if j==0:
		for ind in range(i-1,-1,-1):
			neword1 = [word1[ind]] + neword1
			neword2 = ['#'] + neword2
		return Aligned_words_to_operations(neword1,neword2)

#return the edit distance btw the strings s and t
def String_ED(s,t):
	n = len(s); p = len(t)
	#M[i,j] = edit distance btw s[:i] and t[:j]
	M = np.zeros((n+1,p+1),dtype=int)
	for i in range(1,n+1):
		M[i,0] = i
	for j in range(1,p+1):
		M[0,j] = j
	for i in range(1,n+1):
		for j in range(1,p+1):
			M[i,j] = min([
				M[i-1,j]+1, #deletion
				M[i,j-1]+1, #insertion
				M[i-1,j-1]+int(s[i-1]!=t[j-1]) #substitution
				])
	return M[n,p]
pass
#split the composite node nc1 into the two composite nodes nc2 and nc3 in the diagram diag (modifies diag)
#the splitting is such that nc2 and nc3 are at the same level as nc1 was
def Split_node(diag,nc1,nc2,nc3):
	list_nodes,list_nb,edge_to_arrow = diag
	#find the level and position in list_node at which nc1 is
	level = -1; keep = True; tot_nb = 0; pos = -1
	while keep:
		level += 1
		nb = list_nb[level]
		for ind in range(tot_nb,tot_nb+nb):
			if list_nodes[ind]==nc1:
				pos = ind; keep = False
		tot_nb += nb
	#split nc1 into nc2 and nc3:
	#ngh_in = nodes above nc1; ngh_out = nodes below nc1
	ngh_in = []; ngh_out = []
	for node in list_nodes:
		if (node,nc1) in edge_to_arrow:
			ngh_in.append((node,edge_to_arrow[(node,nc1)]))
		if (nc1,node) in edge_to_arrow:
			ngh_out.append((node,edge_to_arrow[(nc1,node)]))
	for el in ngh_in:
		del edge_to_arrow[(el[0],nc1)]
	for el in ngh_out:
		del edge_to_arrow[(nc1,el[0])]
	for (node,label) in ngh_in:
		for nc in [nc2,nc3]:
			edge_to_arrow[(node,nc)] = label.copy()
	for (node,label) in ngh_out:
		for nc in [nc2,nc3]:
			edge_to_arrow[(nc,node)] = label.copy()
	list_nodes = list_nodes[:pos] + [nc2,nc3] + list_nodes[pos+1:]
	list_nb[level] += 1
	return list_nodes,list_nb,edge_to_arrow

#complete diag1 into a diagram for which it exists a surjective morphism onto diag2
def Diag_completion(diag1,diag2):
	pass
pass

#find a path from n1 to n2 in the network net
def Find_path(n1,n2,net):
	first_arrival = {n1:0}; visited_nodes = {n1}
	border = set(net[n1]); keep = True; level = 1
	while keep:
		if n2 in border:
			keep = False
			first_arrival[n2] = level
		else:
			new_border = set()
			for node in border:
				first_arrival[node] = level
				new_border = new_border.union(set(net[node]))
			visited_nodes = visited_nodes.union(border)
			border = new_border.difference(visited_nodes)
		level += 1
	#collect the path
	path = [n2]; level = first_arrival[n2]
	while level>0:
		ngh = list(net[path[-1]])
		ind = 0
		while first_arrival[ngh[ind]]!=level-1:
			ind += 1
		path.append(ngh[ind])
		level -= 1
	return path[::-1]

#find the label arrows along a path in a diagram; a path is just a list of nodes of diag
def Find_arrows(path,dic_sentence):
	res = ''
	#for each successive pair of nodes in path, find a path in diag connecting the nodes
	for n1,n2 in zip(path,path[1:]):
		#find the label associated to the arrow n1-->n2
		label = Arrow_sentence(dic_sentence[n1],dic_sentence[n2])
		for el in label:
			if not el:
				res += '.'
			for op in el:
				if op[0]=='sub':
					res = res+'S'+op[1]+str(op[2])
				elif op[0]=='ins':
					res = res+'I'+op[1]+str(op[2])
				else:
					res = res+'D'+str(op[1])
			res += '|'
		res = res[:-1]+'\n'
	return res[:-1]

#convert a diagram into two strings with one string (s) encoding the preorder btw nodes and the other
#encoding the label arrows (t)
def Diag_to_string(diag,dic_sentence):
	s = ''; path = []
	#first sort the nodes according to first their level second their value
	list_nodes,list_nb = diag[:2]
	tot_nb = 0
	for nb in list_nb:
		level = []
		for node in list_nodes[tot_nb:tot_nb+nb]:
			if '_' in node:
				level.append(sorted(node.split('_')))
			else:
				level.append([node])
		level.sort()
		string = ''
		for el in level:
			path.append(int(el[0]))
			for val in el:
				string = string+val+'_'
			string = string[:-1]+','
		s = s+string[:-1]+'\n'
		level = sorted(list_nodes)
		tot_nb += nb
	#second find the arrows along the path of the totally ordered nodes (in the increasing direction)
	return s[:-1],Find_arrows(path,dic_sentence)

#return the edit distance btw the diagrams associated to dic1 and dic2:
#to do this we convert the diagrams into strings such that the edit distance btw the strings
#is equal to the edit distance btw the diagrams
def Diag_ED(dic1,dic2):
	diag1 = Sentences_to_diag(dic1)
	diag2 = Sentences_to_diag(dic2)
	s1,t1 = Diag_to_string(diag1,dic1)
	s2,t2 = Diag_to_string(diag2,dic2)
	#average the obtained edit distances knowing that e.g. 2_4 = 4_2 (invariance of the diagram under
	#shuffling of the datasets belonging to the same node)
	return String_ED(s1,s2)+String_ED(t1,t2)

def Get_dic_sentence(obs,list_agg,list_name):
	res = {}
	for num,name in enumerate(list_name):
		print(str(num)+' begins')
		res[num+1] = Describe_scalar(name,obs,list_agg)
	return res
#return {num+1:Describe_scalar(name,obs,list_agg) for num,name in enumerate(list_name)}

#compute the edit distance btw the observables in list_obs and display the similarity matrix
def Sim_diag(list_obs,list_agg,list_name,savefig=''):
	if savefig:
		if savefig[0]!='_':
			savefig = '_'+savefig
	tot_dic = {}
	for obs in list_obs:
		print(str(obs)+' begins')
		dic = Get_dic_sentence(obs,list_agg,list_name)
		print('sentence computed')
		diag = Sentences_to_diag(dic); s,t = Diag_to_string(diag,dic)
		tot_dic[obs] = (s,t)
		print('dic computed')
	nb_obs = len(list_obs); mat = np.zeros((nb_obs,nb_obs))
	print('SED begins')
	for i,obs1 in enumerate(list_obs[:-1]):
		s1,t1 = tot_dic[obs1]
		print(i)
		for j,obs2 in enumerate(list_obs[i+1:]):
			s2,t2 = tot_dic[obs2]
			print(j)
			mat[i,i+j+1] = String_ED(s1,s2)+String_ED(t1,t2)
	print('sim mat computed')
	#convert mat into a similarity matrix (note that mat[i,j]>=0)
	M = np.max(mat)
	for i in range(nb_obs):
		for j in range(i+1,nb_obs):
			mat[i,j] = 1-mat[i,j]/M
			mat[j,i] = mat[i,j]
		mat[i,i] = 1
	np.savetxt('figures/sym_diag/simat_obs.txt')
	#gather the observables in communities to enhance the visibility of the similarity matrix
	net = nx.Graph()
	for i in range(nb_obs):
		for j in range(i+1,nb_obs):
			net.add_edge(i,j,weight=mat[i,j])
	list_com = sorted(nx.community.louvain_communities(net),key=len,reverse=True)
	old_to_new_num = {}; num = 0
	for com in list_com:
		for i in com:
			old_to_new_num[i] = num
			num += 1
	new_mat = np.zeros(mat.shape)
	for i in range(nb_obs):
		for j in range(nb_obs):
			new_mat[old_to_new_num[i],old_to_new_num[j]] = mat[i,j]

	#display the matrix
	fig,ax = plt.subplots(1,1,constrained_layout=True)
	labels = []; fontsize = 12
	for i in range(nb_obs):
		obs = list_obs[old_to_new_num[i]]
		if type(obs)==tuple:
			labels.append(obs[0].replace('0',' '))
		else:
			labels.append(obs.replace('0',' '))
	ax.set_xticks(range(nb_obs))
	ax.set_yticks(range(nb_obs))
	ax.set_xticklabels(labels,rotation=90,fontsize=fontsize)
	ax.set_yticklabels(labels,fontsize=fontsize)
	im = ax.imshow(mat,cmap='gnuplot2')
	plt.colorbar(im,ax=ax)
	plt.savefig('figures/sym_diag/simat_obs'+savefig+'.png')

#compute the edit distance btw the observables in list_obs and display the similarity matrix
def Sim_diag_old(list_obs,list_agg,list_name,savefig=''):
	if savefig:
		if savefig[0]!='_':
			savefig = '_'+savefig
	tot_dic = {obs:Get_dic_sentence(obs,list_agg,list_name) for obs in list_obs}
	nb_obs = len(list_obs); mat = np.zeros((nb_obs,nb_obs))
	for i,obs1 in enumerate(list_obs[:-1]):
		for j,obs2 in enumerate(list_obs[i+1:]):
			mat[i,i+j+1] = Diag_ED(tot_dic[obs1],tot_dic[obs2])
	#convert mat into a similarity matrix (note that mat[i,j]>=0)
	M = np.max(mat)
	for i in range(nb_obs):
		for j in range(i+1,nb_obs):
			mat[i,j] = 1-mat[i,j]/M
			mat[j,i] = mat[i,j]
		mat[i,i] = 1
	np.savetxt('figures/sym_diag/simat_obs.txt')
	#gather the observables in communities to enhance the visibility of the similarity matrix
	net = nx.Graph()
	for i in range(nb_obs):
		for j in range(i+1,nb_obs):
			net.add_edge(i,j,weight=mat[i,j])
	list_com = sorted(nx.community.louvain_communities(net),key=len,reverse=True)
	old_to_new_num = {}; num = 0
	for com in list_com:
		for i in com:
			old_to_new_num[i] = num
			num += 1
	new_mat = np.zeros(mat.shape)
	for i in range(nb_obs):
		for j in range(nb_obs):
			new_mat[old_to_new_num[i],old_to_new_num[j]] = mat[i,j]

	#display the matrix
	fig,ax = plt.subplots(1,1,constrained_layout=True)
	labels = []; fontsize = 12
	for i in range(nb_obs):
		obs = list_obs[old_to_new_num[i]]
		if type(obs)==tuple:
			labels.append(obs[0].replace('0',' '))
		else:
			labels.append(obs.replace('0',' '))
	ax.set_xticks(range(nb_obs))
	ax.set_yticks(range(nb_obs))
	ax.set_xticklabels(labels,rotation=90,fontsize=fontsize)
	ax.set_yticklabels(labels,fontsize=fontsize)
	im = ax.imshow(mat,cmap='gnuplot2')
	plt.colorbar(im,ax=ax)
	plt.savefig('figures/sym_diag/simat_obs'+savefig+'.png')

#in the context of a polynomial regression, convert data X into a matrix M suitable for linear regression
def Data_to_M(X,deg):
	#M[i,j] = X[i]**j
	M = np.ones((len(X),deg+1))
	for j in range(1,deg+1):
		M[:,j] = M[:,j-1]*X
	return M

#return the Maximum Likelihood Estimator (MLE) of the matrix data M and values Y
def Get_MLE(M,Y):
	mat = np.dot(M.T,M); sig = np.linalg.inv(mat); n = len(Y)
	c_hat = np.dot(sig,np.dot(M.T,Y))
	R_hat = np.dot(c_hat.T,np.dot(mat,c_hat))/n
	tau_hat = np.sum(Y**2)/n - R_hat
	return R_hat,tau_hat

#compute the cost of describing the errors when doing the polynomial regression of degree deg on data X,Y
def Cost_errors(X,Y,deg):
	M = Data_to_M(X,deg); n = len(Y)
	R_hat,tau_hat = Get_MLE(M,Y)
	list_val = []
	for k in range(1,n):
		val = (n-k)*np.log2(tau_hat) + k*np.log2(R_hat) - (n-k-1)*np.log2(n-k) - (k-1)*np.log2(k)
		list_val.append(val)
	return 2*min(list_val)

def Log_star(n):
	if n==0:
		return 1
	keep = True; nb = n; res = 0; offset = np.log2(2.865064)
	while keep:
		new_nb = np.log2(nb)
		if new_nb<=0:
			keep = False
		else:
			res += new_nb
			nb = new_nb
	return res + offset

#perform a polynomial fit using Minimum Description Length (MDL) principle
def Display_pol_fit(tab):
	Y = tab[1,:]
	#normalize Y
	m,M = min(Y),max(Y)
	Y = (Y-m)/(M-m)
	#n = nb of data points
	n = len(Y)
	X = np.linspace(0,1,n)
	#deg = degree of the polynom
	deg_MDL = {}
	for deg in range(11):
		#compute the cost of fitting the curve with a polynom of degree deg
		#deg_MDL[deg] = (np.log(deg)+2*np.log(np.log(deg)/np.log(2)))*n/np.log(2) + Cost_errors(X,Y,deg)
		deg_MDL[deg] = Log_star(deg)*n + Cost_errors(X,Y,deg)
	fig,ax = plt.subplots(1,1,constrained_layout=True)
	ax.plot(*zip(*deg_MDL.items()),'.')
	#display the result
	best_deg = min(deg_MDL.keys(),key=lambda deg:deg_MDL[deg])
	#best_deg = 1
	p = np.polyfit(X,Y,best_deg)
	fig,ax = plt.subplots(1,1,constrained_layout=True)
	ax.plot(X,Y,'.')
	XX = np.linspace(0,1,100); YY = np.polyval(p,XX)
	ax.plot(XX,YY,'-')
	plt.show()

#perform a polynomial fit using Minimum Description Length (MDL) principle:
#actually there are sometimes multiple candidates for fitting a curve: all the local minima must
#be returned. Even in the case there is a single global minimum which is also a local minimum, mistakes
#can be made, with presence of small spurious domains
def Get_pol_fit(Y):
	#n = nb of data points
	n = len(Y)
	X = np.linspace(0,1,n)
	#deg = degree of the polynom
	deg_MDL = {}
	for deg in range(11):
		#compute the cost of fitting the curve with a polynom of degree deg
		deg_MDL[deg] = Log_star(deg)*n + Cost_errors(X,Y,deg)
	#return the smallest degree that is zero or a local minimum if existence
	first_locmin = -1; keep = True; deg = 1
	while keep:
		if deg_MDL[deg+1]>deg_MDL[deg] and deg_MDL[deg-1]>deg_MDL[deg]:
			first_locmin = deg
			keep = False
		deg += 1
		if deg==10:
			keep = False
	#del deg_MDL[0]; del deg_MDL[1]
	glob_min = min(deg_MDL.keys(),key=lambda deg:deg_MDL[deg])
	#return np.polyfit(X,Y,glob_min)
	if glob_min==0:
		best_deg = 0
	elif glob_min==10:
		if first_locmin<0:
			best_deg = glob_min
		else:
			best_deg = first_locmin
	else:
		best_deg = glob_min
	return np.polyfit(X,Y,best_deg)

#use cross validation instead of MDL to determine the best degree for the fitting polynom
def Get_pol_fit_V2(Y):
	n = len(Y); X = np.linspace(0,1,n)
	#normalise Y
	m,M = min(Y),max(Y)
	Y = (Y-m)/(M-m)
	dic_errors = {deg:Cross_val(deg,Y) for deg in range(min(len(Y),11))}
	best_deg = min(dic_errors.keys(),key=lambda deg:dic_errors[deg])
	return np.polyfit(X,Y,best_deg)

def Cross_val(deg,Y):
	n = len(Y); X = np.linspace(0,1,n); errors = []
	for pos in range(n):
		XX = np.concatenate((X[:pos],X[pos+1:])); YY = np.concatenate((Y[:pos],Y[pos+1:]))
		pol_fit = np.polyfit(XX,YY,deg)
		errors.append((np.polyval(pol_fit,X[pos])-Y[pos])**2)
	return np.mean(errors)

def Test_cross_val(tab,agg):
	Y = tab[1,:]
	#normalise Y
	m,M = min(Y),max(Y)
	Y = (Y-m)/(M-m)
	#first_deg,second_deg = Get_pol_fit_V2(Y)
	dic_errors = {deg:Cross_val(deg,Y) for deg in range(11)}
	n = len(Y); X = np.linspace(0,1,n)
	#display the data points and the two polynomial fit
	fig,ax = plt.subplots(1,1,constrained_layout=True)
	ax.plot(X,Y,'.',label='data')
	list_marker = ['--','^']; best_deg = min(dic_errors.keys(),key=lambda deg:dic_errors[deg])
	#for deg,marker in zip(dic_errors.keys(),list_marker):
	pol_fit = np.polyfit(X,Y,best_deg)
	ax.plot(X,np.polyval(pol_fit,X),'--',label='deg = '+str(best_deg))
	ax.legend()
	ax.set_title(r'$n = $'+str(agg))
	print('best_deg:',best_deg)

#describe the sign derivative of the polynom "pol_fit" at n points equally spaced on [0,1]
def Pol_to_string(pol_fit,Y):
	n = len(Y); X = np.linspace(0,1,n); digit_to_letter = {0:'0',1:'+',-1:'-'}
	#compute the derivative of the polynom
	deg = len(pol_fit)-1
	der_pol = np.zeros(deg)
	for k,coeff in enumerate(pol_fit[:-1]):
		der_pol[k] = (deg-k)*coeff
	#evaluate the derivative at X
	res = [digit_to_letter[el] for el in np.sign(np.polyval(der_pol,X))]
	#check the first value of the derivative
	YY = np.polyval(pol_fit,X)
	first_letter = digit_to_letter[np.sign(Y[1]-Y[0])]
	if res[0]!=first_letter:
		#we do not allow the polynom to change sign btw the first two points if we have a data curve
		#strictly monotonous at the beginning
		first_sign = np.sign(Y[1:3]-Y[:2])
		if (first_sign>0).all() or (first_sign<0).all():
			res[0] = first_letter
		#fit very close to the data points
		#if math.sqrt((Y[0]-YY[0])**2+(Y[1]-YY[1])**2)/2<0.01:
		#	res[0] = first_letter
	#absorb the non-signficant zeros
	last_letter = res[0]; pos = 0; keep = True
	while keep:
		letter = res[pos]
		if letter=='0':
			res[pos] = last_letter
		else:
			last_letter = letter
		pos += 1
		if pos==len(res):
			keep = False
	return ''.join(res)

#describe the sign derivative of the polynom "pol_fit" at n points equally spaced on [0,1]
def Pol_to_string_V2(pol_fit,Y):
	n = len(Y); X = np.linspace(0,1,n); digit_to_letter = {0:'0',1:'+',-1:'-'}
	YY = np.polyval(pol_fit,X)
	res = [digit_to_letter[el] for el in np.sign(YY[1:]-YY[:-1])]
	first_letter = digit_to_letter[np.sign(Y[1]-Y[0])]
	if res[0]!=first_letter:
		#we do not allow the polynom to change sign btw the first two points if we have a data curve
		#strictly monotonous at the beginning
		first_sign = np.sign(Y[1:3]-Y[:2])
		if (first_sign>0).all() or (first_sign<0).all():
			res[0] = first_letter
	#absorb the non-signficant zeros
	last_letter = res[0]; pos = 0; keep = True
	while keep:
		letter = res[pos]
		if letter=='0':
			res[pos] = last_letter
		else:
			last_letter = letter
		pos += 1
		if pos==len(res):
			keep = False
	return ''.join(res)

#diff with V2: add the same tests on raw data but on fit data to decide whether the fit is flat or not
def Pol_to_string_V3(pol_fit,Y):
	n = len(Y); X = np.linspace(0,1,n); digit_to_letter = {0:'0',1:'+',-1:'-'}
	YY = np.polyval(pol_fit,X)
	m = np.min(YY); M = np.max(YY); der_tab = YY[1:]-YY[:-1]
	if (M-m)/M<0.005:
		return '0'*len(der_tab)
	if (M-m)<=max(abs(der_tab))*1.05:
		return '0'*len(der_tab)
	res = [digit_to_letter[el] for el in np.sign(der_tab)]
	first_letter = digit_to_letter[np.sign(Y[1]-Y[0])]
	if res[0]!=first_letter:
		#we do not allow the polynom to change sign btw the first two points if we have a data curve
		#strictly monotonous at the beginning
		first_sign = np.sign(Y[1:3]-Y[:2])
		if (first_sign>0).all() or (first_sign<0).all():
			res[0] = first_letter
	#absorb the non-signficant zeros
	last_letter = res[0]; pos = 0; keep = True
	while keep:
		letter = res[pos]
		if letter=='0':
			res[pos] = last_letter
		else:
			last_letter = letter
		pos += 1
		if pos==len(res):
			keep = False
	return ''.join(res)

def Get_max_dev(string,Y):
	der_tab = Y[1:]-Y[:-1]; sign_der = np.sign(der_tab); digit_to_letter = {0:'0',-1:'-',1:'+'}
	pos_dev = []
	for pos,letter in enumerate(string):
		if digit_to_letter[sign_der[pos]]!=letter:
			pos_dev.append(pos)
	if not pos_dev:
		return 0
	list_dev = []; start = pos_dev[0]
	for ind,pos in enumerate(pos_dev[:-1]):
		if pos_dev[ind+1]==pos+1 and string[pos+1]==string[pos]:
			if ind==len(pos_dev)-2:
				list_dev.append(abs(Y[start]-Y[pos+2]))
		else:
			list_dev.append(abs(Y[start]-Y[pos+1]))
			start = pos_dev[ind+1]
	if start==pos_dev[-1]:
		list_dev.append(abs(Y[start]-Y[start+1]))
	if not list_dev:
		return 0
	return max(list_dev)

#validate this description: collect all instantaneous deviations and absorb domains as long as the
#maximum deviation is unchanged
def Validate(string,Y):
	#normalize Y
	m,M = min(Y),max(Y)
	Y = (Y-m)/(M-m)
	der_tab = Y[1:]-Y[:-1]; sign_der = np.sign(der_tab); digit_to_letter = {0:'0',-1:'-',1:'+'}
	#the sign cannot change at the last point
	if string[-2]!=string[-1]:
		string = string[:-1]+string[-2]
	#amplitude of the greatest deviation
	max_dev = Get_max_dev(string,Y)
	#for each domain, test whether it can be absorbed or not: absorption if max_dev is unchanged
	#idea: stochastic version: absorption with proba exp(-delta(max_dev)/T) and then decrease T
	keep = True
	while keep:
		domains = Find_domains(string)[1:]
		dom_num = 0; ok = len(domains)>0; cand_string = []
		while ok:
			start,end,sign = domains[dom_num]
			#we see whether max_dev has changed under domain inversion
			new_letter = digit_to_letter[-sign]
			test_string = string[:start]+new_letter*(end-start+1)+string[end+1:]
			#new_max = Get_max_dev(test_string,Y)
			new_max = abs(min(Y[start:end+1])-max(Y[start:end+2]))
			if new_max<=max_dev*(1.05):
				cand_string.append(test_string)
			dom_num += 1
			if dom_num==len(domains):
				ok = False
		if not cand_string:
			keep = False
		else:
			#if several inversions are possible, take the simplest one
			new_string = min(cand_string,key=lambda s:len(String_to_letter(s)))
			if len(String_to_letter(new_string))==len(String_to_letter(string)):
				keep = False
			else:
				string = new_string
	return string

def Validate_V2(old_string,Y,pol_fit):
	print('old_string:',Validate(old_string,Y))
	m,M = min(Y),max(Y)
	Y = (Y-m)/(M-m)
	digit_to_letter = {-1:'-',1:'+'}
	#smooth the polynomial fit by a cubic spline fit
	n = len(Y); X = np.linspace(0,1,n); YY = np.polyval(pol_fit,X)
	#determines the hyperparameter (smoothing degree) by cross-validation
	smooth_factors = [0,0.001,0.005,0.01,0.05,0.1]; mse = {}
	for val in smooth_factors:
		errors = []
		for ind in range(len(X)):
			X_train = np.concatenate((X[:ind],X[ind+1:]))
			Y_train = np.concatenate((Y[:ind],Y[ind+1:]))
			spl = splrep(X_train,Y_train,s=val)
			errors.append((splev(X[ind],spl)-YY[ind])**2)
		mse[val] = np.mean(errors)
	best_smooth_factor = min(mse.keys(),key=lambda s:mse[s])
	print('best factor:',best_smooth_factor)
	spl = UnivariateSpline(X,Y)
	spl.set_smoothing_factor(best_smooth_factor)
	#deduce the new sign derivative from the spline fit
	string = ''.join([digit_to_letter[el] for el in np.sign(spl.derivative()(X[:-1]))])
	der_tab = Y[1:]-Y[:-1]; sign_der = np.sign(der_tab)
	#the sign cannot change at the last point
	if string[-2]!=string[-1]:
		string = string[:-1]+string[-2]
	#amplitude of the greatest deviation
	max_dev = Get_max_dev(string,Y)
	#for each domain, test whether it can be absorbed or not: absorption if max_dev is unchanged
	#idea: stochastic version: absorption with proba exp(-delta(max_dev)/T) and then decrease T
	keep = True
	while keep:
		domains = Find_domains(string)[1:]
		dom_num = 0; ok = len(domains)>0; cand_string = []
		while ok:
			start,end,sign = domains[dom_num]
			#we see whether max_dev has changed under domain inversion
			new_letter = digit_to_letter[-sign]
			test_string = string[:start]+new_letter*(end-start+1)+string[end+1:]
			#new_max = Get_max_dev(test_string,Y)
			new_max = abs(min(Y[start:end+1])-max(Y[start:end+2]))
			if new_max<=max_dev*(1.05):
				cand_string.append(test_string)
			dom_num += 1
			if dom_num==len(domains):
				ok = False
		if not cand_string:
			keep = False
		else:
			#if several inversions are possible, take the simplest one
			new_string = min(cand_string,key=lambda s:len(String_to_letter(s)))
			if len(String_to_letter(new_string))==len(String_to_letter(string)):
				keep = False
			else:
				string = new_string
	print(string)
	#display the data points, the polynomial fit and the spline fit
	fig,ax = plt.subplots(1,1,constrained_layout=True)
	ax.plot(X,Y,'.',label='data')
	ax.plot(X,YY,'--',label='polynom')
	ax.plot(X,spl(X),label='spline')
	ax.legend()
	plt.show()
	exit()
	return string

#idea: the factor 1.05 if new_max<=max_dev*(1.05) could be a variable adjusted by cross-validation
#so as to minimize the prediction error of the derivative sign
#also we could formulate the whole problem of symbolic analysis as the minimisation of a given functional
def Sub_validateV3(string,Y,verbose=False):
	digit_to_letter = {0:'0',-1:'-',1:'+'}
	#the sign cannot change at the last point
	if string[-2]!=string[-1]:
		string = string[:-1]+string[-2]
	if (np.array(list(string))=='0').all():
		return string
	#amplitude of the greatest deviation
	max_dev = Get_max_dev(string,Y)
	if verbose:
		print('max_dev',max_dev)
	#for each domain, test whether it can be absorbed or not: absorption if max_dev is unchanged
	#idea: stochastic version: absorption with proba exp(-delta(max_dev)/T) and then decrease T
	keep = True
	while keep:
		domains = Find_domains(string)[1:]
		dom_num = 0; ok = len(domains)>0; cand_string = []
		if verbose:
			print('domains',domains)
		while ok:
			start,end,sign = domains[dom_num]
			#we see whether max_dev has changed under domain inversion
			new_letter = digit_to_letter[-sign]
			test_string = string[:start]+new_letter*(end-start+1)+string[end+1:]
			new_max = abs(min(Y[start:end+1])-max(Y[start:end+2]))
			if new_max<=max_dev*1.05:
				cand_string.append(test_string)
			dom_num += 1
			if dom_num==len(domains):
				ok = False
		if not cand_string:
			keep = False
		else:
			#if several inversions are possible, take the simplest one
			new_string = min(cand_string,key=lambda s:len(String_to_letter(s)))
			if len(String_to_letter(new_string))==len(String_to_letter(string)):
				keep = False
			else:
				string = new_string
	return string

def Get_shifted_data(Y,pol_fit):
	n = len(Y); X = np.linspace(0,1,n); YY = np.polyval(pol_fit,X)
	#collect the cases where q>=2 consecutive data points are on the same side of the polynom curve
	threshold = 0.01; diff_pol = {}
	for pos in range(n):
		diff = Y[pos]-YY[pos]
		if abs(diff)<threshold:
			pass
		elif diff>0:
			if 1 in diff_pol:
				if pos==diff_pol[1][-1][-1]+1:
					diff_pol[1][-1].append(pos)
				else:
					diff_pol[1].append([pos])
			else:
				diff_pol[1] = [[pos]]
		else:
			if -1 in diff_pol:
				if pos==diff_pol[-1][-1][-1]+1:
					diff_pol[-1][-1].append(pos)
				else:
					diff_pol[-1].append([pos])
			else:
				diff_pol[-1] = [[pos]]
	#remove the isolated deviations from the polynom
	key_to_remove = set(())
	for key,val in diff_pol.items():
		diff_pol[key] = [el for el in val if len(el)>1]
		if not diff_pol[key]:
			key_to_remove.add(key)
	for key in key_to_remove:
		del diff_pol[key]
	#shift the points so that the farthest point from the polynom curve is brought on the polynom
	new_Y = Y.copy()
	for key,val in diff_pol.items():
		for el in val:
			shift = key*max([abs(Y[pos]-YY[pos]) for pos in el])
			for pos in el:
				new_Y[pos] -= shift
	return new_Y

def Get_shifted_data_V2(Y,pol_fit):
	n = len(Y); X = np.linspace(0,1,n); YY = np.polyval(pol_fit,X)
	#collect the cases where q>=2 consecutive data points are on the same side of the polynom curve
	threshold = 0.01; diff_pol = {}
	for pos in range(n):
		diff = Y[pos]-YY[pos]
		if abs(diff)<threshold:
			pass
		elif diff>0:
			if 1 in diff_pol:
				if pos==diff_pol[1][-1][-1]+1:
					diff_pol[1][-1].append(pos)
				else:
					diff_pol[1].append([pos])
			else:
				diff_pol[1] = [[pos]]
		else:
			if -1 in diff_pol:
				if pos==diff_pol[-1][-1][-1]+1:
					diff_pol[-1][-1].append(pos)
				else:
					diff_pol[-1].append([pos])
			else:
				diff_pol[-1] = [[pos]]
	#remove the isolated deviations from the polynom
	isolated_dev = {1:set(),-1:set()}
	key_to_remove = set(())
	for key,val in diff_pol.items():
		diff_pol[key] = [el for el in val if len(el)>1]
		if not diff_pol[key]:
			key_to_remove.add(key)
		#handle the isolated deviations
		for el in val:
			if len(el)==1:
				pos = el[0]
				if pos<len(Y)-2:
					if np.sign(Y[pos+2]-Y[pos])==np.sign(Y[pos+2]-Y[pos+1]) or np.sign(Y[pos+2]-Y[pos])==np.sign(Y[pos]-Y[pos+1]):
						isolated_dev[key].add(pos)
	for key in key_to_remove:
		del diff_pol[key]
	#shift the points so that the farthest point from the polynom curve is brought on the polynom
	new_Y = Y.copy()
	for key,val in diff_pol.items():
		for el in val:
			shift = key*max([abs(Y[pos]-YY[pos]) for pos in el])
			for pos in el:
				new_Y[pos] -= shift
	#shift the isolated points
	for key,val in isolated_dev.items():
		for pos in val:
			new_Y[pos] -= key*abs(Y[pos]-YY[pos])
	return new_Y

def Sub_validateV4(string,Y,verbose=False):
	keep = True
	while keep:
		new_string = Sub_validateV3(string,Y,verbose=verbose)
		if new_string!=string:
			string = new_string
		else:
			keep = False
	return string

def Get_trunc_string(Y,verbose=False):
	pol_fit = Get_pol_fit_V2(Y)
	string = Pol_to_string_V2(pol_fit,Y)
	new_Y = Get_shifted_data_V2(Y,pol_fit)
	string1 = Sub_validateV4(string,Y,verbose=verbose)
	string2 = Sub_validateV4(string,new_Y,verbose=verbose)
	if len(String_to_letter(string1))<len(String_to_letter(string2)):
		string = string1
	else:
		string = string2
	return string

def Validate_V3(string,Y,pol_fit,agg,verbose=False,display=False):
	if verbose:
		print('old_string:\n',Validate(string,Y))
	new_Y = Get_shifted_data(Y,pol_fit)
	string1 = Sub_validateV3(string,Y,verbose=verbose)
	string2 = Sub_validateV3(string,new_Y,verbose=verbose)
	if len(String_to_letter(string1))<len(String_to_letter(string2)):
		string = string1
	else:
		string = string2
	#it can be that the curve is asymptotically flat and so spurious domains appear because of fluctuations
	if len(String_to_letter(string))>2:
		first_letter = string[0]; domains = Find_domains(string)[1:]
		start = domains[0][0]; truncated_string = Get_trunc_string(Y,verbose=verbose)
		if verbose:
			print('truncated string:\n',truncated_string)
		if len(String_to_letter(truncated_string))==1:
			if truncated_string[0]==first_letter or (np.array(list(truncated_string))=='0').all():
				string = first_letter*len(truncated_string)
	if verbose:
		print('new_string:\n',string)
	if display:
		n = len(Y); X = np.linspace(0,1,n)
		#display the data points, the polynomial fit and the spline fit
		fig,ax = plt.subplots(1,1,constrained_layout=True)
		ax.plot(X,Y,'.',label='data')
		ax.plot(X,new_Y,'^',label='shifted data')
		XX = np.linspace(0,1,1000); YY = np.polyval(pol_fit,XX)
		ax.plot(XX,YY,'--',label='polynom')
		ax.legend()
		ax.set_title(r'$n = $'+str(agg))
	return string

def Validate_V4(string,Y,pol_fit,agg,verbose=False,display=False):
	if verbose:
		print('old_string:\n',Validate(string,Y))
	new_Y = Get_shifted_data_V2(Y,pol_fit)
	string1 = Sub_validateV4(string,Y,verbose=verbose)
	string2 = Sub_validateV4(string,new_Y,verbose=verbose)
	if len(String_to_letter(string1))<len(String_to_letter(string2)):
		string = string1
	else:
		string = string2
	#it can be that the curve is asymptotically flat and so spurious domains appear because of fluctuations
	if len(String_to_letter(string))>1:
		first_letter = string[0]; domains = Find_domains(string)[1:]
		start = domains[0][0]; truncated_string = Get_trunc_string(Y[start:],verbose=verbose)
		if verbose:
			print('truncated string:\n',truncated_string)
		if len(String_to_letter(truncated_string))==1:
			if truncated_string[0]==first_letter or (np.array(list(truncated_string))=='0').all():
				string = first_letter*len(string)
			else:
				string = string[:start]+truncated_string
	if verbose:
		print('new_string:\n',string)
	if display:
		n = len(Y); X = np.linspace(0,1,n)
		#display the data points, the polynomial fit and the spline fit
		fig,ax = plt.subplots(1,1,constrained_layout=True)
		ax.plot(X,Y,'.',label='data')
		ax.plot(X,new_Y,'^',label='shifted data')
		XX = np.linspace(0,1,1000); YY = np.polyval(pol_fit,XX)
		ax.plot(XX,YY,'--',label='polynom')
		ax.legend()
		ax.set_title(r'$n = $'+str(agg))
	return string

#convert a curve into a string (effective sign of the curve derivative)
#first interpolate the curve using MDL
##actually take care that modifying a model may lead to improvement on one case and new errors on others:
#we do not want the model with minimum nb of errors but a model which is stricly better, i.e. all correct
#answers are still correct and some wrong answers are rectified
def Stoch_tab_to_string_V2(tab):
	if tab.ndim==2:
		Y = tab[1,:]
	else:
		Y = tab
	der_tab = Y[1:]-Y[:-1]
	#first check for monotonicity
	sign_der = np.sign(der_tab)
	if (sign_der<0).all():
		return '-'*len(der_tab)
	elif (sign_der>0).all():
		return '+'*len(der_tab)
	elif (sign_der==0).all():
		return '0'*len(der_tab)
	'''
	###
	#handle the flat case: we combine different estimators and use a majority rule
	is_flat = 0
	#test of statistical independence btw x and y values
	X = np.asarray(range(len(Y)),dtype=float)
	if distance_correlation_t_test(X,Y).pvalue>0.05:
		is_flat += 1
	#check for the hypothesis of Gaussian white noise (in this case the curve is described as flat)
	if shapiro(der_tab)[1]>0.05 and shapiro(Y)[1]>0.05:
		#check that the tab_derivative is centered around zero and has std_der = sqrt(2)*std_tab
		nb_test = 50; nb_match = 0; scale = math.sqrt(2)*np.std(Y)
		for _ in range(nb_test):
			nb_match += int(ks_2samp(der_tab,np.random.normal(loc=0,scale=scale,size=len(der_tab)))[1]>0.05)
		if nb_match>nb_test/2:
			is_flat += 1
	#test whether there is no rank correlation btw x and y values
	p_value = kendalltau(Y,range(len(Y)),alternative='two-sided')[1]
	if p_value>0.05:
		is_flat += 1
	#test whether there is no rank correlation btw x and y values
	X = list(range(len(Y)))
	if permutation_test((X,),lambda x: spearmanr(x,Y).statistic,permutation_type='pairings').pvalue>0.05:
		is_flat += 1
	if is_flat>=2:
		return '0'*len(der_tab)
	###
	'''
	#if the derivative has accidental zeros, remove them
	pos_zeros = []
	for ind,sign in enumerate(sign_der):
		if sign==0:
			pos_zeros.append(ind)
	if pos_zeros:
		#add a random noise on data points with amplitude equal to the minimum of the non-zero absolute value
		#of the raw derivative
		m = max(min([abs(el) for el in der_tab if el!=0]),1e-10)
		for ind in pos_zeros:
			Y[ind] += (2*rd.random()-1)*m
	#interpolate the curve with a polynom defined on [0,1] using MDL
	pol_fit = Get_pol_fit(Y)
	#if the polynom is constant then the string is full of zeros
	if len(pol_fit)==1:
		return '0'*len(sign_der)
	#get the string description of the polynom
	string = Pol_to_string(pol_fit,Y)[:-1]
	#validate this description: collect all deviations and absorb domains as long as the maximum deviation
	#is unchanged
	return Validate(string,Y)

def Stoch_tab_to_string_V3(tab,agg,verbose=False,display=False):
	if tab.ndim==2:
		Y = tab[1,:]
	else:
		Y = tab
	#first check for monotonicity
	sign_der = np.sign(Y[1:]-Y[:-1])
	if (sign_der<0).all():
		return '-'*len(sign_der)
	elif (sign_der>0).all():
		return '+'*len(sign_der)
	elif (sign_der==0).all():
		return '0'*len(sign_der)
	#normalise Y
	m,M = min(Y),max(Y)
	Y = (Y-m)/(M-m)
	der_tab = Y[1:]-Y[:-1]
	'''
	#if the derivative has accidental zeros, remove them
	pos_zeros = []
	for ind,sign in enumerate(sign_der):
		if sign==0:
			pos_zeros.append(ind)
	if pos_zeros:
		#add a random noise on data points with amplitude equal to the minimum of the non-zero absolute value
		#of the raw derivative
		m = max(min([abs(el) for el in der_tab if el!=0]),1e-10)
		for ind in pos_zeros:
			Y[ind] += (2*rd.random()-1)*m
	'''
	#interpolate the curve with a polynom defined on [0,1] using MDL
	pol_fit = Get_pol_fit(Y)
	#if the polynom is constant then the string is full of zeros
	if len(pol_fit)==1:
		return '0'*len(sign_der)
	#get the string description of the polynom
	string = Pol_to_string(pol_fit,Y)[:-1]
	if verbose:
		print('pol_string:\n',string)
	#validate this description: collect all deviations and absorb domains as long as the maximum deviation
	#is unchanged
	return Validate_V3(string,Y,pol_fit,agg,verbose=verbose,display=display)

#version 6: take as description the simplest one that is compatible with the CI
#sometimes the MDL proposition is wrong so here we test different propositions and choose
#the one that gives rise to the string description closest from the raw data sign derivative
#(we use the string edit distance)
def Stoch_tab_to_string_V4(tab,agg,verbose=False,display=False):
	if tab.ndim==2:
		Y = tab[1,:]
	else:
		Y = tab
	#first check for monotonicity
	sign_der = np.sign(Y[1:]-Y[:-1])
	if (sign_der<0).all():
		return '-'*len(sign_der)
	elif (sign_der>0).all():
		return '+'*len(sign_der)
	elif (sign_der==0).all():
		return '0'*len(sign_der)
	m,M = min(Y),max(Y)
	'''
	if (M-m)/M<0.007:
		print('ok')
		return '0'*len(sign_der)
	'''
	if (M-m)<=max(abs(Y[1:]-Y[:-1]))*1.05:
		return '0'*len(sign_der)
	#normalise Y
	Y = (Y-m)/(M-m)
	der_tab = Y[1:]-Y[:-1]
	#interpolate the curve with a polynom defined on [0,1] using MDL and consider two possibilities
	pol_fit = Get_pol_fit_V2(Y)
	#if the polynom is constant then the string is full of zeros
	if len(pol_fit)==1:
		return '0'*len(sign_der)
	#get the string description of the polynom
	string = Pol_to_string_V2(pol_fit,Y)
	if verbose:
		print('pol_string:\n',string)
	#validate this description: collect all deviations and absorb domains as long as the maximum deviation
	#is unchanged
	return Validate_V4(string,Y,pol_fit,agg,verbose=verbose,display=display)

#subroutine for Stoch_tab_to_string_V5
#optimzation to do: avoid computing multiple times the data matrix (Vandermonde)
def Get_string_V5(Y):
	sign_der = np.sign(Y[1:]-Y[:-1])
	if (sign_der<0).all():
		return '-'*len(sign_der)
	elif (sign_der>0).all():
		return '+'*len(sign_der)
	elif (sign_der==0).all():
		return '0'*len(sign_der)
	m,M = min(Y),max(Y)
	if (M-m)/M<0.005:
		return '0'*len(sign_der)
	if (M-m)<=max(abs(Y[1:]-Y[:-1]))*1.05:
		return '0'*len(sign_der)
	#normalise Y
	Y = (Y-m)/(M-m)
	der_tab = Y[1:]-Y[:-1]
	#interpolate the curve with a polynom defined on [0,1] using MDL and consider two possibilities
	pol_fit = Get_pol_fit_V2(Y)
	#if the polynom is constant then the string is full of zeros
	if len(pol_fit)==1:
		return '0'*len(sign_der)
	return Pol_to_string_V3(pol_fit,Y)

#use Gaussian noise with variance given by the CI of polynomial fit and then choose as
#description the most frequent one
def Stoch_tab_to_string_V5(tab,agg,verbose=False,display=False):
	if tab.ndim==2:
		Y = tab[1,:]
	else:
		Y = tab
	#first check for monotonicity
	sign_der = np.sign(Y[1:]-Y[:-1])
	if (sign_der<0).all():
		return '-'*len(sign_der)
	elif (sign_der>0).all():
		return '+'*len(sign_der)
	elif (sign_der==0).all():
		return '0'*len(sign_der)
	m,M = min(Y),max(Y)
	if (M-m)/M<0.005:
		return '0'*len(sign_der)
	if (M-m)<=max(abs(Y[1:]-Y[:-1]))*1.05:
		return '0'*len(sign_der)
	#normalise Y
	Y = (Y-m)/(M-m)
	der_tab = Y[1:]-Y[:-1]
	#interpolate the curve with a polynom defined on [0,1] using MDL and consider two possibilities
	pol_fit = Get_pol_fit_V2(Y)
	#if the polynom is constant then the string is full of zeros
	if len(pol_fit)==1:
		return '0'*len(sign_der)
	#determine the polynom shell resulting from confidence on the fitting curve
	smoother = PolynomialSmoother(degree=len(pol_fit)-1)
	smoother.smooth(Y)
	#generate intervals
	low_ci,up_ci = smoother.get_intervals('confidence_interval',confidence=0.05)
	#we add Gaussian noise to the data with std equal to half the CI width
	#then we choose as description the most frequent one
	list_std = (up_ci-low_ci)/2; original = Pol_to_string_V3(pol_fit,Y)
	nb_test = 50; dic_sub = {original:1}
	for _ in range(nb_test):
		new_Y = Y + np.random.default_rng().normal(loc=0.0,scale=list_std).flatten()
		string = Get_string_V5(new_Y)
		if string in dic_sub:
			dic_sub[string] += 1
		else:
			dic_sub[string] = 1
	#display the abundancies of obtained strings
	sorted_key = sorted(dic_sub.keys(),key=lambda s:dic_sub[s],reverse=True)
	if display:
		fig,ax = plt.subplots(1,1,constrained_layout=True)
		ax.plot([dic_sub[s] for s in sorted_key],'.')
		ax.plot([0,len(dic_sub)-1],[dic_sub[original]]*2,'--')
	max_pos = 0; keep = True; max_nb = dic_sub[sorted_key[0]]
	while keep:
		new_nb = dic_sub[sorted_key[max_pos]]
		if new_nb!=max_nb:
			keep = False
		else:
			max_pos += 1
			if max_pos==len(dic_sub):
				keep = False
	string = min(sorted_key[:max_pos],key=lambda s:len(String_to_letter(s)))
	if verbose:
		print('original string:')
		print(original)
		print('most frequent strings:')
		for s in sorted_key[:max_pos]:
			print(s)
		print('end of most frequent strings')
		print('chosen string:')
		print(string)
		print('end of function')
	if display:
		plt.show()
	return string

def Test(agg,obs,name,verbose=False,display=False):
	tab = Load_vs_b(name,agg,obs)
	#Test_cross_val(tab,agg); plt.show(); exit()
	#Display_pol_fit(tab)
	string = Stoch_tab_to_string_V5(tab,agg,verbose=verbose,display=display)
	print('final string\n',string)
	Display_symbol(string,tab)
	plt.show(); exit()

#collect all missing values
def Get_missing_obs(list_name,obs_with_arg,list_agg):
	list_obs = Rewrite_listobs(obs_with_arg)
	res = {name:{obs:[] for obs in obs_with_arg} for name in list_name}
	scalar_obs = []; distr_obs = []
	for i,obs in enumerate(list_obs):
		if Get_nature(obs[0])=='scalar':
			scalar_obs.append(i)
		else:
			distr_obs.append(i)
	for name in list_name:
		savename = Get_savename(name)
		for i in scalar_obs:
			obs = list_obs[i]; raw_obs = obs_with_arg[i]
			num = Find_folder(*obs)
			for agg in list_agg:
				try:
					data = np.loadtxt('codata/'+obs[0]+'/'+str(num)+'/Flow_'+savename+'_n'+str(agg)+'.txt')
				except:
					res[name][raw_obs].append(agg)
		for i in distr_obs:
			obs = list_obs[i]; raw_obs = obs_with_arg[i]
			num = Find_folder(*obs)
			for agg in list_agg:
				try:
					data = np.loadtxt('codata/'+obs[0]+'/'+str(num)+'/Flow_'+savename+'_n'+str(agg)+'_b'+str(1)+'.txt')
				except:
					res[name][raw_obs].append(agg)
	#remove empty parts
	key_to_remove = set()
	for name,el in res.items():
		res[name] = {obs:val for obs,val in el.items() if val}
		if not res[name]:
			key_to_remove.add(name)
	for name in key_to_remove:
		del res[name]
	return res

#generate and save figures from obs realizations
def Plot_obs(obs_with_arg,list_name):
	standard_obs = Rewrite_listobs(obs_with_arg)
	scalar_obs = []; distr_obs = []
	for obs in standard_obs:
		if Get_nature(obs[0])=='scalar':
			scalar_obs.append(obs)
		else:
			distr_obs.append(obs)
		Check_dir(obs)
	for name in list_name:
		print(Get_savename(name)+' begins')
		print('\tscalar obs begins')
		for obs in scalar_obs:
			print('\t\t'+obs[0]+' begins')
			Plot_scalar_nb_flow(name,obs,list_b,list_agg)
		print('\tdistr obs begins')
		for obs in distr_obs:
			print('\t\t'+obs[0]+' begins')
			Plot_distr_nb_flow(name,obs,list_b,list_agg)

#do the analysis for the missing values
def Complete_missing_obs(list_name,list_obs,list_agg,list_b):
	missing_obs = Get_missing_obs(list_name,list_obs,list_agg)
	for name,val in missing_obs.items():
		list_agg = set()
		for obs,el in val.items():
			list_agg = list_agg.union(set(el))
		Internalobs_nb_flow(name,list(val.keys()),list_b,list(list_agg))

#visualize the ten most frequent NCTN
def Visu_NCTN(dic_ETN,depth,savefig=''):
	#sort the motifs by decreasing abundancy
	list_seq = sorted(dic_ETN.keys(),key=lambda seq:dic_ETN[seq],reverse=True)
	fig,ax = plt.subplots(2,5,constrained_layout=True,figsize=(12,6))
	fontsize = 16
	for i in range(2):
		for j in range(5):
			num = 5*i+j
			ax[i,j].set_axis_off()
			ax[i,j].set_title(etn_lib.Ordinal(num),fontsize=fontsize)
			etn_lib.Plot_ETN(list_seq[num],depth,ax[i,j])
	plt.savefig('test_GOL/visu_NCTN'+savefig+'.png')
	plt.close()

#note: do we observe patterns when visualizing the adjacency matrix of an empirical TN ? we may perform
#an appropriate permutation of the nodes (i.e. node ordering). Can we find the ordering that maximizes
#the structural complexity of the adjacency matrix ? Or the ordering that maximizes the compressibility of
#the adjacency matrix ?

n = 150; N = n**2; random_init = True
first_image = np.zeros((n,n),dtype=int)
#random initialization
if random_init:
	threshold = 0.1+0.2*rd.random()
	for i in range(n):
		for j in range(n):
			if rd.random()<threshold:
				first_image[i,j] = 1
#draw something as initial condition
pass
lg = Life_game(N,first_image=first_image)
duree = 2000
for num in range(10):
	print('run '+str(num)+' begins')
	data = lg.Evolve(duree,starting_time=400)
	#basic analysis of the resulting TN
	net = tp.Temp_net(data)
	net.Format()
	net.Init(1)
	#draw the edge activity vs time
	fig,ax = Setup_Plot('t','nb of edges',fontsize=14)
	ax.plot([len(val.edges) for val in net.TN.values()])
	ax.set_title('edge activity',fontsize=14)
	plt.savefig('test_GOL/edge_space_weight_run'+str(num)+'.png')
	plt.close()
	#draw the edge duration distribution with log-log scale
	fig,ax = Setup_Plot(r'$\log_{10}(\Delta t)$',r'$\log_{10}(P)$',fontsize=14)
	ax.set_title('edge duration distribution',fontsize=14)
	histo = net.Get_obs(*Rewrite_obs('edge0duration'))
	ax.plot(*Raw_to_binned(histo),'.')
	plt.savefig('test_GOL/edge0duration_run'+str(num)+'.png')
	plt.close()
	#draw the edge interduration distribution with log-log scale
	fig,ax = Setup_Plot(r'$\log_{10}(\Delta t)$',r'$\log_{10}(P)$',fontsize=14)
	ax.set_title('edge interduration distribution',fontsize=14)
	histo = net.Get_obs(*Rewrite_obs('edge0interduration'))
	ax.plot(*Raw_to_binned(histo),'.')
	plt.savefig('test_GOL/edge0interduration_run'+str(num)+'.png')
	plt.close()
	#draw the edge event duration distribution with log-log scale
	fig,ax = Setup_Plot(r'$\log_{10}(n)$',r'$\log_{10}(P)$',fontsize=14)
	ax.set_title('edge event duration distribution',fontsize=14)
	histo = net.Get_obs(*Rewrite_obs('edge0event_duration'))
	ax.plot(*Raw_to_binned(histo),'.')
	plt.savefig('test_GOL/edge0event_duration_run'+str(num)+'.png')
	#draw the cc_size distribution with log-log scale
	fig,ax = Setup_Plot(r'$\log_{10}(n)$',r'$\log_{10}(P)$',fontsize=14)
	ax.set_title('cc size distribution',fontsize=14)
	histo = net.Get_obs(*Rewrite_obs('cc_size'))
	ax.plot(*Raw_to_binned(histo),'.')
	plt.savefig('test_GOL/cc_size_run'+str(num)+'.png')
	#draw the inst_deg distribution with lin-log scale
	fig,ax = Setup_Plot(r'$k$',r'$\log_{10}(P)$',fontsize=14)
	ax.set_title('inst deg distribution',fontsize=14)
	histo = net.Get_obs(*Rewrite_obs('inst_deg'))
	ax.plot(*Raw_to_exp(histo),'.')
	plt.savefig('test_GOL/inst_deg_run'+str(num)+'.png')
	#draw the edge0time_weight distribution with log-log scale
	fig,ax = Setup_Plot(r'$\log_{10}(n)$',r'$\log_{10}(P)$',fontsize=14)
	ax.set_title('edge time weight distribution',fontsize=14)
	histo = net.Get_obs(*Rewrite_obs('edge0time_weight'))
	ax.plot(*Raw_to_binned(histo),'.')
	plt.savefig('test_GOL/edge0time_weight_run'+str(num)+'.png')
	plt.close()
	'''
	#draw the 10 most frequent NCTN of depth 3
	depth = 3
	Visu_NCTN(net.ETN_histo(depth),depth,savefig='_run'+str(num))
	'''
	#draw new initial conditions
	first_image = np.zeros((n,n),dtype=int)
	#random initialization
	if random_init:
		threshold = 0.1+0.2*rd.random()
		for i in range(n):
			for j in range(n):
				if rd.random()<threshold:
					first_image[i,j] = 1
	lg.Refresh(first_image)

exit()
#draw the edge duration distribution with lin-log scale
fig,ax = Setup_Plot(r'$\Delta t$',r'$\log_{10}(P)$',fontsize=14)
ax.set_title('edge duration distribution',fontsize=14)
ax.plot(*Raw_to_exp(histo),'.')
plt.savefig('test_GOL/edge0duration_run_exp'+str(num)+'.png')
plt.close()
n = 20; N = n**2
first_image = np.random.randint(0,2,N).reshape(n,n)
lg = Life_game(N,first_image=first_image)
duree = 10
plt.ion()
fig,ax = plt.subplots(1,1,constrained_layout=True)
im = ax.imshow(first_image,cmap='gnuplot2')
for t in range(duree):
	im.set_data(lg.grid)
	fig.canvas.flush_events()
	fig.canvas.draw()
	time.sleep(1)
	lg.Update()
exit()
pass

'''
#plot the first smoothed timeseries with intervals
plt.figure()
plt.plot(smoother.smooth_data[0], linewidth=3, color='blue')
plt.plot(smoother.data[0], '.k')
plt.fill_between(range(len(smoother.data[0])), low_pi[0], up_pi[0], alpha=0.3, color='blue')
plt.fill_between(range(len(smoother.data[0])), low_ci[0], up_ci[0], alpha=0.3, color='blue')
'''
#check the influence of errors on sentences on the similarity matrix btw observables
#how far am I from mental sanity ?
pass
b_min = 1; b_max = 300; step = 10; nb = (b_max-b_min)//step + 1
list_b = [k*step+b_min for k in range(nb)]
list_name = [('ADM',9,'conf16'),('ADM',18,'conf16'),'conf16',('min_ADM',1),('min_ADM',2),('min_EW',1),('min_EW',2),('min_EW',3),'utah']
list_agg = [1,2,3,4,5,10,20,50,100]
'''
list_obs = ['ICC',('ECTN0motif_error',3),('NCTN0motif_error',3),('NCTN0nb_diff',3),('ECTN0nb_diff',3),('NCTN0nb_tot',3),('ECTN0nb_tot',3)]
for el in ['node','edge']:
	for prefix in ['','inter','event_']:
		for scalar in ['avg','frac']:
			obs = el+'0'+prefix+'duration0'+scalar
			list_obs.append(obs)
#Sim_diag(list_obs,list_agg,list_name,savefig='stoch_V5')
#exit()
for el in ['node','edge']:
	for prefix in ['','inter','event_']:
		obs = Rewrite_obs(el+'0'+prefix+'duration')
		print(str(obs)+" begins")
		for name in list_name:
			print('\t'+Get_savename(name)+' begins')
			Scalar_from_distr(name,obs,list_b,list_agg)
'''

obs = ('NCTN0motif_error', 3)
name = ('ADM',9,'conf16') # ('NCTN0motif_error', 3) ('NCTN0nb_diff', 3) ('ECTN0nb_diff', 3)
#sentence = Describe_scalar(name,obs,list_agg)
#print(sentence); exit()
Test(1,obs,name,verbose=True,display=True)
for agg in list_agg:
	print('n =',agg)
	tab = Load_vs_b(name,agg,obs)
	#Display_pol_fit(tab); exit()
	string = Stoch_tab_to_string_V5(tab,agg,display=True); print(string)
plt.show(); exit()
Display_symbol(string,tab)
exit()

for obs in list_obs:
	print(obs)
	for name in list_name:
		print('\t',name)
		print('\t\t',Describe_scalar(name,obs,list_agg))
exit()


for obs in list_obs:
	print(obs)
	if type(obs)==tuple:
		savefig = obs[0]
	else:
		savefig = obs
	dic_sentence = {}
	for num,name in enumerate(list_name):
		print('\t',name)
		dic_sentence[num+1] = Describe_scalar(name,obs,list_agg)
	diag = Sentences_to_diag(dic_sentence)
	Display_diag(diag,savefig=savefig)
exit()

b_min = 1; b_max = 300; step = 10; nb = (b_max-b_min)//step + 1
list_b = [k*step+b_min for k in range(nb)]
list_agg = [1,2,3,4,5,10,20,30,40,50,100]

obs_with_arg = ['edge0interduration','edge0newborn_duration',('ECTN0time_weight',3),('NCTN0duration',3)]
obs_with_arg += [('ECTN0profile_from_motif',3),('NCTN0profile_from_motif',3),'node_space_weight']
obs_with_arg += ['inst_deg','cc_size','edge_space_weight',('NCTN0motif_error',3),'ICC','avg_cc']
obs_with_arg += [('ECTN0motif_error',3),('NCTN0nb_diff',3),('ECTN0nb_diff',3)]
obs_with_arg += [('NCTN0nb_tot',3),('ECTN0nb_tot',3),'node0time_weight','edge0time_weight']
obs_with_arg += [('NCTN0time_weight',3),'edge0duration','edge0event_duration',('ECTN0event_duration',3)]
obs_with_arg += ['node0interduration','node0newborn_duration','node0duration','node0event_duration']
obs_with_arg += [('ECTN0duration',3),('ECTN0interduration',3),('ECTN0newborn_duration',3)]
obs_with_arg += [('NCTN0interduration',3),('NCTN0newborn_duration',3),('NCTN0event_duration',3)]
obs_with_arg += ['edge0nb_tot','edge0nb_diff']
#####
#obs_with_arg = [('ECTN0profile_from_motif',3)]
obs_with_arg = [('ECTN0time_weight',3),('ECTN0profile_from_motif',3),('ECTN0motif_error',3),('ECTN0nb_diff',3),('ECTN0nb_tot',3),('ECTN0event_duration',3),('ECTN0duration',3),('ECTN0interduration',3),('ECTN0newborn_duration',3)]
