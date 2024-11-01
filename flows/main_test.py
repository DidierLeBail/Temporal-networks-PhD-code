#study reversibility and causality in temporal networks
import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(PROJECT_ROOT)
import Librairies.Temp_net as tp
from Librairies.settings import XP_data,Setup_Plot,Vector_distance,Cosim,Get_versions,Load_instance_param,Get_savename,Raw_to_binned,Load_TN_ADM
from Librairies.atn import ADM_class,Min_EW
import Librairies.ETN as etn_lib

import numpy as np
import math
import matplotlib.pyplot as plt
import networkx as nx
import random as rd
from scipy.optimize import curve_fit
from scipy.stats import ks_2samp
from zipfile import ZipFile,ZIP_DEFLATED

import community.community_louvain as community_louvain

import time

def Check_folder_flow_X(obs_with_arg):
	res = [0]*len(obs_with_arg)
	for i,(obs,arg) in enumerate(obs_with_arg):
		if not os.path.isdir(CURRENT_ROOT+'/codata/'+obs):
			os.mkdir(CURRENT_ROOT+'/codata/'+obs)
			res[i] = 0
			os.mkdir(CURRENT_ROOT+'/codata/'+obs+'/0')
			np.savetxt(CURRENT_ROOT+'/codata/'+obs+'/record.txt',[str(arg),'0'],fmt='%s')
		else:
			tab = np.loadtxt(CURRENT_ROOT+'/codata/'+obs+'/record.txt',dtype=str)
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
				os.mkdir(CURRENT_ROOT+'/codata/'+obs+'/'+str(res[i]))
			np.savetxt(CURRENT_ROOT+'/codata/'+obs+'/record.txt',np.array(list(zip(*record.items())),dtype=str),fmt='%s')
	return res

#draw the distribution of the edge activity derivative
def Draw_edge_der(name,agg):
	net = tp.Temp_net(tp.Load_TN(name))
	net.Init(agg)
	X,Y = zip(*net.Edge_der().items())
	Y = np.asarray(Y,dtype=float)
	Y /= np.sum(Y)
	xlabel = r'$dE$'
	ylabel = r'$P(dE)$'
	fig,ax = Setup_Plot(xlabel,ylabel)
	ax.plot(X,Y,'.')
	plt.show()
	plt.savefig('figures/edge_act/Draw_edge_der_'+name+'.png')

#returns 'scalar'/'distr' if obs is a scalar/distribution observable
def Get_nature(obs):
	scalar_signature = {'ICC','cc_size','nb','error','entropy'}
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

#observable data are stored in 'codata/obs/arg_num/Flow_'+savename+'_n'+str(agg)+'_b'+str(b)+'.txt'
#compute the flow of internal observables (i.e. observables that can be computed
#with only the info of one single state (n,b)) under TS and temporal aggregation
#obs_with_arg = list containing tuples (obs,arg)
def Internalobs_nb_flow(name,obs_with_arg,list_b,list_agg):
	#check the format is correct
	for i,el in enumerate(obs_with_arg):
		if type(el)!=tuple:
			obs_with_arg[i] = (el,())
		elif type(el[1])!=tuple:
			obs_with_arg[i] = (el[0],tuple([el[1]]))
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

#display the results of Internalobs_nb_flow
def Plot_internal_nb_flow(name,obs):
	pass

#compute the distribution of a given observable for different levels of randomization of a dataset
#obs_agg is the aggregation level at which obs should be computed
def Compute_local_distr(name,list_obs,b_max=300,step=10,choice='TS',obs_agg=1):
	if list_obs=='' or list_obs==[]:
		return None
	if type(list_obs)==str:
		list_obs = [list_obs]
	elif type(list_obs)!=list:
		raise ValueError('list_obs should be either a string or a list of strings')
	savename = Get_savename(name)
	net = tp.Temp_net(tp.Load_TN(name))
	net.Init(1)
	if choice=='TS':
		func = net.Local_TS
	elif choice=='TR':
		func = net.Local_TR
	else:
		raise ValueError('choice should be either TS (time shuffling) or TR (time reversal)')
	nb = b_max//step + 1
	X = [k*step+1 for k in range(nb)]
	for b in X:
		print(str(b)+' begins')
		func(b,agg=obs_agg)
		for obs in list_obs:
			histo = net.Get_obs(obs)
			#save data
			np.savetxt('codata/Compute_local_distr_'+choice+'_'+savename+'_'+obs+str(obs_agg)+'_'+str(b)+'.txt',np.array(list(zip(*histo.items()))),fmt='%d')

#compute a given observable for different levels of randomization of a dataset
#obs_agg is the aggregation level at which obs should be computed
def Compute_local_scalar(name,list_obs,b_min=1,b_max=300,step=10,choice='TS',list_agg=[1,2,3,4,5,10,20,50,100]):
	if list_obs=='' or list_obs==[]:
		return None
	if type(list_obs)==str:
		list_obs = [list_obs]
	elif type(list_obs)!=list:
		raise ValueError('list_obs should be either a string or a list of strings')
	if list_agg==0 or list_agg==[]:
		return None
	if type(list_agg)==int:
		list_agg = [list_agg]
	elif type(list_agg)!=list:
		raise ValueError('list_agg should be either an integer or a list of integers')
	savename = Get_savename(name)
	net = tp.Temp_net(tp.Load_TN(name))
	net.Init(1)
	if choice=='TS':
		func = net.Local_TS
	elif choice=='TR':
		func = net.Local_TR
	else:
		raise ValueError('choice should be either TS (time shuffling) or TR (time reversal)')
	nb = (b_max-b_min)//step + 1
	X = [k*step+b_min for k in range(nb)]
	for agg in list_agg:
		res = {obs:[] for obs in list_obs}
		for b in X:
			print(str(b)+' begins')
			func(b,agg=agg)
			for obs in list_obs:
				res[obs].append(net.Get_obs(obs))
		#save data
		for obs,val in res.items():
			np.savetxt('codata/Compute_local_scalar_'+choice+'_'+savename+'_'+obs+str(agg)+'.txt',val)

#compute the dic_ETN similarity btw a locally time shuffled version of the dataset name and
#the original dataset as a fct of the size of the time window, for a given level of aggregation
def Compute_local_dic_ETN(name,b_min=1,b_max=300,step=10,choice='TS',obs_agg=1):
	net = tp.Temp_net(tp.Load_TN(name))
	net.Init(obs_agg)
	if choice=='TS':
		func = net.Local_TS
	elif choice=='TR':
		func = net.Local_TR
	else:
		raise ValueError('choice should be either TS (time shuffling) or TR (time reversal)')
	#ori_dic_ETN = original dic_ETN
	ori_dic_ETN = net.ETN_histo(3,tot=True)
	nb = (b_max-b_min)//step + 1
	X = [k*step+b_min for k in range(nb)]; Y = []
	for b in X:
		print(str(b)+' begins')
		func(b,agg=obs_agg)
		Y.append(Cosim(net.ETN_histo(3,tot=True),ori_dic_ETN))
	#save data
	np.savetxt('codata/Compute_local_dic_ETN_'+choice+'_'+Get_savename(name)+str(obs_agg)+'.txt',Y)

#compute the distribution of a given observable for different levels of aggregation after TS(1,\infty)
def Compute_tot_TS_distr(name,obs,agg_max=100,step=5):
	net = tp.Temp_net(tp.Load_TN(name))
	net.Init(1)
	nb = agg_max//step + 1
	X = [k*step+1 for k in range(nb)]
	savename = Get_savename(name)
	for agg in X:
		print(str(agg)+' begins')
		net.Local_TS(net.info['T'],agg=agg)
		histo = net.Get_obs(obs)
		#save data
		np.savetxt('codata/Compute_tot_TS_distr_'+savename+'_'+obs+str(agg)+'.txt',np.array(list(zip(*histo.items()))),fmt='%d')

#compute the distribution of the edge weight cumulated on windows of several sizes
def Compute_cum_edge_weight(name,cum_min=10,cum_max=300,step=10):
	savename = Get_savename(name)
	net = tp.Temp_net(tp.Load_TN(name))
	net.Init(1)
	nb = (cum_max-cum_min)//step + 1
	list_cum = [k*step+cum_min for k in range(nb)]
	for cum in list_cum:
		print('cum '+str(cum)+' begins')
		histo = net.Get_cum_edge_weight_single(cum)
		#save data
		np.savetxt('codata/Compute_cum_edge_weight_'+savename+'_'+str(cum)+'.txt',np.array(list(zip(*histo.items()))),fmt='%d')

#compute and plot the distribution of the variation of the cumulated edge weight
#dw_ij(b) = dw_ij(b+1)-dw_ij(b) is either 0 or 1
#plot the proba that dw = 1 vs w
def Var_cum_EW(name,list_cum):
	net = tp.Temp_net(tp.Load_TN(name))
	net.Init(1)
	fig,ax = Setup_Plot(r'$\frac{w}{\Delta}$',r'$P(w+1,\Delta+1|w,\Delta)$')
	for cum in list_cum:
		print('cum '+str(cum)+' begins')
		#dic_weight[w] = [n0,n1] where n0 = nb of times we had w at level cum and w at level cum+1
		#and n1 = nb of times we had w at level cum and w+1 at level cum+1
		dic_weight = {}
		cum_TN = net.Get_cum_TN(cum)
		for t,val in cum_TN.items():
			if (t+1)*cum<len(net.TN):
				for edge in val.edges:
					w = val[edge[0]][edge[1]]['weight']
					if w not in dic_weight:
						dic_weight[w] = [0,0]
					if net.TN[(t+1)*cum].has_edge(*edge):
						dic_weight[w][1] += 1
					else:
						dic_weight[w][0] += 1
		X = []; Y = []
		for w,val in dic_weight.items():
			X.append(w/cum)
			Y.append(val[1]/sum(val))
		ax.plot(X,Y,'.',label=r'$\Delta = $'+str(cum))
	ax.legend(fontsize=14)
	plt.savefig(PROJECT_ROOT+'/Articles/Macro_obs/figures/cum_EW/'+Get_savename(name)+'_growth_proba.png')
	plt.close('all')

#plot the result computed by Compute_local_scalar for obs = 'diff_nb_ETN'
#and for obs = 'tot_nb_ETN', to compare with the ETN vector similarity evolution vs b
def Plot_scalar_nb_ETN(name,b_min=1,b_max=300,step=10,choice='TS',list_agg=[]):
	savename = Get_savename(name)
	list_marker = ['s','^','<','>','*','v','1','2','3','4','P','p']
	list_color = ['blue','green','red','black','brown','purple','gray']
	list_b = [k*step+b_min for k in range((b_max-b_min)//step+1)]
	#load the data
	data = {'diff':{},'tot':{}}
	for key in data.keys():
		for agg in list_agg:
			data[key][agg] = np.loadtxt('codata/Compute_local_scalar_'+choice+'_'+savename+'_'+key+'_nb_ETN'+str(agg)+'.txt',dtype=float)
	#######
	#choose a marker per aggregation level and a color per key in data
	agg_to_marker = {agg:marker for agg,marker in zip(list_agg,list_marker)}
	key_to_color = {key:color for key,color in zip(data.keys(),list_color)}
	#first plot : plot the nb of diff and tot ETN vs b for the values of agg in list_agg
	if list_agg:
		fig,ax = Setup_Plot(r'$b$','nb of ETN '+r'$(\log_{10})$')
		for key,color in key_to_color.items():
			for agg,marker in agg_to_marker.items():
				tab = data[key][agg]/data[key][agg][0]
				ax.plot(list_b,np.log10(tab),marker,color=color,label=r'$n = $'+str(agg)+' ('+key+')')
		#######
		ax.legend(fontsize=16,ncol=min(3,len(list_agg)),bbox_to_anchor=(0.5,1.23),loc="upper center")
		if not os.path.isdir('figures/nb_ETN'):
			os.mkdir('figures/nb_ETN')
		if not os.path.isdir('figures/nb_ETN/'+savename):
			os.mkdir('figures/nb_ETN/'+savename)
		plt.savefig('figures/nb_ETN/'+savename+'/Nb_ETN.png')
	#second and third plots: plot the nb of diff ETN vs b for early and whole values of agg
	#fourth and fith plots: plot the nb of tot ETN vs b for early and whole values of agg
	dic_agg = {'early':[1,2,3,4,5],'whole':[1,10,20,50,100]}
	nb_type = {'diff':'nb of diff ETN ','tot':'tot nb of ETN '}
	for key,ylabel in nb_type.items():
		for savefig,list_agg in dic_agg.items():
			fig,ax = Setup_Plot(r'$b$',ylabel+r'$(\log_{10})$')
			for agg,color,marker in zip(list_agg,list_color,list_marker):
				data = np.loadtxt('codata/Compute_local_scalar_'+choice+'_'+savename+'_'+key+'_nb_ETN'+str(agg)+'.txt',dtype=float)
				ax.plot(list_b,np.log10(data),marker,color=color,label=r'$n = $'+str(agg))
			ax.legend(fontsize=14,ncol=min(3,len(list_agg)),bbox_to_anchor=(0.5,1.23),loc="upper center")
			plt.savefig('figures/nb_ETN/'+savename+'/Nb_'+key+'_ETN_'+savefig+'.png')
	plt.close('all')

#plot the result computed by Compute_local_scalar for obs = 'ETN_error'
#plot the ETN error vs b for several values of agg
def Plot_scalar_ETN_error(name,b_min=1,b_max=300,step=10,choice='TS',list_agg=[],savefig=''):
	if savefig!='':
		if savefig[0]!='_':
			savefig = '_'+savefig
	savename = Get_savename(name)
	list_marker = ['s','^','<','>','*','v','1','2','3','4','P','p']
	list_color = ['blue','green','red','black','brown','purple','gray']
	list_b = [k*step+b_min for k in range((b_max-b_min)//step+1)]
	#load the data
	data = {}
	for agg in list_agg:
		data[agg] = np.loadtxt('codata/Compute_local_scalar_'+choice+'_'+savename+'_ETN_error'+str(agg)+'.txt',dtype=float)
	#######
	#first plot : plot the ETN error vs b for the values of agg in list_agg
	fig,ax = Setup_Plot(r'$b$','error on ETN proba')#+r' $(\log_{10})$')
	for (agg,Y),marker,color in zip(data.items(),list_marker,list_color):
			ax.plot(list_b,Y,marker,color=color,label=r'$n = $'+str(agg))
	#######
	ax.legend(fontsize=16,ncol=min(3,len(list_agg)),bbox_to_anchor=(0.5,1.23),loc="upper center")
	if not os.path.isdir('figures/ETN_error'):
		os.mkdir('figures/ETN_error')
	if not os.path.isdir('figures/ETN_error/'+savename):
		os.mkdir('figures/ETN_error/'+savename)
	plt.savefig('figures/ETN_error/'+savename+'/ETN_error_vs_b'+savefig+'.png')
	plt.close('all')

#plot the result computed by Compute_local_scalar for obs = obs
#the first figure displays the obs vs b for the values of agg in list_agg
def Plot_scalar(name,obs,ylabel1,labelfig1,b_min=1,b_max=300,step=10,choice='TS',list_agg=[],savefig=''):
	if savefig!='':
		if savefig[0]!='_':
			savefig = '_'+savefig
	savename = Get_savename(name)
	list_marker = ['s','^','<','>','*','v','1','2','3','4','P','p']
	list_color = ['blue','green','red','black','brown','purple','gray']
	list_b = [k*step+b_min for k in range((b_max-b_min)//step+1)]
	#load the data
	data = {}
	for agg in list_agg:
		data[agg] = np.loadtxt('codata/Compute_local_scalar_'+choice+'_'+savename+'_'+obs+str(agg)+'.txt',dtype=float)
	#######
	#first plot : plot the obs vs b for the values of agg in list_agg
	fig,ax = Setup_Plot(r'$b$',ylabel1)
	for (agg,Y),marker,color in zip(data.items(),list_marker,list_color):
			ax.plot(list_b,Y,marker,color=color,label=r'$n = $'+str(agg))
	#######
	ax.legend(fontsize=16,ncol=min(3,len(list_agg)),bbox_to_anchor=(0.5,1.23),loc="upper center")
	if not os.path.isdir('figures/'+obs):
		os.mkdir('figures/'+obs)
	if not os.path.isdir('figures/'+obs+'/'+savename):
		os.mkdir('figures/'+obs+'/'+savename)
	plt.savefig('figures/'+obs+'/'+savename+'/'+labelfig1+savefig+'.png')
	plt.close('all')

#plot the results computed by Compute_local_dic_ETN
def Plot_local_dic_ETN(name,b_min=1,b_max=300,step=10,choice='TS',list_agg=[]):
	savename = Get_savename(name)
	list_marker = ['s','^','<','>','*','v','1','2','3','4','P','p']
	list_color = ['blue','green','red','black','brown','purple','gray']
	list_b = [k*step+b_min for k in range((b_max-b_min)//step+1)]
	#load the data
	data = {}; tot_ETN_vec = np.ones(len(list_b))
	for agg in range(1,11):
		data[agg] = np.loadtxt('codata/Compute_local_dic_ETN_'+choice+'_'+savename+str(agg)+'.txt',dtype=float)
		tot_ETN_vec *= data[agg]
	#load the ETN vector similarity
	ETN_vec = np.loadtxt('codata/Compute_local_'+choice+'_'+savename+'.txt')
	#######
	#plot the ETN similarity at level agg vs b for the values of agg in list_agg
	fig,ax = Setup_Plot(r'$b$','ETN similarity '+r'$(\log_{10})$')
	for agg,marker,color in zip(list_agg,list_marker,list_color):
		ax.plot(list_b,np.log10(data[agg]),marker,color=color,label=r'$n = $'+str(agg))
	#######
	ax.legend(fontsize=14,ncol=min(3,len(list_agg)),bbox_to_anchor=(0.5,1.23),loc="upper center")
	if not os.path.isdir('figures/dic_ETN'):
		os.mkdir('figures/dic_ETN')
	if not os.path.isdir('figures/dic_ETN/'+savename):
		os.mkdir('figures/dic_ETN/'+savename)
	plt.savefig('figures/dic_ETN/'+savename+'/ETN_sim.png')
	fig,ax = Setup_Plot(r'$b$','ETN similarity '+r'$(\log_{10})$')
	#plot the ETN vector similarity (!!!recall it is computed with the 20 most frequent ETN only!!!)
	ax.plot(ETN_vec[0,:],np.log10(1-ETN_vec[1,:]),list_marker[len(list_agg)],color=list_color[len(list_agg)],label='truncated\nETN vector')
	#plot the ETN vector similarity computed with all the ETN
	ax.plot(list_b,np.log10(tot_ETN_vec),list_marker[len(list_agg)+1],color=list_color[len(list_agg)+1],label='complete\nETN vector')
	ax.legend(fontsize=14,ncol=2,bbox_to_anchor=(0.5,1.23),loc="upper center")
	plt.savefig('figures/dic_ETN/'+savename+'/ETN_vec_sim.png')

def Cum_EW(x,a,b,c):
	return a-b*x-c*np.log10(1-np.power(10,x))

#plot the distribution of the edge weight cumulated on windows of several sizes vs this size
#step = 10; cum_max = 1000; cum_min = 10
def Plot_cum_edge_weight(name,cum_min=10,cum_max=300,step=10,nb_visu=3,savefig='',renormalized=False,fit_cum=True):
	if savefig!='':
		if savefig[0]!='_':
			savefig = '_'+savefig
	savename = Get_savename(name)
	list_marker = ['s','^','<','>','*','v','1','2','3','4','P','p']
	list_color = ['blue','green','red','black','brown','purple','gray']
	list_cum = [k*step+cum_min for k in range((cum_max-cum_min)//step+1)]
	chosen_ind = [(k*(len(list_cum)-1))//(nb_visu-1) for k in range(nb_visu)]
	chosen_cum = [list_cum[ind] for ind in chosen_ind]
	#load the data
	data = {}
	for cum in chosen_cum:
		data[cum] = {}
		tab = np.loadtxt('codata/Compute_cum_edge_weight_'+savename+'_'+str(cum)+'.txt',dtype=int)
		for w,nb in zip(tab[0,:],tab[1,:]):
			data[cum][w] = nb
	#######
	#first plot : complete distributions
	fig,ax = Setup_Plot(r'$\log_{10}(\frac{w^{(\Delta)}}{\Delta})$',r'$\log_{10}(P)$')
	for cum,marker,color in zip(chosen_cum,list_marker,list_color):
		X,Y = Raw_to_binned(data[cum])
		if not renormalized:
			X = np.log10(np.power(10,X)/cum); Y = np.log10(np.power(10,Y)*cum)
		else:
			M = np.max(X); m = np.min(X); X = (X-m)/(M-m)
			M = np.max(Y); m = np.min(Y); Y = (Y-m)/(M-m)
		ax.plot(X,Y,marker,color=color,label=r'$\Delta = $'+str(cum))
		if fit_cum:
			param = curve_fit(Cum_EW,X,Y)[0]
			Y_fit = Cum_EW(X,*param)
			ax.plot(X,Y_fit,'--',color=color)
	#######
	ax.legend(fontsize=14,ncol=min(3,nb_visu),bbox_to_anchor=(0.5,1.23),loc="upper center")
	if not os.path.isdir('figures/cum_edge_weight'):
		os.mkdir('figures/cum_edge_weight')
	if not os.path.isdir('figures/cum_edge_weight/'+savename):
		os.mkdir('figures/cum_edge_weight/'+savename)
	plt.savefig('figures/cum_edge_weight/'+savename+'/Distr'+savefig+'.png')
	#second plot : moments
	#normalise the distributions
	for cum,dic in data.items():
		norm = np.sum(list(dic.values()))
		data[cum] = np.array(list(zip(*dic.items())),dtype=float)
		data[cum][1,:] = data[cum][1,:]/norm
		data[cum][0,:] /= cum
	moments = {'mean':{b:np.sum(val[0,:]*val[1,:]) for b,val in data.items()}}
	moments['std'] = {b:math.sqrt(np.sum((val[0,:]**2)*val[1,:])-moments['mean'][b]**2) for b,val in data.items()}
	list_marker = ['s','^','--']; list_color = ['blue','green','orange']
	fig,ax = Setup_Plot(r'$\log_{10}(\Delta)$ (time period)','moments of '+r'$\frac{w^{(\Delta)}}{\Delta} (\log_{10})$')
	for (label,val),marker,color in zip(moments.items(),list_marker,list_color):
		X,Y = zip(*val.items())
		ax.plot(np.log10(X),np.log10(Y),marker,color=color,label=label)
	ax.legend(fontsize=16,ncol=2,bbox_to_anchor=(0.5,1.23),loc="upper center")
	plt.savefig('figures/cum_edge_weight/'+savename+'/Moments'+savefig+'.png')
	plt.close('all')

#fit of the mean inst deg at aggregation level n after TS(1,\infty)
def Avg_agg_deg(n,a,b,c):
	return a*(n**b-c)

#fit of the mean inst deg vs b after TS(1,b)
def Avg_period_deg(b,p1,p2,p3,p4):
	return p1*(1-p2*(b**p3)*np.exp(-b*p4))

#plot the result computed by Compute_tot_TS_distr for obs = 'inst_deg'
def Plot_tot_TS_distr_inst_deg(name,agg_max,step,nb_visu=3):
	list_marker = ['s','^','<','>','*','v','1','2','3','4','P','p']
	list_color = ['blue','green','red','black','brown','purple','gray']
	savename = Get_savename(name)
	list_agg = [k*step+1 for k in range(agg_max//step + 1)]
	#load the data
	data = {agg:np.loadtxt('codata/Compute_tot_TS_distr_'+savename+'_inst_deg'+str(agg)+'.txt',dtype=float) for agg in list_agg}
	#normalise the distributions
	for agg,val in data.items():
		norm = np.sum(val[1,:])
		data[agg][1,:] /= norm
	#first plot: distribution of instantaneous degree for different levels of aggregation
	chosen_ind = [(k*(len(list_agg)-1))//(nb_visu-1) for k in range(nb_visu)]
	chosen_agg = [list_agg[ind] for ind in chosen_ind]
	base_title = r'$d^{(n)}$'
	fig1,ax1 = Setup_Plot(base_title,r"$\log_{10}(P)$ after TS"+r'$(1,\infty)$')
	for agg,marker,color in zip(chosen_agg,list_marker,list_color):
		ax1.plot(data[agg][0,:],np.log10(data[agg][1,:]),marker,color=color,label='n = '+str(agg))
	ax1.legend(fontsize=16,ncol=min(3,nb_visu),bbox_to_anchor=(0.5,1.23),loc="upper center")
	#save the figure
	if not os.path.isdir('figures/inst_deg'):
		os.mkdir('figures/inst_deg')
	if not os.path.isdir('figures/inst_deg/'+savename):
		os.mkdir('figures/inst_deg/'+savename)
	plt.savefig('figures/inst_deg/'+savename+'/Distr_inst_deg_TS_tot.png')
	#second plot: inst deg mean vs aggregation level with its theoretical fit
	fig2,ax2 = Setup_Plot(r'$n$',r'$<d^{(n)}>$ after TS'+r'$(1,\infty)$')
	moments = {'raw':{agg:np.sum(val[0,:]*val[1,:]) for agg,val in data.items()}}
	param = curve_fit(Avg_agg_deg,*zip(*moments['raw'].items()))[0]
	moments['fit'] = {agg:Avg_agg_deg(agg,*param) for agg in data.keys()}
	#moments['fit'] = {agg:3*math.sqrt(agg)+5 for agg in data.keys()}
	key_to_label = {'raw':'raw','fit':'fit: agg edge weight exp = '+str(np.round(param[1]+1,2))}
	#key_to_label = {'raw':'raw','fit':'fit'}
	list_marker = ['.','--']; list_color = ['blue','orange']
	for (key,val),marker,color in zip(moments.items(),list_marker,list_color):
		ax2.plot(*zip(*val.items()),marker,color=color,label=key_to_label[key])
	ax2.legend(fontsize=16,ncol=2,bbox_to_anchor=(0.5,1.23),loc="upper center")
	plt.savefig('figures/inst_deg/'+savename+'/Moments_inst_deg_TS_tot.png')
	plt.close('all')

#draw the result computed by Compute_local_distr for obs = 'ETN_profile'
#obs_agg is the aggregation level at which obs has been computed
def Plot_local_distr_ETN_profile(name,b_max,step,choice='TS',obs_agg=1):
	savename = Get_savename(name)
	if choice not in {'TS','TR'}:
		raise ValueError('choice should be either TS (time shuffling) or TR (time reversal)')
	xlabel = 'window size'; ylabel = 'profile abundancy'
	fig,ax = Setup_Plot(xlabel,ylabel)
	list_label = ['001','010','011','100','101','110','111']
	list_marker = ['s','^','<','>','*','v','1','2','3','4','P','p']
	list_color = ['blue','green','red','black','brown','purple','gray']
	list_histo = [{} for _ in range(7)]; list_b = [k*step+1 for k in range(b_max//step + 1)]
	for b in list_b:
		tab = np.loadtxt('codata/Compute_local_distr_'+choice+'_'+savename+'_ETN_profile'+str(obs_agg)+'_'+str(b)+'.txt',dtype=int)
		for ind in range(7):
			i = tab[0,ind]
			list_histo[i-1][b] = tab[1,ind]
	for color,marker,label,histo in zip(list_color,list_marker,list_label,list_histo):
		ax.plot(*zip(*histo.items()),marker,label=label,color=color,markersize=10)
	ax.legend(fontsize=16,ncol=4,bbox_to_anchor=(0.5,1.23),loc="upper center")
	if not os.path.isdir('figures/ETN_profile'):
		os.mkdir('figures/ETN_profile')
	if not os.path.isdir('figures/ETN_profile'+'/'+savename):
		os.mkdir('figures/ETN_profile'+'/'+savename)
	plt.savefig('figures/ETN_profile/'+savename+'/Plot_local_distr_'+choice+'_'+str(obs_agg)+'.png')
	plt.close('all')

#draw the result computed by Compute_local_distr for obs = 'inst_deg'
#obs_agg is the aggregation level at which obs has been computed
#nb_visu is the number of values of b for which we want to visualize the inst deg distributions
def Plot_local_distr_inst_deg(name,b_max,step,choice='TS',obs_agg=1,nb_visu=3):
	list_marker = ['s','^','<','>','*','v','1','2','3','4','P','p']
	list_color = ['blue','green','red','black','brown','purple','gray']
	savename = Get_savename(name)
	if choice not in {'TS','TR'}:
		raise ValueError('choice should be either TS (time shuffling) or TR (time reversal)')
	list_b = [k*step+1 for k in range(b_max//step + 1)]
	#load the data
	data = {b:np.loadtxt('codata/Compute_local_distr_'+choice+'_'+savename+'_inst_deg'+str(obs_agg)+'_'+str(b)+'.txt',dtype=float) for b in list_b}
	#normalise the distributions
	for b,val in data.items():
		norm = np.sum(val[1,:])
		data[b][1,:] /= norm
	#first plot: instantaneous degree distribution for different TS periods (b values)
	chosen_ind = [(k*(len(list_b)-1))//(nb_visu-1) for k in range(nb_visu)]
	chosen_b = [list_b[ind] for ind in chosen_ind]
	base_title = r'$d^{('+str(obs_agg)+r')}$'
	fig1,ax1 = Setup_Plot(base_title,r"$\log_{10}(P)$"+' after '+choice+'(1,b)')
	for b,marker,color in zip(chosen_b,list_marker,list_color):
		ax1.plot(data[b][0,:],np.log10(data[b][1,:]),marker,color=color,label='b = '+str(b))
	ax1.legend(fontsize=16,ncol=min(3,nb_visu),bbox_to_anchor=(0.5,1.23),loc="upper center")
	#save the figure
	if not os.path.isdir('figures/inst_deg'):
		os.mkdir('figures/inst_deg')
	if not os.path.isdir('figures/inst_deg/'+savename):
		os.mkdir('figures/inst_deg/'+savename)
	plt.savefig('figures/inst_deg/'+savename+'/Distr_inst_deg_'+choice+'_'+str(obs_agg)+'.png')
	#second plot: inst deg mean and std vs b
	#we also compute and display the exponential fit of the mean inst deg 
	fig2,ax2 = Setup_Plot(choice+' period (b)','moment of '+base_title+' after '+choice+'(1,b)')
	moments = {'mean':{b:np.sum(val[0,:]*val[1,:]) for b,val in data.items()}}
	moments['std'] = {b:math.sqrt(np.sum((val[0,:]**2)*val[1,:])-moments['mean'][b]**2) for b,val in data.items()}
	#fit the inst deg mean with an exponential function
	param = curve_fit(Avg_period_deg,*zip(*moments['mean'].items()),bounds=(0,np.inf))[0]
	print('obs_agg: '+str(obs_agg))
	print('param: '+str(param[0])+', '+str(param[1])+', '+str(param[2])+', '+str(param[3]))
	moments['fit'] = {b:Avg_period_deg(b,*param) for b in data.keys()}
	list_marker = ['s','^','--']; list_color = ['blue','green','orange']
	for (label,val),marker,color in zip(moments.items(),list_marker,list_color):
		ax2.plot(*zip(*val.items()),marker,color=color,label=label)
	ax2.legend(fontsize=16,ncol=3,bbox_to_anchor=(0.5,1.23),loc="upper center")
	plt.savefig('figures/inst_deg/'+savename+'/Moments_inst_deg_'+choice+'_'+str(obs_agg)+'.png')
	plt.close('all')

#compare the fit exponent for the inst deg at agg level 10 with the ETN vector similarity vs TS period
def Compare_inst_deg_to_ETN(name,p4,choice='TS'):
	savename = Get_savename(name)
	tab = np.loadtxt('codata/Compute_local_'+choice+'_'+savename+'.txt')
	fig,ax = Setup_Plot('window size',r'$\log_{10}$(ETN vector similarity)')
	ax.plot(tab[0,:],np.log10(1-tab[1,:]),'.',label='raw',color='blue')
	ax.plot(tab[0,:],-p4*tab[0,:],'--',color='orange',label='inst deg fit')
	ax.legend(fontsize=12)
	plt.show()

#compute the ETN vector (agg_max = 10) distance btw a locally time shuffled version of the dataset name and
#the original dataset as a fct of the size of the time window
def Compute_local(name,b_min=1,b_max=300,step=10,choice='TS'):
	net = tp.Temp_net(tp.Load_TN(name))
	net.Init(1)
	if choice=='TS':
		func = net.Local_TS
	elif choice=='TR':
		func = net.Local_TR
	else:
		raise ValueError('choice should be either TS (time shuffling) or TR (time reversal)')
	#ori_ETN_vector = original ETN_vector
	ori_ETN_vector = net.ETN_vector(10,3)[0]
	nb = (b_max-b_min)//step + 1
	X = [k*step+b_min for k in range(nb)]; Y = [0]
	for b in X[1:]:
		print(str(b)+' begins')
		func(b)
		Y.append(Vector_distance(net.ETN_vector(10,3)[0],ori_ETN_vector))
	#save data
	np.savetxt('codata/Compute_local_'+choice+'_'+Get_savename(name)+'.txt',np.array([X,Y]))

#plot the ETN vector (agg_max = 10) distance btw a locally time shuffled version of the dataset name and
#the original dataset as a fct of the size of the time window
def Plot_local(name,choice='TS'):
	if type(name)==tuple:
		savename = name[0]+str(name[1])+name[2]
	else:
		savename = name
	if choice not in {'TS','TR'}:
		raise ValueError('choice should be either TS (time shuffling) or TR (time reversal)')
	xlabel = 'window size'
	ylabel = r'$\log_{10}$(ETN vector similarity)'
	fig,ax = Setup_Plot(xlabel,ylabel)
	tab = np.loadtxt('codata/Compute_local_'+choice+'_'+savename+'.txt')
	ax.plot(tab[0,:],np.log10(1-tab[1,:]),'.')
	plt.savefig('figures/Plot_local_'+choice+'_'+savename+'.png')

#same as Plot_local_TS but with mutiple datasets on the same plot
def Multiplot_local(list_name,choice='TS',savefig=''):
	if choice not in {'TS','TR'}:
		raise ValueError('choice should be either TS (time shuffling) or TR (time reversal)')
	xlabel = 'window size'
	ylabel = r'$\log_{10}$(ETN vector similarity)'
	markers = ['.','s','^','<','>','*','8','v','1','2','3','4','P','p']
	colors = ['blue','green','red','black','orange','purple','yellow','brown','red','gray','cyan','olive','purple']
	fig,ax = Setup_Plot(xlabel,ylabel)
	for name,marker,color in zip(list_name,markers,colors):
		if type(name)==tuple:
			savename = name[0]+str(name[1])+name[2]
			label = name[0]+'_V'+str(name[1])+'_'+name[2]
		else:
			savename = name
			label = name
		tab = np.loadtxt('codata/Compute_local_'+choice+'_'+savename+'.txt')
		ax.plot(tab[0,:],np.log10(1-tab[1,:]),marker,label=label,color=color)
	ax.legend(fontsize=12,ncol=3)
	plt.savefig('figures/Multiplot_local_'+choice+savefig+'.png')

#compare the value of alpha observed in the version 9 with the inverse characteristic time
#observed in XP datasets
def Comp_alpha_time():
	path_to_V9 = os.path.join(PROJECT_ROOT,'Articles/ADM_class/analysis/ADM_class_V9/')
	time_th = []
	for name in XP_data:
		path = os.path.join(path_to_V9,name+'/best_param.txt')
		best_param = np.loadtxt(path,dtype=str,delimiter=',')
		
		best_param = {best_param[0,i]:float(best_param[1,i])}
		pass
		time_th.append(float(best_param[1,i]))
	#load XP data and estimate the inverse characteristic time
	inv_time = []
	for name in XP_data:
		tab = np.loadtxt('codata/Compute_local_TS_'+name+'.txt')
		dic = {int(np.round(b)):np.log10(1-y) for b,y in zip(tab[0,:],tab[1,:])}
		inv_time.append((dic[51]-dic[251])/200)
	xlabel = 'inverse TS time'
	ylabel = r'$\alpha$'
	fig,ax = Setup_Plot(xlabel,ylabel)
	ax.plot(inv_time,alpha,'.')
	plt.savefig('figures/Comp_alpha_time_ADM_V9.png')

#create the simplest ADM version with edge pruning
#save it as edge_pruning.txt
def Create_simplest_with_edge_pruning():
	version = Get_versions()[18]
	#load instance parameters
	dic_param = Load_instance_param(18,'conf16')
	#load XP info of conf16
	XP_info = {'N':138,'T':3635,'nb of edges':153371,'sigma':0.34,'mu':-0.56}
	#modifies the version
	version['remove'] = 'edge'
	#decide of the new parameter
	dic_param['lambda'] = 22
	#remove the old parameter
	del dic_param['p_d']
	#generate the model instance
	Model = ADM_class(XP_info,**version)
	for param in Model.free_param.keys():
		Model.free_param[param] = dic_param[param]
	Model.Refresh()
	np.savetxt(os.path.join(PROJECT_ROOT,'data/edge_pruning.txt'),Model.Evolve(),fmt='%d')

def Get_min_node_pruning(p_d,N=138,return_model=False):
	version = Get_versions()[7]
	#load instance parameters
	dic_param = Load_instance_param(7,'conf16')
	#load XP info of conf16
	XP_info = {'N':N,'T':3635,'nb of edges':153371,'sigma':0.34,'mu':-0.56}
	#modifies the version
	version['m'] = 'cst'
	version['a'] = 'cst'
	version['c_ij'] = False
	version['update'] = 'linear'
	version['context'] = None
	version['remove'] = 'node'
	#decide of the new parameters
	dic_param['a'] = 0.3
	dic_param['m'] = 1
	dic_param['p_d'] = p_d
	dic_param['p_u'] = 1
	#remove the old parameter
	del dic_param['m_max'], dic_param['a_min'], dic_param['a_max'], dic_param['lambda']
	#generate the model instance
	Model = ADM_class(XP_info,**version)
	for param in Model.free_param.keys():
		Model.free_param[param] = dic_param[param]
	Model.Refresh()
	if return_model:
		return Model
	return Model.Evolve()

def Get_min_edge_pruning(lamb,N=138,return_model=False):
	version = Get_versions()[7]
	#load instance parameters
	dic_param = Load_instance_param(7,'conf16')
	#load XP info of conf16
	XP_info = {'N':N,'T':3635,'nb of edges':153371,'sigma':0.34,'mu':-0.56}
	#modifies the version
	version['m'] = 'cst'
	version['a'] = 'cst'
	version['c_ij'] = False
	version['update'] = 'linear'
	version['context'] = None
	#decide of the new parameter
	dic_param['a'] = 0.1
	dic_param['m'] = 1
	dic_param['lambda'] = lamb
	#remove the old parameter
	del dic_param['m_max'], dic_param['a_min'], dic_param['a_max']
	#generate the model instance
	Model = ADM_class(XP_info,**version)
	for param in Model.free_param.keys():
		Model.free_param[param] = dic_param[param]
	Model.Refresh()
	if return_model:
		return Model
	return Model.Evolve()

#same as Compute_local but with the minimized versions of V7 with node pruning for several values of p_d
def Compute_local_remove_node():
	list_pd = [1,0.1,1e-2,1e-3,1e-4]
	for i,p_d in enumerate(list_pd):
		print(str(i)+' begins')
		data = Get_min_node_pruning(p_d)
		Compute_local((data,'minimized_pd'+str(i)),b_max=300,step=10,choice='TS')

#same as Compute_local but with the minimized versions of V7 with edge pruning for several values of lambda
def Compute_local_remove_edge():
	list_lamb = [100,10,1,1e-1,1e-2]
	for i,lamb in enumerate(list_lamb):
		print(str(i)+' begins')
		data = Get_min_edge_pruning(lamb)
		Compute_local((data,'minimized_lamb'+str(i)),b_max=300,step=10,choice='TS')

def Multiplot_remove(choice):
	if choice=='node':
		list_val = [1,0.1,1e-2,1e-3,1e-4]
		label_chunk = r"$p_{d}^{("
		name_chunk = 'minimized_pd'
	elif choice=='edge':
		list_val = [100,10,1,1e-1,1e-2]
		label_chunk = r"$\lambda^{("
		name_chunk = 'minimized_lamb'
	xlabel = 'window size'
	ylabel = r'$\log_{10}$(ETN vector similarity)'
	markers = ['.','s','^','<','>','*']
	colors = ['blue','green','red','black','orange','purple']
	fig,ax = Setup_Plot(xlabel,ylabel)
	for i,marker,color in zip(range(len(list_val)),markers,colors):
		label = label_chunk+str(i)+r")}$"
		name = name_chunk+str(i)
		tab = np.loadtxt('codata/Compute_local_TS_'+name+'.txt')
		ax.plot(tab[0,:],np.log10(1-tab[1,:]),marker,label=label,color=color)
	ax.legend(fontsize=16)
	plt.savefig('figures/TS/Multiplot_remove_'+choice+'.png')

#same as Compute_local_remove_node but with p_d = 1/N and different values for N
def Compute_pd_N():
	list_N = [100,200,400,800]
	for i,N in enumerate(list_N):
		print(str(i)+' begins (N = '+str(N)+')')
		data = Get_min_node_pruning(1/N,N=N)
		Compute_local((data,'minimized_pd_N'+str(i)),b_max=300,step=10,choice='TS')

def Plot_pd_N():
	list_val = [100,200,400,800]
	name_chunk = 'minimized_pd_N'
	xlabel = 'TS period (b)'
	ylabel = r'$\log_{10}$(ETN vector similarity)'
	markers = ['.','s','^','<','>','*']
	colors = ['blue','green','red','black','orange','purple']
	fig,ax = Setup_Plot(xlabel,ylabel)
	for i,marker,color in zip(range(len(list_val)),markers,colors):
		label = r"$N = $"+str(list_val[i])
		name = name_chunk+str(i)
		tab = np.loadtxt('codata/Compute_local_TS_'+name+'.txt')
		ax.plot(tab[0,:],np.log10(1-tab[1,:]),marker,label=label,color=color)
	ax.legend(fontsize=16)
	plt.savefig('figures/TS/Plot_pd_N_TS.png')

#see whether the knowledge of the edge derivative distribution is enough to recover the edge activity
#distribution
#draw the distribution of the edge activity derivative
def Act_from_der(name,agg):
	net = tp.Temp_net(tp.Load_TN(name))
	net.Init(agg)
	X,Y = zip(*net.Edge_der().items())
	Y = np.asarray(Y,dtype=float)
	Y /= np.sum(Y)
	duree = 3000; act = [0]*duree
	act[0] = 10
	for t in range(1,duree):
		new_act = 0
		while new_act<=0:
			new_act = act[t-1] + Draw_from_tab(X,Y)
		act[t] = new_act
	xlabel = r'$t$'
	ylabel = r'$E(t)$'
	fig,ax = Setup_Plot(xlabel,ylabel)
	ax.plot(act,'.')
	plt.savefig('figures/edge_act/Act_from_der_'+name+'.png')
	plt.show()

#perform the complete analysis of one dataset, avoiding repetitions
def Complete_analysis(name):
	#check which observables remain to compute
	folders = ['cum_edge_weight','dic_ETN','nb_ETN','inst_tri_scalar','ETN_error','ETN_entropy','ETN_profile','inst_deg']
	obs_to_compute = []
	for folder in folders:
		if not os.path.isdir('figures/'+folder+'/'+Get_savename(name)):
			if folder=='nb_ETN':
				obs_to_compute += ['diff_nb_ETN','tot_nb_ETN']
			else:
				obs_to_compute.append(folder)
	if not obs_to_compute:
		return None
	print('obs to compute:')
	print(obs_to_compute)
	name = (tp.Load_TN(name),Get_savename(name))
	agg_inst_deg = [1,2,5,10,30]
	list_agg = [1,2,3,4,5,10,20,50,100]
	scalar_obs = {'inst_tri_scalar','ETN_error','ETN_entropy','diff_nb_ETN','tot_nb_ETN'}
	distr_obs = {'ETN_profile','inst_deg'}
	scalar_obs = list(scalar_obs.intersection(set(obs_to_compute)))
	distr_obs = list(distr_obs.intersection(set(obs_to_compute)))
	for obs_agg in list_agg:
		print('obs_agg '+str(obs_agg)+' begins')
		Compute_local_scalar(name,scalar_obs,b_min=1,b_max=300,step=10,choice='TS',obs_agg=obs_agg)
	Plot_scalar_nb_ETN(name,b_min=1,b_max=300,step=10,choice='TS',list_agg=[1,5,10])
	Plot_scalar_ETN_error(name,b_min=1,b_max=300,step=10,choice='TS',list_agg=[1,10,20,50,100],savefig='whole')
	Plot_scalar_ETN_error(name,b_min=1,b_max=300,step=10,choice='TS',list_agg=[1,2,3,4,5],savefig='early')
	obs_to_ylabel = {'ETN_entropy':'entropy of '+r'$(3,n)$'+'-ETN distribution','inst_tri_scalar':'instantaneous clustering coeff'}
	for obs,ylabel in obs_to_ylabel.items():
		Plot_scalar(name,obs,ylabel,obs+'_vs_b',list_agg=[1,10,20,50,100],savefig='whole')
		Plot_scalar(name,obs,ylabel,obs+'_vs_b',list_agg=[1,2,3,4,5],savefig='early')
	print('scalar_obs done')
	for obs_agg in agg_inst_deg:
		print('obs_agg '+str(obs_agg)+' begins')
		Compute_local_distr(name,distr_obs,b_max=300,step=10,choice='TS',obs_agg=obs_agg)
	for obs_agg in agg_inst_deg:
		Plot_local_distr_ETN_profile(name,300,10,choice='TS',obs_agg=obs_agg)
	for obs_agg in agg_inst_deg[1:]:
		Plot_local_distr_inst_deg(name,300,10,choice='TS',obs_agg=obs_agg,nb_visu=6)
	print('ETN profile and inst deg done')
	if 'dic_ETN' in obs_to_compute:
		for obs_agg in range(1,11):
			print('obs_agg '+str(obs_agg)+' begins')
			Compute_local_dic_ETN(name,b_min=1,b_max=300,step=10,choice='TS',obs_agg=obs_agg)
		Compute_local(name,b_min=1,b_max=300,step=10,choice='TS')
		Plot_local_dic_ETN(name,b_min=1,b_max=300,step=10,choice='TS',list_agg=[1,3,5,10])
	if 'inst_deg' in obs_to_compute:
		Compute_tot_TS_distr(name,'inst_deg',agg_max=100,step=5)
		Plot_tot_TS_distr_inst_deg(name,100,5,nb_visu=6)
	if 'cum_edge_weight' in obs_to_compute:
		Compute_cum_edge_weight(name,cum_min=10,cum_max=1000,step=10)
		Plot_cum_edge_weight(name,cum_min=10,cum_max=1000,step=10,nb_visu=6,savefig='whole_renorm',renormalized=True,fit_cum=False)
		Plot_cum_edge_weight(name,cum_min=10,cum_max=200,step=10,nb_visu=6,savefig='early_renorm',renormalized=True,fit_cum=False)
		Plot_cum_edge_weight(name,cum_min=10,cum_max=1000,step=10,nb_visu=6,savefig='whole_fit',renormalized=False,fit_cum=True)
		Plot_cum_edge_weight(name,cum_min=10,cum_max=200,step=10,nb_visu=6,savefig='early_fit',renormalized=False,fit_cum=True)
		Plot_cum_edge_weight(name,cum_min=10,cum_max=1000,step=10,nb_visu=6,savefig='whole',renormalized=False,fit_cum=False)
		Plot_cum_edge_weight(name,cum_min=10,cum_max=200,step=10,nb_visu=6,savefig='early',renormalized=False,fit_cum=False)
		Var_cum_EW(name,[10,50,100,500,1000])

#draw the figures for the abstract netsci2023
def Netsci2023():
	#first plot: ETN similarity flow, conf16
	Plot_local_dic_ETN('conf16',b_min=1,b_max=300,step=10,choice='TS',list_agg=[1,3,5,10])
	#second plot: ETN similarity flow, ADM9conf16
	Plot_local_dic_ETN(('ADM',9,'conf16'),b_min=1,b_max=300,step=10,choice='TS',list_agg=[1,3,5,10])
	#third plot: triangle density flow, conf16
	Plot_scalar('conf16','inst_tri_scalar','','inst_tri_scalar_vs_b',list_agg=[1,10,20,50,100],savefig='whole')
	#fourth plot: triangle density flow, utah
	Plot_scalar('utah','inst_tri_scalar','','inst_tri_scalar_vs_b',list_agg=[1,10,20,50,100],savefig='whole')

#find the value of p_d such that the ETN vector distance after TS(1,300) matches the conf16 case
#for the minimized 7 version with node pruning
def Find_pd_model(model_name,b=300):
	#load the value of reference
	ETN_vec = np.loadtxt('codata/Compute_local_TS_conf16.txt')
	ind = -1; val = -1
	while val!=b:
		ind += 1
		val = int(np.round(ETN_vec[0,ind]))
	ref = ETN_vec[1,ind]
	#initialize the model
	if model_name=='min_V7':
		model = Get_min_node_pruning(0,return_model=True)
	elif model_name=='min_EW':
		model = Min_EW(138,3635,0.79)
	#tune p_d
	tab = []
	for p_d in np.logspace(1e-3,1e-3,10):
		#compute the artificial data set and determine the ETN vector similarity after TS(1,b)
		model.free_param['p_d'] = p_d
		net = tp.Temp_net(model.Evolve())
		net.Init(1)
		#ori_ETN_vector = ETN_vector without TS (or with TS(1,1))
		ori_ETN_vector = net.ETN_vector(10,3)[0]
		net.Local_TS(b)
		tab.append((p_d,abs(Vector_distance(net.ETN_vector(10,3)[0],ori_ETN_vector)-ref)))
		model.Refresh()
	best_p_d = min(tab,key=lambda el:el[1])[0]
	#return an instance of the model with the best found value for p_d
	model.free_param['p_d'] = best_p_d
	return model.Evolve()

#plot some macro observables to get an overview of the properties of name
def Overview(name):
	net = tp.Temp_net(tp.Load_TN(name))
	net.Init(1)
	nb_edges = [len(net.TN[t].edges) for t in range(len(net.TN))]
	fig,ax = Setup_Plot('time','nb of active edges')
	ax.plot(nb_edges,'.')
	print('nb of temporal edges: '+str(sum(nb_edges)))
	plt.show()

#plot the number of temporal edges vs p_d
#num = version of the Min_EW
def Test_Min_EW(num,pd_min,pd_max):
	model = Min_EW(138,3635,0.79,**Get_versions(choice='min_EW')[num])
	X = np.linspace(pd_min,pd_max,10); Y = []
	for i,p_d in enumerate(X):
		print(str(i)+' begins')
		model.free_param['p_d'] = p_d
		Y.append(np.size(model.Evolve(),0))
		model.Refresh()
	fig,ax = Setup_Plot(r'$p_{d}$','nb of temporal edges')
	ax.plot(X,Y,'.')
	plt.show()

#check the uniformity hypothesis in space (homogeneity) and time (stationarity)
#wrt a given distribution observable at different levels of aggregation
#to do this: sample the observable in space, in time and both, then compute the error distribution
#both in space and time vs the aggregation level
#example with node duration
def Check_uniformity(name):
	list_agg = [1,10,100]
	net = tp.Temp_net(tp.Load_TN(name))
	net.Init()
	print('N = '+str(net.info['N']))
	for agg in list_agg:
		print('n = '+str(agg)+' begins')
		net.Get_TN(agg)
		net.Events()
		#sample the node duration in space-time
		#net.node_event[i][.] = (t_start,t_end)
		space_time_sampling = net.Activity_duration(net.node_event)
		#renormalize the distribution
		norm = sum(list(space_time_sampling.values()))
		for dt in space_time_sampling.keys():
			space_time_sampling[dt] = float(space_time_sampling[dt])/norm
		#sample the node duration in time
		#space_sampling[i] = histo of node duration sampled in the whole timeline of node i
		space_sampling = {}
		for i,val in net.node_event.items():
			space_sampling[i] = {}
			for el in val:
				duration = el[1]-el[0]+1
				if duration in space_sampling[i]:
					space_sampling[i][duration] += 1
				else:
					space_sampling[i][duration] = 1
			#renormalize the distribution
			norm = sum(list(space_sampling[i].values()))
			for dt in space_sampling[i].keys():
				space_sampling[i][dt] = float(space_sampling[i][dt])/norm
		#two measures of heterogeneity:
		#first: compare the distributions point per point
		gap = {dt:0 for dt in space_time_sampling.keys()}
		for i,histo in space_sampling.items():
			for dt,p_space in histo.items():
				gap[dt] += abs(1-p_space/space_time_sampling[dt])
			for dt in set(space_time_sampling.keys()).difference(set(histo.keys())):
				gap[dt] += 1
		for dt in gap.keys():
			gap[dt] /= 138
		#display the results
		fig,ax = Setup_Plot(r'$\Delta t$','dispersion of the proba among nodes')
		ax.plot(*zip(*gap.items()),'.')
		path = PROJECT_ROOT+'/Articles/Macro_obs/figures/check_uniformity'
		if not os.path.isdir(path):
			os.mkdir(path)
		path += '/'+Get_savename(name)
		if not os.path.isdir(path):
			os.mkdir(path)
		plt.savefig(path+'/space_dispersion_method1_agg'+str(agg)+'.png')
		#second: we want to identify two distributions from the same law but with different levels of
		#sampling so we rescale all distributions so that we get the same values at the most probable point
		gap = {dt:0 for dt in space_time_sampling.keys()}
		nb = {dt:0 for dt in space_time_sampling.keys()}
		common_keys = set(space_time_sampling.keys())
		for histo in space_sampling.values():
			common_keys = common_keys.intersection(set(histo.keys()))
		if common_keys:
			most_prob = max(list(common_keys),key=lambda el:space_time_sampling[el])
			print('most_prob: '+str(most_prob))
			for i,histo in space_sampling.items():
				factor = space_time_sampling[most_prob]/histo[most_prob]
				for dt,p_space in histo.items():
					gap[dt] += abs(1-p_space*factor/space_time_sampling[dt])
					nb[dt] += 1
			key_to_remove = {dt for dt in nb.keys() if not nb[dt]}
			for key in key_to_remove:
				del gap[key]
			for dt in gap.keys():
				gap[dt] /= nb[dt]
			#display the results
			fig,ax = Setup_Plot(r'$\Delta t$','dispersion of the proba among nodes')
			ax.plot(*zip(*gap.items()),'.')
			path = PROJECT_ROOT+'/Articles/Macro_obs/figures/check_uniformity/'+Get_savename(name)
			plt.savefig(path+'/space_dispersion_method2_agg'+str(agg)+'.png')

def Compute_KS_uniformity(name,agg_min,agg_max,step):
	nb = (agg_max-agg_min)//step + 1
	list_agg = [k*step+agg_min for k in range(nb)]
	net = tp.Temp_net(tp.Load_TN(name))
	net.Init()
	print('N = '+str(net.info['N']))
	dic_p_value = {}
	for agg in list_agg:
		print('n = '+str(agg)+' begins')
		net.Get_TN(agg)
		#net.node_event[i][.] = (t_start,t_end)
		net.Events()
		#sample the node duration in time
		#space_sampling[i] = observation of node duration sampled in the whole timeline of node i
		space_sampling = {}
		for i,val in net.node_event.items():
			space_sampling[i] = [el[1]-el[0]+1 for el in val]
		#sample the node duration in space-time
		space_time_sampling = []
		for val in space_sampling.values():
			space_time_sampling += val
		#perform a KS test to measure the heterogeneity:
		#compute the p value for every node
		nb_equal = 0
		for i,data in space_sampling.items():
			p_value = ks_2samp(data,space_time_sampling).pvalue
			if p_value>0.05:
				nb_equal += 1
		dic_p_value[agg] = nb_equal/net.info['N']
	path = PROJECT_ROOT+'/Articles/Macro_obs/codata/'
	np.savetxt(path+'Compute_KS_uniformity_'+Get_savename(name)+'.txt',np.array(list(zip(*dic_p_value.items()))))
	
def Plot_KS_uniformity(list_name,savefig=''):
	path = PROJECT_ROOT+'/Articles/Macro_obs/codata/Compute_KS_uniformity_'
	if savefig!='':
		if savefig[0]!='_':
			savefig = '_'+savefig
	list_marker = ['o','s','^','<','>','*','8','v','1','2','3','4','P','p']
	list_color = ['blue','green','red','black','orange','purple','yellow','brown','red','gray','cyan','olive','purple','black']
	fig,ax = Setup_Plot(r'$n$','proportion of identical nodes\nat level '+r'$n$')
	for name,marker,color in zip(list_name,list_marker,list_color):
		savename = Get_savename(name)
		data = np.loadtxt(path+savename+'.txt',dtype=float)
		ax.plot(data[0,:],data[1,:],marker,color=color,label=savename,alpha=0.7)
	ax.legend(fontsize=12,ncol=4,bbox_to_anchor=(0.5,1.23),loc="upper center")
	path = PROJECT_ROOT+'/Articles/Macro_obs/figures/check_uniformity/space_dispersion_method_KS'
	plt.savefig(path+savefig+'.png')

#use minimized V7, for different values of p_d, compute the ETN similarity at levels 1 and 10
#after TS(1,100) and TS(1,225)
def Investigate_min_V7():
	if not os.path.isdir('codata/min_V7'):
		os.mkdir('codata/min_V7')
	#initialize the model
	model = Get_min_node_pruning(0,return_model=True)
	#data[:,0] = value of p_d
	#data[.,:] = [p_d, ETN sim n=1 b=100, ETN sim n=1 b=225, ETN sim n=10 b=100, ETN sim n=10 b=225]
	nb = 10; data = np.zeros((nb,5))
	for i,p_d in enumerate(np.logspace(-3,0,num=nb)):
		print(str(nb-i)+' remaining steps')
		model.free_param['p_d'] = p_d
		data[i,0] = p_d
		net = tp.Temp_net(model.Evolve())
		net.Init(1)
		#ori_dic_ETN = original dic_ETN at level 1
		ori_dic_ETN = net.ETN_histo(3,tot=True)
		for k,b in enumerate([100,225]):
			net.Local_TS(b,agg=1)
			data[i,1+k] = Cosim(net.ETN_histo(3,tot=True),ori_dic_ETN)
		#same at level 10
		net.Get_TN(10)
		#ori_dic_ETN = original dic_ETN at level 10
		ori_dic_ETN = net.ETN_histo(3,tot=True)
		for k,b in enumerate([100,225]):
			net.Local_TS(b,agg=10)
			data[i,3+k] = Cosim(net.ETN_histo(3,tot=True),ori_dic_ETN)
		#reset the model
		model.Refresh()
	#save data
	np.savetxt('codata/min_V7/Investigate_min_V7.txt',data)

#display the results computed by Investigate_min_V7()
def Plot_investigate_min_V7():
	#data[.,:] = [p_d, ETN sim n=1 b=100, ETN sim n=1 b=225, ETN sim n=10 b=100, ETN sim n=10 b=225]
	data = np.loadtxt('codata/min_V7/Investigate_min_V7.txt')
	nb = np.size(data,0)
	#first plot: n=1
	fig,ax = Setup_Plot(r'$\log_{10}(p_{d})$','ETN sim at '+r'$n=1$')
	ax.plot(np.log10(data[:,0]),data[:,1],'.',color='blue',label=r'$b=100$')
	ax.plot(np.log10(data[:,0]),data[:,2],'^',color='orange',label=r'$b=225$')
	#identify p_d that minmizes the ETN similarity
	pd_min = data[min(range(nb),key=lambda i:data[i,2]),0]
	ax.plot([np.log10(pd_min)]*2,list(ax.get_ylim()),'--',color='red',label=r'$p_{d} = $'+str(np.round(pd_min,2)))
	ax.legend(fontsize=14)
	plt.savefig('figures/investigate_min_V7/ETN_sim_p_d_level1.png')
	#second plot: n=10
	fig,ax = Setup_Plot(r'$\log_{10}(p_{d})$','ETN sim at '+r'$n=10$')
	ax.plot(np.log10(data[:,0]),data[:,3],'.',color='blue',label=r'$b=100$')
	ax.plot(np.log10(data[:,0]),data[:,4],'^',color='orange',label=r'$b=225$')
	#identify p_d that minmizes the ETN similarity
	pd_min = data[min(range(nb),key=lambda i:data[i,4]),0]
	ax.plot([np.log10(pd_min)]*2,list(ax.get_ylim()),'--',color='red',label=r'$p_{d} = $'+str(np.round(pd_min,2)))
	ax.legend(fontsize=14)
	plt.savefig('figures/investigate_min_V7/ETN_sim_p_d_level10.png')


#return data_time
def Get_datatime_on_raw_data(data):
	data_time = []
	n1 = 0; n_max = np.size(data,0)
	for n in range(1,n_max):
		t = data[n-1,0]
		if data[n,0]>t:
			data_time += [[t,n1,n]]
			n1 = n
	#take care of the last line of data
	t = data[-1,0]
	data_time += [[t,n1,n_max]]
	return data_time

#perform the local TS(1,b) on the raw data without conversion into a temporal network
def TS_on_raw_data(data,b,data_time):
	#compute the new self.data_time and deduce self.TN at agg level 1
	#determine the times permutation
	list_times = []; nb_blocks = len(data_time)//b
	for k in range(nb_blocks):
		list_times += rd.sample(range(k*b,(k+1)*b),b)
	last_block = len(data_time)-b*nb_blocks
	if last_block>0:
		list_times += rd.sample(range(b*nb_blocks,len(data_time)),last_block)
	new_data_time = [[list_times[el[0]],*el[1:]] for el in data_time]
	new_data_time = sorted(new_data_time,key=lambda el:el[0])
	new_data = np.zeros(data.shape,dtype=int)
	n = 0
	for t,el in enumerate(new_data_time):
		new_data[n:n+el[2]-el[1],1:] = data[el[1]:el[2],1:]
		new_data[n:n+el[2]-el[1],0] = t
		n += el[2]-el[1]
	return new_data

#return the number of nodes without conversion into a temporal network
def Get_N_on_raw_data(data):
	nodes = set(())
	for line in data:
		nodes = nodes.union(set(line[1:]))
	return len(nodes)

#relabel the nodes without conversion into a temporal network (N is the nb of nodes)
def Relabel_on_raw_data(data,N):
	new_labels = list(range(N)); rd.shuffle(new_labels)
	for n in range(np.size(data,0)):
		for k in range(2):
			data[n,k+1] = new_labels[data[n,k+1]]
	return data

#return the file size after compression
def Get_file_size(data):
	np.savetxt('data.txt',data,fmt='%d')
	with ZipFile('Data.zip','w',ZIP_DEFLATED) as zip:
		zip.write('data.txt')
	with ZipFile('Data.zip','r') as zip:
		for info in zip.infolist():
			return info.compress_size

#compute the flow under TS at level 1 of the file size of the temporal network name after compression
def Compute_compressed(name,b_min=1,b_max=300,step=10):
	nb = (b_max-b_min)//step + 1
	list_b = [k*step+b_min for k in range(nb)]; Y = []
	ori_data = tp.Load_TN(name); N = Get_N_on_raw_data(ori_data)
	data_time = Get_datatime_on_raw_data(ori_data)
	print('nb of nodes: '+str(N))
	for b in list_b:
		print(str(b)+' begins')
		#perform TS(1,b)ori_data
		data = TS_on_raw_data(ori_data,b,data_time)
		#compute the file size after compression
		Y.append(Get_file_size(data))
		#permute the nodes in the temporal network to get an error bar on the size of the compressed file
		ori_data = Relabel_on_raw_data(ori_data,N)
	#save the results
	np.savetxt('codata/Compute_compressed_'+Get_savename(name)+'.txt',np.array([list_b,Y],dtype=int),fmt='%d')

#display the results computed by Compute_compressed
def Plot_compressed(list_name,savefig=''):
	if savefig!='':
		if savefig[0]!='_':
			savefig = '_'+savefig
	list_marker = ['.','s','^','<','>','*','8','v','1','2','3','4','P','p']
	list_color = ['blue','green','red','black','orange','purple','yellow','brown','red','gray','cyan','olive','purple','black']
	fig,ax = Setup_Plot(r'$b$','compressed file size after TS'+r'$(1,b)$',fontsize=14)
	for name,marker,color in zip(list_name,list_marker,list_color):
		savename = Get_savename(name)
		data = np.loadtxt('codata/Compute_compressed_'+savename+'.txt',dtype=float)
		#rescale the sizes so that the size is 1 at TS(1,1)
		ax.plot(data[0,:],data[1,:]/data[1,0],marker,color=color,label=savename,alpha=0.7)
	path = 'figures/compressed'
	if not os.path.isdir(path):
		os.mkdir(path)
	ax.legend(fontsize=12,ncol=4,bbox_to_anchor=(0.5,1.23),loc="upper center")
	plt.savefig(path+'/size_vs_b'+savefig+'.png')

#compute a similarity matrix btw all datasets based on edge centered motifs
def EdgeTN_simat(list_name,agg,depth):
	name_to_int = {name:i for i,name in enumerate(list_name)}
	nb = len(name_to_int)
	data_edge = {}; data_node = {}
	for name,i in name_to_int.items():
		print(Get_savename(name)+' begins')
		net = tp.Temp_net(tp.Load_TN(name))
		net.Init(agg=agg)
		data_edge[i] = net.EdgeTN_histo(depth)
		data_node[i] = net.ETN_histo(depth)
	simat_edge = np.ones((nb,nb)); simat_node = np.ones((nb,nb))
	for i in range(nb-1):
		for j in range(i+1,nb):
			simat_edge[i,j] = Cosim(data_edge[i],data_edge[j])
			simat_edge[j,i] = simat_edge[i,j]
			simat_node[i,j] = Cosim(data_node[i],data_node[j])
			simat_node[j,i] = simat_node[i,j]
	np.savetxt(PROJECT_ROOT+'/Articles/ETN_motifs/codata/EdgeTN_simat_agg'+str(agg)+'.txt',simat_edge)
	np.savetxt(PROJECT_ROOT+'/Articles/ETN_motifs/codata/ETN_simat_agg'+str(agg)+'.txt',simat_node)

#compare the power of separation of node and edge centered motifs
def Compare_edge_node_motifs(list_name,agg,depth):
	name_to_int = {name:i for i,name in enumerate(list_name)}
	int_to_name = {i:name for name,i in name_to_int.items()}
	nb = len(name_to_int)
	path = PROJECT_ROOT+'/Articles/ETN_motifs/codata/'
	simat_edge = np.loadtxt(path+'EdgeTN_simat_agg'+str(agg)+'.txt')
	simat_node = np.loadtxt(path+'ETN_simat_agg'+str(agg)+'.txt')
	#plot the edge simat vs node simat
	fig,ax = Setup_Plot('node sim','edge sim')
	X,Y = [],[]
	for i in range(nb):
		for j in range(nb):
			X.append(simat_node[i,j]); Y.append(simat_edge[i,j])
	ax.plot(X,Y,'.')
	plt.show()
	path = PROJECT_ROOT+'/Articles/ETN_motifs/figures/EdgeTN/'
	#detect communities in each similarity matrix
	for simat,savefig in zip([simat_edge,simat_node],['Edge','E']):
		net = nx.Graph()
		for i in range(nb):
			for j in range(i,nb):
				net.add_edge(i,j,weight=simat[i,j])
		dic_comm = community_louvain.best_partition(net)
		list_comm = {}
		for i,comm in dic_comm.items():
			if comm in list_comm:
				list_comm[comm].append(i)
			else:
				list_comm[comm] = [i]
		list_comm = sorted(list_comm.values(),key=len,reverse=True)
		print('nb of comm: '+str(len(list_comm))+' ('+savefig+')')
		print('comm:')
		for comm in list_comm:
			read_comm = []
			for i in comm:
				read_comm.append(Get_savename(int_to_name[i]))
			print(read_comm)
		ordered_i = []
		for comm in list_comm:
			ordered_i += comm
		old_i_to_new = {i:j for j,i in enumerate(ordered_i)}
		new_simat = np.eye(nb)
		for i in range(nb-1):
			for j in range(i+1,nb):
				new_simat[old_i_to_new[i],old_i_to_new[j]] = simat[i,j]
				new_simat[old_i_to_new[j],old_i_to_new[i]] = new_simat[old_i_to_new[i],old_i_to_new[j]]
		fig,ax = plt.subplots(1,1,constrained_layout=True)
		ax.imshow(simat,cmap='gnuplot2')
		plt.savefig(path+savefig+'TN_simat_agg'+str(agg)+'.png')
		plt.close()

'''
test = np.array([[0,0,1],[0,0,2],[1,0,1],[1,0,2],[1,1,2],[2,0,1],[2,1,2]],dtype=int)
net = tp.Temp_net(test); depth = 3; agg = 1
net.Init(agg=1)
NCTN_histo = net.ETN_histo(depth)
print(NCTN_histo)
for seq in NCTN_histo.keys():
	fig,ax = plt.subplots(1,1,constrained_layout=True)
	ax.set_axis_off()
	etn_lib.Plot_ETN(seq,depth,ax)
	plt.savefig('figures/test/'+seq+'NCTN.png')
	plt.close()
exit()
test = {0:'001',1:'010',2:'011'}
print(set(test))
exit()
test = nx.Graph()
list_edges = [(1,0),(1,2),(4,1),(5,3)]
test.add_edges_from(list_edges)
for edge in test.edges:
	if edge[0]<edge[1]:
		i,j = edge
	else:
		i,j = edge[::-1]
	print(edge)
	print(i,j)
exit()
#test = {0:'001',1:'010',2:'011'}
test = {0:list('001'),1:list('010'),2:list('011')}
for u,val in test.items():
	test[u] = ''.join(val)
print(test)
seq_i = ''.join(sorted(test.values()))
print(seq_i)
exit()
test = nx.MultiGraph()
list_edges = [(0,1),(1,2)]
for edge in list_edges:
	test.add_edge(*edge,key=1)
test.add_edge(1,0,key=2)
print(test.edges(2,keys=True))
print(list(test.neighbors(1)))
exit()
name = 'conf16'; agg = 1; depth = 3
net = tp.Temp_net(tp.Load_TN(name))
net.Init(agg=agg)
print('comparison begins')
dic_func = {'baseline':net.ETN_histo}
#dic_func = {'old':net.Get_TN_0,'new':net.Get_TN}
dic_times = {}
nb_test = 1
for key,func in dic_func.items():
	t_start = time.time()
	for _ in range(nb_test):
		histo = func(depth)
	t_end = time.time()
	dic_times[key] = t_end - t_start
for key,duration in dic_times.items():
	print(key+': '+str(duration))
exit()
name = 'conf16'; list_b = [1,5]; list_agg = [1,10]
obs_with_arg = ['edge0interduration','edge0newborn_duration',('ECTN0time_weight',3),('ECTN0profile_from_motif',3),('NCTN0duration',3)]
Internalobs_nb_flow(name,obs_with_arg,list_b,list_agg)
exit()
'''
exit()
depth = 3
for name in ['conf16','utah',('ADM',9,'conf16')]:
	print(Get_savename(name)+' begins')
	net = tp.Temp_net(tp.Load_TN(name))
	net.Init()
	for agg in [1,5]:
		net.Get_TN(agg)
		dic_ETN = net.EdgeTN_histo(depth)
		#sort the motifs by decreasing abundancy
		list_seq = sorted(dic_ETN.keys(),key=lambda seq:dic_ETN[seq],reverse=True)
		fig,ax = plt.subplots(2,5,constrained_layout=True,figsize=(12,6))
		fontsize = 16
		for i in range(2):
			for j in range(5):
				num = 5*i+j
				ax[i,j].set_axis_off()
				ax[i,j].set_title(etn_lib.Ordinal(num),fontsize=fontsize)
				etn_lib.Plot_EdgeTN(list_seq[num],depth,ax[i,j])
		path = PROJECT_ROOT+'/Articles/ETN_motifs/figures/EdgeTN/'
		plt.savefig(path+'most_freq_agg_'+str(agg)+Get_savename(name)+'.png')
		#display the EdgeTN weight histogram
		fig,ax = Setup_Plot(r'$w$',r'$\log_{10}(P)$',fontsize=14)
		ax.plot(*Raw_to_binned(net.Weight_ETN(dic_ETN)),'.',markersize=10,label='edge centered')
		ax.plot(*Raw_to_binned(net.Weight_ETN(net.ETN_histo(depth))),'^',markersize=10,label='node centered')
		ax.legend(fontsize=14)
		path = PROJECT_ROOT+'/Articles/ETN_motifs/figures/EdgeTN/'
		plt.savefig(path+'weight_histo_agg_'+str(agg)+Get_savename(name)+'.png')
		plt.close()
XP_name = ['conf16','conf17','conf18','conf19','french','utah']
XP_name += ['highschool1','highschool2','highschool3','work1','work2']
XP_name += ['hospital','malawi','baboon']
list_models = [('ADM',9,'conf16'),('ADM',18,'conf16'),('min_EW',1),('min_EW',2),('min_EW',3),('min_ADM',2),('min_ADM',1)]
list_name = XP_name+list_models
print('simat begins')
EdgeTN_simat(list_name,1,3)
Compare_edge_node_motifs(list_name,1,3)
exit()

list_models = ['conf16','utah',('ADM',9,'conf16'),('ADM',18,'conf16'),('min_EW',1),('min_EW',2),('min_EW',3),('min_ADM',2),('min_ADM',1)]
list_obs = ['EdgeTN_error','tot_nb_EdgeTN','diff_nb_EdgeTN']
for name in list_models:
	Compute_local_scalar(name,list_obs)
	Plot_scalar_nb_ETN(name)
exit()


agg_max = 100; agg_min = 1; step = 5
for name in list_models:
	print(Get_savename(name)+' begins')
	Compute_KS_uniformity(name,agg_min,agg_max,step)
Plot_KS_uniformity(list_models)
exit()

XP_name = ['conf16','conf17','conf18','conf19','french','utah']
XP_name += ['highschool1','highschool2','highschool3','work1','work2']
XP_name += ['hospital','malawi','baboon']
'''
for name in list_models[4:]:
	print(Get_savename(name)+' begins')
	Compute_compressed(name)
'''
Plot_compressed(list_models,savefig='models')
exit()

#separate the periods of high and low activity and compare them
name = 'utah'
Overview(name)
exit()
#for conf16 we consider as low activities the activities below 53 edges per time
#for utah we take 315 as threshold

#extract the low high activity periods
net = tp.Temp_net(tp.Load_TN(name))
net.Init(1)
low_period = []; high_period = []; period = []; before = 'low'
for t in range(len(net.TN)):
	if len(net.TN[t].edges)<315:
		if before=='low':
			period.append(t)
		else:
			high_period.append(period)
			before = 'low'
			period = [t]
	else:
		if before=='high':
			period.append(t)
		else:
			low_period.append(period)
			before = 'high'
			period = [t]
low_period.sort(key=len,reverse=True)
high_period.sort(key=len,reverse=True)

fig,ax = Setup_Plot('rank','period duration')
ax.plot([len(_)for _ in low_period],'.',color='blue',label='low')
ax.plot([len(_)for _ in high_period],'^',color='orange',label='high')
ax.legend(fontsize=14)
path = PROJECT_ROOT+'/Articles/Macro_obs/figures/low_high_period/'+Get_savename(name)
if not os.path.isdir(path):
	os.mkdir(path)
plt.savefig(path+'/period_duration.png')
plt.close()

#extract one sub-period of each type with same durations
low_data = []; high_data = []
for t,t1 in zip(low_period[0],high_period[0]):
	for edge in net.TN[t].edges:
		low_data.append([t,*edge])
	for edge in net.TN[t1].edges:
		high_data.append([t1,*edge])
low_net = tp.Temp_net(np.asarray(low_data,dtype=int)); high_net = tp.Temp_net(np.asarray(high_data,dtype=int))
low_net.Format(); high_net.Format()
low_net.Prepare(); high_net.Prepare()

#compare the statistical properties of the two TN
low_net.Ref_obs(type_obs='distribution'); high_net.Ref_obs(type_obs='distribution')
for obs in low_net.dic_obs['distribution'].keys():
	if 'duration' in obs:
		xlabel = r'$\log_{10}(\Delta t)$'
	elif 'weight' in obs:
		xlabel = r'$\log_{10}(w)$'
	else:
		xlabel = r'$\log_{10}(n)$'
	fig,ax = Setup_Plot(xlabel,r'$\log_{10}(P)$')
	ax.plot(*Raw_to_binned(low_net.dic_obs['distribution'][obs]),'.',color='blue',label='low period')
	ax.plot(*Raw_to_binned(high_net.dic_obs['distribution'][obs]),'^',color='blue',label='high period')
	ax.legend(fontsize=14)
	plt.savefig(PROJECT_ROOT+'/Articles/Macro_obs/figures/low_high_period/'+Get_savename(name)+'/'+obs+'.png')
	plt.close()

#compute the average nb of newborn edges for different data sets (the frac parameter in Min_EW class)

#min_EW does possess memory but seems invariant under time shuffling
#it is the same reason than the minimized 7 version: there is no removal of nodes (temperature is zero)
#hence we need to introduce a removal rate
#consider the minimized 7 version: consider one edge pruning and one node pruning such that
#after TS(1,300) we end up with the same value as conf16 for the ETN vector distance
#then compare the flows in these two cases
pass
#triadic or avalanche mechanism for pruning? (if a strong bond breaks it could give rise to additional
#breaks)

exit()
Complete_analysis(('min_EW',3))

#model with heterogeneous node activities

XP_name = ['conf16','conf17','conf18','conf19','french','utah']
XP_name += ['highschool1','highschool2','highschool3','work1','work2']
XP_name += ['hospital','malawi','baboon']
#priority_names = ['conf16','utah',('ADM',9,'conf16'),('ADM',18,'conf16')]
priority_names = []
for name in priority_names:
	print(Get_savename(name)+' begins')
	Complete_analysis(name)
