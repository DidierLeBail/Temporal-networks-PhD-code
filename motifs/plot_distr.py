import main
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
#FIGPATH = os.path.join(PROJECT_ROOT,'manuscript/chapter2/obs_distr/')
FIGPATH = os.path.join(PROJECT_ROOT,'manuscript/images/defence/')

#generate a TN from Dyn_EW in atn.py with name as reference
def Get_dyn_EW(name,where=None):
	motifs = main.Motifs_tp(name,where=where)
	motifs.Get_TN(1)
	ext_act = [len(G.edges) for G in motifs.TN]
	N = 138; frac = 0.79
	model = main.atn.Dyn_EW(N,frac,ext_act)
	model.Evolve(t_min=3000)
	savepath = os.path.join(PROJECT_ROOT,'data/EW_'+name+'.txt')
	np.savetxt(savepath,np.array(model.events,dtype=int),fmt='%d')
	print('TN saved at :\n'+savepath)

#generate a TN where G(t) = ER with same nb of edges as G_{name}(t)
#but the edges have distinct proba so that the weighted aggregated network is preserved
def Get_disordered_ER(name,where=None):
	mtp = main.Motifs_tp(name,where=where)
	mtp.Get_TN(1)
	agg_net = nx.Graph()
	for g in mtp.TN:
		for edge in g.edges:
			if agg_net.has_edge(*edge):
				agg_net[edge[0]][edge[1]]['weight'] += 1
			else:
				agg_net.add_edge(*edge,weight=1)
	norm = sum([agg_net[edge[0]][edge[1]]['weight'] for edge in agg_net.edges])
	print('aggregated network computed')
	#generate the disordered ER network
	events = []; timeline = [len(g.edges) for g in mtp.TN]
	for t,nb_edges in enumerate(timeline):
		for edge in agg_net.edges:
			if main.rd.random()<agg_net[edge[0]][edge[1]]['weight']*nb_edges/norm:
				events.append([t,*edge])
	print("events computed")
	mtp_er = main.Motifs_tp(np.array(events,dtype=int))
	mtp_er.Get_TN(1)
	return mtp,mtp_er

'''
list_names = ['conf16','conf17','conf18','conf19','highschool1','highschool2','highschool3']
list_names += ['work1','work2','malawi','hospital','baboon','utah','french']

for name in list_names:
	print(name)
	mtp = main.Motifs_tp(name)
	mtp.TN = nx.Graph()
	mtp.TN.add_edges_from(mtp.data[:,1:])
	print('nb of edges',len(mtp.TN.edges))
exit()
'''

fontsize = 15; xlabel = r"$\log_{10}(n)$"; ylabel = r"$\log_{10}(P)$"
name = 'utah'
mtp,mtp_er = Get_disordered_ER(name)

'''
#visualize aggregated network
net = nx.Graph()
for graph in mtp.TN:
	for edge in graph.edges:
		if net.has_edge(*edge):
			net[edge[0]][edge[1]]['weight'] += 1
		else:
			net.add_edge(*edge,weight=1)
net_er = nx.Graph()
for graph in mtp_er.TN:
	for edge in graph.edges:
		if net_er.has_edge(*edge):
			net_er[edge[0]][edge[1]]['weight'] += 1
		else:
			net_er.add_edge(*edge,weight=1)
nx.write_gexf(net,FIGPATH+'agg_net'+name+'.gexf')
nx.write_gexf(net_er,FIGPATH+'agg_net'+name+'_ER.gexf')
exit()
'''

fig,ax = main.Setup_Plot(xlabel,ylabel,fontsize=fontsize)
ax.plot(*main.Raw_to_binned(mtp.Get_cc_size()),'o',label="original network")
ax.plot(*main.Raw_to_binned(mtp_er.Get_cc_size()),'<',label="disordered network")
ax.legend(fontsize=fontsize)
plt.savefig(FIGPATH+"cc_size"+name+"agg1.png");exit()
list_agg = [5,10,30,50]
for agg in list_agg:
	print(agg)
	mtp.Get_TN(agg)
	mtp_er.Get_TN(agg)
	fig,ax = main.Setup_Plot(xlabel,ylabel,fontsize=fontsize)
	ax.plot(*main.Raw_to_binned(mtp.Get_cc_size()),'o',label="original network")
	ax.plot(*main.Raw_to_binned(mtp_er.Get_cc_size()),'<',label="disordered network")
	ax.legend(fontsize=fontsize)
	plt.savefig(FIGPATH+"cc_size"+name+"agg"+str(agg)+".png")
exit()

#generate a TN where G(t) = ER with same nb of edges as G_{name}(t)
#then compute the resulting cc size distribution
def cc_size_from_varying_ER(name,where=None):
	motifs = main.Motifs_tp(name,where=where)
	motifs.Get_TN(1)
	histo = {}
	for nb_edges in [len(G.edges()) for G in motifs.TN]:
		G = nx.gnm_random_graph(motifs.N,nb_edges)
		list_cc = nx.connected_components(G)
		for cc in list_cc:
			n = len(cc)
			if n>1:
				if n in histo:
					histo[n] += 1
				else:
					histo[n] = 1
	return main.Raw_to_binned(histo)
def node_degree_from_varying_ER(name,where=None):
	motifs = main.Motifs_tp(name,where=where)
	motifs.Get_TN(1)
	histo = {}
	for nb_edges in [len(G.edges()) for G in motifs.TN]:
		G = nx.gnm_random_graph(motifs.N,nb_edges)
		for i in G.nodes:
			n = G.degree(i)
			if n in histo:
				histo[n] += 1
			else:
				histo[n] = 1
	del histo[0]
	X,Y = zip(*main.Norm_dic_histo(histo).items())
	return (X,np.log10(Y))
def edge_time_weight_from_varying_ER(name,where=None):
	motifs = main.Motifs_tp(name,where=where)
	motifs.Get_TN(1)
	data = []
	for t,nb_edges in enumerate([len(G.edges()) for G in motifs.TN]):
		G = nx.gnm_random_graph(motifs.N,nb_edges)
		for edge in G.edges():
			data.append([t,*edge])
	motifs = main.Motifs_tp(np.array(data,dtype=int))
	motifs.Get_TN(1)
	return main.Raw_to_binned(main.Time_weight_histo(motifs.Edge_event_train()))

def Draw_nb_active_edges(name,where):
	motifs = main.Motifs_tp(name,where=where)
	motifs.Get_TN(1)
	nb_edges = [len(G.edges) for G in motifs.TN]
	fontsize = 15
	fig,ax = main.Setup_Plot(r'$t$',r'$|G(t)|$',fontsize=fontsize)
	ax.plot(nb_edges,'.')
	plt.savefig(FIGPATH+name+'_nb_edges.png')
	plt.close()

def func(k,a,b,c):
	return a -b*k -c*np.power(10,k)

def func_deg(k,a,b):
	return a + k*(b-np.log10(k))

def Get_poisson_degree(list_k,list_val):
	k = np.asarray(list_k)
	y = np.asarray(list_val)
	p_opt = curve_fit(func_deg,k,y,p0=(1,0),bounds=([-np.inf,-np.inf],[np.inf,1/np.log(10)]))[0]
	print('mean degree',10**(p_opt[1]-1/np.log(10)))
	return func_deg(k,*p_opt)

def Get_cc_size_min_EW(list_k,list_val):
	'''
	N,duree,frac = 138,8000,0.79
	#analysis parameters
	e_r = 2e-2
	e_c = frac/duree
	c = N*e_c*(1+e_r)/(2*(e_c+e_r))
	print('c =',c)
	k = np.power(10,np.asarray(list_k))
	return 4+np.log10(k**(-1.5)*(c*np.exp(1-c))**k/(c*np.sqrt(2*np.pi)))
	'''
	k = np.asarray(list_k)
	y = np.asarray(list_val)
	p_opt = curve_fit(func,k,y,p0=(1,1.5,1),bounds=([-np.inf,0,0],[np.inf,3,np.inf]))[0]
	print('exponent',p_opt[1])
	return func(k,*p_opt)

def get_dic_obs():
	xlabels = [r'$\log_{10}(n)$',r'$\log_{10}(\tau)$',r'$\log_{10}(\Delta\tau)$',r'$k$',r'$\log_{10}(|b|)$',r'$\log_{10}(w)$']
	dic_obs = {}
	for prefix in ['node','edge']:
		for suffix,k in zip(['duration','interduration','train_weak_duration','time_weight'],[1,2,4,5]):
			dic_obs[prefix+'_'+suffix] = xlabels[k]
	for obs,k in zip(['cc_size_in_nodes','node_degree','NCTN_weight','ECTN_weight'],[0,3,5,5]):
		dic_obs[obs] = xlabels[k]
	#,'lineage_size_in_nodes','lineage_size_in_time_steps']
	print('nb of observables:',len(dic_obs))
	return dic_obs

def get_common_obs():
	xlabels = [r'$\log_{10}(n)$',r'$\log_{10}(\tau)$',r'$\log_{10}(\Delta\tau)$',r'$k$',r'$\log_{10}(|b|)$',r'$\log_{10}(w)$']
	dic_obs = {}
	for suffix,k in zip(['duration','interduration','train_weak_duration','time_weight'],[1,2,4,5]):
		dic_obs['edge_'+suffix] = xlabels[k]
	for obs,k in zip(['cc_size_in_nodes','node_degree','NCTN_weight','ECTN_weight'],[0,3,5,5]):
		dic_obs[obs] = xlabels[k]
	print('nb of observables:',len(dic_obs))
	return dic_obs

def get_distr(list_names,dic_obs):
	#distr[obs] = list of the distributions of the observable obs for every TN in list_names
	distr = {obs:[] for obs in dic_obs}
	for el in list_names:
		if type(el)==tuple:
			name,where = el
		else:
			name = el
			where = None

		print(name)
		
		motifs = main.Motifs_tp(name,where=where)
		motifs.Get_TN(1)
		#node_event = motifs.Node_event_train()
		edge_event = motifs.Edge_event_train()
		#dic_NCTN = motifs.Get_dic_NCTN(3)
		#dic_ECTN = motifs.Get_dic_ECTN(3)

		print('\tdata computed')
		
		#compute the edge duration, interduration, train weak duration and time weight
		#distr['edge_duration'].append(main.Raw_to_binned(main.Duration_histo(edge_event)))
		#distr['edge_interduration'].append(main.Raw_to_binned(main.Interduration_histo(edge_event)))
		#x,y = zip(*main.Train_weak_duration_histo(edge_event).items())
		#distr['edge_train_weak_duration'].append((np.log10(x),np.log10(y)))
		distr['edge_train_weak_duration'].append(main.Raw_to_binned(main.Train_weak_duration_histo(edge_event)))
		#distr['edge_time_weight'].append(main.Raw_to_binned(main.Time_weight_histo(edge_event)))
		'''
		#compute the node duration, interduration, train weak duration and time weight
		distr['node_duration'].append(main.Raw_to_binned(main.Duration_histo(node_event)))
		distr['node_interduration'].append(main.Raw_to_binned(main.Interduration_histo(node_event)))
		distr['node_train_weak_duration'].append(main.Raw_to_binned(main.Train_weak_duration_histo(node_event)))
		distr['node_time_weight'].append(main.Raw_to_binned(main.Time_weight_histo(node_event)))
		
		#compute the node degree
		X,Y = zip(*motifs.Get_direct_deg_histo().items())
		distr['node_degree'].append((X,np.log10(Y)))
		
		#compute the size in nodes of connected components
		#X,Y = zip(*motifs.Get_cc_size().items())
		#distr['cc_size_in_nodes'].append((np.log10(X),np.log10(Y)))
		distr['cc_size_in_nodes'].append(main.Raw_to_binned(motifs.Get_cc_size()))
		
		#compute the NCTN and ECTN space-time weights
		distr['NCTN_weight'].append(main.Raw_to_binned(main.Get_weight_histo(dic_NCTN)))
		distr['ECTN_weight'].append(main.Raw_to_binned(main.Get_weight_histo(dic_ECTN)))
		'''
	return distr

#for each observable plot the distributions obtained for every TN on the same figure
def Plot_all_distr(distr,list_names,dic_obs,additional_data={},savepath='chapter2/obs_distr/'):
	fontsize = 15
	for obs,val in distr.items():
		fig,ax = main.Setup_Plot(dic_obs[obs],r'$\log_{10}(P)$',fontsize=fontsize)
		for data,name,marker,color in zip(val,list_names,main.LIST_MARKER,main.LIST_COLOR):
			if type(name)==tuple:
				label = name[0]
			else:
				label = name
			ax.plot(*data,marker,color=color,label=label)
		if obs in additional_data:
			for data in additional_data[obs]:
				ax.plot(*data,'--',color='gray')
		ax.legend(fontsize=fontsize)
		plt.savefig(FIGPATH+savepath+obs+'.png')
		plt.close()

#generate a TN where G(t) = ER with same nb of edges as G_{name}(t)
#then compute the resulting cc size distribution and compare it with name for name in
#['conf16','highschool3']
def Fig_cc_size_ER():
	list_names = ['conf16','highschool3']
	dic_obs = get_dic_obs()
	distr = get_distr(list_names,dic_obs)
	fontsize = 15
	obs = 'cc_size_in_nodes'; val = distr[obs]
	fig,ax = main.Setup_Plot(dic_obs[obs],r'$\log_{10}(P)$',fontsize=fontsize)
	for data,name,marker,color in zip(val,list_names,main.LIST_MARKER,main.LIST_COLOR):
		ax.plot(*data,marker,color=color,label=name)
	ind = len(list_names)
	for name,marker,color in zip(list_names,main.LIST_MARKER[ind:],main.LIST_COLOR[ind:]):
		data = cc_size_from_varying_ER(name)
		ax.plot(*data,marker,color=color,label='ER_'+name)
	ax.legend(fontsize=fontsize)
	plt.show()
def Fig_node_degree_ER():
	list_names = ['conf16','utah']
	dic_obs = get_dic_obs()
	distr = get_distr(list_names,dic_obs)
	fontsize = 15
	obs = 'node_degree'; val = distr[obs]
	fig,ax = main.Setup_Plot(dic_obs[obs],r'$\log_{10}(P)$',fontsize=fontsize)
	for data,name,marker,color in zip(val,list_names,main.LIST_MARKER,main.LIST_COLOR):
		ax.plot(*data,marker,color=color,label=name)
	ind = len(list_names)
	#list_k,list_val = zip(*sorted(list(zip(*val[1])),key=lambda el:el[0]))
	#ax.plot(list_k,Get_poisson_degree(list_k,list_val),'--',color=color,label='Poisson law')
	for name,marker,color in zip(list_names,main.LIST_MARKER[ind:],main.LIST_COLOR[ind:]):
		data = node_degree_from_varying_ER(name)
		ax.plot(*data,marker,color=color,label='ER_'+name)
		#list_k,list_val = zip(*sorted(list(zip(*data)),key=lambda el:el[0]))
		#ax.plot(list_k,Get_poisson_degree(list_k,list_val),'--',color=color,label='Poisson law')
	ax.legend(fontsize=fontsize)
	plt.show()
def Fig_edge_weight_ER():
	list_names = ['conf16','utah']
	dic_obs = get_dic_obs()
	distr = get_distr(list_names,dic_obs)
	fontsize = 15
	obs = 'edge_time_weight'; val = distr[obs]
	fig,ax = main.Setup_Plot(dic_obs[obs],r'$\log_{10}(P)$',fontsize=fontsize)
	for data,name,marker,color in zip(val,list_names,main.LIST_MARKER,main.LIST_COLOR):
		ax.plot(*data,marker,color=color,label=name)
	ind = len(list_names)
	for name,marker,color in zip(list_names,main.LIST_MARKER[ind:],main.LIST_COLOR[ind:]):
		data = edge_time_weight_from_varying_ER(name)
		ax.plot(*data,marker,color=color,label='ER_'+name)
	ax.legend(fontsize=fontsize)
	plt.show()
def Fig_edge_weight_min_EW():
	list_names = ['conf16','utah',('min_EW3_RS.txt',os.path.join(os.path.dirname(__file__),'min_EW3_RS.txt'))]
	dic_obs = get_dic_obs()
	distr = get_distr(list_names,dic_obs)
	fontsize = 15
	obs = 'edge_time_weight'; val = distr[obs]
	fig,ax = main.Setup_Plot(dic_obs[obs],r'$\log_{10}(P)$',fontsize=fontsize)
	for data,name,marker,color in zip(val,list_names,main.LIST_MARKER,main.LIST_COLOR):
		if type(name)==tuple:
			label = name[0]
		else:
			label = name
		ax.plot(*data,marker,color=color,label=label)
	ax.legend(fontsize=fontsize)
	plt.show()

#the edge time weight is not log-normal distributed
def Fig_chap2_distr():
	list_names = ['conf16','french','utah','highschool3','work2']
	dic_obs = get_common_obs()
	distr = get_distr(list_names,dic_obs)
	additional_data = {}

	#add a fit to check a predicted functional form
	obs = 'edge_train_weak_duration'
	print(obs)
	additional_data[obs] = []
	for data in distr[obs]:
		list_val = sorted(list(zip(*data)),key=lambda el:el[0])
		x,y = zip(*list_val)
		x = np.asarray(x)
		y = np.asarray(y)
		p_opt = curve_fit(func,x,y,p0=(1,0,1),bounds=([-np.inf,-5,0],[np.inf,5,np.inf]))[0]
		print('\texponent',p_opt[1])
		print('\tdecay rate',p_opt[2])
		additional_data[obs].append((x,func(x,*p_opt)))
	'''
	#add straight lines to get lower and upper bounds for the exponent
	obs_to_slopes = {'edge_duration':(2,3),'edge_interduration':(1,2),'edge_time_weight':(1,2)}
	obs_to_slopes['NCTN_weight'] = (1.5,2.5) ; obs_to_slopes['ECTN_weight'] = (1.5,2.5)
	for obs,val in obs_to_slopes.items():
		additional_data[obs] = []
		x_0 = 0; list_y = [np.max(data[1]) for data in distr[obs]]
		y_f = np.min([np.min(data[1]) for data in distr[obs]])
		x_f = np.max([np.max(data[0]) for data in distr[obs]])
		for slope,y_0 in zip(val,(np.max(list_y),np.min(list_y))):
			y_1 = y_0 - slope*(x_f-x_0)
			if y_f<y_1:
				y_2 = y_1
				x_2 = x_f
			else:
				x_2 = (y_0-y_f)/slope + x_0
				y_2 = y_f
			additional_data[obs].append(([x_0,x_2],[y_0,y_2]))
	'''
	Plot_all_distr(distr,list_names,dic_obs,additional_data=additional_data)

#check the nb of contacts per edge and the edge time weight are proportional to each other 
def Check_nb_train_edge_weight_old():
	list_names = ['conf16','french','utah','highschool3','work2']
	distr = []
	for el in list_names:
		if type(el)==tuple:
			name,where = el
		else:
			name = el
			where = None
		print(name)
		motifs = main.Motifs_tp(name,where=where)
		motifs.Get_TN(1)
		edge_event = motifs.Edge_event_train()
		#compute w(e)/n(e) for each edge e
		tab = []
		for val in edge_event.values():
			w = 0
			for el in val:
				duration = el[1]-el[0]+1
				w += duration
			tab.append(w/len(val))
		distr.append(tab.copy())
	#plot the result
	fontsize = 15; savepath = 'chapter2/obs_distr/Check_nb_train_edge_weight'
	fig,ax = main.Setup_Plot(r"$\frac{w}{n}$",r'$P$',fontsize=fontsize)
	for data,name,marker,color in zip(distr,list_names,main.LIST_MARKER,main.LIST_COLOR):
		if type(name)==tuple:
			label = name[0]
		else:
			label = name
		ax.hist(data,bins='auto',histtype='step',linewidth=4,color=color,label=label,density=True)
	ax.legend(fontsize=fontsize)
	plt.savefig(FIGPATH+savepath+'.png')
	plt.close()

#check the nb of contacts per edge and the edge time weight are proportional to each other 
def Check_nb_train_edge_weight():
	list_names = ['conf16','french','utah','highschool3','work2']
	distr = []; edge_weight = []
	for el in list_names:
		if type(el)==tuple:
			name,where = el
		else:
			name = el
			where = None
		print(name)
		motifs = main.Motifs_tp(name,where=where)
		motifs.Get_TN(1)
		edge_event = motifs.Edge_event_train()
		#compute the histo of n
		histo = {}
		for val in edge_event.values():
			n = len(val)
			if n in histo:
				histo[n] += 1
			else:
				histo[n] = 1
		'''
		tab = sorted(list(histo.items()),key=lambda el:el[0])
		x,y = zip(*tab)
		distr.append((np.log10(x),np.log10(y)))
		tab = sorted(list(main.Time_weight_histo(edge_event).items()),key=lambda el:el[0])
		x,y = zip(*tab)
		edge_weight.append((np.log10(x),np.log10(y)))
		'''
		distr.append(main.Raw_to_binned(histo))
		edge_weight.append(main.Raw_to_binned(main.Time_weight_histo(edge_event)))
	#plot the result
	fontsize = 15; savepath = 'chapter2/obs_distr/Check_nb_train_edge_weight'
	fig,ax = main.Setup_Plot(r"$\log_{10}(n)$",r'$\log_{10}(P)$',fontsize=fontsize)
	for data,data2,name,marker,color in zip(distr,edge_weight,list_names,main.LIST_MARKER,main.LIST_COLOR):
		if type(name)==tuple:
			label = name[0]
		else:
			label = name
		ax.plot(*data,marker,color=color,label=label)
		x = data2[0]; y = np.asarray(data2[1]) - data2[1][0] + data[1][0]
		ax.plot(x,y,'--',color=color,alpha=0.5)
	ax.legend(fontsize=fontsize)
	plt.savefig(FIGPATH+savepath+'.png')
	plt.close()

#check the edge_train_weak_duration
def Check_edge_train_weak_duration():
	list_names = ['conf16','french','utah','highschool3','work2']
	distr = []
	for el in list_names:
		if type(el)==tuple:
			name,where = el
		else:
			name = el
			where = None
		print(name)
		motifs = main.Motifs_tp(name,where=where)
		motifs.Get_TN(1)
		edge_event = motifs.Edge_event_train()
		#compute Z_{\Delta}(n)
		n_max = 0
		for val in edge_event.values():
			n = len(val)
			if n_max<n:
				n_max = n
		tab_Z = np.zeros(n_max,dtype=int)
		for n in range(1,n_max+1):
			for val in edge_event.values():
				c = len(val)
				if c>=n:
					tab_Z[n-1] += c - n + 1
		histo = {n+1:nb for n,nb in enumerate(tab_Z)}
		x,y = zip(*histo.items())
		distr.append((x,np.log10(y)))
	#plot the result
	fontsize = 15; savepath = 'chapter2/obs_distr/Check_edge_train_weak_duration'
	fig,ax = main.Setup_Plot(r"$\log_{10}(n)$",r'$\log_{10}(P)$',fontsize=fontsize)
	for data,name,marker,color in zip(distr,list_names,main.LIST_MARKER,main.LIST_COLOR):
		if type(name)==tuple:
			label = name[0]
		else:
			label = name
		ax.plot(*data,marker,color=color,label=label)
	ax.legend(fontsize=fontsize)
	plt.savefig(FIGPATH+savepath+'.png')
	plt.close()

Check_edge_train_weak_duration();exit()
#Check_nb_train_edge_weight();exit()

Fig_chap2_distr();exit()

where_RS = os.path.join(os.path.dirname(__file__),'min_EW3_RS.txt')
list_names = ['conf16','EW_conf16','utah',('min_EW3_RS',where_RS),'min_EW3']
#list_names = ['conf16','french','utah','highschool3','work2','min_EW3']

dic_obs = get_dic_obs()
distr = get_distr(list_names,dic_obs)
Plot_all_distr(distr,list_names,dic_obs);exit()

####################################################

fontsize = 15
obs = 'node_degree'; val = distr[obs]
fig,ax = main.Setup_Plot(dic_obs[obs],r'$\log_{10}(P)$',fontsize=fontsize)
for data,name,marker,color in zip(val,list_names,main.LIST_MARKER,main.LIST_COLOR):
	ax.plot(*data,marker,color=color,label=name[0])
ax.legend(fontsize=fontsize)
plt.show()
exit()
list_k,list_val = zip(*sorted(list(zip(*val[0])),key=lambda el:el[0]))
ax.plot(list_k,Get_poisson_degree(list_k,list_val),'--',label='Poisson law')
#x_m = np.min(val[0][0]); y_M = np.max(val[0][1]); x_M = np.max(val[0][0])
#ax.plot([x_m,x_M],[y_M,y_M-2*(x_M-x_m)],'--',color='k',label='th')
ax.legend(fontsize=fontsize)
#plt.savefig(FIGPATH+obs+'min_EW.png')
plt.show()
#plt.close()
exit()

