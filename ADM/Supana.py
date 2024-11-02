from Global import *
import networkx as nx
import Temp_net
import atn
from scipy.stats import truncnorm
from scipy.stats import ks_2samp,kstest
from community import community_louvain

XP_data = np.loadtxt('XP_data.txt',dtype=str)

def Load_instance_param(version_nb,name):
	model = 'ADM_class_V'+str(version_nb)
	set_int_param = {'m_max','m','c'}
	dic_param = {}
	#load parameters
	best_param = np.loadtxt('analysis/'+model+'/'+name+'/best_param.txt',dtype=str,delimiter=',')
	for i,param in enumerate(best_param[0,:]):
		if param in set_int_param:
			dic_param[param] = int(best_param[1,i])
		else:
			dic_param[param] = float(best_param[1,i])
	return dic_param

def Load_XP_info(name):
	global_info = np.loadtxt('analysis/'+name+'/global_info.txt',dtype=str,delimiter=',')
	XP_info = {}
	for i in range(len(global_info[0,:])):
		x = global_info[0,i]; y = global_info[1,i]
		if x in {'N','T','nb of edges'}:
			XP_info[x] = int(y)
		else:
			XP_info[x] = float(y)
	return XP_info

#for the dataset of name dataset, compute and save the observed node activity distribution
def Node_acti(dataset):
	if type(dataset)==tuple:
		#then we analyze the model version dataset[0] w.r.t. the reference of name dataset[1]
		path = 'ADM_class_V'+str(dataset[0])+'/'; name = dataset[1]
		#load instance parameters
		dic_param = Load_instance_param(dataset[0],name)
		#load XP info
		XP_info = Load_XP_info(name)
		#generate the model instance
		Model = atn.ADM_class(XP_info,**versions[dataset[0]])
		for param in Model.free_param.keys():
			Model.free_param[param] = dic_param[param]
		Model.Refresh()
		TN_data = Model.Evolve()
	else:
		path = ''; name = dataset
		#then we analyze the empirical reference of name dataset
		TN_data = np.loadtxt('data/'+name+'.txt',dtype=int)
	#analyze the resulting temporal network
	temp_net = Temp_net.Temp_net(TN_data)
	temp_net.Get_data_time()
	temp_net.Get_TN(1)
	temp_net.Get_info()
	temp_net.Events()
	Save_distribution('analysis/'+path+name,'node_weight',temp_net.Weight(temp_net.node_event))

def Load_node_weight(dataset,option='raw'):
	if type(dataset)==tuple:
		path = 'ADM_class_V'+str(dataset[0])+'/'; name = dataset[1]
	else:
		path = ''; name = dataset
	data = np.loadtxt('analysis/'+path+name+'/distribution/node_weight.txt',dtype=int)
	tab = []
	for i in range(np.size(data,1)):
		tab += [data[0,i]]*data[1,i]
	tab = np.asarray(tab,dtype=float)
	#T = duration of the data set
	T = float(Load_XP_info(name)['T'])
	tab /= T
	if option=='log':
		tab = np.log10(tab)
	elif option=='lin-rescaled':
		m = np.min(tab); M = np.max(tab)
		tab = (tab-m)/(M-m)
	elif option=='log-rescaled':
		tab = np.log10(tab)
		m = np.min(tab); M = np.max(tab)
		tab = (tab-m)/(M-m)
	return tab

#draw the node weight distribution
def Display_node_weight(dataset,option='raw',return_fig=False,hist_label=None):
	#display the results
	if option=='raw':
		xlabel = r"$a$"
	elif option=='log':
		xlabel = r"$\log_{10}(a)$"
	elif option=='rescaled':
		xlabel = "rescaled log10 activity"
	ylabel = r"$P$"; tab = Load_node_weight(dataset,option=option)
	fig,ax = Setup_Plot(xlabel,ylabel)
	if hist_label is None:
		ax.hist(tab,density=True)
	else:
		ax.hist(tab,density=True,label=hist_label)
	#ax.set_yscale('log', nonposy='clip')
	if type(dataset)==tuple:
		namefig = 'V'+str(dataset[0])+dataset[1]
	else:
		namefig = dataset
	if return_fig:
		return fig,ax
	else:
		plt.savefig('figures/node_weight/distr_'+option+'_'+namefig+'.png')

#perform the KS test to check whether the rescaled node weight is drawn from the same distribution
#for all empirical data sets
def XP_KS(option='raw'):
	nb_xp = len(XP_data)
	simat = np.eye(nb_xp,dtype=int)
	for i,name1 in enumerate(XP_data):
		tab1 = Load_node_weight(name1,option=option)
		for j,name2 in enumerate(XP_data[i+1:],start=i+1):
			tab2 = Load_node_weight(name2,option=option)
			p_value = ks_2samp(tab1,tab2)[1]
			#if tab1 and tab2 are sampled from the same distribution
			if p_value>0.05:
				simat[i,j] = 1
			else:
				simat[i,j] = 0
			simat[j,i] = simat[i,j]
	fontsize = 16
	fig,ax = plt.subplots(1,1,constrained_layout=True)
	ax.set_xticks(range(nb_xp))
	ax.set_yticks(range(nb_xp))
	ax.set_yticklabels(XP_data,fontsize=fontsize)
	ax.set_xticklabels(XP_data,rotation=90,fontsize=fontsize)
	ax.imshow(simat,cmap='gnuplot2')
	plt.savefig('figures/node_weight/KS_simat_'+option+'.png')

#perform the KS test to compare the rescaled node weights of model instances
#associated to the same reference name
def model_KS(name,option='raw'):
	network = nx.Graph()
	network.add_nodes_from(range(1,20))
	for v1 in range(1,20):
		tab1 = Load_node_weight((v1,name),option=option)
		for v2 in range(v1+1,20):
			tab2 = Load_node_weight((v2,name),option=option)
			p_value = ks_2samp(tab1,tab2)[1]
			#if tab1 and tab2 are sampled from the same distribution
			if p_value>0.05:
				network.add_edge(v1,v2)
	node_to_comm = community_louvain.best_partition(network)
	comm_to_node = {}
	for node,comm in node_to_comm.items():
		if comm in comm_to_node:
			comm_to_node[comm].add(node)
		else:
			comm_to_node[comm] = {node}
	comm = sorted(list(comm_to_node.values()),key=len,reverse=True)
	labels = [v1 for el in comm for v1 in el]
	nb = 19; simat = np.eye(nb,dtype=int)
	for i,v1 in enumerate(labels):
		for j,v2 in enumerate(labels[i+1:],start=i+1):
			if network.has_edge(v1,v2):
				simat[i,j] = 1
			else:
				simat[i,j] = 0
			simat[j,i] = simat[i,j]
	fontsize = 14
	fig,ax = plt.subplots(1,1,constrained_layout=True)
	ax.set_xticks(range(nb))
	ax.set_yticks(range(nb))
	ax.set_yticklabels(labels,fontsize=fontsize)
	ax.set_xticklabels(labels,fontsize=fontsize)
	ax.imshow(simat,cmap='gnuplot2')
	plt.savefig('figures/node_weight/KS_simat_'+option+'_'+name+'.png')

def CDF_power_law(y,a_min,a_max):
	if type(y)==np.ndarray:
		res = np.zeros(y.shape)
		for i,val in enumerate(y):
			if val>=a_max:
				res[i] = 1
			elif val>0:
				res[i] = log(val/a_min)/log(a_max/a_min)
		return res
	else:
		if y<=0:
			return 0
		elif y>=a_max:
			return 1
		return log(y/a_min)/log(a_max/a_min)

#perform the KS test to compare the distribution of node intrinsic activity in a model instance
#with its node weight distribution
#return 1 if the two distributions are considered as identical, 0 else
def KS_intr(version_nb,name):
	#load the parameters of the distribution of the node intrinsic activity
	dic_param = Load_instance_param(version_nb,name)
	if 'a_max' in dic_param:
		a_min,a_max = dic_param['a_min'],dic_param['a_max']
		#load the node weight distribution
		node_weight = Load_node_weight((version_nb,name),option='raw')
		m = np.min(node_weight); M = np.max(node_weight)
		#rescale it btw a_min and a_max
		node_weight = a_min + (node_weight-m)*(a_max-a_min)/(M-m)
		#if p_value<0.05, the two distributions are considered as different from each other
		return int(kstest(node_weight,CDF_power_law,args=(a_min,a_max))[1]>0.05)
	else:
		a = dic_param['a']
		node_weight = Load_node_weight((version_nb,name),option='raw')
		return int(ks_2samp(node_weight,[a]*len(node_weight))[1]>0.05)

#for a model instance, display on the same figure the node weight
#and node intrinsic activity histograms
def Display_both(version_nb,name):
	fig,ax = Display_node_weight((version_nb,name),return_fig=True,hist_label='node weight')
	#load the parameters of the node intrinsic activity distribution
	dic_param = Load_instance_param(version_nb,name)
	if 'a_min' in dic_param:
		a_min,a_max = dic_param['a_min'],dic_param['a_max']
		tab = []
		for _ in range(1000):
			tab.append(a_min*(a_max/a_min)**rd.random())
		ax.hist(tab,density=True,alpha=0.5,bins=10,label='intrinsic activity')
	else:
		a = dic_param['a']
		ax.plot([a]*2,[*ax.get_ylim()],label='intrinsic activity')
	ax.legend(fontsize=16)
	namefig = 'V'+str(version_nb)+name
	plt.savefig('figures/node_weight/distr_both_'+namefig+'.png')

#plot the node activity histogram for aggregation level agg
def Get_node_act(agg,name):
	for cand,val in dts.listdts.items():
		if name in val:
			pathres = cand
	N = dts.listdts[pathres][name]
	raw_data = Raw_data(dts.listdatapath[pathres]+name+".txt",N)
	raw_data.get_data_time()
	node_act = np.zeros(N,dtype=float)
	for t,event in raw_data.Get_dicevent(agg).items():
		for i in event.nodes:
			node_act[i] += 1
	print('duree = '+str(t))
	plt.figure()
	detail_title = '\ndataset : '+name
	plt.title('histogram of the log of the node activity and its fit'+detail_title)
	plt.xlabel(r"$\log_{10}(a)$")
	plt.ylabel(r"$P$")
	tab = np.log10(node_act)-np.log10(t)
	values,bins = plt.hist(tab,density=True,label='XP histo')[:2]

	a = np.min(tab); b = np.max(tab)
	#estimate the parameter mu
	bin_min = len(bins)-3
	while values[bin_min]>values[-1]:
		bin_min -= 1
	mu_tab = [el for el in tab if el>=bins[bin_min]]
	mu = np.mean(mu_tab); sig = 2*sqrt(np.var(mu_tab))
	nb = 100
	X = np.linspace(np.min(tab),np.max(tab),nb)
	Y = Trunc_density(X,mu,sig,a,b)
	rescale = np.max(values)/np.max(Y); Y *= rescale
	plt.plot(X,Y,'--',label='fit pdf')
	plt.plot([bins[bin_min],bins[bin_min]],[0,1.05*np.max(values)],'--')

	aa, bb = (a - mu) / sig, (b - mu) / sig
	tab = truncnorm.rvs(aa,bb,loc=mu,scale=sig,size=10000)
	plt.hist(tab,density=True,bins=50,alpha=0.5,label='fit histo')
	plt.legend()
	plt.show()
