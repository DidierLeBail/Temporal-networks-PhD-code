import numpy as np
import networkx as nx
import random as rd
from Librairies.settings import PROJECT_ROOT,Load_TN_ADM,Load_TN_Min_EW,Load_TN_min_ADM

'''
observables which depend on the TN at distinct states (n,b) are computed externally to the class
These are:
 - ETN autosimilarity, KS test of any distr obs btw different states
'''
#load TN data of a given name
def Load_TN(name):
	if type(name)==tuple:
		if type(name[0])==str:
			#name should be of the form ('ADM',version_nb,reference)
			if name[0]=='ADM':
				return Load_TN_ADM(name[1],name[2])
			#name should be of the form ('min_EW',version_nb)
			elif name[0]=='min_EW':
				return Load_TN_Min_EW(name[1])
			elif name[0]=='min_ADM':
				return Load_TN_min_ADM(name[1])
		#name should be of the form (dataset,name of the dataset)
		elif len(name)==2 and type(name[0])==np.ndarray:
			return name[0]
	return np.loadtxt(PROJECT_ROOT+'/data/'+name+'.txt',dtype=int)

#an observable is written as 'object'0'object property we want to sample', e.g.
#'NCTN0nb_diff' or 'edge0event_duration' or 'ICC' or 'node_space_weight'
def Obs_to_chunks(obs):
	if '0' in obs:
		obj1,prop = obs.split('0')
		if 'duration' in prop or 'events' in prop:
			obj2 = 'train'
		elif 'time_weight' in prop or 'motif' in prop or 'entropy' in prop or 'nb' in prop:
			obj2 = 'time_weight'
		return ([obj1,obj2],prop)
	return ([],obs)

def All_NCTN_profiles(depth):
	P_to_int = {}; profile = ['0']*depth
	for num in range(1,2**depth):
		ind = depth-1
		while profile[ind]=='1':
			profile[ind] = '0'
			ind -= 1
		profile[ind] = '1'
		P_to_int[''.join(profile)] = num
	return P_to_int

def All_ECTN_profiles(depth):
	Psat_to_int = {}; profile = ['0']*depth
	for num in range(1,4**depth):
		ind = depth-1
		while profile[ind]=='3':
			profile[ind] = '0'
			ind -= 1
		profile[ind] = str(int(profile[ind])+1)
		Psat_to_int[''.join(profile)] = num
	Pcentral_to_int = {}; profile = ['0']*depth
	for num in range(4**depth+1,4**depth+2**depth):
		ind = depth-1
		while profile[ind]=='1':
			profile[ind] = '0'
			ind -= 1
		profile[ind] = '1'
		Pcentral_to_int[''.join(profile)] = num
	return Psat_to_int,Pcentral_to_int

class Temp_net:
	"""
	observables of reference :
	- size of connected components of the interaction graph
	- activity duration of nodes, edges
	- newborn activity of edges
	- delayed activity duration of events
	- interactivity duration of nodes, edges
	- aggregated weight of (2,1)-ETN, (3,1)-ETN, edges
	- clustering coefficient of the static network
	- unweighted degree distribution of the static network
	- degree assortativity of the static network
	- (3,n)-ETN for n = 1,...,10
	"""
	def __init__(self,dataset):
		#allows to keep some info in memory to fasten computations
		#memory[(num,arg)] = result of the method identified by the integer num run with the arguments *arg
		self.memory = {'node':{},'edge':{}}
		self.Compute_object = {'edge':{'train':self.Edge_event_train,'time_weight':self.Edge_dic_weight}}
		self.Compute_object['node'] = {'train':self.Node_event_train,'time_weight':self.Node_dic_weight}
		self.Compute_object['NCTN'] = {'train':self.ETN_event_train,'time_weight':self.ETN_histo}
		self.Compute_object['ECTN'] = {'train':self.EdgeTN_event_train,'time_weight':self.EdgeTN_histo}
		self.Get_histo = {'duration':self.Duration_histo,'interduration':self.Interduration_histo,'event_duration':self.Events_duration_histo}
		self.Get_histo['time_weight'] = self.Weight_histo
		self.Get_histo['newborn_duration'] = self.Newborn_duration_histo
		self.Get_histo['profile_from_motif'] = self.Profile_from_motif
		self.Get_histo['profile_from_events'] = self.Profile_from_events
		self.Get_histo['inst_deg'] = self.Get_inst_deg
		self.Get_histo['cc_size'] = self.Get_cc_size
		self.Get_histo['cum_edge_weight'] = self.Get_cum_edge_weight_single
		self.Get_histo['node_space_weight'] = self.Get_node_space_weight
		self.Get_histo['edge_space_weight'] = self.Get_edge_space_weight
		self.Get_val = {'motif_error':self.Get_motif_error,'entropy':self.Get_entropy}
		self.Get_val['nb_diff'] = self.Get_nb_diff
		self.Get_val['nb_tot'] = self.Get_nb_tot
		self.Get_val['ICC'] = self.Get_val_ICC
		#dic_obs[type_obs][obs] = observable
		self.dic_obs = {'point':{},'distribution':{},'vector':{}}
		#measurement data
		self.data = dataset
		#interaction temporal network
		self.TN = {}
		#data_time[i] = [t_i, n_1, n_p] with t_i the (i+1)th time appearing in data
		#n_1 the first line of occurrence of t_i and n_p-1 the last one
		self.data_time = []
		#info contains :
		#number of nodes
		#number of temporal edges
		#number of timestamps
		#minimum node activity
		#maximum node activity
		#the parameters mu and sigma of the distribution of the node activity
		self.info = {}	
		#weighted aggregated static network
		self.static_net = nx.Graph()
		#Ind_to_edge[ind] = (i,j) corresponding to identifier ind
		self.Ind_to_edge = []

	#replace self.data by its formatted version, i.e. time begins at 0, two consecutive times
	#are separated by one and nodes are numeroted from 0 to nb of nodes-1
	def Format(self):
		self.Get_data_time()
		node_to_int = {}; num = 0
		for n in range(np.size(self.data,0)):
			for k in self.data[n,1:]:
				if not k in node_to_int:
					node_to_int[k] = num
					num += 1
		for t,el in enumerate(self.data_time):
			self.data[el[1]:el[2],0] = t
		for n in range(np.size(self.data,0)):
			for k in range(1,3):
				self.data[n,k] = node_to_int[self.data[n,k]]

	#compute self.data_time
	def Get_data_time(self):
		self.data_time = []
		n1 = 0; n_max = np.size(self.data,0)
		for n in range(1,n_max):
			t = self.data[n-1,0]
			if self.data[n,0]>t:
				self.data_time += [[t,n1,n]]
				n1 = n
		#take care of the last line of data
		t = self.data[-1,0]
		self.data_time += [[t,n1,n_max]]

	#extract the profile histo with given depth from the events train of obj
	def Profile_from_events(self,key_mem):
		obj,depth = key_mem
		P_to_int = All_NCTN_profiles(depth)
		P_to_int['0'*depth] = 0
		histo = {num:0 for num in P_to_int.values()}
		for val in self.memory[obj]['train'].values():
			last_time = 0
			tot_profile = ''
			for el in val:
				tot_profile += '0'*(el[0]-last_time)
				tot_profile += '1'*(el[1]-el[0]+1)
				last_time = el[1]+1
			for t in range(len(tot_profile)-depth+1):
				num = P_to_int[tot_profile[t:t+depth]]
				histo[num] += 1
		return histo

	#extract the edge profile histo from dic_ETN
	def Profile_from_NCTN(self,depth):
		P_to_int = All_NCTN_profiles(depth)
		histo = {i:0 for i in P_to_int.values()}
		for seq,nb in self.memory[('NCTN',depth)]['time_weight'].items():
			for i in range(len(seq)//depth):
				histo[P_to_int[seq[i*depth:(i+1)*depth]]] += nb
		return histo

	#extract the edge profile histo from dic_EdgeTN
	def Profile_from_ECTN(self,depth):
		Psat_to_int,Pcentral_to_int = All_ECTN_profiles(depth)
		histo = {i:0 for i in list(Psat_to_int.values())+list(Pcentral_to_int.values())}
		for seq,nb in self.memory[('ECTN',depth)]['time_weight'].items():
			histo[Pcentral_to_int[seq[:depth]]] += nb
			seq_sat = seq[depth:]
			for i in range(len(seq_sat)//depth):
				histo[Psat_to_int[seq_sat[i*depth:(i+1)*depth]]] += nb
		return histo

	#extract the edge profile histo from either dic_EdgeTN or dic_ETN
	def Profile_from_motif(self,key_mem):
		obj,depth = key_mem
		if obj=='NCTN':
			return self.Profile_from_NCTN(depth)
		elif obj=='ECTN':
			return self.Profile_from_ECTN(depth)

	#compute the total nb of ETN
	def Get_nb_tot(self,obj):
		return sum(list(self.memory[obj]['time_weight'].values()))

	#compute the nb of different ETN
	def Get_nb_diff(self,obj):
		return len(self.memory[obj]['time_weight'])

	def Clear_memory(self):
		self.memory = {'node':{},'edge':{}}

	#returns the histo of the total nb of nodes active at the same time
	def Get_node_space_weight(self):
		histo = {}
		for val in self.TN.values():
			n = len(val.nodes)
			if n in histo:
				histo[n] += 1
			else:
				histo[n] = 1
		return histo

	#returns the histo of the total nb of edges active at the same time
	def Get_edge_space_weight(self):
		histo = {}
		for val in self.TN.values():
			n = len(val.edges)
			if n in histo:
				histo[n] += 1
			else:
				histo[n] = 1
		return histo

	def Get_obs(self,obs,arg):
		#break the observable into meaningful chunks: the first chunk gives the object to compute
		#and the second tells how to compute the observable using this object
		#observable standard encoding: list of objects to compute followed by the property of the objects
		#we want to sample (note that there may be no object to compute, i.e. chunk_obj may be an empty list)
		chunk_obj,chunk_prop = Obs_to_chunks(obs)
		if chunk_obj:
			if not arg:
				key_mem = chunk_obj[0]
			else:
				key_mem = (chunk_obj[0],*arg)
			if not key_mem in self.memory:
				self.memory[key_mem] = {chunk_obj[1]:self.Compute_object[chunk_obj[0]][chunk_obj[1]](*arg)}
			elif chunk_obj[1] not in self.memory[key_mem]:
				self.memory[key_mem][chunk_obj[1]] = self.Compute_object[chunk_obj[0]][chunk_obj[1]](*arg)
			tot_arg = tuple([chunk_obj[0]])+arg
		else:
			tot_arg = arg
		if not tot_arg:
			if chunk_prop in self.Get_histo:
				return self.Get_histo[chunk_prop]()
			elif chunk_prop in self.Get_val:
				return self.Get_val[chunk_prop]()
		elif len(tot_arg)==1:
			key_mem = tot_arg[0]
		else:
			key_mem = tot_arg
		if chunk_prop in self.Get_histo:
			return self.Get_histo[chunk_prop](key_mem)
		elif chunk_prop in self.Get_val:
			return self.Get_val[chunk_prop](key_mem)

	def Get_val_ICC(self):
		return np.mean([nx.average_clustering(G) for G in self.TN.values()])

	def Get_entropy(self,obj):
		list_proba = np.asarray(list(self.memory[obj]['time_weight'].values()),dtype=float)
		norm = np.sum(list_proba)
		list_proba = list_proba/norm
		return np.sum(-list_proba*np.log10(list_proba))

	#return the error made on ETN proba by assuming independence of activity profiles
	#as well as independence btw nb of satellites and activity profiles
	def Get_motif_error(self,key_mem):
		obj,depth = key_mem
		if obj=='NCTN':
			abundancy,list_proba = self.NCTN_xpth_probas(depth)
		elif obj=='ECTN':
			abundancy,list_proba = self.ECTN_xpth_probas(depth) 
		return np.sqrt(np.sum((abundancy-list_proba)**2)/len(list_proba))

	def Get_cum_edge_weight_single(self,cum):
		nb_blocks = len(self.TN)//cum
		#histo[w] = nb of occurrences such that an edge is active w times on a time interval of length cum
		histo = {}
		for t in range(nb_blocks):
			cum_net = nx.Graph()
			for tau in range(t*cum,(t+1)*cum):
				for edge in self.TN[tau].edges:
					if cum_net.has_edge(*edge):
						cum_net[edge[0]][edge[1]]['weight'] += 1
					else:
						cum_net.add_edge(*edge,weight=1)
			for edge in cum_net.edges:
				w = cum_net[edge[0]][edge[1]]['weight']
				if w in histo:
					histo[w] += 1
				else:
					histo[w] = 1
		return histo

	def NCTN_histo_sat(self,depth):
		histo_sat = {}
		for seq,nb in self.memory[('NCTN',depth)]['time_weight'].items():
			nb_sat = len(seq)//depth
			if nb_sat in histo_sat:
				histo_sat[nb_sat] += nb
			else:
				histo_sat[nb_sat] = nb
		return histo_sat

	def ECTN_histo_sat(self,depth):
		histo_sat = {}
		for seq,nb in self.memory[('ECTN',depth)]['time_weight'].items():
			nb_sat = len(seq)//depth-1
			if nb_sat in histo_sat:
				histo_sat[nb_sat] += nb
			else:
				histo_sat[nb_sat] = nb
		return histo_sat

	#local time shuffling (TS) of self.TN with time window b
	#modifies self.TN as well as self.data_time but data is preserved
	def Local_TS(self,b,agg=1):
		#compute self.data_time
		self.Get_data_time()
		#compute the new self.data_time and deduce self.TN at agg level 1
		#determine the times permutation
		list_times = []; nb_blocks = len(self.data_time)//b
		for k in range(nb_blocks):
			list_times += rd.sample(range(k*b,(k+1)*b),b)
		last_block = len(self.data_time)-b*nb_blocks
		if last_block>0:
			list_times += rd.sample(range(b*nb_blocks,len(self.data_time)),last_block)
		new_data_time = [[list_times[el[0]],*el[1:]] for el in self.data_time]
		self.data_time = sorted(new_data_time,key=lambda el:el[0])
		self.Get_TN(agg)

	#local time reversal (TR) of self.TN with time window b
	#modifies self.TN as well as self.data_time but data is preserved
	def Local_TR(self,b,agg=1):
		#compute self.data_time
		self.Get_data_time()
		#compute the new self.data_time and deduce self.TN at agg level 1
		#determine the times permutation
		list_times = []; nb_blocks = len(self.data_time)//b
		for k in range(nb_blocks):
			list_times += [(k+1)*b-l-1 for l in range(b)]
		last_block = len(self.data_time)-b*nb_blocks
		if last_block>0:
			list_times += [len(self.data_time)-l-1 for l in range(last_block)]
		new_data_time = [[list_times[el[0]],*el[1:]] for el in self.data_time]
		self.data_time = sorted(new_data_time,key=lambda el:el[0])
		self.Get_TN(agg)

	#assume that self.Prepare() has been run
	#compute the reference observables
	def Ref_obs(self,norm=False,type_obs='all'):
		if type_obs=='all':
			keys = ['distribution','point','vector']
		elif type(type_obs)==list:
			keys = type_obs
		elif type(type_obs)==str:
			keys = [type_obs]
		for key in keys:
			self.Compute_ref_obs[key]()
		
		#normalize the distributions
		if norm:
			for name_obs,vec in self.dic_obs['distribution'].items():
				norm = sum(list(vec.values()))
				for key in vec.keys():
					self.dic_obs['distribution'][name_obs][key] /= norm

	#prepare the temporal network for analysis (computation of observables of reference)
	def Prepare(self):
		self.Get_data_time()
		self.Get_TN(1)
		self.Get_info()
		self.Static_net()
		self.Events()

	#minimum intialization to get the temporal network at aggregation level agg
	def Init(self,agg=None):
		self.Get_data_time()
		self.info['N'] = len(set(self.data[:,1]).union(set(self.data[:,2])))
		self.Ind_to_edge = [(i,j) for i in range(self.info['N']-1) for j in range(i+1,self.info['N'])]
		if type(agg)==int:
			self.Get_TN(agg)

	#compute the square interaction network
	def Square_TN(self):
		res = {}
		adj_mat = np.zeros((self.info['N'],self.info['N']),dtype=int)
		for t,event in self.TN.items():
			square_event = nx.Graph()
			for edge in event.edges:
				adj_mat[edge[0]][edge[1]] = 1
				adj_mat[edge[1]][edge[0]] = 1
			mat = np.dot(adj_mat,adj_mat)
			for i in range(self.info['N']):
				for j in range(i+1,self.info['N']):
					if mat[i,j]:
						square_event.add_edge(i,j)
			res[t] = square_event
			#reinitialize the adjacency matrix
			for edge in event.edges:
				adj_mat[edge[0]][edge[1]] = 0
				adj_mat[edge[1]][edge[0]] = 0
		self.TN = res

	#compute the interaction network at the power n
	def Power_TN(self,n):
		res = {}
		adj_mat = np.zeros((self.info['N'],self.info['N']),dtype=int)
		for t,event in self.TN.items():
			square_event = nx.Graph()
			for edge in event.edges:
				adj_mat[edge[0]][edge[1]] = 1
				adj_mat[edge[1]][edge[0]] = 1
			mat = np.eye(self.info['N'],dtype=int)
			for _ in range(n):
				mat = np.dot(mat,adj_mat)
			for i in range(self.info['N']):
				for j in range(i+1,self.info['N']):
					if mat[i,j]:
						square_event.add_edge(i,j)
			res[t] = square_event
			#reinitialize the adjacency matrix
			for edge in event.edges:
				adj_mat[edge[0]][edge[1]] = 0
				adj_mat[edge[1]][edge[0]] = 0
		self.TN = res

	#prepare to partial analysis (compute the observables of evaluation)
	def Partial_prepare(self,N,duration):
		self.info['N'] = N
		self.info['T'] = duration
		self.Get_data_time()
		self.Get_TN(1)
		self.info['nb of edges'] = len(self.data)

	#return the cumulated interaction graph at level cum
	def Get_cum_TN(self,cum):
		#split the timeline into blocks of size cum
		new_nb = len(self.TN)//cum
		cum_TN = {t:nx.Graph() for t in range(new_nb)}
		for t in range(new_nb):
			for k in range(cum):
				n1,n2 = self.data_time[t*cum+k][1:]
				for n in range(n1,n2):
					i,j = self.data[n,1:]
					if cum_TN[t].has_edge(i,j):
						cum_TN[t][i][j]['weight'] += 1
					else:
						cum_TN[t].add_edge(i,j,weight=1)
		return cum_TN

	#compute the interaction graph at aggregation level agg
	def Get_TN(self,agg):
		nb_time = len(self.data_time)
		new_nb = nb_time//agg
		#self.TN[t] = aggregated graph of interactions on t^th time interval
		self.TN = {t:nx.Graph() for t in range(new_nb)}
		for t in range(new_nb):
			for k in range(agg):
				n1,n2 = self.data_time[t*agg+k][1:]
				for n in range(n1,n2):
					self.TN[t].add_edge(*self.data[n,1:],key=self.Edge_to_ind(*self.data[n,1:]))
		#take care of the last aggregation window
		if new_nb*agg<nb_time:
			self.TN[new_nb] = nx.Graph()
			for k in range(nb_time-agg*new_nb):
				n1,n2 = self.data_time[agg*new_nb+k][1:]
				for n in range(n1,n2):
					self.TN[new_nb].add_edge(*self.data[n,1:],key=self.Edge_to_ind(*self.data[n,1:]))
		self.info['T'] = len(self.TN)

	#compute the number of timestamps, nodes and temporal edges
	#as well as the parameters of the node activity distribution viewed
	#as the exponential of a two sided truncated Gaussian variable
	def Get_info(self):
		self.info['T'] = len(self.TN)
		self.info['nb of edges'] = len(self.data)
		nodes = set(())
		for events in self.TN.values():
			nodes = nodes.union(set(events.nodes))
		self.info['N'] = len(nodes)
		activity = np.zeros(self.info['N'],dtype=float)
		for events in self.TN.values():
			for node in events.nodes:
				activity[node] += 1
		self.info['a_min'] = np.min(activity)/self.info['T']
		self.info['a_max'] = np.max(activity)/self.info['T']
		tab = np.log10(activity)-np.log10(self.info['T'])
		values,bins = np.histogram(tab,density=True)
		#estimate the parameters mu and sigma
		bin_min = len(bins)-3
		while values[bin_min]>values[-1]:
			bin_min -= 1
		mu_tab = [el for el in tab if el>=bins[bin_min]]
		self.info['mu'] = np.mean(mu_tab)
		self.info['sigma'] = 2*np.sqrt(np.var(mu_tab))

	#compute the weighted aggregated static network
	def Static_net(self):
		self.static_net = nx.Graph()
		for events in self.TN.values():
			for edge in events.edges:
				if self.static_net.has_edge(*edge):
					self.static_net[edge[0]][edge[1]]['weight'] += 1
				else:
					self.static_net.add_edge(*edge,weight=1)

	#compute the observables associated to the static network :
	# - unweighted degree distribution
	# - edge weight distribution
	# - correlation between the two (degree assortativity)
	# - clustering coefficient
	def Get_static_obs(self):
		histo_degree = {}
		for i in self.static_net.nodes:
			degree = self.static_net.degree(i)
			if degree in histo_degree:
				histo_degree[degree] += 1
			else:
				histo_degree[degree] = 1
		histo_weight = {}
		for edge in self.static_net.edges:
			weight = self.static_net[edge[0]][edge[1]]['weight']
			if weight in histo_weight:
				histo_weight[weight] += 1
			else:
				histo_weight[weight] = 1
		clustering_coeff = nx.average_clustering(self.static_net)
		deg_assortativity = nx.degree_pearson_correlation_coefficient(self.static_net,weight='weight')
		self.static_obs_done = [histo_weight,histo_degree,clustering_coeff,deg_assortativity]

	def Edge_to_ind(self,i,j):
		if i>j:
			i,j = j,i
		return i*(2*self.info['N']-i-1)//2 + j-i-1

	#compute the histogram of the edge activity derivative
	def Edge_der(self):
		#histo[dE] = nb of times t such that E(t+1)-E(t) = dE
		histo = {}
		for t in range(len(self.TN)-1):
			dE = len(self.TN[t+1].edges)-len(self.TN[t].edges)
			if dE in histo:
				histo[dE] += 1
			else:
				histo[dE] = 1
		return histo

	#from dic_ETN deduce the th and XP relations btw a edge centered motif and its time_weight
	def ECTN_xpth_probas(self,depth):
		Psat_to_int,Pcentral_to_int = All_ECTN_profiles(depth)
		#determine the distribution for the number of satellites on the recorded EdgeTN
		histo_sat = self.ECTN_histo_sat(depth)
		norm = sum(list(histo_sat.values()))
		for key in histo_sat.keys():
			histo_sat[key] /= norm
		#different distribution for the central edge and the satellites profiles
		histo_profiles = self.Profile_from_ECTN(depth)
		#normalize the profile proba
		norm_central = sum([histo_profiles[i] for i in Pcentral_to_int.values()])
		norm_sat = sum([histo_profiles[i] for i in Psat_to_int.values()])
		for i in Pcentral_to_int.values():
			histo_profiles[i] /= norm_central
		for i in Psat_to_int.values():
			histo_profiles[i] /= norm_sat
		#determine the proba of each observed ETN and plot it against its observed abundancy
		list_seq,abundancy = zip(*self.memory[('ECTN',depth)]['time_weight'].items())
		list_proba = []
		for seq in list_seq:
			nb_sat = len(seq)//depth-1
			proba = histo_sat[nb_sat]
			proba *= histo_profiles[Pcentral_to_int[seq[:depth]]]
			seq_sat = seq[depth:]
			for i in range(nb_sat):
				proba *= histo_profiles[Psat_to_int[seq_sat[i*depth:(i+1)*depth]]]
			list_proba.append(proba)
		abundancy = np.asarray(abundancy,dtype=float)
		norm = np.sum(abundancy)
		abundancy = abundancy/norm
		return abundancy,np.asarray(list_proba,dtype=float)

	#from dic_ETN deduce the th and XP relations btw a node centered motif and its time_weight
	def NCTN_xpth_probas(self,depth):
		Psat_to_int = All_NCTN_profiles(depth)
		#determine the distribution for the number of satellites (aggregated degree on depth)
		histo_sat = self.NCTN_histo_sat(depth)
		norm = sum(list(histo_sat.values()))
		for key in histo_sat.keys():
			histo_sat[key] /= norm
		#determine the distribution for the activity profiles
		histo_profiles = self.Profile_from_NCTN(depth)
		#normalize the profile proba
		norm = sum(list(histo_profiles.values()))
		for key in histo_profiles.keys():
			histo_profiles[key] /= norm
		#determine the proba of each observed ETN and plot it against its observed abundancy
		list_seq,abundancy = zip(*self.memory[('NCTN',depth)]['time_weight'].items())
		list_proba = []
		for seq in list_seq:
			nb_sat = len(seq)//depth
			proba = histo_sat[nb_sat]
			for i in range(nb_sat):
				proba *= histo_profiles[P_to_int[seq[i*depth:(i+1)*depth]]]
			list_proba.append(proba)
		abundancy = np.asarray(abundancy,dtype=float)
		norm = np.sum(abundancy)
		abundancy = abundancy/norm
		return abundancy,np.asarray(list_proba,dtype=float)

	#return the error made on the ETN probability by considering activity profiles as stat
	#independent, this error being computed at a given aggregation level and
	#on a time window of a given length and averaged over the whole dataset timeline
	#return also the density of instantaneous triangles (interaction graph clustering coefficient)
	#averaged over the same time windows
	#for both the error and clustering coeff, return both the mean and std over the time windows
	def Get_ETN_error_tri(self,inst_ETN_histo,inst_triangles,depth,window):
		error = []; triangle = []
		#handle the first window
		merged_ETN = {}; merged_tri = 0
		for tau in range(window):
			merged_tri += inst_triangles[tau]
			dic_ETN = inst_ETN_histo[tau]
			for seq,nb in dic_ETN.items():
				if seq in merged_ETN:
					merged_ETN[seq] += nb
				else:
					merged_ETN[seq] = nb
		abundancy,list_proba = self.From_ETN_to_stat(merged_ETN,depth)
		error.append(np.sqrt(np.sum((abundancy-list_proba)**2))/len(list_proba))
		triangle.append(merged_tri/window)
		for t in range(1,len(self.TN)-window-depth+2):
			#remove the time t-1
			merged_tri -= inst_triangles[t-1]
			dic_ETN = inst_ETN_histo[t-1]
			for seq,nb in dic_ETN.items():
				merged_ETN[seq] -= nb
				if merged_ETN[seq]<=0:
					del merged_ETN[seq]
			#add the time t-1+window
			merged_tri += inst_triangles[t-1+window]
			dic_ETN = inst_ETN_histo[t-1+window]
			for seq,nb in dic_ETN.items():
				if seq in merged_ETN:
					merged_ETN[seq] += nb
				else:
					merged_ETN[seq] = nb
			abundancy,list_proba = self.From_ETN_to_stat(merged_ETN,depth)
			error.append(np.sqrt(np.sum((abundancy-list_proba)**2))/len(list_proba))
			triangle.append(merged_tri/window)
		return {'mean':np.mean(error),'std':np.std(error)},{'mean':np.mean(triangle),'std':np.std(triangle)}

	#assumes self.Get_TN has been run
	#compute the histogram for instantaneous node degree
	def Get_inst_deg(self):
		histo = {}
		for val in self.TN.values():
			#count the nb of nodes with degree zero (these are the inactive nodes)
			nb = self.info['N']-len(val.nodes)
			if nb:
				histo[0] = nb
			for i in val.nodes:
				deg = val.degree(i)
				if deg in histo:
					histo[deg] += 1
				else:
					histo[deg] = 1
		return histo

	#compute the train of events for nodes
	def Node_event_train(self):
		node_event = {}
		active_nodes = set(self.TN[0].nodes)
		node_starting_time = {i:0 for i in active_nodes}
		for t in range(1,len(self.TN)):
			current_nodes = set(self.TN[t].nodes)
			#edges active at t and not at t-1
			new_nodes = current_nodes.difference(active_nodes)
			#edges active at t-1 and not at t
			finished_nodes = active_nodes.difference(current_nodes)
			for ind in finished_nodes:
				if ind in node_event:
					node_event[ind].append((node_starting_time[ind],t-1))
				else:
					node_event[ind] = [(node_starting_time[ind],t-1)]
				del node_starting_time[ind]
			for ind in new_nodes:
				node_starting_time[ind] = t
			#update the set of active nodes
			active_nodes = current_nodes
		for ind in active_nodes:
			if ind in node_event:
				node_event[ind].append((node_starting_time[ind],self.info['T']-1))
			else:
				node_event[ind] = [(node_starting_time[ind],self.info['T']-1)]
		return node_event

	#compute the train of events for edges
	#edge_event[ind][k] = (t_0,t_f) with t_0 = starting time of the k^th event for edge of identifier ind
	#and t_f = ending time
	def Edge_event_train(self):
		edge_event = {}
		#edge_starting_time[ind] = last starting time of the activation of the edge of identifier ind
		edge_starting_time = {}
		#active_edges = set of edges active at current time
		active_edges = set(())
		for edge in self.TN[0].edges:
			ind = self.Edge_to_ind(*edge)
			active_edges.add(ind)
			edge_starting_time[ind] = 0
		for t in range(1,len(self.TN)):
			#edges active at t
			current_edges = {self.Edge_to_ind(*edge) for edge in self.TN[t].edges}
			#edges active at t and not at t-1
			new_edges = current_edges.difference(active_edges)
			#edges active at t-1 and not at t
			finished_edges = active_edges.difference(current_edges)
			for ind in finished_edges:
				if ind in edge_event:
					edge_event[ind].append((edge_starting_time[ind],t-1))
				else:
					edge_event[ind] = [(edge_starting_time[ind],t-1)]
				del edge_starting_time[ind]
			for ind in new_edges:
				edge_starting_time[ind] = t
			#update the set of active edges
			active_edges = current_edges
		#take care of the last timestamp
		for ind in active_edges:
			if ind in edge_event:
				edge_event[ind].append((edge_starting_time[ind],self.info['T']-1))
			else:
				edge_event[ind] = [(edge_starting_time[ind],self.info['T']-1)]
		return edge_event

	#res[ind] = time weight of edge ind
	def Edge_dic_weight(self):
		res = {}
		for val in self.TN.values():
			for edge in val.edges:
				ind = self.Edge_to_ind(*edge)
				if ind in res:
					res[ind] += 1
				else:
					res[ind] = 1
		return res

	#res[i] = time weight of node i
	def Node_dic_weight(self):
		res = {i:0 for i in range(self.info['N'])}
		for val in self.TN.values():
			for i in val.nodes:
				res[ind] += 1
		return res

	#compute the weight of events occurring in event_train
	#the weight of an event is just the total number of time steps it occurred in event_train
	def Weight_histo(self,obj):
		histo = {}
		for weight in self.memory[obj]['time_weight'].values():
			if weight in histo:
				histo[weight] += 1
			else:
				histo[weight] = 1
		return histo

	#compute the not normalized histogram of the activity duration
	#for the list of events event_train
	def Duration_histo(self,obj):
		histo = {}
		for val in self.memory[obj]['train'].values():
			for el in val:
				duration = el[1]-el[0]+1
				if duration in histo:
					histo[duration] += 1
				else:
					histo[duration] = 1
		return histo

	#compute the not normalized histogram of the newborn activity duration
	#for the list of events event_train
	def Newborn_duration_histo(self,obj):
		histo = {}
		for key in self.memory[obj]['train'].keys():
			el = self.memory[obj]['train'][key][0]
			duration = el[1]-el[0]+1
			if duration in histo:
				histo[duration] += 1
			else:
				histo[duration] = 1
		return histo

	#compute the not normalized histogram of the interactivity duration
	#for the list of events event_train
	def Interduration_histo(self,obj):
		histo = {}
		for val in self.memory[obj]['train'].values():
			for n in range(1,len(val)):
				interduration = val[n][0]-val[n-1][1]-1
				if interduration in histo:
					histo[interduration] += 1
				else:
					histo[interduration] = 1
		return histo

	#compute the not normalized histogram of the delayed activity duration for events
	#for the list of events event_train
	def Events_duration_histo(self,obj,delay=4):
		histo = {}
		for val in self.memory[obj]['train'].values():
			if len(val)==1:
				if 1 in histo:
					histo[1] += 1
				else:
					histo[1] = 1
			else:
				#nb of consecutive events
				nb = 1
				for n in range(1,len(val)):
					interduration = val[n][0]-val[n-1][1]-1
					if interduration<delay:
						nb += 1
					else:
						if nb in histo:
							histo[nb] += 1
						else:
							histo[nb] = 1
						nb = 1
		return histo

	#compute the observables of evaluation
	def Evaluation_obs(self):
		self.eval_obs['vector']['ETN3'] = self.ETN_vector(5,3)[0]

	#compute the distribution of the size of connected components of the interaction graph
	def Get_cc_size(self):
		cc_histo = {}
		for G in self.TN.values():
			list_cc = nx.connected_components(G)
			for cc in list_cc:
				n = len(cc)
				if n in cc_histo:
					cc_histo[n] += 1
				else:
					cc_histo[n] = 1
		return cc_histo

	#compute the sequence of the motif starting at time t of central edge ind
	def Compute_EdgeTNS(self,t,ind,depth):
		#we want to keep the info of which node of the central edge i--j a given satellite has interacted
		#Hence we need to distinguish i and j, i.e. we want an order relation such that i>=j e.g.
		#Then we associate 1 to an interaction with i, 2 to an interaction with j, 3 with both and 0 else.
		#the order relation we use derives from the node centered ETN of i and j
		#here is the encoding: the depth first letters are the activity profile of the central edge
		#then we choose to represent i by 1 and j by 2 iif j>=i
		#note: i and j are equivalent iif the resulting sequence if invariant under 1,2<-->2,1
		edge = self.Ind_to_edge[ind]
		#central_profile = activity profile of the central edge
		central_profile = ['0']*depth
		#ETN of the nodes from the central edge
		node_seq_i = self.Compute_ETNS(t,edge[0],depth)
		node_seq_j = self.Compute_ETNS(t,edge[1],depth)
		#code for i and j
		if node_seq_i<=node_seq_j:
			node_to_letter = {edge[0]:1,edge[1]:2}
		else:
			node_to_letter = {edge[1]:1,edge[0]:2}
		#encode the activity profiles of the satellites: dic_s[u] = activity profile of satellite u
		dic_s = {}
		for tau in range(depth):
			if self.TN[t+tau].has_edge(*edge):
				central_profile[tau] = '1'
			#ngh0 = set of ngh of i other than j or ngh of j
			#ngh01 = set of common ngh of i and j
			ngh0 = set(()); ngh01 = set(()); ngh1 = set(())
			if edge[1] in self.TN[t+tau]:
				ngh1 = {u for u in self.TN[t+tau][edge[1]]}
			if edge[0] in ngh1:
				ngh1.remove(edge[0])
			if edge[0] in self.TN[t+tau]:
				for u in self.TN[t+tau][edge[0]]:
					if u in ngh1:
						ngh01.add(u)
					else:
						ngh0.add(u)
			if edge[1] in ngh0:
				ngh0.remove(edge[1])
			ngh1 = ngh1.difference(ngh01)
			for u in ngh01.union(ngh0).union(ngh1):
				if not u in dic_s:
					dic_s[u] = ['0']*depth
			for u in ngh01:
				dic_s[u][tau] = '3'
			for u in ngh0:
				dic_s[u][tau] = str(node_to_letter[edge[0]])
			for u in ngh1:
				dic_s[u][tau] = str(node_to_letter[edge[1]])
		return ''.join(central_profile)+''.join(sorted([''.join(val) for val in dic_s.values()]))

	#compute the sequence of the motif starting at time t of central node v
	def Compute_ETNS(self,t,v,depth):
		#dic_s[u][tau] = '1' if u is a satellite of v at time t+tau and '0' else
		dic_s = {}
		for tau in range(depth):
			#if v is active (i.e. has at least one satellite)
			if v in self.TN[t+tau]:
				for u in self.TN[t+tau][v]:
					if u in dic_s:
						dic_s[u][tau] = '1'
					else:
						dic_s[u] = ['0']*depth
						dic_s[u][tau] = '1'
		return ''.join(sorted([''.join(val) for val in dic_s.values()]))

	def Add_ETN(self,histo,central_nodes,t,depth):
		for v in central_nodes:
			seq = self.Compute_ETNS(t,v,depth)
			if seq in histo:
				histo[seq] += 1
			else:
				histo[seq] = 1
		return histo

	def Add_EdgeTN(self,histo,central_edges,t,depth):
		for ind in central_edges:
			seq = self.Compute_EdgeTNS(t,ind,depth)
			if seq in histo:
				histo[seq] += 1
			else:
				histo[seq] = 1
		return histo

	#compute the edge-centered ETN histogram for a given depth and aggregation level
	#central edges are edges active at least once on depth consecutive time steps
	def EdgeTN_histo(self,depth):
		#histo[seq] = nb of occurrences of EdgeTN seq in the whole temporal network
		histo = {}; nb_time = len(self.TN)
		list_times = range(nb_time-depth+1)
		#cumulate the interactions on depth time steps to keep track of the central edges
		dic_central_edges = {}
		for tau in range(depth):
			for edge in self.TN[tau].edges:
				ind = self.Edge_to_ind(*edge)
				if ind in dic_central_edges:
					dic_central_edges[ind] += 1
				else:
					dic_central_edges[ind] = 1
		histo = self.Add_EdgeTN(histo,dic_central_edges.keys(),0,depth)
		for t in list_times[1:]:
			#remove the edges from time t-1
			for edge in self.TN[t-1].edges:
				ind = self.Edge_to_ind(*edge)
				dic_central_edges[ind] -= 1
			#add the edges from time t+depth-1
			for edge in self.TN[t+depth-1].edges:
				ind = self.Edge_to_ind(*edge)
				if ind in dic_central_edges:
					dic_central_edges[ind] += 1
				else:
					dic_central_edges[ind] = 1
			edges_to_remove = {ind for ind,nb in dic_central_edges.items() if nb<=0}
			for ind in edges_to_remove:
				del dic_central_edges[ind]
			#update the EdgeTN histogram
			histo = self.Add_EdgeTN(histo,dic_central_edges.keys(),t,depth)
		return histo

	#return the set of ETN of given depth that are active at time t
	def Get_current_ETN(self,t,depth):
		central_nodes = set(())
		for tau in range(depth):
			central_nodes = central_nodes.union(set(self.TN[t+tau].nodes))
		return {self.Compute_ETNS(t,v,depth) for v in central_nodes}

	#return the set of EdgeTN of given depth that are active at time t
	def Get_current_EdgeTN(self,t,depth):
		central_edges = set(())
		for tau in range(depth):
			for edge in self.TN[t+tau].edges:
				central_edges.add(self.Edge_to_ind(*edge))
		return {self.Compute_EdgeTNS(t,ind,depth) for ind in central_edges}

	#compute the train of events for EdgeTN
	def EdgeTN_event_train(self,depth):
		ETN_event = {}
		#active_ETN = set of ETN active at current time
		active_ETN = self.Get_current_EdgeTN(0,depth)
		#ETN_starting_time[seq] = last starting time of the activation of the ETN of signature seq
		ETN_starting_time = {seq:0 for seq in active_ETN}
		for t in range(1,len(self.TN)-depth+1):
			#ETN active at t
			current_ETN = self.Get_current_EdgeTN(t,depth)
			#ETN active at t and not at t-1
			new_ETN = current_ETN.difference(active_ETN)
			#ETN active at t-1 and not at t
			finished_ETN = active_ETN.difference(current_ETN)
			for seq in finished_ETN:
				if seq in ETN_event:
					ETN_event[seq].append((ETN_starting_time[seq],t-1))
				else:
					ETN_event[seq] = [(ETN_starting_time[seq],t-1)]
				del ETN_starting_time[seq]
			for seq in new_ETN:
				ETN_starting_time[seq] = t
			#update the set of active ETN
			active_ETN = current_ETN
		#take care of the last timestamp
		for seq in active_ETN:
			if seq in ETN_event:
				ETN_event[seq].append((ETN_starting_time[seq],self.info['T']-1))
			else:
				ETN_event[seq] = [(ETN_starting_time[seq],self.info['T']-1)]
		return ETN_event

	#compute the train of events for ETN
	def ETN_event_train(self,depth):
		ETN_event = {}
		#active_ETN = set of ETN active at current time
		active_ETN = self.Get_current_ETN(0,depth)
		#ETN_starting_time[seq] = last starting time of the activation of the ETN of signature seq
		ETN_starting_time = {seq:0 for seq in active_ETN}
		for t in range(1,len(self.TN)-depth+1):
			#ETN active at t
			current_ETN = self.Get_current_ETN(t,depth)
			#ETN active at t and not at t-1
			new_ETN = current_ETN.difference(active_ETN)
			#ETN active at t-1 and not at t
			finished_ETN = active_ETN.difference(current_ETN)
			for seq in finished_ETN:
				if seq in ETN_event:
					ETN_event[seq].append((ETN_starting_time[seq],t-1))
				else:
					ETN_event[seq] = [(ETN_starting_time[seq],t-1)]
				del ETN_starting_time[seq]
			for seq in new_ETN:
				ETN_starting_time[seq] = t
			#update the set of active ETN
			active_ETN = current_ETN
		#take care of the last timestamp
		for seq in active_ETN:
			if seq in ETN_event:
				ETN_event[seq].append((ETN_starting_time[seq],self.info['T']-1))
			else:
				ETN_event[seq] = [(ETN_starting_time[seq],self.info['T']-1)]
		return ETN_event

	#compute the ETN histogram for a given depth and aggregation level
	#we restrict to motifs whose central node is active at the beginning or at the end of the motif
	#for motifs with depth 2, this includes all motifs but the empty one, and for motifs with depth n>2,
	#this excludes all motifs reducible to a motif of depth less than n
	#however if tot=True, we take these motifs into account as well
	def ETN_histo(self,depth,window='sliding',tot=True):
		#histo[seq] = nb of occurrences of ETN seq in the whole temporal network
		histo = {}; nb_time = len(self.TN)
		if window=='sliding':
			list_times = range(nb_time-depth+1)
		elif window=='jumping':
			list_times = range(0,nb_time-depth+1,depth)
		else:
			raise ValueError("choices for window are either sliding or jumping")
		if not tot:
			for t in list_times:
				#build the set of central nodes
				central_nodes = set(self.TN[t].nodes).union(set(self.TN[t+depth-1].nodes))
				#for each central node, compute the sequence of the associated motif
				histo = self.Add_ETN(histo,central_nodes,t,depth)
		else:
			for t in list_times:
				central_nodes = set(())
				for tau in range(depth):
					central_nodes = central_nodes.union(set(self.TN[t+tau].nodes))
				histo = self.Add_ETN(histo,central_nodes,t,depth)
		return histo

	#compute the ETN vector for various aggregation levels and fixed depth
	#assume that the current aggregation level is 1
	def ETN_vector(self,agg_max,depth):
		giant_vec = {agg:{} for agg in range(1,agg_max+1)}
		ETN_agg1 = self.ETN_histo(depth)
		#retain only the 20 most frequent ETN
		most_freq = sorted(ETN_agg1.keys(),key=lambda seq:ETN_agg1[seq],reverse=True)[:20]
		giant_vec[1] = {seq:ETN_agg1[seq] for seq in most_freq}
		for agg in range(2,agg_max+1):
			self.Get_TN(agg)
			dic = self.ETN_histo(depth)
			#retain only the 20 most frequent ETN
			most_freq = sorted(dic.keys(),key=lambda seq:dic[seq],reverse=True)[:20]
			giant_vec[agg] = {seq:dic[seq] for seq in most_freq}
		return giant_vec,ETN_agg1

	#dic_ETN is the result returned by self.ETN_histo
	#plot the number of distinct motifs containing n satellites vs n if choice=='sat'
	#plot the number of distinct motifs having w as weight vs w if choice=='weight'
	#plot the number of distinct motifs having nb vertical edges vs nb if choice=='edge'
	#plot the ETN weight distribution in the log-log plane :
	#use ETN_nb(depth,agg,choice='weight')
	def ETN_nb(self,dic_ETN,depth,choice='weight'):
		if choice=='weight':
			data = self.Weight_ETN(dic_ETN)
		elif choice=='sat':
			data = {}
			for seq in dic_ETN.keys():
				nb_sat = len(seq)//depth
				if nb_sat in data:
					data[nb_sat] += 1
				else:
					data[nb_sat] = 1
		elif choice=='edge':
			data = {}
			for seq in dic_ETN.keys():
				nb = sum([int(letter) for letter in seq])
				if nb in data:
					data[nb] += 1
				else:
					data[nb] = 1
		else:
			raise ValueError('choice should be either weight, sat or edge')
		return data

	#compute all the ETN observables
	def Get_ETN_obs(self):
		#compute the (2,1)-ETN
		ETN2_agg1 = self.ETN_histo(2)
		#ETN2_weight[n] = nb of motifs of depth 2 that have realized n times in the whole temporal network
		ETN2_weight = {}
		for n in ETN2_agg1.values():
			if n in ETN2_weight:
				ETN2_weight[n] += 1
			else:
				ETN2_weight[n] = 1

		#compute the (3,agg)-ETN for agg = 1,...,10
		giant_vec,ETN3_agg1 = self.ETN_vector(10,3)
		#ETN3_weight[n] = nb of motifs of depth 3 that have realized n times in the whole temporal network
		ETN3_weight = {}
		for n in ETN3_agg1.values():
			if n in ETN3_weight:
				ETN3_weight[n] += 1
			else:
				ETN3_weight[n] = 1
		return ETN2_weight,ETN3_weight,giant_vec

	#compute self.Distr_obs
	def Get_distr_obs(self):
		self.Distr_obs['cc_size'] = self.Get_cc_size
		self.Distr_obs['ETN'] = self.ETN_histo
		self.Distr_obs['ETN_nb'] = self.ETN_nb
		self.Distr_obs['edge_act_dur'] = 0
		self.Distr_obs['edge_act_inter'] = 0
		self.Distr_obs['node_act_dur'] = 0
		self.Distr_obs['node_act_inter'] = 0
		self.Distr_obs['edge_der'] = self.Edge_der
		self.Distr_obs['deg_inst'] = self.Get_inst_deg
		self.Distr_obs['edge_newborn'] = 0

#compute and save the distribution for instantaneous node degree
def Compute_distr_obs(name_obs,name,agg,arg=()):
	net = Temp_net(Load_TN(name))
	net.Init(agg)
	net.Get_distr_obs()
	histo = net.Distr_obs[name_obs](*arg)
	np.savetxt('figures/'+name_obs+'/codata/'+name+'_agg'+str(agg)+'.txt',np.array(list(zip(*histo.items()))),fmt='%d')

#compute and plot the distribution for instantaneous node degree
#together with the plots from XP data in the (X,Y) plane (lin,log)
def Plot_inst_deg(name,agg,XP_data):
	histo = {}
	#compute new data
	net = Temp_net(Load_TN(name))
	net.Init(agg)
	tab = net.Get_inst_deg()
	norm = float(np.sum(list(tab.values())))
	histo[name] = {key:float(val)/norm for key,val in tab.items()}

	#load XP data
	for XP_name in XP_data:
		tab = np.loadtxt('figures/deg_inst/codata/Compute_inst_deg_'+XP_name+'_agg'+str(agg)+'.txt',dtype=int)
		norm = float(np.sum(tab[1,:]))
		histo[XP_name] = {tab[0,i]:float(tab[1,i])/norm for i in range(np.size(tab,1))}

	#plot data
	xlabel = 'instantaneous node degree'
	ylabel = r"$\log_{10}(P)$"
	fig,ax = Setup_Plot(xlabel,ylabel)
	X,Y = zip(*histo[name].items())
	ax.plot(X,np.log10(Y),'^',color='red',label=name)
	for XP_name in XP_data:
		X,Y = zip(*histo[XP_name].items())
		ax.plot(X,np.log10(Y),'.',color='blue',alpha=0.5)
	ax.legend(fontsize=16)
	plt.savefig('figures/deg_inst/Plot_inst_deg_'+name+'_agg'+str(agg)+'.png')
