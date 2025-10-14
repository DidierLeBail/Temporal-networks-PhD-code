import numpy as np
import networkx as nx
import random as rd
from libs.settings import PROJECT_ROOT,Load_TN_ADM,Load_TN_Min_EW,Load_TN_min_ADM

'''
observables which depend on the TN at distinct states (n,b) are computed externally to the class
These are:
 - ETN autosimilarity
 - KS test of any distr obs btw different states
'''
#load TN data of a given name, this TN is not a randomization of another
def Load_TN_no_random(name):
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

#load TN data of a given name
def Load_TN(name):
	if type(name)==tuple:
		if type(name[-1])==str:
			if 'randomized' in name[-1]:
				if len(name)==2:
					data = Load_TN_no_random(name[0])
				else:
					data = Load_TN_no_random(name[:-1])
				#randomization type
				num_random = int(name[-1][10:])
				#strong time shuffling: the timeline of each edge is shuffled
				if num_random==1:
					#collect the number of times each edge is active, i.e. the time aggregated network
					time_agg = nx.Graph()
					for n in range(np.size(data,0)):
						i,j = data[n,1:]
						if time_agg.has_edge(i,j):
							time_agg[i][j]['weight'] += 1
						else:
							time_agg.add_edge(i,j,weight=1)
					#draw new activity times for each edge
					list_times = list(range(data[-1,0]+1)) #list of available times
					new_TN = {} #new_TN[t] = set of edges active at time t
					for edge in time_agg.edges:
						new_times = rd.sample(list_times,time_agg[edge[0]][edge[1]]['weight'])
						for t in new_times:
							if t in new_TN:
								new_TN[t].add(edge)
							else:
								new_TN[t] = {edge}
					#put new_TN under the form of a formatted measurement table
					new_data = []
					for num,t in enumerate(sorted(new_TN.keys())):
						for edge in new_TN[t]:
							new_data.append([num,*edge])
					return np.asarray(new_data,dtype=int)
				#weaker time shuffling: the global edge activity timeline is preserved
				#however, co-occurent edges before this shuffling may not be after
				elif num_random==2:
					#get data_time
					data_time = []
					n1 = 0; n_max = np.size(data,0)
					for n in range(1,n_max):
						t = data[n-1,0]
						if data[n,0]>t:
							data_time += [[n1,n]]
							n1 = n
					#take care of the last line of data
					t = data[-1,0]
					data_time += [[n1,n_max]]
					#shuffle the measurement table
					rng = np.random.default_rng()
					new_data = rng.shuffle(data)
					for t,el in enumerate(data_time):
						new_data[el[0]:el[1],0] = t
					return new_data
				elif num_random==3:
					pass
					#keep contact and intercontact the same ; i.e. shuffle edge events
				elif num_random==4:
					pass
					#reattribute the edge activity profiles among edges
			return Load_TN_no_random(name)
		return Load_TN_no_random(name)
	return Load_TN_no_random(name)

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
	"""
	def __init__(self,dataset):
		#allows to keep some info in memory to hopefully fasten computations
		self.memory = {}
		for obj in ['node','edge',('NCTN',3),('ECTN',3)]:
			self.memory[obj] = {}
			for el in ['train','time_weight']:
				self.memory[obj][el] = {}
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
		self.Get_val['avg_cc'] = self.Get_val_avg_cc
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
		#number of timestamps
		self.info = {}

	#replace self.data by its formatted version, i.e. time begins at 0, two consecutive times
	#are separated by one and nodes are numeroted from 0 to nb of nodes-1
	#and self-loops are removed
	def Format(self,duration=np.inf):
		self.Get_data_time()
		first_line = max(0,len(self.data_time)-duration)
		self.data = self.data[self.data_time[first_line][1]:,:]
		#remove any self-loop
		valid_lines = []
		for n in range(np.size(self.data,0)):
			if self.data[n,1]!=self.data[n,2]:
				valid_lines.append(n)
		self.data = self.data[valid_lines,:]
		#recompute data_time
		self.Get_data_time()
		#relabel the nodes
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
		for key1,val in self.memory.items():
			for key2 in val.keys():
				self.memory[key1][key2] = {}

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
				self.memory[key_mem] = {'train':{},'time_weight':{}}
				self.memory[key_mem][chunk_obj[1]] = self.Compute_object[chunk_obj[0]][chunk_obj[1]](*arg)
			elif not self.memory[key_mem][chunk_obj[1]]:
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

	#compute the average nb of cc per time step
	def Get_val_avg_cc(self):
		return np.mean([len(list(nx.connected_components(G))) for G in self.TN.values()])

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

	#sliding time shuffling (STS) with range b
	#followed by sliding time aggregation of level agg
	#modifies self.TN as well as self.data_time but data is preserved
	def Sliding_transfo(self,b,agg):
		#compute self.data_time
		self.Get_data_time()
		nb_time = len(self.data_time)
		#sliding time shuffling
		list_times = list(range(nb_time))
		for k in range(nb_time-b+1):
			list_times[k:k+b] = rd.sample(list_times[k:k+b],b)
		new_data_time = [[list_times[el[0]],*el[1:]] for el in self.data_time]
		self.data_time = sorted(new_data_time,key=lambda el:el[0])
		#sliding time aggregation
		new_nb = nb_time-agg+1
		#self.TN[t] = aggregated graph of interactions on t^th time interval
		self.TN = {t:nx.Graph() for t in range(new_nb)}
		#initial graph
		G = nx.Graph()
		for k in range(agg):
			for n in range(*self.data_time[k][1:]):
				i,j = self.data[n,1:]
				if G.has_edge(i,j):
					G[i][j]['weight'] += 1
				else:
					G.add_edge(i,j,weight=1)
		for t in range(new_nb-1):
			self.TN[t].add_edges_from(G.edges)
			#remove edges from time t
			for n in range(*self.data_time[t][1:]):
				i,j = self.data[n,1:]
				if G[i][j]['weight']==1:
					G.remove_edge(i,j)
				else:
					G[i][j]['weight'] -= 1
			#add edges from time t+agg
			for n in range(*self.data_time[t+agg][1:]):
				i,j = self.data[n,1:]
				if G.has_edge(i,j):
					G[i][j]['weight'] += 1
				else:
					G.add_edge(i,j,weight=1)
		self.TN[new_nb-1].add_edges_from(G.edges)
		self.info['T'] = new_nb

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

	#minimum intialization to get the temporal network at aggregation level agg
	def Init(self,agg=None):
		self.Get_data_time()
		self.info['N'] = len(set(self.data[:,1]).union(set(self.data[:,2])))
		if type(agg)==int:
			self.Get_TN(agg)

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
		new_nb = len(self.data_time)//agg
		#self.TN[t] = aggregated graph of interactions on t^th time interval
		self.TN = {t:nx.Graph() for t in range(new_nb)}
		for t in range(new_nb):
			for k in range(agg):
				n1,n2 = self.data_time[t*agg+k][1:]
				for n in range(n1,n2):
					self.TN[t].add_edge(*self.data[n,1:])
		#take care of the last aggregation window
		if new_nb*agg<len(self.data_time):
			self.TN[new_nb] = nx.Graph()
			for k in range(len(self.data_time)-agg*new_nb):
				n1,n2 = self.data_time[agg*new_nb+k][1:]
				for n in range(n1,n2):
					self.TN[new_nb].add_edge(*self.data[n,1:])
		self.info['T'] = len(self.TN)

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
	def ECTN_xpth_probas(self,depth,show_seq=True):
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
		if not show_seq:
			return abundancy,np.asarray(list_proba,dtype=float)
		return list_seq,abundancy,np.asarray(list_proba,dtype=float)

	#from dic_ETN deduce the th and XP relations btw a node centered motif and its time_weight
	def NCTN_xpth_probas(self,depth,show_seq=False):
		P_to_int = All_NCTN_profiles(depth)
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
		if show_seq:
			return list_seq,abundancy,np.asarray(list_proba,dtype=float)
		return abundancy,np.asarray(list_proba,dtype=float)

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
			if edge[0]<edge[1]:
				ind = edge
			else:
				ind = edge[::-1]
			active_edges.add(ind)
			edge_starting_time[ind] = 0
		for t in range(1,len(self.TN)):
			#edges active at t
			current_edges = set(())
			for edge in self.TN[t].edges:
				if edge[0]<edge[1]:
					ind = edge
				else:
					ind = edge[::-1]
				current_edges.add(ind)
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
				if edge[0]<edge[1]:
					ind = edge
				else:
					ind = edge[::-1]
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
				res[i] += 1
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
		#if no realization occurs, we put an impossible one
		if not histo:
			return {0.1:1}
		return histo

	def Weight_ETN(self,values):
		histo = {}
		for weight in values:
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
		#if no realization occurs, we put an impossible one
		if not histo:
			return {0.1:1}
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
		#if no realization occurs, we put an impossible one
		if not histo:
			return {0.1:1}
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
		#if no realization occurs, we put an impossible one
		if not histo:
			return {0.1:1}
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
		#if no realization occurs, we put an impossible one
		if not histo:
			return {0.1:1}
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
		i,j = ind
		#encode the activity profiles of the satellites: dic_s[u] = activity profile of satellite u
		dic_s_i = {}; dic_s_j = {}; dic_s = {}
		for tau in range(depth):
			if i in self.TN[t+tau]:
				ngh_i = set(self.TN[t+tau].neighbors(i))
			else:
				ngh_i = set(())
			if j in self.TN[t+tau]:
				ngh_j = set(self.TN[t+tau].neighbors(j))
			else:
				ngh_j = set(())
			for u in ngh_i:
				if u in dic_s_i:
					dic_s_i[u][tau] = '1'
				else:
					dic_s_i[u] = ['0']*depth
					dic_s_i[u][tau] = '1'
				if u in dic_s:
					dic_s[u][tau] = '1'
				else:
					dic_s[u] = ['0']*depth
					dic_s[u][tau] = '1'
			for u in ngh_j:
				if u in dic_s_j:
					dic_s_j[u][tau] = '1'
				else:
					dic_s_j[u] = ['0']*depth
					dic_s_j[u][tau] = '1'
				if u in dic_s:
					dic_s[u][tau] = '2'
				else:
					dic_s[u] = ['0']*depth
					dic_s[u][tau] = '2'
			for u in ngh_i.intersection(ngh_j):
				dic_s[u][tau] = '3'
		central_profile = ''.join(dic_s_i[j])
		del dic_s_i[j]; del dic_s_j[i]; del dic_s[i]; del dic_s[j]
		seq_i = ''.join(sorted([''.join(val) for val in dic_s_i.values()]))
		seq_j = ''.join(sorted([''.join(val) for val in dic_s_j.values()]))
		if seq_i>seq_j:
			#exchange '1' with '2'
			for u in dic_s:
				for tau in range(depth):
					if dic_s[u][tau]=='1':
						dic_s[u][tau] = '2'
					elif dic_s[u][tau]=='2':
						dic_s[u][tau] = '1'
		return central_profile+''.join(sorted([''.join(val) for val in dic_s.values()]))

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
		return None

	def Add_EdgeTN(self,histo,central_edges,t,depth):
		for ind in central_edges:
			seq = self.Compute_EdgeTNS(t,ind,depth)
			if seq in histo:
				histo[seq] += 1
			else:
				histo[seq] = 1
		return None

	#compute dic_EdgeTN alias dic_ECTN
	def EdgeTN_histo(self,depth):
		#histo[seq] = nb of occurrences of EdgeTN seq in the whole temporal network
		histo = {}; list_times = range(len(self.TN)-depth+1)
		#cumulate the interactions on depth time steps to keep track of the central edges
		dic_central_edges = {}
		for tau in range(depth):
			for edge in self.TN[tau].edges:
				if edge[0]<edge[1]:
					ind = edge
				else:
					ind = edge[::-1]
				if ind in dic_central_edges:
					dic_central_edges[ind] += 1
				else:
					dic_central_edges[ind] = 1
		self.Add_EdgeTN(histo,dic_central_edges.keys(),0,depth)
		for t in list_times[1:]:
			#remove the edges from time t-1
			for edge in self.TN[t-1].edges:
				if edge[0]<edge[1]:
					ind = edge
				else:
					ind = edge[::-1]
				dic_central_edges[ind] -= 1
			#add the edges from time t+depth-1
			for edge in self.TN[t+depth-1].edges:
				if edge[0]<edge[1]:
					ind = edge
				else:
					ind = edge[::-1]
				if ind in dic_central_edges:
					dic_central_edges[ind] += 1
				else:
					dic_central_edges[ind] = 1
			edges_to_remove = {ind for ind,nb in dic_central_edges.items() if nb<=0}
			for ind in edges_to_remove:
				del dic_central_edges[ind]
			#update the EdgeTN histogram
			self.Add_EdgeTN(histo,dic_central_edges.keys(),t,depth)
		return histo

	#compute the sequence of the motif starting at time t of central edge ind
	def Compute_EdgeTNS_without_rep(self,t,ind,depth):
		i,j = ind; set_nodes = {i,j}
		#encode the activity profiles of the satellites: dic_s[u] = activity profile of satellite u
		dic_s_i = {}; dic_s_j = {}; dic_s = {}
		for tau in range(depth):
			if i in self.TN[t+tau]:
				ngh_i = set(self.TN[t+tau].neighbors(i))
			else:
				ngh_i = set(())
			if j in self.TN[t+tau]:
				ngh_j = set(self.TN[t+tau].neighbors(j))
			else:
				ngh_j = set(())
			for u in ngh_i:
				if u in dic_s_i:
					dic_s_i[u][tau] = '1'
				else:
					dic_s_i[u] = ['0']*depth
					dic_s_i[u][tau] = '1'
					set_nodes.add(u)
				if u in dic_s:
					dic_s[u][tau] = '1'
				else:
					dic_s[u] = ['0']*depth
					dic_s[u][tau] = '1'
			for u in ngh_j:
				if u in dic_s_j:
					dic_s_j[u][tau] = '1'
				else:
					dic_s_j[u] = ['0']*depth
					dic_s_j[u][tau] = '1'
					set_nodes.add(u)
				if u in dic_s:
					dic_s[u][tau] = '2'
				else:
					dic_s[u] = ['0']*depth
					dic_s[u][tau] = '2'
			for u in ngh_i.intersection(ngh_j):
				dic_s[u][tau] = '3'
		central_profile = ''.join(dic_s_i[j])
		del dic_s_i[j]; del dic_s_j[i]; del dic_s[i]; del dic_s[j]
		seq_i = ''.join(sorted([''.join(val) for val in dic_s_i.values()]))
		seq_j = ''.join(sorted([''.join(val) for val in dic_s_j.values()]))
		if seq_i>seq_j:
			#exchange '1' with '2'
			for u in dic_s:
				for tau in range(depth):
					if dic_s[u][tau]=='1':
						dic_s[u][tau] = '2'
					elif dic_s[u][tau]=='2':
						dic_s[u][tau] = '1'
		return (tuple(sorted(list(set_nodes))),central_profile+''.join(sorted([''.join(val) for val in dic_s.values()])))

	def Add_EdgeTN_without_rep(self,histo,central_edges,t,depth):
		real_ECTN = set(())
		for ind in central_edges:
			real_ECTN.add(self.Compute_EdgeTNS_without_rep(t,ind,depth))
		for el in real_ECTN:
			seq = el[1]
			if seq in histo:
				histo[seq] += 1
			else:
				histo[seq] = 1
		return None

	def ECTN_without_rep(self,depth):
		#histo[seq] = nb of occurrences of EdgeTN seq in the whole temporal network
		histo = {}; list_times = range(len(self.TN)-depth+1)
		#cumulate the interactions on depth time steps to keep track of the central edges
		dic_central_edges = {}
		for tau in range(depth):
			for edge in self.TN[tau].edges:
				if edge[0]<edge[1]:
					ind = edge
				else:
					ind = edge[::-1]
				if ind in dic_central_edges:
					dic_central_edges[ind] += 1
				else:
					dic_central_edges[ind] = 1
		self.Add_EdgeTN_without_rep(histo,dic_central_edges.keys(),0,depth)
		for t in list_times[1:]:
			#remove the edges from time t-1
			for edge in self.TN[t-1].edges:
				if edge[0]<edge[1]:
					ind = edge
				else:
					ind = edge[::-1]
				dic_central_edges[ind] -= 1
			#add the edges from time t+depth-1
			for edge in self.TN[t+depth-1].edges:
				if edge[0]<edge[1]:
					ind = edge
				else:
					ind = edge[::-1]
				if ind in dic_central_edges:
					dic_central_edges[ind] += 1
				else:
					dic_central_edges[ind] = 1
			edges_to_remove = {ind for ind,nb in dic_central_edges.items() if nb<=0}
			for ind in edges_to_remove:
				del dic_central_edges[ind]
			#update the EdgeTN histogram
			self.Add_EdgeTN_without_rep(histo,dic_central_edges.keys(),t,depth)
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
				if edge[0]<edge[1]:
					ind = edge
				else:
					ind = edge[::-1]
				central_edges.add(ind)
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

	#compute dic_ETN alias dic_NCTN
	def ETN_histo(self,depth):
		#histo[seq] = nb of occurrences of ETN seq in the whole temporal network
		histo = {}
		for t in range(len(self.TN)-depth+1):
			central_nodes = set(())
			for tau in range(depth):
				central_nodes = central_nodes.union(set(self.TN[t+tau].nodes))
			self.Add_ETN(histo,central_nodes,t,depth)
		return histo

	#compute the sequence of the motif starting at time t of central node v
	def Compute_ETNS_without_rep(self,t,v,depth):
		#dic_s[u][tau] = '1' if u is a satellite of v at time t+tau and '0' else
		dic_s = {}; set_nodes = {v}
		for tau in range(depth):
			#if v is active (i.e. has at least one satellite)
			if v in self.TN[t+tau]:
				for u in self.TN[t+tau][v]:
					if u in dic_s:
						dic_s[u][tau] = '1'
					else:
						dic_s[u] = ['0']*depth
						dic_s[u][tau] = '1'
						set_nodes.add(u)
		return (tuple(sorted(list(set_nodes))),''.join(sorted([''.join(val) for val in dic_s.values()])))

	def Add_ETN_without_rep(self,histo,central_nodes,t,depth):
		real_NCTN = set(())
		for v in central_nodes:
			real_NCTN.add(self.Compute_ETNS_without_rep(t,v,depth))
		for el in real_NCTN:
			seq = el[1]
			if seq in histo:
				histo[seq] += 1
			else:
				histo[seq] = 1
		return None

	#compute dic_ETN alias dic_NCTN without repetitions of the same sub-temporal graph 
	def NCTN_without_rep(self,depth):
		histo = {}
		for t in range(len(self.TN)-depth+1):
			central_nodes = set(())
			for tau in range(depth):
				central_nodes = central_nodes.union(set(self.TN[t+tau].nodes))
			self.Add_ETN_without_rep(histo,central_nodes,t,depth)
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

	pass
	#compute dic_NCTN but takes into account only the isolated motifs
	def NCTN_isolated(self,depth):
		#histo[seq] = nb of occurrences of ETN seq in the whole temporal network
		histo = {}
		for t in range(1,len(self.TN)-depth):
			central_nodes = set(())
			for tau in range(depth):
				central_nodes = central_nodes.union(set(self.TN[t+tau].nodes))
			remove_node = set(())
			for node in central_nodes:
				if node in self.TN[t-1] or node in self.TN[t+depth]:
					remove_node.add(node)
			self.Add_ETN(histo,central_nodes.difference(remove_node),t,depth)
		return histo

	#compute dic_ECTN but takes into account only the isolated motifs
	def ECTN_isolated(self,depth):
		#histo[seq] = nb of occurrences of EdgeTN seq in the whole temporal network
		histo = {}; list_times = range(len(self.TN)-depth+1)
		#cumulate the interactions on depth time steps to keep track of the central edges
		dic_central_edges = {}
		for tau in range(depth):
			for edge in self.TN[tau].edges:
				if edge[0]<edge[1]:
					ind = edge
				else:
					ind = edge[::-1]
				if ind in dic_central_edges:
					dic_central_edges[ind] += 1
				else:
					dic_central_edges[ind] = 1
		self.Add_EdgeTN(histo,dic_central_edges.keys(),0,depth)
		for t in list_times[1:]:
			#remove the edges from time t-1
			for edge in self.TN[t-1].edges:
				if edge[0]<edge[1]:
					ind = edge
				else:
					ind = edge[::-1]
				dic_central_edges[ind] -= 1
			#add the edges from time t+depth-1
			for edge in self.TN[t+depth-1].edges:
				if edge[0]<edge[1]:
					ind = edge
				else:
					ind = edge[::-1]
				if ind in dic_central_edges:
					dic_central_edges[ind] += 1
				else:
					dic_central_edges[ind] = 1
			edges_to_remove = {ind for ind,nb in dic_central_edges.items() if nb<=0}
			for ind in edges_to_remove:
				del dic_central_edges[ind]
			#update the EdgeTN histogram
			self.Add_EdgeTN(histo,dic_central_edges.keys(),t,depth)
		return histo
