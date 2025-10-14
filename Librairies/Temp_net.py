from os.path import join
import numpy as np
import networkx as nx
import random as rd

from Librairies import atn
from Librairies.utils import ROOT_DIR
ADM_DIR = join(ROOT_DIR,'data/ADM')
DATA_DIR = join(ROOT_DIR,'data')

#return info about the dataset of savename equal to ref_name
#these info are used e.g. to tune ADM models
def load_XP_info(ref_name):
	global_info = np.loadtxt(join(ADM_DIR,'ref_data/'+ref_name+'/global_info.txt'),dtype=str,delimiter=',')
	XP_info = {}
	for i in range(len(global_info[0,:])):
		x = global_info[0,i]; y = global_info[1,i]
		if x in {'N','T','nb of edges'}:
			XP_info[x] = int(y)
		else:
			XP_info[x] = float(y)
	return XP_info

class TN_name:
	"""docstring for TN_name."""

	def __init__(self):
		#savename is how the TN should be called (it is a string)
		self.savename = ""
		self.versions = {}
		self.get_savename()
		self.get_versions()

	#return the tij data
	def load_TN(self):
		pass

	#savename is a string
	def get_savename(self):
		pass

	#initialize self.versions
	def get_versions(self):
		pass

class ADM_name(TN_name):
	"""docstring for ADM_name."""

	def __init__(self,version_nb,ref_name):
		#gives the ADM model considered
		self.version_nb = version_nb
		#savename of the dataset used to tune the ADM model (give the model instance)
		self.ref_name = ref_name

		TN_name.__init__(self)

	def get_savename(self):
		self.savename = 'ADM'+str(self.version_nb)+self.ref_name

	def load_TN(self):
		#load instance parameters
		dic_param = self.load_instance_param()

		#load XP info
		XP_info = load_XP_info(self.ref_name)

		#generate the model instance
		model = atn.ADM_class(XP_info,**self.versions[self.version_nb])
		for param in model.free_param.keys():
			model.free_param[param] = dic_param[param]
		model.refresh()
		return model.evolve()

	def get_versions(self):
		#version 1 = basis version
		#versions 2 to 13 are adjacent to 1
		#versions 14 and beyond are combinations of multiple adjacent versions

		# basis version :
		# m_{i}, a_{i}, alpha_{i}, contextual interactions are neutral, c_{i,j} used, constant egonet growth,
		# removal of edges depending on their weight, Alpha update process
		self.versions[1] = {
			'm': 'random',
			'a':'power',
			'update': 'alpha,i',
			'context': 'neutral',
			'c_ij': True,
			'egonet_growth': 'cst',
			'remove': 'edge'
		}
		for k in range(2, 20):
			self.versions[k] = {**self.versions[1]}

		# 2nd version: linear reinforcement process and no gradual decay
		self.versions[2]['update'] = 'linear'

		# 3rd version: random removal of nodes
		self.versions[3]['remove'] = 'node'

		# 4th version: varying egonet growth
		self.versions[4]['egonet_growth'] = 'var'

		# 5th version: c_{i,j} = 1
		self.versions[5]['c_ij'] = False

		# 6th version: intentional and contextual interactions are equivalent
		self.versions[6]['context'] = 'equivalent'

		# 7th version: contextual interactions are pure noise
		self.versions[7]['context'] = 'noise'

		# 8th version: no contextual interactions
		self.versions[8]['context'] = None

		# 9th version: alpha
		self.versions[9]['update'] = 'alpha'

		# 10th version: alpha_{i}, beta_{i}
		self.versions[10]['update'] = 'alpha,beta,i'

		# 11th version: alpha_{i,j}, beta_{i,j}
		self.versions[11]['update'] = 'alpha,beta,ij'

		# 12th version: a
		self.versions[12]['a'] = 'cst'

		# 13th version: m
		self.versions[13]['m'] = 'cst'

		# 14th version: ori_ADM, i.e. the following combination : versions 2+3+4+5+8+13
		self.versions[14] = {
			**self.versions[14],
			**{'update': 'linear', 'm': 'cst', 'context': None, 'c_ij': False, 'egonet_growth': 'var', 'remove': 'node'}
		}

		# 15th version: 2+5+8+13 (linear reinforcement process and linear decay process)
		self.versions[15] = {
			**self.versions[15],
			**{'update': 'linear', 'm': 'cst', 'context': None, 'c_ij': False}
		}

		# 16th version: 5+8+11+13 (transitive initialization for weights of social ties)
		self.versions[16] = {
			**self.versions[16],
			**{'update': 'alpha,beta,ij', 'm': 'cst', 'context': None, 'c_ij': False}
		}

		# 17th version: 3+5+8+9+12+13 (simplest version with exponential Hebbian process)
		self.versions[17] = {
			**self.versions[17],
			**{'update': 'alpha', 'm': 'cst', 'a':'cst', 'context': None, 'c_ij': False, 'remove': 'node'}
		}

		# 18th version: 2+3+5+8+12+13 (simplest version with linear Hebbian process)
		self.versions[18] = {
			**self.versions[18],
			**{'update': 'linear', 'm': 'cst', 'a':'cst', 'context': None, 'c_ij': False, 'remove': 'node'}
		}
		self.versions[18] = {}

		# 19th version: 7+9+13 (best expected version)
		self.versions[19] = {
			**self.versions[19],
			**{'update': 'alpha', 'm': 'cst', 'context': 'noise'}
		}

	# return the ADM parameters tuned wrt the dataset of savename equal to ref_name
	def load_instance_param(self):
		model = 'ADM_class_V'+str(self.version_nb)
		set_int_param = {'m_max','m','c'}
		dic_param = {}

		#load parameters
		best_param = np.loadtxt(join(ADM_DIR,'models/'+model+'/'+self.ref_name+'/best_param.txt'),dtype=str,delimiter=',')
		for i,param in enumerate(best_param[0,:]):
			if param in set_int_param:
				dic_param[param] = int(best_param[1,i])
			else:
				dic_param[param] = float(best_param[1,i])
		return dic_param

class EW_name(TN_name):
	"""docstring for EW_name."""

	def __init__(self, arg):
		TN_name.__init__(self)
		self.version_nb = version_nb

	def get_savename(self):
		self.savename = 'min_EW'+str(self.version_nb)

	def get_versions(self):
		#if shift=True and removal'=None then
		#the number of temporal edges is O(duree**2) instead of O(duree)
		#so the analysis is barely doable. One way to do it would be to parallelize the computation of
		#observables by using the CPT clusters
		#On the contrary, if shift=False and 'removal'!=None then
		#the data set will only contain newborn activations in the stationary state
		self.versions[1] = {'shift': False, 'removal': None, 'newborn': 'random'}
		self.versions[2] = {'shift': True, 'removal': 'node_unif', 'newborn': 'random'}
		self.versions[3] = {'shift': True, 'removal': 'edge_unif', 'newborn': 'random'}

	def load_TN(self):
		return atn.Min_EW(138, 3635, 0.79, **self.versions[self.version_nb]).evolve()

class XP_name(TN_name):
	"""formatted empirical TN"""

	def __init__(self,ref_name):
		self.ref_name = ref_name
		TN_name.__init__(self)

	def load_TN(self):
		return np.loadtxt(join(DATA_DIR,'empirical/'+self.savename+'.txt'),dtype=int)

	def get_savename(self):
		self.savename = self.ref_name

class Min_ADM_name(TN_name):
	"""docstring for Min_ADM_name."""

	def __init__(self,version_nb):
		#gives the ADM model considered
		self.version_nb = version_nb
		TN_name.__init__(self)

	def get_savename(self):
		self.savename = 'min_ADM'+str(self.version_nb)

	def get_versions(self):
		self.versions = {'m':'random','a':'power','update':'alpha,i','context':'noise','c_ij':True,'egonet_growth':'cst','remove':'edge'}

	def load_TN(self):
		#load XP info of conf16
		XP_info = {'N':138,'T':3635,'nb of edges':153371,'sigma':0.34,'mu':-0.56}
		#modifies the version
		self.versions['m'] = 'cst'
		self.versions['a'] = 'cst'
		self.versions['c_ij'] = False
		self.versions['update'] = 'linear'
		self.versions['context'] = None
		self.versions['remove'] = 'node'
		#decide of the parameters
		dic_param = {}
		dic_param['a'] = 0.3
		dic_param['m'] = 1
		if self.version_nb==2:
			p_d = 0.1
		elif self.version_nb==1:
			p_d = 0.02
		dic_param['p_d'] = p_d
		dic_param['p_u'] = 1
		dic_param['p_g'] = 0.08498
		#generate the model instance
		model = atn.ADM_class(XP_info,**version)
		for param in model.free_param.keys():
			model.free_param[param] = dic_param[param]
		model.refresh()
		return model.evolve()

class Temp_net:
	"""
    input: t_ij table
	"""
	def __init__(self,t_ij):
		self.data = t_ij.copy()
		#nb of nodes
		self.N = len(set(self.data[:,1]).union(set(self.data[:,2])))
		#interaction temporal network
		self.TN = []
		#data_time[i] = [t_i, n_1, n_p] with t_i the (i+1)th time appearing in data
		#n_1 the first line of occurrence of t_i and n_p-1 the last one
		self.data_time = []

	#compute the interaction graph at aggregation level agg (sliding aggregation)
	def get_TN(self,agg=1,data_time=True):
		#compute self.data_time
		if data_time:
			self.get_data_time()
		nb_time = len(self.data_time)
		new_nb = nb_time-agg+1
		#self.TN[t] = aggregated graph of interactions on t^th time interval
		self.TN = [nx.Graph() for _ in range(new_nb)]
		#initial graph
		G = nx.Graph()
		for k in range(agg):
			for n in range(*self.data_time[k]):
				i,j = self.data[n,1:]
				if G.has_edge(i,j):
					G[i][j]['weight'] += 1
				else:
					G.add_edge(i,j,weight=1)
		for t in range(new_nb-1):
			self.TN[t].add_edges_from(G.edges)
			#remove edges from time t
			for n in range(*self.data_time[t]):
				i,j = self.data[n,1:]
				if G[i][j]['weight']==1:
					G.remove_edge(i,j)
				else:
					G[i][j]['weight'] -= 1
			#add edges from time t+agg
			for n in range(*self.data_time[t+agg]):
				i,j = self.data[n,1:]
				if G.has_edge(i,j):
					G[i][j]['weight'] += 1
				else:
					G.add_edge(i,j,weight=1)
		self.TN[new_nb-1].add_edges_from(G.edges)
	
	#compute the number of timestamps, nodes and temporal edges
	#as well as the parameters of the node activity distribution viewed
	#as the exponential of a two sided truncated Gaussian variable
	def get_info(self):
		info = {}
		info['T'] = len(self.TN)
		info['nb of edges'] = len(self.data)
		info['N'] = self.N
		activity = np.zeros(self.N,dtype=float)
		for events in self.TN:
			for node in events.nodes:
				activity[node] += 1
		info['a_min'] = np.min(activity)/info['T']
		info['a_max'] = np.max(activity)/info['T']
		tab = np.log10(activity)-np.log10(info['T'])
		values,bins = np.histogram(tab,density=True)
		#estimate the parameters mu and sigma
		bin_min = len(bins)-3
		while values[bin_min]>values[-1]:
			bin_min -= 1
		mu_tab = [el for el in tab if el>=bins[bin_min]]
		info['mu'] = np.mean(mu_tab)
		info['sigma'] = 2*np.sqrt(np.var(mu_tab))
		return info

	#compute self.data_time
	def get_data_time(self):
		self.data_time = []
		n1 = 0; n_max = np.size(self.data,0)
		for n in range(1,n_max):
			if self.data[n,0]>self.data[n-1,0]:
				self.data_time.append([n1,n])
				n1 = n
		#take care of the last line of data
		self.data_time.append([n1,n_max])

	#replace self.data by its formatted version, i.e. time begins at 0, two consecutive times
	#are separated by one and nodes are numeroted from 0 to nb of nodes-1
	def format(self,max_T=np.inf):
		self.get_data_time()
		last_line = min(max_T,len(self.data_time))
		self.data = self.data[:self.data_time[last_line-1][1],:]
		#remove any self-loop
		valid_lines = []
		for n in range(np.size(self.data,0)):
			if self.data[n,1]!=self.data[n,2]:
				valid_lines.append(n)
		self.data = self.data[valid_lines,:]
		#recompute data_time
		self.get_data_time()
		#relabel the nodes
		node_to_int = {}; num = 0
		for n in range(np.size(self.data,0)):
			for k in self.data[n,1:]:
				if not k in node_to_int:
					node_to_int[k] = num
					num += 1
		for t,el in enumerate(self.data_time[:last_line]):
			self.data[el[0]:el[1],0] = t
		for n in range(np.size(self.data,0)):
			for k in range(1,3):
				self.data[n,k] = node_to_int[self.data[n,k]]

	#compute the interaction graph at aggregation level agg (sliding aggregation)
	def sliding_time_aggregation(self,agg,data_time=True):
		#compute self.data_time
		if data_time:
			self.get_data_time()
		nb_time = len(self.data_time)
		new_nb = nb_time-agg+1
		#self.TN[t] = aggregated graph of interactions on t^th time interval
		self.TN = [nx.Graph() for _ in range(new_nb)]
		#initial graph
		G = nx.Graph()
		for k in range(agg):
			for n in range(*self.data_time[k]):
				i,j = self.data[n,1:]
				if G.has_edge(i,j):
					G[i][j]['weight'] += 1
				else:
					G.add_edge(i,j,weight=1)
		for t in range(new_nb-1):
			self.TN[t].add_edges_from(G.edges)
			#remove edges from time t
			for n in range(*self.data_time[t]):
				i,j = self.data[n,1:]
				if G[i][j]['weight']==1:
					G.remove_edge(i,j)
				else:
					G[i][j]['weight'] -= 1
			#add edges from time t+agg
			for n in range(*self.data_time[t+agg]):
				i,j = self.data[n,1:]
				if G.has_edge(i,j):
					G[i][j]['weight'] += 1
				else:
					G.add_edge(i,j,weight=1)
		self.TN[new_nb-1].add_edges_from(G.edges)

	#return the fully aggregated network (weighted or not)
	def fully_agg(self,is_weighted=False):
		static_net = nx.Graph()
		if is_weighted:
			for events in self.TN:
				for edge in events.edges():
					if static_net.has_edge(*edge):
						static_net[edge[0]][edge[1]]['weight'] += 1
					else:
						static_net.add_edge(*edge,weight=1)
		else:
			for events in self.TN:
				static_net.add_edges_from(events.edges())
		return static_net
