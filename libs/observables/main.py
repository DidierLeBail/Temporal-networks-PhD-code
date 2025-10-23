"""
obj_train is a list of time series. Each series is associated to the spatial part
of a given object in a temporal graph, and the values of the series give
the times of activity of this object.
Example: if an element of node_train is [1,23,24], this means that one node
has been active at times 1, 23 and 24 only.
For NCTN or ECTN, the definition is different (because otherwise makes no sense
for obs like the duration e.g.):
If an element of NCTN_train is [1,23,24], this means that their is one NCTN
(of given sequence) that is active in the temporal graph at times 1, 23 and 24 only.
However, these inclusions can share distinct spatial parts.

obj_to_space_weight is an integer-valued time series, where
obj_to_space_weight[t] = nb of inclusions of obj in the temporal graph at time t
For example:
node_to_space_weight[t] = nb of nodes active at t
NCTN_to_space_weight[t] = nb of NCTN active at t

obj_to_time_weight is a dictionary such that:
node_to_time_weight[i] = nb of activation times of the node i
obj_to_weight is a dictionary such that:
NCTN_to_weight[seq] = total nb of activation times of the NCTN seq (here the weight is the space-time weight)
"""
from typing import Literal, Union, List, Tuple
import numpy as np
import networkx as nx
import os, sys
ROOT_DIR = os.path.dirname(__file__)

for _ in range(2):
	ROOT_DIR = os.path.dirname(ROOT_DIR)
sys.path.append(ROOT_DIR)

from libs.temporal_network import Graph_interaction, Table_interaction

pass
# part of the API
def sample_motifs(temp_net,
	motif_type: Literal['NCTN', 'ECTN'],
	truncate: Union[int, None]=None
):
	pass

class Histog(dict):
	def __init__(self, *args, **kwargs):
		dict.__init__(self, *args, **kwargs)

	def increase_count(self, key):
		"""Increase by 1 the value dic[key]
		and add the non-existing key."""
		try:
			self[key] += 1
		except KeyError:
			self[key] = 1

	def to_exp(self):
		"""return X, np.log10(Y) after normalization of the histo (X, Y)"""
		X, Y = zip(*self.items())
		Y = np.asarray(Y, dtype=float)
		# normalization
		norm = np.sum(Y)
		Y[:] /= norm
		# transfo
		return X, np.log10(Y)

	def truncate(self, nb_of_items: Union[int, None]):
		"""Keep only the most frequent items in self.
		"""
		if nb_of_items is not None:
			return Histog(sorted(self.items(), key=lambda el: el[1])[:nb_of_items])
		return self

	def reduce(self):
		r"""Return the histogram of the nb of occurrences of self.
		(count the nb of occurrences of each value in self.values())
		"""
		res = Histog()
		for count in self.values():
			res.increase_count(count)
		return res

class Obj_obs:
	"""pass"""

	def __init__(self, *args, **kwargs):
		pass
	
	@classmethod
	def meth_builder(cls):
		"""add some methods at runtime"""
		def meth_body(self, name):
			raise AttributeError(f"method '{name}' not implemented for {self.__class__.__name__}")
		
		for name in [
			'duration',
			'interduration',
			'delayed_duration',
			'space_time_weight',
			'space_weight',
			'time_weight',
			'total_number',
			'diff_number'
		]:
			meth_name = "sample_" + name
			setattr(cls, meth_name, lambda self, temp_net: meth_body(self, name))

	def sample_inclusions(self, temp_net):
		pass

def deco_add_reduce(sample_fct):
	r"""Add the `reduce` keyword to `sample_fct`."""
	def new_fct(*args, reduce: bool = False, **kwargs):
		if reduce:
			return sample_fct(*args, **kwargs).reduce()
		else:
			return sample_fct(*args, **kwargs)
	new_fct.__name__ = sample_fct.__name__
	new_fct.__doc__ = sample_fct.__doc__
	return new_fct

@deco_add_reduce
def sample_space_time_weight(inclusions):
	r"""Returns the histogram of the space-time weight distribution.
	"""
	return Histog([(key, len(val)) for key, val in inclusions.items()])

@deco_add_reduce
def sample_time_weight(inclusions):
	r"""For each object (key of `inclusions`), count the number of times
	this object includes for a given space component.
	Said otherwise res[obj][space_comp] = nb of inclusions of obj with
	space_comp.

	If there is a single key in `inclusions`, and that this key is the empty string,
	return an histogram with a single level.
	Otherwise, return an histogram with 2 levels.
	"""
	res = Histog()
	try:
		for incl in inclusions[""]:
			res.increase_count(incl[0])
	except KeyError:
		for obj, incls in inclusions.items():
			for incl in incls:
				res.increase_count((obj, incl[0]))
	return res

@deco_add_reduce
def sample_space_weight(inclusions):
	r"""For each object (key of `inclusions`), res[obj][time_comp] = nb of inclusions of obj with
	time_comp.

	If there is a single key in `inclusions`, and that this key is the empty string,
	return an histogram with a single level.
	Otherwise, return an histogram with 2 levels.
	"""
	res = Histog()
	try:
		for incl in inclusions[""]:
			res.increase_count(incl[1])
	except KeyError:
		for obj, incls in inclusions.items():
			for incl in incls:
				res.increase_count((obj, incl[1]))
	return res

class NCTN_obs(Obj_obs):
	r"""Node-Centered Temporal Neighborhood.

	NCTN, originally named ETN for Egocentric Temporal Neighborhood,
	can be seen as a cylinder of radius 1, defined by a center and a height.
	
	Parameters
	----------
	depth : int
		height of the cylinder ; number of time steps
		we track interactions with the central node
	
	Attributes
	----------
	depth : int

	References
	----------
	These motifs are introduced in this paper.

	Examples
	--------
	pass
	"""

	def __init__(self, depth: int, *args, **kwargs):
		Obj_obs.__init__(self, *args, **kwargs)
		self.depth = depth
	
	def _NCTN_string(self,
		temp_net: Graph_interaction,
		central_node: int,
		starting_time: int
	):
		"""
		Parameters
		----------
		temp_net : Graph_interaction
			The temporal network in which we mine motifs for.
		inclusion : Tuple[int, int]
			Pair (v, t) where v is the central node of the inclusion
			and t its starting time.

		Returns
		-------
		nctn_str : str
			The string representration of the NCTN included in `temp_net`
			at `(v, t)`.
		"""
		# dic_s[u][tau] = '1' if u is a satellite of v at time t + tau and '0' else
		dic_s = {}
		for tau in range(self.depth):
			# if v is active (i.e. has at least one satellite)
			if central_node in temp_net[starting_time + tau]:
				for u in temp_net[starting_time + tau][central_node]:
					if u in dic_s:
						dic_s[u][tau] = '1'
					else:
						dic_s[u] = ['0'] * self.depth
						dic_s[u][tau] = '1'
		return ''.join(sorted([''.join(val) for val in dic_s.values()]))

	def sample_inclusions(self, temp_net: Graph_interaction):
		r"""Collect all maximal inclusions (number of occurrences) of NCTNs in `temp_net`.

		A maximal inclusion is defined by a space and a time component.
		In the case of NCTNs, the space component is the central node (cylinder center)
		and the time component is the starting time of the inclusion (cylinder base).
		"""
		# res[s] = [(i, t)], where s is the string representation of a NCTN
		# and (i, t) labels its maximal inclusions in `temp_net`
		inclusions = {}
		for t in range(temp_net.duration - self.depth + 1):
			central_nodes = set().union(
				*(set(temp_net[t + tau].nodes) for tau in range(self.depth))
			)
			for v in central_nodes:
				nctn_str = self._NCTN_string(temp_net, v, t)
				try:
					inclusions[nctn_str].append((v, t))
				except KeyError:
					inclusions[nctn_str] = [(v, t)]
		return inclusions

class Edge_obs(Obj_obs):
	r"""Contrary to NCTNs or ECTNs, there is only one edge (up to temporal graph isomorphism).

	Thus, the inclusions of an edge in a temporal network are all associated to the same
	object, which we will denote as the empty string '' or "".

	Notes
	-----
	- Sampling the time weight in a temporal network yields its fully weighted aggregated graph.
	- Sampling the inclusions yields the same data as the original temporal network.
	- Sampling the space weight yields the activity level of the network
	(nb of interactions across time).
	"""
	def sample_inclusions(self, temp_net: Graph_interaction):
		r"""An inclusion here is given by ((i, j), t),
		where i < j to ensure the unicity of the inclusion.
		"""
		inclusions = {"": []}
		for t, snapshot in enumerate(temp_net):
			inclusions[""].extend([(tuple(sorted(edge)), t) for edge in snapshot.edges])
		return inclusions

pass
#############################################################################
#values is produced e.g. via dic.values() and should contain integers
#returns histo, where histo[val] = nb of occurrences of val in values
def sample_values(values):
	histo = Histog()
	for val in values:
		histo.increase_data((val,))
	return histo

#TN is a temporal graph viewed as a list of graphs
def get_node_to_space_weight(TN):
	return [len(G.nodes()) for G in TN]

#TN is a temporal graph viewed as a list of graphs
def get_edge_to_space_weight(TN):
	return [len(G.edges()) for G in TN]

#TN is a temporal graph viewed as a list of graphs
def get_NCTN_to_space_weight(TN,depth):
	res = []
	pass
	return res

#TN is a temporal graph viewed as a list of graphs
def get_ECTN_to_space_weight(TN,depth):
	res = []
	pass
	return res

#returns an integer-valued histogram (called histo)
#returns the time_weight histo for edges or nodes but the space-time weight for NCTN or ECTN
#the input is obj_train or obj_to_time_weight; if obj==edge:
#histo[w] = nb of edges that have been active at exactly w time steps
#2 edges differ by their spatial part
#if obj==NCTN:
#histo[w] = nb of NCTN that have been active w times in total (space-time weight)
#2 NCTN differ by their string sequence
def natural_weight(obj_train=None,obj_to_weight=None):
	if obj_to_weight is not None:
		return sample_values(obj_to_weight.values())
	pass

#compute the unnormalized histogram of the duration
#from obj_train
def duration(obj_train):
	histo = {}
	for val in obj_train:
		for el in val:
			increase_dic(histo,(el[1]-el[0]+1,))
	return histo

#compute the unnormalized histogram of the newborn duration
#from obj_train
def newborn_duration(obj_train):
	histo = {}
	for val in obj_train:
		el = val[0]
		increase_dic(histo,(el[1]-el[0]+1,))
	return histo

#compute the unnormalized histogram of the interduration
#from obj_train
def interduration(obj_train):
	histo = {}
	for val in obj_train:
		for n in range(1,len(val)):
			increase_dic(histo,(val[n][0]-val[n-1][1]-1,))
	return histo

#compute the unnormalized histogram of the weak (delayed) event duration
#from obj_train
def weak_event_duration(obj_train,delay=4):
	histo = {}
	for val in obj_train:
		if len(val)==1:
			increase_dic(histo,(1,))
		else:
			#nb of consecutive events
			nb = 1
			for n in range(1,len(val)):
				interduration = val[n][0]-val[n-1][1]-1
				if interduration<delay:
					nb += 1
				else:
					increase_dic(histo,(nb,))
					nb = 1
	return histo

#compute the distribution of the size of connected components
#of the interaction graph TN (list of graphs)
def cc_size(TN):
	histo = {}
	for G in TN:
		for cc in nx.connected_components(G):
			increase_dic(histo,(len(cc),))
	return histo

def nb_tot(obj_to_time_weight):
	return sum(list(obj_to_time_weight.values()))
def nb_diff(obj_to_time_weight):
	return len(obj_to_time_weight)


#return node_train
#node_train[ind][k] = (t_0,t_f) with t_0 = starting time of the k^th event for node of identifier ind
#and t_f = ending time
def get_node_train(TN):
	node_event = {}
	active_nodes = set(TN[0].nodes())
	node_starting_time = {i:0 for i in active_nodes}
	for t in range(1,len(TN)):
		current_nodes = set(TN[t].nodes())
		#nodes active at t and not at t-1
		new_nodes = current_nodes.difference(active_nodes)
		#nodes active at t-1 and not at t
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
	#take care of the last timestamp
	for ind in active_nodes:
		if ind in node_event:
			node_event[ind].append((node_starting_time[ind],len(TN)-1))
		else:
			node_event[ind] = [(node_starting_time[ind],len(TN)-1)]
	return list(node_event.values())

#return edge_train
#edge_event[ind][k] = (t_0,t_f) with t_0 = starting time of the k^th event for edge of identifier ind
#and t_f = ending time
def get_edge_train(TN):
	edge_event = {}
	#edge_starting_time[ind] = last starting time of the activation of the edge of identifier ind
	edge_starting_time = {}
	#active_edges = set of edges active at current time
	active_edges = set(())
	for edge in TN[0].edges():
		if edge[0]<edge[1]:
			ind = edge
		else:
			ind = edge[::-1]
		active_edges.add(ind)
		edge_starting_time[ind] = 0
	for t in range(1,len(TN)):
		#edges active at t
		current_edges = set(())
		for edge in TN[t].edges():
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
			edge_event[ind].append((edge_starting_time[ind],len(TN)-1))
		else:
			edge_event[ind] = [(edge_starting_time[ind],len(TN)-1)]
	return list(edge_event.values())

def get_NCTN_train(TN,depth):
	pass
def get_ECTN_train(TN,depth):
	pass

#return the unnormalized histogram for instantaneous node degree
#where the degree zero is excluded
def node_deg(TN):
	histo = {}
	for G in TN:
		for i in G.nodes:
			increase_dic(histo,(G.degree(i),))
	return histo
