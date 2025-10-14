import numpy as np
import networkx as nx
import random as rd
from Librairies.utils import increase_dic

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

#values is produced e.g. via dic.values() and should contain integers
#returns histo, where histo[val] = nb of occurrences of val in values
def sample_values(values):
	histo = {}
	for val in values:
		increase_dic(histo,(val,))
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
